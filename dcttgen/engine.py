"""The one training loop (contract section 9). fit() trains any nn.Module whose forward(batch: dict) returns a dict with "loss".

All step counts are OPTIMISER steps; gradient accumulation only changes how many micro-batches make one step.
"""
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, GradientAccumulationPlugin, set_seed

_STEP_DIR = re.compile(r"step_(\d{7})$")


def lr_at(step: int, t) -> float:
    """D10. Learning rate of the optimiser step about to run; step = optimiser steps already taken (0-based)."""
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps  # linear warm-up: lr/warmup_steps ... lr
    progress = min(1.0, (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps))
    return t.lr_min + 0.5 * (t.lr - t.lr_min) * (1.0 + math.cos(math.pi * progress))  # cosine: lr -> lr_min


def make_optimizer(model, t) -> torch.optim.AdamW:
    """AdamW with the values of plan section 3.6. Weight decay on matrices and embeddings (ndim >= 2), none on biases and norm gains."""
    params = [p for p in model.parameters() if p.requires_grad]  # model.parameters() lists a tied tensor once
    groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": t.weight_decay},
              {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=t.lr, betas=tuple(t.betas), eps=t.eps)


def latest_checkpoint(run_dir) -> str | None:
    found = [(int(m.group(1)), p) for p in Path(run_dir).glob("step_*") if (m := _STEP_DIR.match(p.name))]
    return str(max(found)[1]) if found else None


def load_weights(module: torch.nn.Module, checkpoint: str) -> None:
    """Load <checkpoint>/model.safetensors (contract 7.4) into an unwrapped module, strictly.

    Use this and not module.load_state_dict(load_file(...)): save_state drops duplicate tied tensors (lm_head = embed_tokens),
    and a checkpoint written from a DistributedDataParallel model carries a "module." prefix on every key.
    """
    from safetensors.torch import load_file, load_model, save_file

    path = Path(checkpoint) / "model.safetensors"
    state = load_file(str(path))
    if state and all(k.startswith("module.") for k in state):
        tmp = path.with_name("model.unwrapped.safetensors")
        save_file({k[len("module."):]: v for k, v in state.items()}, str(tmp), metadata={"format": "pt"})
        path = tmp
    try:
        load_model(module, str(path), strict=True)  # knows which missing keys are tied aliases
    finally:
        if path.name == "model.unwrapped.safetensors":
            path.unlink()


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def _set_rng_state(s) -> None:
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2])
    if s[3] is not None:
        torch.cuda.set_rng_state_all(s[3])


def _add(sums: dict, out: dict, device) -> None:
    for k, v in out.items():
        v = torch.as_tensor(v, dtype=torch.float32, device=device).detach().mean()
        sums[k] = sums.get(k, 0.0) + v


def _mean(acc: Accelerator, sums: dict, n: int) -> dict:
    """Mean over n micro-batches and over processes, with one collective call."""
    keys = sorted(sums)
    vec = acc.reduce(torch.stack([torch.as_tensor(sums[k]) for k in keys]) / n, reduction="mean")
    return dict(zip(keys, vec.tolist()))


def _log(acc: Accelerator, path: Path, rec: dict) -> None:
    if acc.is_main_process:
        line = json.dumps(rec)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def evaluate(acc: Accelerator, model, loader, seed: int) -> dict:
    """Mean of every scalar the model returns. A fixed RNG stream makes the numbers comparable between checkpoints
    (the RF loss draws random times and noise) and, being forked, leaves the training RNG stream untouched."""
    model.eval()
    sums, n = {}, 0
    devices = [acc.device.index] if acc.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in loader:
            _add(sums, model(batch), acc.device)
            n += 1
    model.train()
    # ponytail: mean of per-batch means, not token-weighted; even_batches may repeat a few samples - return a count and weight if it matters
    return {"val_" + k: v for k, v in _mean(acc, sums, n).items()} if n else {}


def _save(acc: Accelerator, run: Path, step: int, epoch: int, seen: int, keep_last: int) -> None:
    """Write run/step_<7 digits>/ atomically: into a temporary directory first, renamed once complete (a crash never
    leaves a half-written step_* directory for the next resume to pick up). Every process takes part in save_state."""
    final, tmp = run / f"step_{step:07d}", run / f".tmp_step_{step:07d}"
    acc.wait_for_everyone()
    acc.save_state(str(tmp))  # model.safetensors, optimizer.bin, random_states_<rank>.pkl (+ scaler.pt for fp16)
    acc.wait_for_everyone()
    if acc.is_main_process:
        (tmp / "trainer_state.json").write_text(json.dumps({"step": step, "epoch": epoch, "batch_in_epoch": seen}))
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        for old in sorted(run.glob("step_*"))[:-keep_last]:  # keep_last = 0: the slice is empty, everything is kept
            shutil.rmtree(old)
    acc.wait_for_everyone()


def fit(model: torch.nn.Module, train_loader, val_loader, cfg, run_dir: str) -> None:
    """Train model for cfg.train.max_steps optimiser steps; resumes from the newest step_* directory in run_dir.

    train_loader: a torch DataLoader over a map-style dataset whose order depends only on the epoch (fit calls set_epoch).
    val_loader: the same kind, or None.
    """
    t, run = cfg.train, Path(run_dir)
    bad = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:  # eps = 1e-9 underflows to 0 in float16, and bf16 weights lose small updates: keep float32 master weights
        raise TypeError(f"trainable parameters must be float32 ({bad[0]} is not); set train.mixed_precision for half precision")
    acc = Accelerator(
        mixed_precision=t.mixed_precision,  # explicit argument beats ACCELERATE_MIXED_PRECISION from accelerate launch
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=t.grad_accum, sync_with_dataloader=False),
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=t.seed),
    )
    set_seed(t.seed, device_specific=True)
    opt = make_optimizer(model, t)
    model, opt, train_loader = acc.prepare(model, opt, train_loader)
    if val_loader is not None:
        val_loader = acc.prepare(val_loader)
    model.train()

    if acc.is_main_process:
        run.mkdir(parents=True, exist_ok=True)
        for stale in run.glob(".tmp_step_*"):
            shutil.rmtree(stale, ignore_errors=True)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    acc.wait_for_everyone()
    step = epoch = seen = 0  # optimiser steps done; epoch; micro-batches already consumed in this epoch
    ckpt = latest_checkpoint(run)
    if ckpt:
        acc.load_state(ckpt)  # model, optimizer, RNG streams of every process, GradScaler
        s = json.loads((Path(ckpt) / "trainer_state.json").read_text())
        step, epoch, seen = s["step"], s["epoch"], s["batch_in_epoch"]
        acc.print(f"resumed from {ckpt}")
    rng = _rng_state() if ckpt and seen else None  # the RNG exactly as it was when the checkpoint was written

    log, sums, n_micro, t_last, s_last = run / "log.jsonl", {}, 0, time.time(), step
    while step < t.max_steps:
        train_loader.set_epoch(epoch)
        start = seen
        for batch in (acc.skip_first_batches(train_loader, seen) if seen else train_loader):
            if rng is not None:  # first batch after a resume: starting this iterator drew a worker seed from the torch RNG,
                _set_rng_state(rng)  # which the uninterrupted run did not draw here, so put the RNG back as saved
                rng = None
            seen += 1
            with acc.accumulate(model):  # no gradient all-reduce on the non-final micro-batches
                out = model(batch)  # always the wrapped module itself, never a method: DDP and autocast hook forward()
                acc.backward(out["loss"])  # divides by grad_accum
                if acc.sync_gradients:  # last micro-batch of the window: this call performs the optimiser step
                    gnorm = acc.clip_grad_norm_(model.parameters(), t.grad_clip or float("inf"))
                    for g in opt.param_groups:
                        g["lr"] = lr_at(step, t)
                opt.step()  # no-ops until sync_gradients
                opt.zero_grad()
            _add(sums, out, acc.device)
            n_micro += 1
            if not acc.sync_gradients:
                continue
            step += 1
            hook = getattr(acc.unwrap_model(model), "on_step_end", None)
            if hook:
                hook(step)  # e.g. the EMA update of the RF Transformer
            last = step >= t.max_steps
            if last or step % t.log_every == 0 or step % t.eval_every == 0 or step % t.save_every == 0:
                rec = {"split": "train", "step": step, "epoch": epoch, "lr": lr_at(step - 1, t), "grad_norm": float(gnorm),
                       "seconds_per_step": (time.time() - t_last) / (step - s_last), **_mean(acc, sums, n_micro)}
                loss = rec["loss"]
                if not math.isfinite(loss):  # every process sees the same reduced value, so all raise together
                    raise FloatingPointError(f"loss is {loss} at step {step}; the last checkpoint is intact")
                _log(acc, log, rec)
                sums, n_micro, t_last, s_last = {}, 0, time.time(), step
            if val_loader is not None and (last or step % t.eval_every == 0):
                _log(acc, log, {"split": "val", "step": step, **evaluate(acc, model, val_loader, t.seed)})
            if last or step % t.save_every == 0:
                _save(acc, run, step, epoch, seen, t.keep_last)
            if last:
                break
        else:  # the loader ran dry: next epoch
            if start == 0 and seen == 0:
                raise RuntimeError("train_loader yielded no batches")
            if rng is not None:  # the checkpoint was written at the last batch of its epoch: nothing was left to skip
                _set_rng_state(rng)
                rng = None
            epoch, seen = epoch + 1, 0
