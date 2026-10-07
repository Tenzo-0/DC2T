"""python -m dc2t.lm.train --config C --phase pretrain|finetune [--override a.b=value ...] [--name N] [--probe]

Builds the vocabulary, the model and the two loaders, then hands them to the engine's fit() (chapter 04). Nothing here trains by itself."""
from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dc2t.lm.data import PHASES, ClipDataset, TokenBudgetSampler, collate
from dc2t.lm.model import load_lm
from dc2t.lm.sequence import build_document
from dc2t.lm.vocab import Vocab
from dc2t.plan import INSTRUMENTS, Plan, plan_sections


def make_loader(cfg, vocab: Vocab, split: str, phase: str, shuffle: bool) -> DataLoader:
    ds = ClipDataset(cfg, vocab, split, phase)
    sampler = TokenBudgetSampler(ds.lengths, cfg.lm.max_tokens, shuffle)      # the engine calls loader.batch_sampler.set_epoch()
    return DataLoader(ds, batch_sampler=sampler, collate_fn=partial(collate, pad_id=vocab.pad), num_workers=cfg.lm.num_workers)


def probe(cfg, vocab: Vocab, model, seconds: int = 300, steps: int = 3) -> None:
    """Speed and peak memory of one training step on a synthetic document (random codes, the longest legal plan and caption)."""
    dev = next(model.parameters()).device
    sections = plan_sections(seconds)
    plan = Plan(120, seconds, sections, ["calm"], list(INSTRUMENTS))
    ids, labels = build_document(vocab, torch.randint(0, vocab.V, (vocab.K, seconds * vocab.frame_rate)), sections, "word " * 200, plan)
    batch = {k: v.to(dev) for k, v in collate([{"input_ids": ids, "labels": labels}], vocab.pad).items()}
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)       # lr 0: the weights stay as they are, the optimiser state is allocated as in training
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = model(batch)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if dev.type == "cuda" else "n/a (CPU)"
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters | {ids.numel()} tokens | {dt:.2f} s/step | "
          f"{ids.numel() / dt:,.0f} tokens/s | peak GPU memory {peak}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value", help="repeatable; e.g. lm.checkpoint=runs/lm_pretrain/<name>/step_0010000")
    ap.add_argument("--name", help="run directory name; default <codec.tag>-<backbone>")
    ap.add_argument("--probe", action="store_true", help="time one training step on a synthetic 300 s document and exit")
    args = ap.parse_args(argv)
    from dc2t.config import load_config      # chapter 04
    from dc2t.engine import fit              # chapter 04
    cfg = load_config(args.config, args.override)
    vocab = Vocab.build(cfg)
    model = load_lm(cfg, vocab)
    if args.probe:
        probe(cfg, vocab, model.to("cuda" if torch.cuda.is_available() else "cpu"))
        return
    train, val = make_loader(cfg, vocab, "train", args.phase, True), make_loader(cfg, vocab, "val", args.phase, False)
    longest = max(train.dataset.lengths + val.dataset.lengths)
    if longest > model.lm.config.max_position_embeddings:
        raise SystemExit(f"the longest document has {longest} tokens but {cfg.lm.backbone} positions only {model.lm.config.max_position_embeddings}; "
                         f"set lm.max_seconds or use a backbone with a longer context")
    run_dir = Path(cfg.paths.runs) / f"lm_{args.phase}" / (args.name or f"{cfg.codec.tag}-{Path(cfg.lm.backbone).name.lower()}")
    if os.environ.get("RANK", "0") == "0":      # `accelerate launch` sets RANK; only one process writes the vocabulary files
        vocab.save(run_dir)
    fit(model, train, val, cfg, str(run_dir))


if __name__ == "__main__":
    main()
