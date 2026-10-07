"""Run: pytest -q tests/test_engine.py   or   python tests/test_engine.py   (needs accelerate; CPU only)"""
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dc2t.config import ROOT, load_config  # noqa: E402
from dc2t.engine import fit, latest_checkpoint, load_weights, lr_at, make_optimizer  # noqa: E402

BASE = str(ROOT / "configs" / "base.yaml")
V, D, T = 17, 8, 9


class Seqs(Dataset):
    """Token sequences obeying next = (cur + 3) % V: learnable by a bigram model, and a pure function of the index."""

    def __init__(self, n=20):
        self.start = torch.arange(n) % V

    def __len__(self):
        return len(self.start)

    def __getitem__(self, i):
        return {"ids": (self.start[i] + 3 * torch.arange(T)) % V}


class TinyLM(nn.Module):
    """Tied embedding and head (like Qwen2.5), dropout (so that RNG restoration matters), an on_step_end hook."""

    def __init__(self, dropout=0.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb, self.mix, self.norm, self.drop = nn.Embedding(V, D), nn.Linear(D, D), nn.LayerNorm(D), nn.Dropout(dropout)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight
        self.calls, self.crash_at, self.nan_at = [], None, None

    def forward(self, batch):
        ids = batch["ids"]
        logits = self.head(self.drop(self.norm(torch.tanh(self.mix(self.emb(ids[:, :-1]))))))
        loss = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1))
        if self.nan_at is not None and len(self.calls) + 1 == self.nan_at:
            loss = loss * float("nan")
        return {"loss": loss, "acc": (logits.argmax(-1) == ids[:, 1:]).float().mean()}

    def on_step_end(self, step):
        self.calls.append(step)
        if step == self.crash_at:
            raise KeyboardInterrupt("simulated crash")


def cfg_for(*overrides):
    return load_config(BASE, ["train.mixed_precision=no", "train.log_every=1", "train.eval_every=1000", "train.save_every=1000",
                              "train.warmup_steps=2", *overrides])


def loader(n=20, batch_size=4, shuffle=True):
    return DataLoader(Seqs(n), batch_size=batch_size, shuffle=shuffle)


def records(run, split="train"):
    lines = (Path(run) / "log.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["split"] == split]


def weights(run, step):
    return load_file(str(Path(run) / f"step_{step:07d}" / "model.safetensors"))


def tmp():
    return Path(tempfile.mkdtemp())


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{exc.__name__} not raised")


def test_lr_boundaries():
    t = cfg_for("train.warmup_steps=10", "train.max_steps=110").train
    close = lambda a, b: abs(a - b) < 1e-12  # noqa: E731
    assert close(lr_at(0, t), 3e-4 / 10) and close(lr_at(9, t), 3e-4) and close(lr_at(10, t), 3e-4)  # warm-up ends at the peak
    assert close(lr_at(60, t), (3e-4 + 3e-5) / 2)  # halfway through the cosine
    assert close(lr_at(110, t), 3e-5) and close(lr_at(10**6, t), 3e-5)  # D10: the floor is lr_min, and it stays there
    assert all(lr_at(s, t) >= lr_at(s + 1, t) for s in range(10, 110))
    assert close(lr_at(0, cfg_for("train.warmup_steps=0").train), 3e-4)


def test_weight_decay_groups():
    m = TinyLM()
    m.mix.bias.requires_grad = False  # a frozen tensor is in no group
    groups = make_optimizer(m, cfg_for().train).param_groups
    decay, none = groups
    assert decay["weight_decay"] == 0.1 and none["weight_decay"] == 0.0
    assert {id(p) for p in decay["params"]} == {id(m.emb.weight), id(m.mix.weight)}  # tied tensor listed once
    assert {id(p) for p in none["params"]} == {id(m.norm.weight), id(m.norm.bias)}  # norm gain and bias, no decay
    assert groups[0]["betas"] == (0.9, 0.999) and groups[0]["eps"] == 1e-9


def test_loss_falls_and_checkpoint_layout():
    run = tmp()
    cfg = cfg_for("train.max_steps=30", "train.grad_accum=1", "train.save_every=10", "train.lr=0.03", "train.lr_min=0.003")
    fit(TinyLM(), loader(), None, cfg, run)
    rec = records(run)
    assert len(rec) == 30 and rec[-1]["loss"] < 0.5 * rec[0]["loss"]  # the model learns next = cur + 3
    assert [r["lr"] for r in rec[:3]] == [lr_at(s, cfg.train) for s in range(3)]  # the logged lr is the schedule
    steps = sorted(p.name for p in Path(run).glob("step_*"))
    assert steps == ["step_0000020", "step_0000030"]  # step_0000010 pruned by keep_last=2
    for name in ("model.safetensors", "optimizer.bin", "trainer_state.json", "random_states_0.pkl"):
        assert (Path(run) / "step_0000030" / name).exists(), name
    assert json.loads((Path(run) / "step_0000030" / "trainer_state.json").read_text())["step"] == 30
    assert not list(Path(run).glob(".tmp_step_*")) and (Path(run) / "config.yaml").exists()
    assert latest_checkpoint(run) == str(Path(run) / "step_0000030")


def test_resume_is_exact_across_an_epoch_boundary():
    # 10 samples, batch 4 -> 3 micro-batches per epoch; grad_accum 2 -> accumulation windows straddle epoch boundaries.
    # save_every=3 writes the checkpoint at the last batch of epoch 1; save_every=2 writes it one batch into epoch 1.
    for save_every, crash_at in ((3, 4), (2, 3)):
        cfg = cfg_for("train.max_steps=6", "train.grad_accum=2", f"train.save_every={save_every}", "train.lr=0.03")
        straight, crashed = tmp(), tmp()
        fit(TinyLM(dropout=0.2), loader(10), None, cfg, straight)
        broken = TinyLM(dropout=0.2)
        broken.crash_at = crash_at  # dies after the checkpoint was written, while the next step is in flight
        assert raises(KeyboardInterrupt, fit, broken, loader(10), None, cfg, crashed) == "simulated crash"
        assert latest_checkpoint(crashed).endswith(f"step_{save_every:07d}")
        fit(TinyLM(dropout=0.2, seed=99), loader(10), None, cfg, crashed)  # new objects, other init: everything comes from disk
        a, b = weights(straight, 6), weights(crashed, 6)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)  # bit-identical, dropout masks included
        assert [r["step"] for r in records(crashed)] == [1, 2, 3, 4, 5, 6]  # no gap, no repeated step
        assert [r["loss"] for r in records(straight)] == [r["loss"] for r in records(crashed)]


def test_accumulation_equals_the_big_batch():
    one, two = tmp(), tmp()
    fit(TinyLM(), loader(8, 8, shuffle=False), None, cfg_for("train.max_steps=3", "train.grad_accum=1", "train.save_every=3"), one)
    fit(TinyLM(), loader(8, 4, shuffle=False), None, cfg_for("train.max_steps=3", "train.grad_accum=2", "train.save_every=3"), two)
    a, b = weights(one, 3), weights(two, 3)
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)  # same update from 2 x 4 as from 1 x 8


def test_on_step_end_counts_optimiser_steps_not_batches():
    m = TinyLM()
    fit(m, loader(), None, cfg_for("train.max_steps=5", "train.grad_accum=2", "train.save_every=5"), tmp())
    assert m.calls == [1, 2, 3, 4, 5]  # the hook runs on the unwrapped module, once per optimiser step


def test_validation_leaves_the_training_stream_alone():
    cfg = cfg_for("train.max_steps=6", "train.grad_accum=1", "train.save_every=6", "train.eval_every=2")
    with_val, without = tmp(), tmp()
    fit(TinyLM(dropout=0.2), loader(10), loader(8, shuffle=False), cfg, with_val)
    fit(TinyLM(dropout=0.2), loader(10), None, cfg, without)
    a, b = weights(with_val, 6), weights(without, 6)
    assert all(torch.equal(a[k], b[k]) for k in a)  # validation drew random numbers, but from a forked stream
    val = records(with_val, "val")
    assert [r["step"] for r in val] == [2, 4, 6] and all("val_loss" in r and "val_acc" in r for r in val)


def test_half_precision_parameters_are_refused():
    msg = raises(TypeError, fit, TinyLM().half(), loader(), None, cfg_for("train.max_steps=1"), tmp())
    assert "float32" in msg
    assert str(torch.tensor(1e-9, dtype=torch.float16).item()) == "0.0"  # why: Adam eps vanishes in float16


def test_non_finite_loss_stops_the_run_with_the_last_checkpoint_intact():
    run, m = tmp(), TinyLM()
    m.nan_at = 7
    msg = raises(FloatingPointError, fit, m, loader(), None, cfg_for("train.max_steps=10", "train.grad_accum=1", "train.save_every=5"), run)
    assert "step 7" in msg and sorted(p.name for p in Path(run).glob("step_*")) == ["step_0000005"]


def test_an_empty_loader_is_an_error_not_an_endless_loop():
    assert "no batches" in raises(RuntimeError, fit, TinyLM(), DataLoader(Seqs(0), batch_size=4), None, cfg_for("train.max_steps=2"), tmp())


def test_checkpoints_load_back_into_a_tied_model_even_with_a_ddp_prefix():
    run = tmp()
    fit(TinyLM(), loader(), None, cfg_for("train.max_steps=3", "train.grad_accum=1", "train.save_every=3"), run)
    ckpt = Path(latest_checkpoint(run))
    saved = load_file(str(ckpt / "model.safetensors"))
    assert "head.weight" not in saved and "emb.weight" in saved  # save_state stores a tied tensor once ...
    assert "head.weight" in raises(RuntimeError, TinyLM().load_state_dict, saved)  # ... so a plain load_state_dict fails
    fresh = TinyLM(seed=5)
    load_weights(fresh, str(ckpt))
    assert fresh.head.weight is fresh.emb.weight and torch.equal(fresh.emb.weight, saved["emb.weight"])
    prefixed = tmp()  # what an 8-GPU run writes if accelerate saves the DDP wrapper: every key starts with "module."
    save_file({"module." + k: v for k, v in saved.items()}, str(prefixed / "model.safetensors"), metadata={"format": "pt"})
    other = TinyLM(seed=6)
    load_weights(other, str(prefixed))
    assert torch.equal(other.mix.weight, fresh.mix.weight) and not (prefixed / "model.unwrapped.safetensors").exists()


def test_overfit_one_batch():
    batch = {"ids": torch.stack([Seqs()[i]["ids"] for i in range(4)])}
    run = tmp()
    fit(TinyLM(), DataLoader([{"ids": r} for r in batch["ids"]], batch_size=4), None,
        cfg_for("train.max_steps=150", "train.grad_accum=1", "train.save_every=150", "train.lr=0.05", "train.lr_min=0.01"), run)
    rec = records(run)
    assert rec[0]["loss"] > 2.0 and rec[-1]["loss"] < 0.05 and rec[-1]["acc"] == 1.0  # ln(17) = 2.83 -> memorised


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{Path(__file__).name}: {len(tests)} passed")
