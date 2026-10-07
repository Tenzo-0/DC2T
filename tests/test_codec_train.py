"""CPU tests for the codec training module and the tokenisation job. The frozen MuCodec stack and the RVQ are small stand-ins
with the released shapes and return signatures. Run: pytest -q tests/test_codec_train.py   or   python tests/test_codec_train.py"""
import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dc2t.codec.tokenize import tokenize
from dc2t.codec.train import CodecTrainer, fit_latent_norm

ns = types.SimpleNamespace
K, V, W = 2, 8, 8  # codebooks, codebook size, code frames per window


def make_cfg(root=".", **rf):
    return ns(paths=ns(data_root=str(root)), audio=ns(sample_rate=32000, frame_rate=25),
              codec=ns(tag="k2v8", num_codebooks=K, codebook_size=V, window_frames=W),
              rf=ns(**{**dict(width=32, depth=1, heads=2, cond_drop=0.5, ema_decay=0.9, w_commit=0.25, w_codebook=1.0), **rf}))


class Frozen:  # stands in for codec.Released: features Float[B, 1024, T], latents Float[B, 16, T / 2, 32] with mean 3 and std 2
    device = torch.device("cpu")

    def features(self, x48):
        B, N = x48.shape
        return x48.view(B, N // 1920, 1920).mean(-1)[:, None, :] * torch.arange(1, 1025).view(1, -1, 1)

    def latents(self, x48):
        B, N = x48.shape
        return 3.0 + 2.0 * x48.view(B, N // 3840, 3840)[:, :, :512].reshape(B, N // 3840, 16, 32).permute(0, 2, 1, 3)


class TinyRVQ(nn.Module):  # a trainable residual quantiser with the 6-tuple that MuCodec's ResidualVectorQuantize.forward returns
    def __init__(self):
        super().__init__()
        self.inp, self.out, self.books = nn.Linear(1024, 4), nn.Linear(4, 1024), nn.Parameter(torch.randn(K, V, 4))

    def forward(self, z):  # z Float[B, 1024, T]
        e = self.inp(z.transpose(1, 2))
        res, q, codes, commit, codebook = e, 0, [], 0, 0
        for k in range(K):
            idx = torch.cdist(res, self.books[k].expand(len(z), -1, -1)).argmin(-1)
            qk = self.books[k][idx]
            commit, codebook = commit + F.mse_loss(res, qk.detach()), codebook + F.mse_loss(qk, res.detach())
            q, res, codes = q + res + (qk - res).detach(), res - qk.detach(), codes + [idx]
        return self.out(q).transpose(1, 2), torch.stack(codes, 1), None, commit, codebook, None


def batch(B=4, seed=0):
    return {"wav48": torch.randn(B, W * 1920, generator=torch.Generator().manual_seed(seed))}


def test_forward_trains_rvq_and_rf_and_leaves_the_frozen_stack_out():
    torch.manual_seed(0)
    m = CodecTrainer(make_cfg(), TinyRVQ(), Frozen())
    fit_latent_norm(m.norm, Frozen(), [batch(8, s) for s in range(20)], batches=20)
    assert abs(m.norm.mean.mean() - 3) < 0.05 and abs(m.norm.std.mean() - 2) < 0.05          # the statistics of Frozen.latents
    out = m(batch())
    assert out["loss"].ndim == 0 and out["loss"].requires_grad and set(out) == {"loss", "rf", "commit", "codebook", "ppl_k0", "ppl_k1"}
    assert abs(out["rf"] - 2.0) < 0.2            # zero-initialised output, unit-variance target: E|eps - x0|^2 = 2 exactly when the latents are standardised
    assert all(1.0 <= out[f"ppl_k{k}"] <= V for k in range(K))
    out["loss"].backward()
    grads = {n.split(".")[0] for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads == {"rvq", "rf"}                # the RF loss reaches the RVQ through the condition; the EMA copy gets no gradient
    assert {k.split(".")[0] for k in m.state_dict()} == {"rvq", "rf", "ema", "norm"}        # the prefixes Codec.load reads; no frozen weights
    assert all(not p.requires_grad for p in m.ema.parameters())
    m.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        a = m(batch())["loss"]
        torch.manual_seed(1)
        assert torch.equal(a, m(batch())["loss"])                                              # validation is repeatable under a seed


def test_ema_follows_the_trained_weights():
    m = CodecTrainer(make_cfg(), TinyRVQ(), Frozen())
    with torch.no_grad():
        m.rf.out.weight.fill_(1.0)
    m.on_step_end(1)
    assert torch.allclose(m.ema.out.weight, torch.full_like(m.ema.out.weight, 0.1))            # (1 - 0.9) of the way
    m.on_step_end(2)
    assert torch.allclose(m.ema.out.weight, torch.full_like(m.ema.out.weight, 0.19))


def test_a_few_steps_reduce_the_loss():
    torch.manual_seed(0)
    m = CodecTrainer(make_cfg(width=64, cond_drop=0.0), TinyRVQ(), Frozen())
    fit_latent_norm(m.norm, Frozen(), [batch(8, s) for s in range(4)], batches=4)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-3)
    hist = []
    for step in range(60):
        loss = m(batch(8, step))["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        m.on_step_end(step + 1)
        hist.append(loss.item())
    assert sum(hist[-10:]) / 10 < 0.9 * sum(hist[:10]) / 10, (hist[:3], hist[-3:])


class FakeCodec:
    num_codebooks, codebook_size, frame_rate, sample_rate = K, V, 25, 32000

    def __init__(self, fp="aaaa", bad=False):
        self.fp, self.bad, self.calls = fp, bad, 0

    def fingerprint(self):
        return self.fp

    def encode(self, wav):  # Float[1, N] -> Long[1, K, N / 1280]
        self.calls += 1
        T = wav.shape[1] // 1280 - self.bad
        return (torch.arange(K * T).view(1, K, T) + int(wav.abs().sum() > 0)) % V


def test_tokenize_writes_contract_files_is_resumable_and_refuses_another_rvq():
    rows = [{"clip_id": f"c{i}", "duration": 30 + i, "audio": f"audio/c{i}.flac"} for i in range(5)]
    read = lambda row: torch.zeros(row["duration"] * 32000)
    with tempfile.TemporaryDirectory() as d:
        cfg, codec = make_cfg(d), FakeCodec()
        assert tokenize(cfg, codec, rows, read, (0, 2)) == (3, 0) and tokenize(cfg, codec, rows, read, (1, 2)) == (2, 0)   # two shards
        out = Path(d) / "codes" / "k2v8"
        assert sorted(p.name for p in out.glob("*.npy")) == [f"c{i}.npy" for i in range(5)] and not list(out.glob(".*.tmp"))
        for row in rows:
            c = np.load(out / f"{row['clip_id']}.npy")
            assert c.dtype == np.int16 and c.shape == (K, 25 * row["duration"]) and c.min() >= 0 and c.max() < V
        assert json.loads((out / "meta.json").read_text())["fingerprint"] == "aaaa"
        assert tokenize(cfg, codec, rows, read) == (0, 5) and codec.calls == 5                       # a re-run encodes nothing
        for bad_call in (lambda: tokenize(cfg, FakeCodec("bbbb"), rows, read),                      # tokens of another RVQ live here
                         lambda: tokenize(make_cfg(Path(d) / "x"), FakeCodec(bad=True), rows, read),  # the codec returned T - 1 frames
                         lambda: tokenize(make_cfg(Path(d) / "y"), codec, rows, lambda row: torch.zeros(7))):   # the audio has the wrong length
            try:
                bad_call()
                raise AssertionError("expected a refusal")
            except (RuntimeError, ValueError):
                pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_codec_train: {len(tests)} tests passed")
