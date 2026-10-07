"""python -m dc2t.codec.train --config C [--override a.b=value ...] [--name N] [--fit-norm]

Trains the K x V RVQ and the Rectified Flow Transformer together, on frozen MuEncoder features and frozen Mel-VAE latents
(plan 3.4, 3.6). Run it once with --fit-norm (one process), then without it (any number of processes, through fit())."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dc2t.codec.codec import VAE_SR, Released, to_rate
from dc2t.codec.rf import LatentNorm, RFConfig, RFTransformer, pair_frames, rf_loss, sample_t


class CodecTrainer(nn.Module):
    """forward(batch) -> loss dict (contract 9). Its state-dict prefixes are the ones Codec.load reads: rvq., rf., ema., norm.
    The frozen MuEncoder / Mel-VAE stack sits in a plain list, so it is not saved, not optimised and not wrapped by DDP."""

    def __init__(self, cfg, rvq: nn.Module, released):
        super().__init__()
        self.rvq = rvq                                              # trainable: MuCodec's ResidualVectorQuantize with K and V from config
        self.rf = RFTransformer(RFConfig.from_cfg(cfg.rf))          # trainable
        self.ema = copy.deepcopy(self.rf).requires_grad_(False)     # what Codec.decode uses
        self.norm = LatentNorm()                                    # buffers, filled by fit_latent_norm before training
        self._frozen = [released]
        r = cfg.rf
        self.V, self.cond_drop, self.ema_decay, self.w_commit, self.w_codebook = cfg.codec.codebook_size, r.cond_drop, r.ema_decay, r.w_commit, r.w_codebook

    def forward(self, batch: dict) -> dict:
        """batch["wav48"]: Float[B, W * 1920], mono 48 kHz windows of W = codec.window_frames code frames."""
        released, wav = self._frozen[0], batch["wav48"]
        with torch.no_grad():
            feats = released.features(wav).float()                      # Float[B, 1024, W]      frozen MuEncoder
            x0 = self.norm.normalize(released.latents(wav).float())     # Float[B, 16, W/2, 32]  frozen Mel-VAE, standardised
        zq, codes, _, commit, codebook, *_ = self.rvq(feats)            # zq Float[B, 1024, W] (straight-through), codes Long[B, K, W]
        cond = pair_frames(zq.transpose(1, 2))                          # Float[B, W/2, 2048]: code frames 2i, 2i+1 -> latent frame i
        B = x0.shape[0]
        drop = torch.rand(B, device=x0.device) < self.cond_drop if self.training else None   # teach the null condition (guidance)
        rf = rf_loss(self.rf, x0, cond, sample_t(B).to(x0.device), torch.randn_like(x0), drop)
        out = {"loss": rf + self.w_commit * commit + self.w_codebook * codebook,
               "rf": rf.detach(), "commit": commit.detach(), "codebook": codebook.detach()}
        for k in range(codes.shape[1]):                                 # per-level perplexity inside this batch: a collapsing codebook shows here first
            p = torch.bincount(codes[:, k].reshape(-1), minlength=self.V).float()
            p = p / p.sum()
            out[f"ppl_k{k}"] = torch.exp(-(p * p.clamp_min(1e-12).log()).sum())
        return out

    @torch.no_grad()
    def on_step_end(self, step: int) -> None:                           # fit() calls this after every optimiser step
        for e, p in zip(self.ema.parameters(), self.rf.parameters()):
            e.lerp_(p, 1.0 - self.ema_decay)


@torch.no_grad()
def fit_latent_norm(norm: LatentNorm, released, loader, batches: int) -> None:
    """Per-(channel, mel-bin) mean and std of the Mel-VAE latents over `batches` batches (MuCodec's Feature2DProcessor)."""
    s, s2, n = 0.0, 0.0, 0
    for _, batch in zip(range(batches), loader):
        x = released.latents(batch["wav48"].to(released.device)).float().cpu()      # Float[B, 16, L, 32]
        s, s2, n = s + x.mean((0, 2)), s2 + x.pow(2).mean((0, 2)), n + 1           # each Float[16, 32]
    if n == 0:
        raise ValueError("the loader gave no batches")
    mean = s / n
    norm.mean.copy_(mean[:, None])
    norm.std.copy_((s2 / n - mean ** 2).clamp_min(1e-8).sqrt()[:, None])


class WindowDataset(Dataset):
    """One window of codec.window_frames code frames per clip, as mono 48 kHz audio: {"wav48": Float[W * 1920]}.
    The window starts on the code-frame grid, at a random frame (fixed=True: frame 0, for validation and statistics).
    A clip shorter than the window is zero-padded at the end; Codec.encode pads the same way."""

    def __init__(self, cfg, split: str, fixed: bool = False):
        self.root = Path(cfg.paths.data_root)
        text = (self.root / "manifest" / f"{split}.jsonl").read_text(encoding="utf-8")
        self.rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.sr, self.fr, self.W, self.fixed = cfg.audio.sample_rate, cfg.audio.frame_rate, cfg.codec.window_frames, fixed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        import soundfile as sf
        row, spf = self.rows[i], self.sr // self.fr
        T = row["duration"] * self.fr
        start = 0 if self.fixed or T <= self.W else int(torch.randint(0, T - self.W + 1, ()))
        y, sr = sf.read(str(self.root / row["audio"]), start=start * spf, frames=min(self.W, T) * spf, dtype="float32")
        if sr != self.sr or y.ndim != 1:
            raise ValueError(f"{row['audio']}: expected mono {self.sr} Hz, got {sr} Hz with shape {y.shape}")
        y = F.pad(torch.from_numpy(y), (0, self.W * spf - len(y)))
        return {"wav48": to_rate(y, self.sr, VAE_SR)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--name", help="run directory name; default codec.tag")
    ap.add_argument("--fit-norm", action="store_true", help="compute the latent statistics, write latent_norm.pt and exit")
    args = ap.parse_args(argv)
    from dc2t.config import load_config      # chapter 04
    from dc2t.engine import fit              # chapter 04
    cfg = load_config(args.config, args.override)
    t = cfg.train
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    released = Released.load(cfg, device)
    del released.mu.model.cfm_wrapper           # the released flow transformer is not used on this path
    model = CodecTrainer(cfg, released.new_rvq(cfg.codec.num_codebooks, cfg.codec.codebook_size), released)
    run_dir = Path(cfg.paths.runs) / "codec" / (args.name or cfg.codec.tag)
    stats = run_dir / "latent_norm.pt"
    loader = lambda split, fixed: DataLoader(WindowDataset(cfg, split, fixed), batch_size=t.batch_size, shuffle=not fixed,
                                             num_workers=t.num_workers, drop_last=not fixed)
    if args.fit_norm:
        run_dir.mkdir(parents=True, exist_ok=True)
        fit_latent_norm(model.norm, released, loader("train", True), cfg.rf.norm_batches)
        torch.save(model.norm.state_dict(), stats)
        print(f"wrote {stats}: mean {model.norm.mean.mean():.3f}, std {model.norm.std.mean():.3f}")
        return
    if not stats.is_file():
        raise SystemExit(f"{stats} is missing: run this command once with --fit-norm before training")
    model.norm.load_state_dict(torch.load(stats))
    fit(model, loader("train", False), loader("val", True), cfg, str(run_dir))


if __name__ == "__main__":
    main()
