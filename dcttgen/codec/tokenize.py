"""python -m dcttgen.codec.tokenize --config C [--override a.b=value ...] [--shard i/n]

Writes data/codes/<codec.tag>/<clip_id>.npy (contract 7.3) for every clip of manifest/all.jsonl.
Safe to re-run (finished clips are skipped) and to run as n shards at once, one per GPU: --shard 0/8 ... --shard 7/8."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dcttgen.codec.codec import claim_tag_dir


def tokenize(cfg, codec, rows, read_audio, shard: tuple[int, int] = (0, 1)) -> tuple[int, int]:
    """rows: manifest rows. read_audio(row) -> Float[duration * sample_rate] mono. Returns (written, skipped)."""
    K, V, fr, sr = codec.num_codebooks, codec.codebook_size, codec.frame_rate, codec.sample_rate
    if V > 2 ** 15:
        raise ValueError("codes are stored as int16 (contract 7.3): codebook_size must not exceed 32768")
    d = Path(cfg.paths.data_root) / "codes" / cfg.codec.tag
    claim_tag_dir(d, {"fingerprint": codec.fingerprint(), "tag": cfg.codec.tag, "K": K, "V": V})   # refuses another RVQ's directory
    written = skipped = 0
    for i, row in enumerate(rows):
        if i % shard[1] != shard[0]:
            continue
        out = d / f"{row['clip_id']}.npy"
        if out.exists():
            skipped += 1
            continue
        wav = read_audio(row)
        if wav.ndim != 1 or wav.numel() != row["duration"] * sr:
            raise ValueError(f"{row['clip_id']}: {tuple(wav.shape)} samples, expected ({row['duration'] * sr},)")
        codes = codec.encode(wav[None])[0]                                       # Long[K, T]
        if tuple(codes.shape) != (K, row["duration"] * fr) or int(codes.min()) < 0 or int(codes.max()) >= V:
            raise ValueError(f"{row['clip_id']}: the codec returned {tuple(codes.shape)}, expected {(K, row['duration'] * fr)} with values in [0, {V})")
        tmp = d / f".{row['clip_id']}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:                                               # a crash never leaves a half-written .npy
            np.save(f, codes.cpu().numpy().astype(np.int16))
        os.replace(tmp, out)
        written += 1
    return written, skipped


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip starting at i")
    args = ap.parse_args(argv)
    import soundfile as sf
    from dcttgen.codec.codec import Codec
    from dcttgen.config import load_config      # chapter 04
    cfg = load_config(args.config, args.override)
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit("--shard must be i/n with 0 <= i < n")
    root = Path(cfg.paths.data_root)
    rows = [json.loads(line) for line in (root / "manifest" / "all.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    codec = Codec.load(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"), decoder=False)
    read = lambda row: torch.from_numpy(sf.read(str(root / row["audio"]), dtype="float32")[0])
    written, skipped = tokenize(cfg, codec, rows, read, (i, n))
    print(f"shard {i}/{n}: wrote {written}, skipped {skipped} existing")


if __name__ == "__main__":
    main()
