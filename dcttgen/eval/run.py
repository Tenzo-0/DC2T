"""python -m dcttgen.eval.run --config C --out runs/eval/NAME --stage codes|audio|score [--override a.b=value ...]

codes (language-model environment): one code matrix per validation prompt -> OUT/codes/<clip_id>.npy, and OUT/prompts.json
audio (codec environment):          OUT/gen/<clip_id>.wav from those codes, and the reference clips -> OUT/ref/<clip_id>.wav
score (metrics environment):        FD-openl3, KL-PaSST and CLAP score through stable-audio-metrics -> OUT/metrics.json
Every stage is resumable: finished clips are skipped."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from dcttgen.plan import Plan, plan_sections


def select(rows: list[dict], n: int, seed: int) -> list[dict]:
    """The captioned clips, in a fixed order; n > 0 keeps a seeded random subset of that size."""
    rows = sorted((r for r in rows if r.get("caption")), key=lambda r: r["clip_id"])
    return rows if not n or n >= len(rows) else sorted(random.Random(seed).sample(rows, n), key=lambda r: r["clip_id"])


def eval_plan(row: dict, seconds: int) -> Plan:
    """seconds = 0: the clip's own plan (the research plan's protocol). Otherwise a whole piece of that length with the
    clip's bpm, moods and instruments."""
    if not seconds:
        return Plan.from_manifest(row)
    return Plan(row["bpm"], seconds, plan_sections(seconds), row["moods"], row["instruments"])


def centre(wav, seconds: int, sr: int):
    """The middle `seconds` of a reference clip (the whole clip if seconds = 0 or the clip is shorter)."""
    n = seconds * sr
    if not seconds or len(wav) <= n:
        return wav
    a = (len(wav) - n) // 2
    return wav[a:a + n]


def _prompts(cfg) -> list[dict]:
    e, root = cfg.eval, Path(cfg.paths.data_root)
    text = (root / "manifest" / f"{e.split}.jsonl").read_text(encoding="utf-8")
    rows = select([json.loads(line) for line in text.splitlines() if line.strip()], e.num_prompts, e.seed)
    if not rows:
        raise ValueError(f"manifest/{e.split}.jsonl has no captioned clips")
    return rows


def make_codes(cfg, out, to_codes) -> int:
    """to_codes(prompt, cfg, seed=..., **plan fields) -> (plan, codes Long[K, T]) is infer.prompt_to_codes. Returns the number made."""
    e, out, made = cfg.eval, Path(out), 0
    rows = _prompts(cfg)
    (out / "codes").mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        path = out / "codes" / f"{row['clip_id']}.npy"
        if path.exists():
            continue
        fields = {}
        if e.plan_mode == "given":
            p = eval_plan(row, e.seconds)
            fields = dict(duration=p.duration, bpm=p.bpm, moods=p.moods, instruments=p.instruments)
        _, codes = to_codes(row["caption"], cfg, seed=e.seed + i, **fields)      # seed + index: reproducible, and unchanged by a resume
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(codes).astype(np.int16))
        os.replace(tmp, path)
        made += 1
    (out / "prompts.json").write_text(json.dumps({r["clip_id"]: r["caption"] for r in rows}, indent=1), encoding="utf-8")
    return made


def make_audio(cfg, out, to_wav, read, write) -> int:
    """to_wav(codes, cfg, seed=...) -> (wav, sr) is infer.codes_to_wav; read(path) -> (wav, sr); write(path, wav, sr)."""
    import torch
    e, root, out, made = cfg.eval, Path(cfg.paths.data_root), Path(out), 0
    (out / "gen").mkdir(parents=True, exist_ok=True)
    (out / "ref").mkdir(exist_ok=True)
    for i, row in enumerate(_prompts(cfg)):
        gen, ref = out / "gen" / f"{row['clip_id']}.wav", out / "ref" / f"{row['clip_id']}.wav"
        if not ref.exists():
            wav, sr = read(root / row["audio"])
            write(ref, centre(wav, e.seconds, sr), sr)
        if gen.exists():
            continue
        codes = torch.from_numpy(np.load(out / "codes" / f"{row['clip_id']}.npy").astype(np.int64))
        wav, sr = to_wav(codes, cfg, seed=e.seed + i)
        write(gen, wav, sr)
        made += 1
    return made


def score(cfg, out, metrics=None) -> dict:
    """metrics = (openl3_fd, passt_kld, clap_score); by default they are imported from the stable-audio-metrics clone."""
    e, out = cfg.eval, Path(out)
    prompts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    missing = [i for i in prompts if not (out / "gen" / f"{i}.wav").exists() or not (out / "ref" / f"{i}.wav").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prompts have no generated or reference file (first: {missing[:3]}); finish the audio stage")
    if metrics is None:
        sys.path.insert(0, str(Path(e.metrics_repo).resolve()))
        from src.clap_score import clap_score
        from src.openl3_fd import openl3_fd
        from src.passt_kld import passt_kld
        metrics = (openl3_fd, passt_kld, clap_score)
    fd, kld, clap = metrics
    gen, ref, f = str(out / "gen"), str(out / "ref"), e.fd
    res = {"n": len(prompts),
           "fd_openl3": float(fd(channels=f.channels, samplingrate=f.sample_rate, content_type=f.content_type, openl3_hop_size=f.hop,
                                 eval_path=gen, ref_path=ref, batching=f.batch)),
           "kl_passt": float(kld(ids=sorted(prompts), eval_path=gen, ref_path=ref, collect="mean")),
           "clap_score": float(clap(prompts, gen, clap_model=e.clap_model))}
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=("codes", "audio", "score"))
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    args = ap.parse_args(argv)
    from dcttgen.config import load_config
    cfg = load_config(args.config, args.override)
    if args.stage == "score":
        print(json.dumps(score(cfg, args.out), indent=1))
    elif args.stage == "codes":
        from dcttgen.infer import prompt_to_codes
        print(f"generated {make_codes(cfg, args.out, prompt_to_codes)} code files in {args.out}/codes")
    else:
        import soundfile as sf
        from dcttgen.infer import codes_to_wav
        read = lambda path: sf.read(str(path), dtype="float32")
        write = lambda path, wav, sr: sf.write(str(path), wav.numpy() if hasattr(wav, "numpy") else wav, sr, subtype="PCM_16")
        print(f"decoded {make_audio(cfg, args.out, codes_to_wav, read, write)} pieces into {args.out}/gen")


if __name__ == "__main__":
    main()
