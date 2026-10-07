"""Quality filter (plan 3.5): vocal gate, then per-recording Frechet Audio Distance from three embeddings."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

from dc2t.data.common import write_jsonl
from dc2t.data.features import vocal_fraction


def centre_excerpt(y: np.ndarray, seconds: int, sr: int) -> np.ndarray:
    """At most `seconds` of audio from the middle of y. Equal-length excerpts make per-file FAD comparable:
    a Gaussian fitted to few frames inflates the score, so a short file would otherwise look worse."""
    n = seconds * sr
    a = max(0, (len(y) - n) // 2)
    return y[a:a + n]


def parse_fadtk_csv(text: str) -> dict[str, float]:
    """fadtk --indiv writes 'path,score' lines without a header, best score first. Returns {file stem: score}.
    Lines whose score is not a number are skipped; the caller must treat a missing file as a failure."""
    out = {}
    for line in text.splitlines():
        if "," in line:
            path, _, score = line.rpartition(",")
            try:
                out[Path(path).stem] = float(score)
            except ValueError:
                pass
    return out


def combine_scores(per_model: dict[str, dict[str, float]], ids) -> tuple[dict[str, float], list[str]]:
    """per_model[model][recording_id] = that model's per-file FAD. Returns (z, failed).
    Each model's scores go through log, then (x - median) / (1.4826 * MAD): a robust z-score, so a model whose
    scores are 1000 times larger cannot dominate, and one catastrophic file cannot stretch the scale.
    The combined score is the mean of the z-scores (the plan's 'mean'); high = far from the reference = worse.
    A recording missing from any model, or with a non-finite or non-positive score, is returned in `failed`
    (fadtk drops files that raise an error without telling the caller)."""
    ok = [i for i in sorted(ids) if all(i in s and np.isfinite(s[i]) and s[i] > 0 for s in per_model.values())]
    z = np.zeros(len(ok))
    for s in per_model.values():
        x = np.log([s[i] for i in ok])
        med = np.median(x)
        z += (x - med) / (1.4826 * np.median(np.abs(x - med)) or 1.0)
    z /= len(per_model)
    return dict(zip(ok, z.tolist())), sorted(set(ids) - set(ok))


def select_discard(z: dict[str, float], drop_fraction: float = 0.1, z_max: float | None = None) -> set[str]:
    """R5: discard the HIGH tail (a high FAD means far from the reference). z_max, if set, wins over drop_fraction."""
    if z_max is not None:
        return {i for i, v in z.items() if v > z_max}
    return set(sorted(z, key=z.get, reverse=True)[:round(drop_fraction * len(z))])


def run_fadtk(model: str, baseline, eval_dir, csv, workers: int = 8) -> None:
    """One call of the fadtk command line (not executed in this guide: see the chapter)."""
    shutil.rmtree(Path(baseline) / "stats", ignore_errors=True)   # fadtk caches mu/cov per directory and never invalidates
    subprocess.run(["fadtk", model, str(baseline), str(eval_dir), str(csv), "--indiv", "-w", str(workers)], check=True)


def run(cfg, fad=run_fadtk) -> list[dict]:
    """Vocal gate -> excerpts -> fadtk per model -> combine -> work/gate.jsonl and work/selected.jsonl.
    `fad(model, baseline_dir, eval_dir, csv)` is injectable so that the stage can be tested without fadtk."""
    import soundfile as sf
    root, d = Path(cfg.paths.data_root), cfg.data
    work = root / "work"
    rows, kept = [], {}
    for p in sorted((work / "rec").glob("*.flac")):
        frac = vocal_fraction(np.load(work / "embed" / f"{p.stem}.npz")["p_voice"], d.vocal_prob)
        rows.append({"recording_id": p.stem, "vocal_fraction": frac, "status": "vocal" if frac > d.vocal_max_recording else "ok"})
        if frac <= d.vocal_max_recording:
            kept[p.stem] = p
    pool = work / "fad" / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    for f in pool.glob("*.flac"):                                         # a recording that left the pool must not be scored,
        if f.stem not in kept:                                            # nor may its embeddings enter the baseline statistics
            f.unlink()
            for e in pool.glob(f"embeddings/*/{f.stem}.npy"):
                e.unlink()
    for rid, p in kept.items():
        if not (pool / f"{rid}.flac").exists():
            y, sr = sf.read(p, dtype="int16")
            sf.write(pool / f"{rid}.flac", centre_excerpt(y, d.fad_excerpt_s, sr), sr, format="FLAC", subtype="PCM_16")
    baseline = pool
    if d.fad_reference == "gold":                                         # data/gold.txt: one recording_id per line
        baseline = work / "fad" / "gold"
        shutil.rmtree(baseline, ignore_errors=True)
        baseline.mkdir(parents=True)
        for rid in (root / "gold.txt").read_text().split():
            shutil.copy2(pool / f"{rid}.flac", baseline / f"{rid}.flac")  # FileNotFoundError if a gold recording was gated out
    tag = hashlib.sha1("\n".join(sorted(kept)).encode()).hexdigest()[:8]   # a different pool gets different CSV names
    per_model = {}
    for model in d.fad_models:
        csv = work / "fad" / f"{model}-{tag}.csv"
        if not csv.exists():
            fad(model, baseline, pool, csv)
            shutil.rmtree(pool / "convert", ignore_errors=True)           # fadtk's resampled WAV copies: gigabytes, never reused
        per_model[model] = parse_fadtk_csv(csv.read_text())
    z, _ = combine_scores(per_model, kept)
    gone = select_discard(z, d.fad_drop_fraction, d.fad_z_max)
    for r in rows:
        rid = r["recording_id"]
        if rid in z:
            r["fad_z"], r["status"] = z[rid], "fad_high" if rid in gone else "ok"
        elif rid in kept:
            r["status"] = "fad_failed"                                    # missing from a CSV: never silently kept
    write_jsonl(work / "gate.jsonl", rows)                                # every recording with its verdict
    write_jsonl(work / "selected.jsonl", [r for r in rows if r["status"] == "ok"])
    return rows
