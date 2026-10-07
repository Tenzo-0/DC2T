"""The manifest (contract 7.2): assemble the rows, enforce every rule, split by recording, write the three files, report.

python -m dc2t.data.manifest --config C      validates manifest/all.jsonl and the audio files; exit code 0 = milestone M1 is met
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from dc2t.data.common import read_jsonl, write_jsonl
from dc2t.data.vocab import MOODS
from dc2t.plan import MIN_CLIP_S, POSITIONS, Plan, max_clip_seconds, plan_sections

FIELDS = ("clip_id", "recording_id", "position", "audio", "duration", "bpm", "moods", "instruments", "sections", "caption", "split")
CLIP_ID = re.compile(r"[A-Za-z0-9_]+")
MAX_CAPTION_CHARS = 600


def split_of(group_id: str, val_percent: float = 2.0) -> str:
    """98 % / 2 % by a hash of the recording (or of its group): the same answer on every machine and after new
    recordings are added, and every clip of one recording lands on the same side."""
    bucket = int(hashlib.sha1(("split:" + group_id).encode()).hexdigest(), 16) % 10_000
    return "val" if bucket < round(val_percent * 100) else "train"


def validate_row(row, audio=None, sample_rate=None) -> list[str]:
    """Problems of one row ([] = valid). audio = (samplerate, channels, frames) of its file, when it was probed."""
    if not isinstance(row, dict):
        return ["the row is not a JSON object"]
    missing = [f for f in FIELDS if f not in row]
    if missing:
        return [f"missing fields {missing}"]
    bad, cid, pos, dur = [], row["clip_id"], row["position"], row["duration"]
    if not isinstance(cid, str) or not CLIP_ID.fullmatch(cid):
        bad.append("clip_id must match [A-Za-z0-9_]+")
    if not isinstance(row["recording_id"], str) or not row["recording_id"]:
        bad.append("recording_id must be a non-empty string")
    if row["audio"] != f"audio/{cid}.flac":
        bad.append("audio must be 'audio/<clip_id>.flac'")
    whole_seconds = isinstance(dur, int) and not isinstance(dur, bool)
    if pos not in POSITIONS:
        bad.append(f"position must be one of {POSITIONS}")
    elif not whole_seconds or not MIN_CLIP_S <= dur <= max_clip_seconds(pos):
        bad.append(f"duration must be an integer in [{MIN_CLIP_S}, {max_clip_seconds(pos)}] for position {pos}")
    try:
        plan = Plan.from_manifest(row)          # bpm range; sections order, caps and sum; moods form; instruments
    except ValueError as e:
        bad.append(str(e))
        plan = None
    if plan is not None:
        labels = [name for name, _ in plan.sections]
        if "main" not in labels:
            bad.append("sections must contain main")
        if "intro" in labels and pos not in ("whole", "first"):
            bad.append("an intro needs position whole or first")
        if "outro" in labels and pos not in ("whole", "last"):
            bad.append("an outro needs position whole or last")
        if any(m not in MOODS for m in plan.moods):
            bad.append(f"moods must come from the vocabulary {MOODS}")
    cap = row["caption"]
    if cap is not None and (not isinstance(cap, str) or not cap.strip() or re.search(r"[\r\n]", cap) or len(cap) > MAX_CAPTION_CHARS):
        bad.append(f"caption must be null or one non-empty line of at most {MAX_CAPTION_CHARS} characters")
    if row["split"] not in ("train", "val"):
        bad.append("split must be train or val")
    if audio is not None:
        sr, channels, frames = audio
        if channels != 1 or (sample_rate and sr != sample_rate) or (whole_seconds and frames != dur * sr):
            bad.append(f"audio is {channels} channel(s), {sr} Hz, {frames} samples; expected mono, {sample_rate} Hz "
                       f"and exactly duration x rate samples")
    return bad


def probe_flac(path) -> tuple[int, int, int]:
    import soundfile as sf
    info = sf.info(str(path))
    return info.samplerate, info.channels, info.frames


def validate_manifest(rows, data_root=None, sample_rate=None, probe=probe_flac) -> list[str]:
    """Every problem of a manifest: the row rules, unique clip_id, one split per recording and, when data_root is
    given, the audio files themselves (contract 7.1)."""
    problems, seen, split_by_rec = [], set(), {}
    for n, row in enumerate(rows, start=1):
        is_row = isinstance(row, dict)
        cid = row.get("clip_id", f"line {n}") if is_row else f"line {n}"
        audio = None
        if data_root is not None and is_row and isinstance(row.get("audio"), str):
            path = Path(data_root) / row["audio"]
            if path.is_file():
                audio = probe(path)
            else:
                problems.append(f"{cid}: {row['audio']} does not exist")
        problems += [f"{cid}: {b}" for b in validate_row(row, audio, sample_rate)]
        if not is_row:
            continue
        if cid in seen:
            problems.append(f"{cid}: duplicate clip_id")
        seen.add(cid)
        rec = row.get("recording_id")
        if split_by_rec.setdefault(rec, row.get("split")) != row.get("split"):
            problems.append(f"{cid}: recording {rec} is in both splits")
    return problems


def build_rows(clips, features, instruments, captions, groups=None, val_percent: float = 2.0):
    """clips: the rows of clips.cut_recording. features[clip_id] = {"bpm": int, "moods": [...]}.
    instruments[clip_id] = [...]. captions[clip_id] = str; a clip without one gets a null caption (LM pre-training only).
    groups[recording_id] = group id (recordings of one session share a split). Returns (rows, dropped): a clip without
    features or without any annotated instrument cannot satisfy the contract and is dropped, with the reason."""
    rows, dropped = [], []
    for c in clips:
        cid = c["clip_id"]
        if cid not in features:
            dropped.append((cid, "no features"))
            continue
        if not instruments.get(cid):
            dropped.append((cid, "no annotated instrument"))
            continue
        group = (groups or {}).get(c["recording_id"]) or c["recording_id"]
        rows.append({"clip_id": cid, "recording_id": c["recording_id"], "position": c["position"], "audio": c["audio"],
                     "duration": c["duration"], "bpm": features[cid]["bpm"], "moods": features[cid]["moods"],
                     "instruments": instruments[cid],
                     "sections": [list(s) for s in plan_sections(c["duration"], c["position"])],
                     "caption": captions.get(cid), "split": split_of(group, val_percent)})
    return rows, dropped


def write_manifests(rows, data_root, sample_rate=None, probe=probe_flac) -> None:
    """Writes manifest/all.jsonl, train.jsonl and val.jsonl - or nothing at all if a single rule is broken."""
    problems = validate_manifest(rows, data_root, sample_rate, probe)
    if problems:
        raise ValueError(f"{len(problems)} manifest problems; the first ones: {problems[:5]}")
    d = Path(data_root) / "manifest"
    write_jsonl(d / "all.jsonl", rows)
    for split in ("train", "val"):
        write_jsonl(d / f"{split}.jsonl", [r for r in rows if r["split"] == split])


def report(rows) -> dict:
    """The dataset in numbers: sizes, the split, and the distribution of every plan field."""
    hours = lambda rs: round(sum(r["duration"] for r in rs) / 3600, 2)
    band = lambda x, w: f"{x // w * w}-{x // w * w + w - 1}"
    count = lambda items: dict(sorted(Counter(items).items()))
    return {"clips": len(rows), "hours": hours(rows), "recordings": len({r["recording_id"] for r in rows}),
            "val_clips": sum(r["split"] == "val" for r in rows), "val_hours": hours([r for r in rows if r["split"] == "val"]),
            "captioned": sum(r["caption"] is not None for r in rows),
            "position": count(r["position"] for r in rows),
            "duration_s": count(band(r["duration"], 60) for r in rows),
            "bpm": count(band(r["bpm"], 20) for r in rows),
            "moods": count(m for r in rows for m in r["moods"]),
            "instruments": count(i for r in rows for i in r["instruments"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    from dc2t.config import load_config      # chapter 04
    cfg = load_config(args.config)
    rows = read_jsonl(Path(cfg.paths.data_root) / "manifest" / "all.jsonl")
    problems = validate_manifest(rows, cfg.paths.data_root, cfg.audio.sample_rate)
    for p in problems[:50]:
        print("PROBLEM", p)
    print(json.dumps(report(rows), indent=1))
    print(f"{len(rows)} rows, {len(problems)} problems")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
