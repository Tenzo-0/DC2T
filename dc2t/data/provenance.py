"""data/provenance.csv: one row per raw recording. The pipeline only touches recordings whose rights are cleared."""
import csv
import re
from datetime import date
from pathlib import Path

COLUMNS = ("recording_id", "raw_file", "source_type", "source_ref", "rights", "rights_ref", "obtained_on", "group_id", "notes")
SOURCE_TYPES = ("video_site", "musician", "expert", "archive", "other")
RIGHTS = ("licensed_open", "written_consent", "permission_pending", "unknown")
MEDIA = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mp4", ".mkv", ".wma", ".aiff"}
ID = re.compile(r"[A-Za-z0-9_]+")


def load_provenance(data_root, raw_dir: str = "raw", allowed=("licensed_open", "written_consent")):
    """Returns (cleared, problems). cleared: rows whose rights are in `allowed`, with 'raw_path' added.
    problems: strings. Any problem should stop the run: every raw file needs exactly one valid row."""
    root = Path(data_root)
    raw = (root / raw_dir).resolve()
    with open(root / "provenance.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            return [], [f"provenance.csv lacks columns {missing}"]
        rows = list(reader)
    problems, cleared, seen_ids, seen_files = [], [], set(), set()
    for n, row in enumerate(rows, start=2):                      # line 1 is the header
        bad = []
        rid = row["recording_id"].strip()
        if not ID.fullmatch(rid):
            bad.append("recording_id must match [A-Za-z0-9_]+")
        if rid in seen_ids:
            bad.append("duplicate recording_id")
        path = (raw / row["raw_file"].strip()).resolve()
        if not path.is_relative_to(raw) or not path.is_file():
            bad.append(f"raw_file {row['raw_file']!r} is not a file inside {raw_dir}/")
        if row["source_type"] not in SOURCE_TYPES:
            bad.append(f"source_type must be one of {SOURCE_TYPES}")
        if row["rights"] not in RIGHTS:
            bad.append(f"rights must be one of {RIGHTS}")
        if row["rights"] in allowed:                              # cleared rows must be fully documented
            if not row["source_ref"].strip() or not row["rights_ref"].strip():
                bad.append("source_ref and rights_ref are required when rights are cleared")
            try:
                date.fromisoformat(row["obtained_on"])
            except ValueError:
                bad.append("obtained_on must be an ISO date (YYYY-MM-DD)")
        problems += [f"provenance.csv line {n} ({rid or '?'}): {b}" for b in bad]
        seen_ids.add(rid)
        seen_files.add(path)
        if not bad and row["rights"] in allowed:
            cleared.append({**row, "recording_id": rid, "raw_path": path})
    for p in sorted(raw.rglob("*")):                              # a media file nobody wrote down is an error
        if p.is_file() and p.suffix.lower() in MEDIA and p.resolve() not in seen_files:
            problems.append(f"orphan: {p.relative_to(raw)} has no row in provenance.csv")
    return cleared, problems
