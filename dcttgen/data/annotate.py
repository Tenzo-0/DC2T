"""Manual instrument annotation: file format, propagation from recordings to clips, agreement between annotators.

One CSV per annotator. One row per recording; one column per canonical instrument, plus `other` and `notes`.
Cell: empty or 0 = not heard; 1 = heard (anywhere); 'a-b;c-d' = heard only between those seconds of the
standardised recording (work/rec/<recording_id>.flac)."""
import csv
import hashlib
from pathlib import Path

from dcttgen.data.vocab import INSTRUMENTS

COLUMNS = ("recording_id", *INSTRUMENTS, "other", "notes")      # `other`: a loud instrument that is not one of the six


def parse_cell(cell: str):
    """None = absent, [(0, inf)] = present throughout, otherwise a list of (from_s, to_s) spans."""
    cell = (cell or "").strip()
    if cell in ("", "0"):
        return None
    if cell == "1":
        return [(0.0, float("inf"))]
    spans = []
    for part in cell.split(";"):
        a, _, b = part.partition("-")
        spans.append((float(a), float(b)))                       # ValueError on anything else: fix the sheet
        if not 0 <= spans[-1][0] < spans[-1][1]:
            raise ValueError(f"bad span {part!r}")
    return spans


def clip_instruments(row: dict, start: int, duration: int, min_overlap: int = 10) -> list[str]:
    """Instruments of one clip: a recording-level '1' is inherited by every clip; a span counts when it overlaps
    the clip [start, start + duration) by at least min(min_overlap, duration) seconds. Canonical order."""
    out = []
    for name in INSTRUMENTS:
        spans = parse_cell(row.get(name, ""))
        if spans and sum(max(0.0, min(b, start + duration) - max(a, start)) for a, b in spans) >= min(min_overlap, duration):
            out.append(name)
    return out


def load_annotations(paths) -> dict[str, dict]:
    """Merge annotator files in the given order; a later file replaces an earlier row for the same recording
    (so the adjudicated file goes last)."""
    merged = {}
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                merged[row["recording_id"].strip()] = row
    return merged


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(rows_a: dict[str, dict], rows_b: dict[str, dict]) -> dict[str, dict]:
    """Per instrument, over the recordings both annotators did: presence agreement, kappa and the disagreements."""
    both = sorted(set(rows_a) & set(rows_b))
    out = {}
    for name in (*INSTRUMENTS, "other"):
        a = [parse_cell(rows_a[r].get(name, "")) is not None for r in both]
        b = [parse_cell(rows_b[r].get(name, "")) is not None for r in both]
        out[name] = {"n": len(both), "positives": sum(a) + sum(b), "kappa": cohen_kappa(a, b) if both else float("nan"),
                     "disagree": [r for r, x, y in zip(both, a, b) if x != y]}
    return out


def in_overlap_set(recording_id: str, percent: int = 10) -> bool:
    """Both annotators do these recordings (a reproducible 10 %); the check above measures their agreement on them."""
    return int(hashlib.sha1(("overlap:" + recording_id).encode()).hexdigest(), 16) % 100 < percent


def write_sheet(path, recording_ids) -> None:
    """Empty annotation sheet for the given recordings (one per annotator)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for rid in recording_ids:
            w.writerow({"recording_id": rid})
