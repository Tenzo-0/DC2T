import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import csv
import tempfile
from pathlib import Path

from dc2t.data.provenance import COLUMNS, load_provenance


def write_csv(root: Path, rows: list[dict]):
    with open(root / "provenance.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


GOOD = {"recording_id": "yt_a1", "raw_file": "a.m4a", "source_type": "video_site", "source_ref": "https://example.org/v/a1",
        "rights": "written_consent", "rights_ref": "consent/2026-001.pdf", "obtained_on": "2026-03-01"}


def setup(rows, files=("a.m4a", "b.wav")):
    d = tempfile.TemporaryDirectory()
    root = Path(d.name)
    (root / "raw" / "sub").mkdir(parents=True)
    for f in files:
        (root / "raw" / f).write_bytes(b"x")
    write_csv(root, rows)
    return d, root


def test_cleared_rows_pass_and_pending_rows_are_skipped():
    pending = {**GOOD, "recording_id": "yt_b2", "raw_file": "b.wav", "rights": "permission_pending", "rights_ref": ""}
    d, root = setup([GOOD, pending])
    with d:
        cleared, problems = load_provenance(root)
        assert problems == []
        assert [r["recording_id"] for r in cleared] == ["yt_a1"]            # pending is recorded but never processed


def test_every_violation_is_reported():
    cases = {
        "recording_id must match": {**GOOD, "recording_id": "bad id!"},
        "rights must be one of": {**GOOD, "rights": "yes"},
        "source_type must be one of": {**GOOD, "source_type": "radio"},
        "ISO date": {**GOOD, "obtained_on": "01/03/2026"},
        "rights_ref are required": {**GOOD, "rights_ref": ""},
        "not a file inside raw/": {**GOOD, "raw_file": "../outside.wav"},
    }
    for msg, row in cases.items():
        d, root = setup([row], files=("a.m4a",))
        with d:
            cleared, problems = load_provenance(root)
            assert cleared == [] and any(msg in p for p in problems), (msg, problems)
    d, root = setup([GOOD, {**GOOD, "raw_file": "b.wav"}])
    with d:
        assert any("duplicate recording_id" in p for p in load_provenance(root)[1])
    d, root = setup([GOOD])                                                  # b.wav has no row
    with d:
        assert any("orphan: b.wav" in p for p in load_provenance(root)[1])


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
