import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import tempfile
from pathlib import Path

from dc2t.data.annotate import (agreement, clip_instruments, cohen_kappa, in_overlap_set, load_annotations,
                                   parse_cell, write_sheet)
from dc2t.data.vocab import INSTRUMENTS


def test_parse_cell():
    inf = float("inf")
    assert [parse_cell(c) for c in ("", "0", " ", None)] == [None] * 4
    assert parse_cell("1") == [(0.0, inf)]
    assert parse_cell("120-300;400.5-520") == [(120.0, 300.0), (400.5, 520.0)]
    for bad in ("yes", "5-3", "-4-9", "10-"):
        try:
            parse_cell(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_propagation_to_clips():
    row = {"zither": "1", "two-string fiddle": "0", "bamboo flute": "100-200", "monochord": "270-300", "gong ban": ""}
    # a clip inherits a recording-level 1; canonical order; absent instruments never appear
    assert clip_instruments(row, 0, 100) == ["zither"]
    # the flute plays 100-200 s: a clip must overlap it by 10 s
    assert clip_instruments(row, 0, 105) == ["zither"]                    # only 5 s of overlap
    assert clip_instruments(row, 0, 120) == ["zither", "bamboo flute"]    # 20 s of overlap
    assert clip_instruments(row, 190, 100) == ["zither", "monochord", "bamboo flute"]   # flute: exactly 10 s; monochord 270-290: 20 s
    assert clip_instruments(row, 200, 100) == ["zither", "monochord"]     # monochord 270-300 is inside [200, 300)
    assert clip_instruments({"monochord": "1"}, 0, 30) == ["monochord"]
    assert clip_instruments({}, 0, 30) == []                              # the caller must drop a clip with no instruments
    assert list(INSTRUMENTS) == ["moon-shaped lute", "two-string fiddle", "zither", "monochord", "bamboo flute", "gong ban"]


def test_kappa_and_agreement():
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == 0.5
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    assert cohen_kappa([True, True], [True, True]) == 1.0                  # no variation at all: perfect, not a division by zero
    a = {"r1": {"zither": "1"}, "r2": {"zither": "1"}, "r3": {"zither": "0"}, "r4": {"zither": "0"}, "only_a": {"zither": "1"}}
    b = {"r1": {"zither": "1"}, "r2": {"zither": "0"}, "r3": {"zither": ""}, "r4": {"zither": "0"}}
    got = agreement(a, b)["zither"]
    assert got["n"] == 4 and got["kappa"] == 0.5 and got["disagree"] == ["r2"]   # recordings only one annotator did are ignored


def test_files_merge_later_wins_and_overlap_is_reproducible():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_sheet(d / "a.csv", ["r1", "r2"])
        write_sheet(d / "b.csv", ["r2"])
        (d / "c.csv").write_text("recording_id,zither,notes\nr2,1,adjudicated\n", encoding="utf-8")
        merged = load_annotations([d / "a.csv", d / "b.csv", d / "c.csv"])
        assert set(merged) == {"r1", "r2"} and merged["r2"]["notes"] == "adjudicated" and merged["r2"]["zither"] == "1"
    ids = [f"rec{i}" for i in range(2000)]
    share = sum(in_overlap_set(i) for i in ids) / len(ids)
    assert 0.08 < share < 0.12 and [in_overlap_set(i) for i in ids] == [in_overlap_set(i) for i in ids]


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
