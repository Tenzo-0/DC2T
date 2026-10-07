"""Run: pytest -q tests/test_data_manifest.py   or   python tests/test_data_manifest.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dc2t.data.common import read_jsonl  # noqa: E402
from dc2t.data.manifest import build_rows, report, split_of, validate_manifest, validate_row, write_manifests  # noqa: E402

CAPTION = "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute"


def good(**kw):  # the example row of contract 7.2
    row = {"clip_id": "yt_3fA9c_00", "recording_id": "yt_3fA9c", "position": "first", "audio": "audio/yt_3fA9c_00.flac",
           "duration": 270, "bpm": 80, "moods": ["uplifting", "joyful"],
           "instruments": ["zither", "two-string fiddle", "moon-shaped lute"],
           "sections": [["intro", 30], ["main", 240]], "caption": CAPTION, "split": "train"}
    return {**row, **kw}


def test_contract_example_row_is_valid():
    assert validate_row(good()) == []
    assert validate_row(good(caption=None)) == []                                   # allowed: LM pre-training only
    assert validate_row(good(), audio=(32000, 1, 270 * 32000), sample_rate=32000) == []


def test_one_violation_per_rule():
    cases = {
        "clip_id must": good(clip_id="bad id", audio="audio/bad id.flac"),
        "recording_id must": good(recording_id=""),
        "audio must": good(audio="audio/other.flac"),
        "position must": good(position="start"),
        "duration must be an integer": good(duration=271, sections=[["intro", 30], ["main", 241]]),   # a first clip is at most 270 s
        "bpm must": good(bpm=20),
        "sum of the sections": good(sections=[["intro", 30], ["main", 200]]),
        "sections must follow": good(sections=[["main", 240], ["intro", 30]]),
        "[main] must be an integer in 1..240": good(position="whole", duration=300, sections=[["intro", 10], ["main", 280], ["outro", 10]]),
        "sections must contain main": good(duration=30, sections=[["intro", 30]]),
        "an intro needs": good(position="last"),
        "an outro needs": good(sections=[["main", 240], ["outro", 30]]),
        "moods must come from": good(moods=["groovy"]),
        "moods must be": good(moods=[]),
        "instruments must": good(instruments=["piano"]),
        "caption must": good(caption="two\nlines"),
        "split must": good(split="test"),
        "missing fields ['bpm']": {k: v for k, v in good().items() if k != "bpm"},
    }
    for needle, row in cases.items():
        problems = validate_row(row)
        assert any(needle in p for p in problems), (needle, problems)
    assert validate_row("not a row") == ["the row is not a JSON object"]
    short = validate_row(good(), audio=(32000, 1, 270 * 32000 - 1), sample_rate=32000)   # one sample short: T != 25 x duration later
    stereo = validate_row(good(), audio=(44100, 2, 270 * 44100), sample_rate=32000)
    assert len(short) == 1 and "exactly duration x rate samples" in short[0] and "2 channel(s), 44100 Hz" in stereo[0]


def test_split_is_stable_and_about_two_percent():
    ids = [f"rec_{i}" for i in range(20000)]
    val = sum(split_of(i) == "val" for i in ids)
    assert 330 <= val <= 470, val                                                   # 2 % of 20,000 = 400
    assert split_of("rec_0") == split_of("rec_0") == split_of("rec_0", 2.0)
    assert split_of("x", 0.0) == "train" and split_of("x", 100.0) == "val"
    assert {split_of(i, 50.0) for i in ids[:100]} == {"train", "val"}


def _clips():
    mk = lambda rec, i, pos, dur: {"clip_id": f"{rec}_{i:02d}", "recording_id": rec, "position": pos, "start": 0,
                                   "duration": dur, "audio": f"audio/{rec}_{i:02d}.flac"}
    return [mk("a", 0, "first", 270), mk("a", 1, "middle", 240), mk("a", 2, "last", 100), mk("b", 0, "whole", 150), mk("c", 0, "whole", 30)]


def test_build_write_and_validate():
    clips = _clips()
    feats = {c["clip_id"]: {"bpm": 80, "moods": ["calm"]} for c in clips if c["clip_id"] != "c_00"}
    inst = {c["clip_id"]: ["zither"] for c in clips}
    inst["a_01"] = []                                                               # nobody heard one of the six instruments
    rows, dropped = build_rows(clips, feats, inst, {"b_00": "A calm Don ca tai tu piece played by zither"}, groups={"a": "session1", "b": "session1"})
    assert dropped == [("a_01", "no annotated instrument"), ("c_00", "no features")]
    assert [r["clip_id"] for r in rows] == ["a_00", "a_02", "b_00"]
    assert rows[0]["sections"] == [["intro", 30], ["main", 240]] and rows[1]["sections"] == [["main", 80], ["outro", 20]]
    assert rows[2]["sections"] == [["intro", 30], ["main", 90], ["outro", 30]]      # the plan's own example
    assert len({r["split"] for r in rows}) == 1                                     # one group -> one split
    assert rows[0]["caption"] is None and rows[2]["caption"].startswith("A calm")
    seconds = {r["audio"]: r["duration"] for r in rows}
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "audio").mkdir()
        for r in rows:
            (root / r["audio"]).write_bytes(b"")
        probe = lambda p: (32000, 1, seconds[f"audio/{p.name}"] * 32000)
        write_manifests(rows, root, 32000, probe)
        on_disk = read_jsonl(root / "manifest" / "all.jsonl")
        parts = read_jsonl(root / "manifest" / "train.jsonl") + read_jsonl(root / "manifest" / "val.jsonl")
        assert on_disk == rows and sorted(r["clip_id"] for r in parts) == ["a_00", "a_02", "b_00"]
        assert validate_manifest(on_disk, root, 32000, probe) == []
        other = "val" if rows[0]["split"] == "train" else "train"
        leak = validate_manifest([rows[0], {**rows[1], "split": other}], root, 32000, probe)
        assert leak == ["a_02: recording a is in both splits"]
        assert validate_manifest([rows[0], rows[0]], root, 32000, probe) == ["a_00: duplicate clip_id"]
        (root / "audio" / "b_00.flac").unlink()
        assert validate_manifest(rows, root, 32000, probe) == ["b_00: audio/b_00.flac does not exist"]
        try:
            write_manifests(rows, root, 32000, probe)                               # refuses, and leaves the old files alone
            raise AssertionError("an invalid manifest was written")
        except ValueError as e:
            assert "1 manifest problems" in str(e)
    rep = report(rows)
    assert rep["clips"] == 3 and rep["recordings"] == 2 and rep["hours"] == round(520 / 3600, 2) and rep["captioned"] == 1
    assert rep["position"] == {"first": 1, "last": 1, "whole": 1} and rep["instruments"] == {"zither": 3}
    assert rep["duration_s"] == {"120-179": 1, "240-299": 1, "60-119": 1} and rep["bpm"] == {"80-99": 3}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_data_manifest: {len(tests)} tests passed")
