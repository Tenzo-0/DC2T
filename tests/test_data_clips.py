import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import tempfile
from pathlib import Path

import numpy as np

from dcttgen.data.clips import boundary_costs, clip_count, cut_recording, plan_clips, positions
from dcttgen.plan import MIN_CLIP_S, max_clip_seconds

try:
    import soundfile as sf
except ImportError:      # the audio tests are skipped (they pass vacuously) without soundfile
    sf = None


def check_plan(total, clips):
    assert [c.position for c in clips] == positions(len(clips))             # whole | first, middle..., last
    assert sum(c.duration for c in clips) == total                          # nothing lost, nothing added
    assert [c.start for c in clips] == [sum(d.duration for d in clips[:i]) for i in range(len(clips))]
    for c in clips:                                                         # every clip inside its limits
        assert MIN_CLIP_S <= c.duration <= max_clip_seconds(c.position), (total, clips)
    assert len(clips) == clip_count(total)                                  # as few boundaries as possible


def test_plan_clips_every_length():
    rng = np.random.default_rng(0)
    for total in range(0, 3001):
        if total < MIN_CLIP_S:
            assert plan_clips(total) == []                                  # too short: dropped, never padded
            continue
        check_plan(total, plan_clips(total))
        check_plan(total, plan_clips(total, cost=rng.random(total + 1), slack=int(rng.integers(0, 40))))


def test_plan_clips_known_values():
    def shape(t):
        return [(c.position, c.duration) for c in plan_clips(t)]
    assert shape(29) == []
    assert shape(30) == [("whole", 30)]
    assert shape(300) == [("whole", 300)]
    assert shape(301) == [("first", 150), ("last", 151)]          # no 1-second remainder: 301 = 150 + 151
    assert shape(540) == [("first", 270), ("last", 270)]
    assert [c.duration for c in plan_clips(541)] == [180, 180, 181]
    assert shape(780) == [("first", 270), ("middle", 240), ("last", 270)]
    assert len(plan_clips(781)) == 4


def test_cuts_prefer_quiet_seconds():
    cost = np.ones(331)
    cost[175] = 0.0                                                # a pause at 175 s; the even cut is at 165 s
    assert [c.duration for c in plan_clips(330, cost, slack=15)] == [175, 155]
    cost = np.ones(331)
    cost[100] = 0.0                                                # outside +-15 s of 165: ignored
    assert [c.duration for c in plan_clips(330, cost, slack=15)] == [165, 165]
    y = np.ones(8000 * 100) * 1000.0
    y[8000 * 40 - 2000: 8000 * 40 + 2000] = 0.0                    # 0.5 s of silence centred on second 40
    c = boundary_costs(y, 8000, 100)
    assert c[40] == 0.0 and c[39] > 0 and c[41] > 0 and c.shape == (101,)


def write_recording(path, seconds, sr, seed=0):
    x = (np.random.default_rng(seed).normal(0, 3000, int(seconds * sr))).astype(np.int16)
    sf.write(path, x, sr, format="FLAC", subtype="PCM_16")
    return x


def test_cut_exact_samples():
    if sf is None:
        return print("skipped: soundfile not installed")
    for sr, seconds in [(8000, 700.9), (32000, 61.4)]:           # not whole seconds on purpose
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            x = write_recording(d / "rec.flac", seconds, sr)
            rows = cut_recording(d / "rec.flac", d / "audio", "r1", d / "r1.json")
            assert sum(r["duration"] for r in rows) == int(seconds)  # the last 0.9 / 0.4 s is dropped
            parts = []
            for r in rows:
                info = sf.info(d / "audio" / f"{r['clip_id']}.flac")
                assert (info.channels, info.samplerate, info.frames) == (1, sr, r["duration"] * sr)
                assert info.frames % (sr // 25) == 0               # whole 25 Hz code frames
                parts.append(sf.read(d / "audio" / f"{r['clip_id']}.flac", dtype="int16")[0])
            assert np.array_equal(np.concatenate(parts), x[: int(seconds) * sr])   # bit-exact slices


def test_cut_is_resumable_and_stale_proof():
    if sf is None:
        return print("skipped: soundfile not installed")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_recording(d / "rec.flac", 700, 8000)
        rows = cut_recording(d / "rec.flac", d / "audio", "r1", d / "r1.json")
        assert len(rows) == 3
        stamp = (d / "audio" / "r1_00.flac").stat().st_mtime_ns
        assert cut_recording(d / "rec.flac", d / "audio", "r1", d / "r1.json") == rows
        assert (d / "audio" / "r1_00.flac").stat().st_mtime_ns == stamp          # untouched on a re-run
        (d / "audio" / "r1_01.flac").unlink()                                    # a lost clip is rewritten
        assert cut_recording(d / "rec.flac", d / "audio", "r1", d / "r1.json") == rows
        assert (d / "audio" / "r1_01.flac").exists()
        write_recording(d / "rec.flac", 100, 8000, seed=1)                       # the recording changed: 1 clip now
        assert [r["clip_id"] for r in cut_recording(d / "rec.flac", d / "audio", "r1", d / "r1.json")] == ["r1_00"]
        assert sorted(p.name for p in (d / "audio").iterdir()) == ["r1_00.flac"]   # r1_01, r1_02 removed


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
