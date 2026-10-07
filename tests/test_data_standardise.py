import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    from dc2t.data.standardise import standardise, standardise_one
except ImportError:        # needs soundfile, soxr, librosa (and ffmpeg): skipped on a machine without them
    sf = None

OPTS = dict(sr=32000, peak=0.8, top_db=50.0, min_seconds=30)


def have_tools():
    return sf is not None and shutil.which("ffmpeg") and shutil.which("ffprobe")


def plucks(seconds, sr, amp=0.2):
    """Decaying 440/660 Hz notes, one every 0.5 s: energetic like music, easy to trim."""
    t = np.arange(int(seconds * sr)) / sr
    env = np.exp(-6 * (t % 0.5))
    return (amp * env * (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 660 * t))).astype(np.float32)


def write_wav(path, left, right, sr):
    sf.write(path, np.stack([left, right], axis=1), sr, subtype="FLOAT")


def test_output_is_mono_32k_trimmed_and_peak_normalised():
    if not have_tools():
        return print("skipped: needs soundfile, soxr, librosa, ffmpeg")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sr_in = 44100
        music = plucks(34, sr_in)
        pad = np.zeros(3 * sr_in, dtype=np.float32)
        write_wav(d / "in.wav", np.concatenate([pad, music, pad]), np.concatenate([pad, 0.5 * music, pad]), sr_in)
        stats = standardise(d / "in.wav", d / "out.flac", **OPTS)
        info = sf.info(d / "out.flac")
        assert (info.format, info.subtype, info.channels, info.samplerate) == ("FLAC", "PCM_16", 1, 32000)
        assert abs(info.duration - 34.0) < 0.1                       # 3 s of digital silence removed at each end
        y, _ = sf.read(d / "out.flac", dtype="float32")
        assert abs(np.abs(y).max() - 0.8) < 0.002                     # loudest sample at the configured peak
        assert np.abs(y[:3200]).max() > 0.1 and np.abs(y[-3200:]).max() > 0.01   # no silence left at either end
        assert stats["source_rate"] == 44100 and abs(stats["seconds"] - info.duration) < 1e-3


def test_downmix_is_the_mean_and_bad_input_is_rejected():
    if not have_tools():
        return print("skipped: needs soundfile, soxr, librosa, ffmpeg")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        m = plucks(40, 32000)
        write_wav(d / "anti.wav", m, -m, 32000)                       # (L + R) / 2 = 0: nothing is left
        write_wav(d / "short.wav", m[: 32000 * 10], m[: 32000 * 10], 32000)
        write_wav(d / "zero.wav", 0 * m, 0 * m, 32000)
        for name, msg in [("anti", "silent"), ("zero", "silent"), ("short", "shorter than 30 s")]:
            try:
                standardise(d / f"{name}.wav", d / f"{name}.flac", **OPTS)
                raise AssertionError(f"{name} should have been rejected")
            except ValueError as e:
                assert msg in str(e), (name, str(e))
            assert not (d / f"{name}.flac").exists() and not list(d.glob("*.tmp"))   # nothing half-written is left


def test_decodes_a_compressed_48k_file():
    if not have_tools():
        return print("skipped: needs soundfile, soxr, librosa, ffmpeg")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        m = plucks(45, 48000)
        write_wav(d / "in.wav", m, m, 48000)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(d / "in.wav"), "-c:a", "aac", "-b:a", "128k",
                        str(d / "in.m4a")], check=True)
        stats = standardise(d / "in.m4a", d / "out.flac", **OPTS)
        info = sf.info(d / "out.flac")
        assert (info.channels, info.samplerate) == (1, 32000) and stats["source_rate"] == 48000
        assert 44.0 < info.duration < 46.5                           # AAC adds a little padding; trimming removes most


def test_standardise_one_is_resumable_and_notices_changes():
    if not have_tools():
        return print("skipped: needs soundfile, soxr, librosa, ffmpeg")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        m = plucks(40, 32000)
        write_wav(d / "in.wav", m, m, 32000)
        item = (d / "in.wav", d / "out.flac", d / "out.json", OPTS)
        standardise_one(item)
        stamp = (d / "out.flac").stat().st_mtime_ns
        standardise_one(item)
        assert (d / "out.flac").stat().st_mtime_ns == stamp          # same input, same settings: nothing redone
        standardise_one((d / "in.wav", d / "out.flac", d / "out.json", {**OPTS, "peak": 0.5}))
        y, _ = sf.read(d / "out.flac")
        assert abs(np.abs(y).max() - 0.5) < 0.002                    # a changed setting redoes the work


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
