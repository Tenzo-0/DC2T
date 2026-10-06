"""Standardise one raw recording: decode -> mono -> sample_rate -> trim silence -> peak level -> 16-bit FLAC."""
import hashlib
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf
import soxr

from dcttgen.data.common import fingerprint, load_marker, save_marker, tmp_name


def decode(path) -> tuple[np.ndarray, int, str]:
    """Any audio or video file -> (Float32[N, 2] at the file's own rate, that rate, ffmpeg's stderr).
    First audio stream only; mono sources are duplicated to two channels. Needs ffmpeg and ffprobe on PATH."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                            "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    if not probe.stdout.split():
        raise ValueError("no audio stream")
    # ponytail: the whole recording is held in RAM (about 2.8 GB per hour of 48 kHz stereo) - split longer files first
    ff = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "2",
                         "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(ff.stdout, dtype="<f4").reshape(-1, 2), int(probe.stdout.split()[0]), ff.stderr.decode()[:200]


def standardise(raw, out, *, sr: int, peak: float, top_db: float, min_seconds: int) -> dict:
    """Write `out` (FLAC, mono, `sr` Hz, 16-bit, loudest sample at `peak`) and return its statistics."""
    x, rate, warn = decode(raw)
    y = x.mean(axis=1)                                           # (L + R) / 2, the downmix MuCodec's own encoder uses
    if rate != sr:
        y = soxr.resample(y, rate, sr, quality="HQ")             # the filter behind librosa's default res_type
    if np.abs(y).max(initial=0.0) < 1e-3:
        raise ValueError("silent: peak below -60 dBFS")
    _, (a, b) = librosa.effects.trim(y, top_db=top_db)           # drop head and tail quieter than top_db below the loudest frame
    y = y[a:b]
    if len(y) < min_seconds * sr:
        raise ValueError(f"{len(y) / sr:.1f} s after trimming, shorter than {min_seconds} s")
    top = float(np.abs(y).max())
    y = np.clip(y * (peak / top), -1.0, 1.0)                     # one gain per recording: clips keep their relative levels
    tmp = tmp_name(out)
    sf.write(tmp, y, sr, format="FLAC", subtype="PCM_16")
    os.replace(tmp, out)
    return {"seconds": len(y) / sr, "source_rate": rate, "gain_db": float(20 * np.log10(peak / top)), "ffmpeg_stderr": warn}


def sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def standardise_one(item) -> None:
    """item = (raw, out, marker, params). Skips when the marker matches the raw file and the settings (resumable)."""
    raw, out, marker, params = item
    key = {"raw": fingerprint(raw), **params}
    if load_marker(marker, key) is not None and os.path.exists(out):
        return
    stats = standardise(raw, out, **params)
    save_marker(marker, key, {**stats, "raw_sha1": sha1(raw)})
