"""Cut a standardised recording into whole-second clips (decision D4)."""
import os
import re
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

import numpy as np

from dcttgen.data.common import fingerprint, load_marker, save_marker, tmp_name
from dcttgen.plan import MIN_CLIP_S, max_clip_seconds


@dataclass(frozen=True)
class Clip:
    position: str    # whole | first | middle | last
    start: int       # seconds from the start of the standardised recording
    duration: int    # seconds


def positions(n: int) -> list[str]:
    return ["whole"] if n == 1 else ["first"] + ["middle"] * (n - 2) + ["last"]


def clip_count(total: int) -> int:
    """Fewest clips that can hold `total` seconds: whole 300 s, first/last 270 s, middle 240 s."""
    n = 1
    while sum(max_clip_seconds(p) for p in positions(n)) < total:
        n += 1
    return n


def split_even(total: int, caps: list[int]) -> list[int]:
    """Split `total` seconds into len(caps) integer parts, as equal as the caps allow (tightest cap first)."""
    parts, left = [0] * len(caps), total
    for k, i in enumerate(sorted(range(len(caps)), key=lambda i: caps[i])):
        parts[i] = min(caps[i], left // (len(caps) - k))
        left -= parts[i]
    assert left == 0, "total exceeds the sum of the caps"
    return parts


def plan_clips(total: int, cost=None, slack: int = 15) -> list[Clip]:
    """Plan the clips of a recording that has `total` whole seconds. Returns [] if total < MIN_CLIP_S.

    The clip count is the minimum that fits the caps, and the seconds are divided as evenly as the caps allow,
    so there is never a short remainder. If `cost` (cost[t] = loudness at whole second t, len total + 1) is
    given, each boundary moves to the quietest second within +-slack of its even position, never leaving
    a clip outside [MIN_CLIP_S, cap] and never making the rest of the recording unplannable."""
    if total < MIN_CLIP_S:
        return []
    n = clip_count(total)
    pos = positions(n)
    caps = [max_clip_seconds(p) for p in pos]
    dur = split_even(total, caps)
    if cost is not None and n > 1:
        nominal, start, dur = list(accumulate(dur))[:-1], 0, []
        for i, nom in enumerate(nominal):
            flo = max(start + MIN_CLIP_S, total - sum(caps[i + 1:]))        # feasible: this clip within limits
            fhi = min(start + caps[i], total - (n - i - 1) * MIN_CLIP_S)    # and the rest still plannable
            lo, hi = max(flo, nom - slack), min(fhi, nom + slack)           # never empty: see the chapter
            b = min(range(lo, hi + 1), key=lambda t: (cost[t], abs(t - nom)))
            dur.append(b - start)
            start = b
        dur.append(total - start)
    clips, s = [], 0
    for p, d in zip(pos, dur):
        clips.append(Clip(p, s, d))
        s += d
    return clips


def boundary_costs(y: np.ndarray, sr: int, total: int, window_s: float = 0.5) -> np.ndarray:
    """Mean power in a window centred on each whole second 0..total (low = a quiet place to cut). y: [N]."""
    h = int(window_s * sr / 2)
    out = np.empty(total + 1)
    for t in range(total + 1):
        w = y[max(0, t * sr - h): t * sr + h].astype(np.float64)
        out[t] = float(np.mean(w * w)) if len(w) else 0.0
    return out


def cut_recording(rec_flac, out_dir, recording_id: str, marker, slack: int = 15) -> list[dict]:
    """Write <out_dir>/<recording_id>_<NN>.flac for every clip and return one row per clip.
    Slices are taken on the int16 samples, so they are bit-exact copies with exactly duration * sr samples.
    `marker` (a JSON path) makes the call resumable and stale-proof: see common.load_marker."""
    import soundfile as sf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {"rec": fingerprint(rec_flac), "slack": slack}
    rows = load_marker(marker, key)
    if rows is not None and all((out_dir / f"{r['clip_id']}.flac").exists() for r in rows):
        return rows
    x, sr = sf.read(rec_flac, dtype="int16")                  # x: Int16[N], mono
    total = len(x) // sr                                       # the last fraction of a second is dropped
    rows = []
    for i, c in enumerate(plan_clips(total, boundary_costs(x, sr, total), slack)):
        clip_id = f"{recording_id}_{i:02d}"
        path = out_dir / f"{clip_id}.flac"
        tmp = tmp_name(path)
        sf.write(tmp, x[c.start * sr:(c.start + c.duration) * sr], sr, format="FLAC", subtype="PCM_16")
        os.replace(tmp, path)
        rows.append({"clip_id": clip_id, "recording_id": recording_id, "position": c.position,
                     "start": c.start, "duration": c.duration, "audio": f"audio/{clip_id}.flac"})
    keep = {f"{r['clip_id']}.flac" for r in rows}
    for p in out_dir.iterdir():                                # clips of an earlier, different plan
        if re.fullmatch(re.escape(recording_id) + r"_\d+\.flac", p.name) and p.name not in keep:
            p.unlink()
    save_marker(marker, key, rows)
    return rows
