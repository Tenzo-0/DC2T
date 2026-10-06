# DcttGen Implementation Guide — 01. Data Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this chapter task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** from a folder of raw recordings to `data/audio/*.flac` and `data/manifest/{all,train,val}.jsonl` that pass the contract's validator.
**Architecture:** a chain of small stages, each reading files and writing files under `data/work/`, each safe to re-run. Decisions (which clip, which mood, which split) are plain functions tested on CPU; the heavy tools (ffmpeg, Essentia, fadtk, the GPT-4o API) are called at the edges.
**Tech stack:** `ffmpeg`/`ffprobe`, `soundfile`, `soxr`, `librosa`, `numpy`; Essentia with TensorFlow models and `fadtk` 1.1.0 (both Linux); the `openai` Python SDK.
**Spec:** plan §1, §2.1, §3.5, §4.1 · contract: [00 — Overview, Spec and Shared Contracts](00-overview-and-contracts.md)

## Global constraints

- Output audio: FLAC, mono, `cfg.audio.sample_rate` Hz, **exactly `duration × sample_rate` samples** (contract §7.1). One sample off breaks `T = 25 × duration` in every later chapter.
- Manifest rows follow contract §7.2 exactly; `python -m dcttgen.data.manifest --config C` exiting with code 0 is the acceptance test of milestone M1.
- Clip durations are whole seconds in `[30, max_clip_seconds(position)]`; `plan_sections` and `max_clip_seconds` are imported from `dcttgen/plan.py` (chapter 03, contract §8.3), never re-implemented.
- Only recordings whose rights are written down as cleared enter the pipeline.
- Every stage is resumable and notices when its input changed: an item is skipped only if a marker written *after* its outputs matches the input file and the settings.
- API keys come from the environment (`OPENAI_API_KEY`); they are never written to a config, a log or the manifest.
- Essentia's TensorFlow models and `fadtk` run on Linux. Everything else, and every test, also runs on Windows.

## Review focus

| # | Input or condition | Expected behaviour | Test |
|---|---|---|---|
| 1 | A recording of any length from 30 s to hours | Every clip within its limits, durations summing to the total, positions `first, middle…, last`; never a short remainder | `test_plan_clips_every_length` |
| 2 | A clip file one sample short, or stereo, or at another rate | The manifest is refused | `test_cut_exact_samples`, `test_one_violation_per_rule` |
| 3 | A stage is re-run after an upstream file or a setting changed | The item is redone, not skipped; stale clips of an earlier plan are deleted | `test_cut_is_resumable_and_stale_proof`, `test_standardise_one_is_resumable_and_notices_changes` |
| 4 | The FAD tool silently drops a file, or returns a non-finite score | The recording is marked failed — never silently kept | `test_missing_or_invalid_scores_are_failures_not_passes` |
| 5 | GPT-4o names an instrument that was not annotated, mentions singing, or emits text that looks like a control token | The caption is rejected and the complaint fed back; after the last attempt the clip keeps its template caption | `test_validator_rejects_each_kind_of_bad_caption`, `test_gpt_step_feeds_complaints_back_and_gives_up_cleanly` |

A sixth, checked by `test_build_write_and_validate`: two clips of one recording in different splits is reported as leakage.

## 1. Background and design

### 1.1 What the plan says

**Plan §3.5:** collect recordings; resample to 32 kHz mono; score every file with Fréchet Audio Distance from VGGish, CLAP and EnCodec embeddings and average the three; discard poor files; `librosa` for bpm and duration; Essentia for moods; annotate six instruments by listening; GPT-4o writes a caption from those fields. About 28,000 clips, 2,000 hours. **Plan §4.1:** 98 % / 2 % split.

What the plan leaves open, and what this chapter decides:

| Open point | Decision | Why |
|---|---|---|
| How instrumental audio is obtained (the plan criticises a dataset for keeping vocals) | A voice classifier gates out recordings with singing; no source separation | separation leaves artefacts that the codec would learn; see open question 1 |
| How long recordings become clips | Contract D4: whole-second clips that know their `position` | — |
| How three FAD scores on different scales become one | Robust z-score of the log of each, then the mean | a raw mean is the EnCodec score alone |
| Which tail is discarded | The **high** tail (contract R5) | low FAD = close to the reference = good |
| The reference set for FAD | The dataset itself, or a hand-picked `gold.txt` | there is no other Don ca tai tu corpus |
| The mood vocabulary | 13 words voted for by Essentia's 56 mood/theme labels | contract §7.2 needs a closed list |

### 1.2 The stages and the files

```
data/
  raw/                          your recordings, any format ffmpeg reads
  provenance.csv                one row per raw file: source and rights                       (you write this)
  annotations/*.csv             instrument sheets, one per annotator; the adjudicated one last (you fill these)
  gold.txt                      optional: recording ids that are known to be good
  work/
    rec/<recording_id>.flac     1  standardise   whole recording: mono, 32 kHz, trimmed, peak 0.8
    embed/<recording_id>.npz    2  embed         Essentia: p_voice [n], mood [n, 56]
    gate.jsonl, selected.jsonl  3  gate          vocal gate, then FAD; one verdict per recording
    clips.jsonl                 4  cut           one row per clip
    features.jsonl              5  features      bpm, bpm_confidence, moods per clip
    captions.jsonl              7  captions      GPT-4o cache
  audio/<clip_id>.flac          4  cut           the clips
  manifest/{all,train,val}.jsonl  8  manifest
```

Stage 6 is human: two people fill the instrument sheets. Quality is judged per **recording** (stages 2–3) before cutting, so a rejected recording costs no clip storage and all clips of a recording share one verdict.

### 1.3 External tools — what was verified

| Tool | Fact | How it was checked |
|---|---|---|
| ffmpeg / ffprobe | the decode commands in `standardise.py` | **Verified**: executed (ffmpeg 7.1), `test_decodes_a_compressed_48k_file` |
| `soxr`, `librosa`, `soundfile` | `soxr.resample(quality="HQ")`, `librosa.effects.trim`, `librosa.beat.beat_track`, `librosa.onset.onset_strength`, `librosa.autocorrelate` | **Verified**: executed by the tests with the current releases |
| `fadtk` | version 1.1.0, `requires-python >=3.10,<3.14`; command `fadtk <model> <baseline> <eval> [csv] [--inf] [--indiv] [-w N]`; model names `vggish`, `clap-laion-audio`, `clap-laion-music`, `clap-2023`, `encodec-emb`, `encodec-emb-48k`; it needs the SoX and ffmpeg executables; it caches statistics in `<dataset>/stats/<model>` and resampled copies in `<dataset>/convert/<rate>`; a file that raises an error is logged and left out of the CSV | **Verified** by reading `pyproject.toml`, `__main__.py`, `model_loader.py` and `fad.py` of `github.com/microsoft/fadtk`. **Not executed.** |
| Essentia models | `discogs-effnet-bs64-1` (16 kHz, algorithm `TensorflowPredictEffnetDiscogs`, embeddings at output `PartitionedCall:1`, 1280-d); `mtg_jamendo_moodtheme-discogs-effnet-1` (`TensorflowPredict2D`, input `model/Placeholder`, output `model/Sigmoid`, 56 classes); `voice_instrumental-discogs-effnet-1` (output `model/Softmax`, classes `["instrumental", "voice"]`) | **Verified** from the `.json` metadata published with each model at `essentia.upf.edu/models/`. **Not executed.** |
| OpenAI SDK | `client.responses.create(model=, instructions=, input=, text=, temperature=, max_output_tokens=)`, `response.output_text`, `response.usage.input_tokens` / `output_tokens` | **Verified** against the SDK's README and `response_create_params.py`. **No API call was made**; prices are **Unverified**. |

### 1.4 Decisions in this chapter

| | Decision | Reason | Config key |
|---|---|---|---|
| 1.1 | Rights gate: only `licensed_open` and `written_consent` rows are processed; a raw file without a row is an error | a heritage project must be able to show where every recording came from | — |
| 1.2 | One gain per recording, loudest sample at 0.8; head and tail quieter than 50 dB below the loudest frame are trimmed | MuCodec's own encoder scales peaks above 0.8 down, so 0.8 keeps input and reconstruction at the same level; one gain per recording keeps the relative level of its clips | `data.peak`, `data.top_db` |
| 1.3 | Vocal gate: a recording is dropped if more than 20 % of its patches have P(voice) > 0.5 | see §1.1 | `data.vocal_prob`, `data.vocal_max_recording` |
| 1.4 | Fewest clips that fit, seconds divided evenly, each cut moved to the quietest second within ± 15 s | no short remainder; cuts fall between phrases | `data.cut_slack_s` |
| 1.5 | FAD on equal-length centre excerpts (60 s) | a Gaussian fitted to few frames inflates the score, so short files would look worse | `data.fad_excerpt_s` |
| 1.6 | Discard the worst 10 % by combined score | a starting point; calibrate by listening to the files around the threshold | `data.fad_drop_fraction`, `data.fad_z_max` |
| 1.7 | Tempo is folded into 40–200 bpm and carries a confidence; below 0.3 it is flagged for a human | this music has free-rhythm passages and beat trackers answer anyway | `data.bpm_min_confidence` |
| 1.8 | Every clip gets a deterministic template caption; GPT-4o captions replace it when they pass validation | the pilot needs no API and costs nothing; a failed GPT call never leaves a clip without a caption | `data.caption_model` |
| 1.9 | The split is a hash of the recording's group id | stable when data is added; no leakage | `data.val_percent` |

**A note on validity.** The tempo estimator assumes a steady pulse; Essentia's mood model was trained on Western production music. Their outputs are *labels the model will be conditioned on*, not ground truth. Have someone who knows the music review a random 100 clips of the pilot before trusting them at scale (open question 4).

## 2. File map

| File | Responsibility |
|---|---|
| `dcttgen/data/common.py` | atomic writes, JSONL, markers for safe re-runs, a failure-tolerant stage runner |
| `dcttgen/data/vocab.py` | the mood vocabulary and its votes |
| `dcttgen/data/provenance.py` | `provenance.csv`: rights gate |
| `dcttgen/data/standardise.py` | decode → mono → resample → trim → level → FLAC |
| `dcttgen/data/embed.py` | Essentia models → per-patch voice and mood activations |
| `dcttgen/data/quality.py` | vocal gate and per-recording FAD |
| `dcttgen/data/clips.py` | clip planning and sample-exact cutting |
| `dcttgen/data/features.py` | tempo with confidence, mood words, patch slicing |
| `dcttgen/data/annotate.py` | instrument sheets, propagation to clips, annotator agreement |
| `dcttgen/data/captions.py` | template caption, caption validator, GPT-4o step with cache |
| `dcttgen/data/manifest.py` | row assembly, validator, split, report |
| `dcttgen/data/pipeline.py` | the command-line driver (Task 9) |
| `tests/test_data_*.py` | 33 tests |

## 3. Tasks

The data tests print `N tests passed` when run directly. Tests that need `soundfile`, `soxr`, `librosa` or `ffmpeg` print `skipped: …` when those are absent — read the output, a skip is not a pass.

### Task 1: shared helpers and the rights gate

**Files:** Create `dcttgen/data/__init__.py` (empty), `dcttgen/data/common.py`, `dcttgen/data/provenance.py` · Test `tests/test_data_provenance.py`

**Interfaces:**
- Produces: `read_jsonl`, `write_jsonl`, `tmp_name`, `fingerprint`, `load_marker`, `save_marker`, `run_stage(items, fn, *, workers=1, failed_path=None) -> (done, failed)`; `load_provenance(data_root, raw_dir="raw", allowed=…) -> (cleared_rows, problems)`.

`provenance.csv` has the columns `recording_id, raw_file, source_type, source_ref, rights, rights_ref, obtained_on, group_id, notes`. `rights` is one of `licensed_open`, `written_consent`, `permission_pending`, `unknown`; only the first two are processed, and those rows must carry a `source_ref` (the URL or the person), a `rights_ref` (the licence name, or where the signed consent is kept) and an ISO date. `group_id` ties together recordings of one session so that they share a split.

**On rights.** A recording downloaded from a video site is, by default, `unknown` or `permission_pending`: being publicly viewable is not a licence to train on it or to redistribute it. A recording made with a musician needs their written consent to this specific use. Write it down per recording now; reconstructing it later for 28,000 clips is not possible.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_data_provenance.py</code> — 69 lines (click to expand)</summary>

```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import csv
import tempfile
from pathlib import Path

from dcttgen.data.provenance import COLUMNS, load_provenance


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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_provenance.py` → `ModuleNotFoundError: No module named 'dcttgen.data'`
- [ ] **Step 3: implement**

**`dcttgen/data/common.py`** — 77 lines

```python
"""Helpers shared by the data stages: atomic writes, JSONL, markers for safe re-runs, a failure-tolerant runner."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def tmp_name(path) -> Path:
    """Sibling temp path. Write there, then os.replace(): a crash never leaves a half-written final file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    tmp = tmp_name(path)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def fingerprint(path) -> list[int]:
    """[size, mtime_ns] of an input file: cheap, and it changes whenever the file is rewritten."""
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def load_marker(path, key):
    """The payload saved by save_marker if it was saved for the same `key` (inputs + parameters), else None.
    A stage skips an item only when its marker matches, so changing an upstream file or a setting redoes it."""
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m["payload"] if m.get("key") == json.loads(json.dumps(key)) else None


def save_marker(path, key, payload) -> None:
    """Write the marker LAST, after every output of the item exists: a crash then leaves no marker."""
    tmp = tmp_name(path)
    tmp.write_text(json.dumps({"key": key, "payload": payload}), encoding="utf-8")
    os.replace(tmp, path)


class _Safe:
    """Call fn(item); turn an exception into a record. A class, not a closure, so it can be pickled."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, item):
        try:
            self.fn(item)
            return None
        except Exception as e:   # noqa: BLE001 - one bad file must not stop a 2,000-hour run
            return {"item": str(item), "error": f"{type(e).__name__}: {e}"}


def run_stage(items, fn, *, workers: int = 1, failed_path=None) -> tuple[int, int]:
    """Run fn(item) over all items; failures are recorded in failed_path (JSONL) and the rest still run.
    Returns (done, failed). With workers > 1, fn must be a module-level function."""
    items, safe = list(items), _Safe(fn)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(safe, items))
    else:
        results = [safe(i) for i in items]
    failed = [r for r in results if r]
    if failed_path is not None:
        write_jsonl(failed_path, failed)
    return len(items) - len(failed), len(failed)
```

**`dcttgen/data/provenance.py`** — 55 lines

```python
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_provenance.py` → `2 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/data tests/test_data_provenance.py && git commit -m "feat: data helpers and the rights gate"`

### Task 2: standardising a recording

**Files:** Create `dcttgen/data/standardise.py` · Test `tests/test_data_standardise.py`

**Interfaces:**
- Consumes: `ffmpeg` and `ffprobe` on `PATH`; `cfg.audio.sample_rate`, `cfg.data.peak`, `cfg.data.top_db`; the helpers of Task 1.
- Produces: `decode(path) -> (Float32[N, 2], rate, stderr)`; `standardise(raw, out, *, sr, peak, top_db, min_seconds) -> dict`; `standardise_one((raw, out, marker, params))` for `run_stage`.

Order matters: downmix **before** resampling (one resampler pass), trim **before** levelling (silence must not set the gain), and write 16-bit FLAC last. The downmix is `(L + R) / 2`, the same one MuCodec's encoder applies (`model.py:237`).

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_data_standardise.py</code> — 109 lines (click to expand)</summary>

```python
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
    from dcttgen.data.standardise import standardise, standardise_one
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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_standardise.py` → `skipped: …` four times (the module cannot be imported yet)
- [ ] **Step 3: implement**

**`dcttgen/data/standardise.py`** — 62 lines

```python
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_standardise.py` → `4 tests passed` **with no `skipped:` line**
- [ ] **Step 5: commit** — `git add dcttgen/data/standardise.py tests/test_data_standardise.py && git commit -m "feat: standardise recordings to mono 32 kHz FLAC"`

### Task 3: cutting recordings into clips

**Files:** Create `dcttgen/data/clips.py` · Test `tests/test_data_clips.py`

**Interfaces:**
- Consumes: `MIN_CLIP_S`, `max_clip_seconds` from `dcttgen/plan.py`.
- Produces: `plan_clips(total, cost=None, slack=15) -> list[Clip]` with `Clip(position, start, duration)`; `boundary_costs(y, sr, total)`; `cut_recording(rec_flac, out_dir, recording_id, marker, slack=15) -> list[dict]` — rows with `clip_id, recording_id, position, start, duration, audio`.

How a recording of `total` whole seconds is planned:

1. The clip count `n` is the smallest for which the caps (300 for a single clip; otherwise 270, 240, …, 240, 270) can hold `total`.
2. The seconds are divided as evenly as the caps allow. So a 301 s recording becomes 150 + 151, not 270 + 31.
3. Each boundary then moves to the quietest whole second within ± `slack` of its even position, restricted to positions that keep this clip within `[30, cap]` and leave the rest plannable. That window is never empty: the even split is always inside it.

A recording shorter than 30 s gives no clip. The fraction of a second at the end is dropped. Clips are sliced from the 16-bit samples, so they are bit-exact copies with exactly `duration × sample_rate` samples.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_data_clips.py</code> — 111 lines (click to expand)</summary>

```python
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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_clips.py` → `ModuleNotFoundError: No module named 'dcttgen.data.clips'`
- [ ] **Step 3: implement**

**`dcttgen/data/clips.py`** — 110 lines

```python
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_clips.py` → `5 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/data/clips.py tests/test_data_clips.py && git commit -m "feat: whole-second clips with positions"`

### Task 4: Essentia activations, tempo and moods

**Files:** Create `dcttgen/data/vocab.py`, `dcttgen/data/embed.py`, `dcttgen/data/features.py` · Test `tests/test_data_features.py`

**Interfaces:**
- Consumes: the three Essentia model files and their `.json` metadata in one directory; `librosa`.
- Produces: `MOODS` (13 words) and `MOOD_VOTES`; `embed(flac, out, models_dir)` writing `p_voice Float[n]` and `mood Float[n, 56]`; `mood_labels(models_dir)`; `bpm_and_confidence(y, sr) -> (bpm, confidence)`; `fold_bpm`; `pick_moods(p, labels, k=3) -> (words, top_score)`; `vocal_fraction(p_voice, thr)`; `clip_patches(a, total_s, start, duration)`.

The mood vocabulary — the closed list contract §7.2 refers to: `calm`, `relaxing`, `meditative`, `melancholic`, `sad`, `romantic`, `emotional`, `dreamy`, `hopeful`, `uplifting`, `joyful`, `energetic`, `dramatic`. Each word is voted for by one or more of the model's 56 labels (`MOOD_VOTES`); a clip's activations are averaged over its patches, the best word is always kept, and up to two more are kept if they score at least half as much.

The manifest stores `bpm` as an integer: round the folded tempo; if the confidence is below `data.bpm_min_confidence`, list the clip for review rather than dropping it.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_data_features.py</code> — 91 lines (click to expand)</summary>

```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import numpy as np

from dcttgen.data.features import bpm_and_confidence, clip_patches, fold_bpm, pick_moods, vocal_fraction
from dcttgen.data.vocab import MOOD_VOTES, MOODS

# The 56 classes of mtg_jamendo_moodtheme-discogs-effnet-1, copied from the "classes" list of its JSON file
# (https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json)
LABELS = ["action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas", "commercial",
          "cool", "corporate", "dark", "deep", "documentary", "drama", "dramatic", "dream", "emotional", "energetic",
          "epic", "fast", "film", "fun", "funny", "game", "groovy", "happy", "heavy", "holiday", "hopeful", "inspiring",
          "love", "meditative", "melancholic", "melodic", "motivational", "movie", "nature", "party", "positive",
          "powerful", "relaxing", "retro", "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space", "sport",
          "summer", "trailer", "travel", "upbeat", "uplifting"]


def test_vocabulary_is_closed_and_votes_only_for_real_labels():
    assert len(LABELS) == 56 and len(set(LABELS)) == 56
    assert all(l in LABELS for votes in MOOD_VOTES.values() for l in votes)      # no word votes for a label that is not there
    assert all(w.isalpha() and w.islower() for w in MOODS)                      # one lowercase word each (contract 7.2)
    assert "uplifting" in MOODS and "joyful" in MOODS                           # the plan's own example moods


def p_with(**labels):
    p = np.full(56, 0.01)
    for name, v in labels.items():
        p[LABELS.index(name)] = v
    return p


def test_pick_moods():
    assert pick_moods(p_with(happy=0.4, uplifting=0.3, calm=0.01), LABELS) == (["joyful", "uplifting"], 0.4)
    # the runner-up is dropped when it is below half the best score ...
    assert pick_moods(p_with(sad=0.4, calm=0.1), LABELS)[0] == ["sad"]
    # ... or below the absolute floor; at most three words
    assert pick_moods(p_with(sad=0.04, calm=0.03), LABELS)[0] == ["sad"]
    four = p_with(sad=0.4, calm=0.39, hopeful=0.38, dream=0.37, romantic=0.36)
    assert pick_moods(four, LABELS)[0] == ["sad", "calm", "hopeful"]
    # a flat output still gives one word: the manifest needs at least one mood per clip
    words, top = pick_moods(np.zeros(56), LABELS)
    assert len(words) == 1 and words[0] in MOODS and top == 0.0
    # a label vote is by name, so a different label order must give the same answer
    order = list(reversed(range(56)))
    assert pick_moods(p_with(happy=0.4)[order], [LABELS[i] for i in order])[0] == ["joyful"]


def test_fold_bpm():
    assert [fold_bpm(x) for x in (0, -5, 120, 20, 300, 39.9, 200, 250)] == [0.0, 0.0, 120, 40, 150, 79.8, 200, 125]


def test_vocal_fraction_and_patch_slicing():
    assert vocal_fraction(np.array([0.9, 0.1, 0.6, 0.2])) == 0.5
    assert vocal_fraction(np.array([])) == 1.0                                  # no evidence is treated as vocal
    a = np.arange(100)                                                          # 100 patches over a 100 s recording
    assert list(clip_patches(a, 100.0, 20, 30)) == list(range(20, 50))
    assert list(clip_patches(a, 100.0, 70, 30)) == list(range(70, 100))
    assert len(clip_patches(np.arange(3), 100.0, 0, 30)) >= 1                   # never empty


def plucks(times, seconds, sr):
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    n, t = int(0.4 * sr), np.arange(int(0.4 * sr)) / sr
    for k, t0 in enumerate(times):
        i = int(t0 * sr)
        if i + n <= len(y):
            y[i:i + n] += (np.exp(-8 * t) * np.sin(2 * np.pi * 220 * (1 + 0.25 * (k % 3)) * t)).astype(np.float32)
    return y


def test_bpm_confidence_separates_steady_from_free_rhythm():
    try:
        import librosa  # noqa: F401
    except ImportError:
        return print("skipped: librosa not installed")
    sr = 32000
    steady, conf_s = bpm_and_confidence(plucks(np.arange(0.5, 60, 0.6), 60, sr), sr)             # 100 bpm
    assert abs(steady - 100) < 4 and conf_s > 0.6, (steady, conf_s)
    rng = np.random.default_rng(0)
    _, conf_f = bpm_and_confidence(plucks(np.sort(rng.uniform(0.5, 59, 100)), 60, sr), sr)      # no pulse
    assert conf_f < 0.3 < conf_s, (conf_f, conf_s)
    assert bpm_and_confidence(np.zeros(sr * 30, dtype=np.float32), sr) == (0.0, 0.0)             # silence: nothing to track


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_features.py` → `ModuleNotFoundError: No module named 'dcttgen.data.features'`
- [ ] **Step 3: implement**

**`dcttgen/data/vocab.py`** — 21 lines

```python
"""Closed vocabularies of the manifest (contract 7.2)."""
from dcttgen.plan import INSTRUMENTS  # noqa: F401 - the six canonical names, in the contract's order; defined once, in plan.py

# Mood word -> labels of Essentia's mtg_jamendo_moodtheme model whose probabilities vote for it.
# Every label below is one of the model's 56 classes (checked by tests/test_data_features.py).
MOOD_VOTES = {
    "calm": ("calm", "soft"),
    "relaxing": ("relaxing",),
    "meditative": ("meditative",),
    "melancholic": ("melancholic",),
    "sad": ("sad",),
    "romantic": ("romantic", "love"),
    "emotional": ("emotional",),
    "dreamy": ("dream",),
    "hopeful": ("hopeful",),
    "uplifting": ("uplifting", "inspiring", "positive", "motivational"),
    "joyful": ("happy", "fun", "upbeat"),
    "energetic": ("energetic",),
    "dramatic": ("dramatic", "drama"),
}
MOODS = tuple(MOOD_VOTES)   # the closed mood vocabulary: one lowercase word each
```

**`dcttgen/data/features.py`** — 60 lines

```python
"""Per-clip musical features. The tempo comes from librosa; the vocal and mood decisions are plain numpy
functions applied to the outputs of Essentia's models (embed.py), so they run and are tested without TensorFlow."""
import numpy as np

from dcttgen.data.vocab import MOOD_VOTES


def fold_bpm(bpm: float, lo: float = 40.0, hi: float = 200.0) -> float:
    """Tempo estimators often answer half or double the felt tempo: halve or double until the value is in [lo, hi]."""
    if bpm <= 0:
        return 0.0
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def bpm_and_confidence(y: np.ndarray, sr: int, hop: int = 512) -> tuple[float, float]:
    """y: Float[N] mono. Returns (bpm, confidence in [0, 1]); (0.0, 0.0) if there is nothing to track.
    Confidence is the normalised autocorrelation of the onset envelope at one or two beat periods: close to 1 for
    a steady pulse, close to 0 for free rhythm. (beat_track's own beats are no evidence: its dynamic programme
    returns evenly spaced beats even for music that has none.)"""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, _ = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0 or len(env) < 16 or not np.any(env):
        return 0.0, 0.0
    ac = librosa.autocorrelate(env - env.mean(), max_size=len(env) // 2)
    if ac[0] <= 0:
        return 0.0, 0.0
    ac = ac / ac[0]
    lag = 60.0 * sr / (hop * bpm)                                       # frames per beat
    peaks = [ac[int(round(m * lag)) - 1: int(round(m * lag)) + 2].max() for m in (1, 2) if int(round(m * lag)) + 2 < len(ac)]
    return fold_bpm(bpm), float(np.clip(max(peaks, default=0.0), 0.0, 1.0))


def pick_moods(p: np.ndarray, labels: list[str], k: int = 3, min_p: float = 0.05, rel: float = 0.5):
    """p: Float[56], mtg_jamendo_moodtheme's sigmoid output averaged over a clip's patches. labels: the model's 56
    class names in output order (read them from its JSON file; never type them). Returns (words, top score):
    1..k words of the closed vocabulary. The best word is always kept (the manifest needs at least one); the others
    need score >= min_p and >= rel * best, which stops a weak runner-up from being written as a mood."""
    idx = {name: i for i, name in enumerate(labels)}
    score = {w: max(float(p[idx[name]]) for name in votes) for w, votes in MOOD_VOTES.items()}
    ranked = sorted(score, key=score.get, reverse=True)
    top = score[ranked[0]]
    return [ranked[0]] + [w for w in ranked[1:k] if score[w] >= min_p and score[w] >= rel * top], top


def vocal_fraction(p_voice: np.ndarray, thr: float = 0.5) -> float:
    """p_voice: Float[n], P(voice) per ~1 s patch. Fraction of patches above thr. No patches = no evidence: 1.0."""
    return float(np.mean(p_voice > thr)) if len(p_voice) else 1.0


def clip_patches(a: np.ndarray, total_s: float, start: int, duration: int) -> np.ndarray:
    """a: [n, ...], one row per patch over a recording of total_s seconds. The rows inside [start, start + duration)."""
    n = len(a)
    i, j = int(start * n / total_s), int(np.ceil((start + duration) * n / total_s))
    return a[i:max(j, i + 1)]
```

**`dcttgen/data/embed.py`** — 31 lines

```python
"""Essentia models over a standardised recording -> work/embed/<recording_id>.npz with
    p_voice Float[n]     P(voice) of each patch, from voice_instrumental-discogs-effnet-1
    mood    Float[n, 56] mtg_jamendo_moodtheme-discogs-effnet-1 activations of each patch
Linux only (essentia-tensorflow publishes no Windows wheel). Not executed in this guide: the node names, class
order and sample rate below are read from the .json metadata published next to each model file."""
import json
from pathlib import Path

import numpy as np

EFFNET = "discogs-effnet-bs64-1"
MOOD = "mtg_jamendo_moodtheme-discogs-effnet-1"
VOICE = "voice_instrumental-discogs-effnet-1"


def mood_labels(models_dir) -> list[str]:
    """The 56 class names in output order: read them from the model's own metadata, never type them."""
    return json.loads((Path(models_dir) / f"{MOOD}.json").read_text(encoding="utf-8"))["classes"]


def embed(flac, out, models_dir) -> None:
    from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    m = Path(models_dir)
    audio = MonoLoader(filename=str(flac), sampleRate=16000, resampleQuality=4)()                       # the models' sample rate
    emb = TensorflowPredictEffnetDiscogs(graphFilename=str(m / f"{EFFNET}.pb"), output="PartitionedCall:1")(audio)   # Float[n, 1280]
    mood = TensorflowPredict2D(graphFilename=str(m / f"{MOOD}.pb"), input="model/Placeholder", output="model/Sigmoid")(emb)     # Float[n, 56]
    voice = TensorflowPredict2D(graphFilename=str(m / f"{VOICE}.pb"), input="model/Placeholder", output="model/Softmax")(emb)   # Float[n, 2]
    classes = json.loads((m / f"{VOICE}.json").read_text(encoding="utf-8"))["classes"]                  # ["instrumental", "voice"]
    tmp = Path(out).with_name(Path(out).name + ".tmp.npz")
    np.savez(tmp, p_voice=np.asarray(voice)[:, classes.index("voice")], mood=np.asarray(mood))
    tmp.replace(out)
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_features.py` → `5 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/data/vocab.py dcttgen/data/embed.py dcttgen/data/features.py tests/test_data_features.py && git commit -m "feat: tempo with confidence and a closed mood vocabulary"`

**Not executed:** `embed.py` (Essentia has no Windows wheel). The node names, class order and sample rate in it are the ones in the models' own metadata (§1.3); `MonoLoader(filename=, sampleRate=, resampleQuality=)` is **Unverified** — compare with the usage snippet on each model's page before the first run. Download the `.pb` and `.json` files from the URLs in the metadata, for example `https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb`.

### Task 5: the quality gate

**Files:** Create `dcttgen/data/quality.py` · Test `tests/test_data_quality.py`

**Interfaces:**
- Consumes: `work/rec/*.flac`, `work/embed/*.npz`; `cfg.data.vocal_*`, `cfg.data.fad_*`; the `fadtk` command.
- Produces: `combine_scores(per_model, ids) -> (z, failed)`; `select_discard(z, drop_fraction, z_max)`; `parse_fadtk_csv`; `centre_excerpt`; `run(cfg, fad=run_fadtk) -> rows`, writing `work/gate.jsonl` (every recording with `status`: `ok`, `vocal`, `fad_high`, `fad_failed`) and `work/selected.jsonl`.

How the three scores become one: for each embedding, take the log of the per-file scores, subtract the median and divide by 1.4826 × the median absolute deviation. The combined score is the mean of the three. The log and the robust scale stop one embedding — or one catastrophic file — from deciding everything; `test_fad_discards_the_high_tail_and_no_embedding_dominates` multiplies one model's scores by 1,000 and gets the same verdicts.

The stage works around three behaviours of `fadtk` (§1.3): it deletes the baseline's cached statistics before each call, removes the gigabytes of resampled copies afterwards, and treats a recording missing from a CSV as `fad_failed`.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_data_quality.py</code> — 103 lines (click to expand)</summary>

```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dcttgen.data.common import read_jsonl
from dcttgen.data.quality import centre_excerpt, combine_scores, parse_fadtk_csv, run, select_discard

try:
    import soundfile as sf
except ImportError:
    sf = None


def test_parse_fadtk_csv():
    text = "/p/pool/r001.flac,12.5\n/p/pool/r002.flac,nan_x\n/p/pool/r003.flac,0.25\n\n"
    assert parse_fadtk_csv(text) == {"r001": 12.5, "r003": 0.25}          # headerless; unparsable scores are skipped
    assert parse_fadtk_csv("C:\\data\\pool\\r9.flac,3.0") == {"r9": 3.0}


def scores(ids, worst, scale):
    """100 plausible per-file scores; the first `worst` files are 8 times worse than the rest."""
    base = np.random.default_rng(1).lognormal(1.0, 0.2, len(ids))
    return {i: float(scale * b * (8 if k < worst else 1)) for k, (i, b) in enumerate(zip(ids, base))}


def test_fad_discards_the_high_tail_and_no_embedding_dominates():
    ids = [f"r{i:03d}" for i in range(100)]
    per_model = {"vggish": scores(ids, 5, 1.0), "clap-laion-music": scores(ids, 5, 1000.0), "encodec-emb": scores(ids, 5, 0.01)}
    z, failed = combine_scores(per_model, ids)
    assert failed == [] and len(z) == 100
    assert select_discard(z, drop_fraction=0.05) == set(ids[:5])          # R5: the 5 worst files, not the 5 best
    assert select_discard(z, z_max=5.0) == set(ids[:5])                     # bad files sit at z of about 11-14, good ones below 3
    assert min(z[i] for i in ids[:5]) > max(z[i] for i in ids[5:])        # the bad files are separated, not just ranked first
    # scale invariance: x1000 on one model changes nothing, whereas a raw mean would be decided by that model alone
    z2, _ = combine_scores({**per_model, "encodec-emb": {i: v * 1000 for i, v in per_model["encodec-emb"].items()}}, ids)
    assert max(abs(z[i] - z2[i]) for i in ids) < 1e-9
    raw = {i: np.mean([m[i] for m in per_model.values()]) for i in ids}
    assert max(abs(raw[i] - per_model["clap-laion-music"][i] / 3) for i in ids) < 0.1 * max(raw.values())


def test_missing_or_invalid_scores_are_failures_not_passes():
    ids = ["a", "b", "c", "d"]
    per_model = {"m1": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "m2": {"a": 1.0, "b": 2.0, "c": 3.0}}   # d missing from m2
    z, failed = combine_scores(per_model, ids)
    assert failed == ["d"] and set(z) == {"a", "b", "c"}
    per_model["m2"]["d"] = float("nan")
    assert combine_scores(per_model, ids)[1] == ["d"]
    per_model["m2"]["d"] = -0.3                                           # a negative FAD is numerical garbage
    assert combine_scores(per_model, ids)[1] == ["d"]


def test_centre_excerpt():
    y = np.arange(100)
    assert list(centre_excerpt(y, 4, 10)) == list(range(30, 70))
    assert list(centre_excerpt(y, 20, 10)) == list(range(100))           # shorter than the excerpt: whole file


def make_cfg(root, **data):
    d = dict(vocal_prob=0.5, vocal_max_recording=0.2, fad_excerpt_s=3, fad_models=["vggish", "encodec-emb"],
             fad_reference="pool", fad_drop_fraction=0.15, fad_z_max=None)
    return SimpleNamespace(paths=SimpleNamespace(data_root=str(root)), data=SimpleNamespace(**{**d, **data}))


def test_stage_applies_both_gates_and_resumes():
    if sf is None:
        return print("skipped: soundfile not installed")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "work" / "rec").mkdir(parents=True)
        (root / "work" / "embed").mkdir(parents=True)
        for k in range(8):                                                # r0..r7, 6 s each at 8 kHz
            sf.write(root / "work" / "rec" / f"r{k}.flac", np.zeros(48000, dtype=np.int16) + k, 8000, format="FLAC", subtype="PCM_16")
            voice = np.full(100, 0.9 if k == 7 else 0.05)                 # r7 is a vocal recording
            np.savez(root / "work" / "embed" / f"r{k}.npz", p_voice=voice)
        calls = []

        def fake_fad(model, baseline, eval_dir, csv):                     # stands in for the fadtk command line
            calls.append(model)
            files = sorted(Path(eval_dir).glob("*.flac"))
            assert {f.stem for f in files} == {f"r{k}" for k in range(7)}  # the vocal recording is not scored
            assert all(sf.info(f).duration == 3.0 for f in files)          # excerpts of fad_excerpt_s seconds
            csv.write_text("".join(f"{f},{10.0 if f.stem == 'r3' else 1.0 + int(f.stem[1:]) / 100}\n" for f in files))

        rows = {r["recording_id"]: r for r in run(make_cfg(root), fad=fake_fad)}
        assert rows["r7"]["status"] == "vocal" and rows["r3"]["status"] == "fad_high"
        assert [r["status"] for k, r in sorted(rows.items()) if k not in ("r3", "r7")] == ["ok"] * 6
        assert sorted(r["recording_id"] for r in read_jsonl(root / "work" / "selected.jsonl")) == ["r0", "r1", "r2", "r4", "r5", "r6"]
        assert calls == ["vggish", "encodec-emb"]
        run(make_cfg(root), fad=fake_fad)
        assert calls == ["vggish", "encodec-emb"]                         # same pool: the CSVs are reused, fadtk is not called


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_quality.py` → `ModuleNotFoundError: No module named 'dcttgen.data.quality'`
- [ ] **Step 3: implement**

**`dcttgen/data/quality.py`** — 113 lines

```python
"""Quality filter (plan 3.5): vocal gate, then per-recording Frechet Audio Distance from three embeddings."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

from dcttgen.data.common import write_jsonl
from dcttgen.data.features import vocal_fraction


def centre_excerpt(y: np.ndarray, seconds: int, sr: int) -> np.ndarray:
    """At most `seconds` of audio from the middle of y. Equal-length excerpts make per-file FAD comparable:
    a Gaussian fitted to few frames inflates the score, so a short file would otherwise look worse."""
    n = seconds * sr
    a = max(0, (len(y) - n) // 2)
    return y[a:a + n]


def parse_fadtk_csv(text: str) -> dict[str, float]:
    """fadtk --indiv writes 'path,score' lines without a header, best score first. Returns {file stem: score}.
    Lines whose score is not a number are skipped; the caller must treat a missing file as a failure."""
    out = {}
    for line in text.splitlines():
        if "," in line:
            path, _, score = line.rpartition(",")
            try:
                out[Path(path).stem] = float(score)
            except ValueError:
                pass
    return out


def combine_scores(per_model: dict[str, dict[str, float]], ids) -> tuple[dict[str, float], list[str]]:
    """per_model[model][recording_id] = that model's per-file FAD. Returns (z, failed).
    Each model's scores go through log, then (x - median) / (1.4826 * MAD): a robust z-score, so a model whose
    scores are 1000 times larger cannot dominate, and one catastrophic file cannot stretch the scale.
    The combined score is the mean of the z-scores (the plan's 'mean'); high = far from the reference = worse.
    A recording missing from any model, or with a non-finite or non-positive score, is returned in `failed`
    (fadtk drops files that raise an error without telling the caller)."""
    ok = [i for i in sorted(ids) if all(i in s and np.isfinite(s[i]) and s[i] > 0 for s in per_model.values())]
    z = np.zeros(len(ok))
    for s in per_model.values():
        x = np.log([s[i] for i in ok])
        med = np.median(x)
        z += (x - med) / (1.4826 * np.median(np.abs(x - med)) or 1.0)
    z /= len(per_model)
    return dict(zip(ok, z.tolist())), sorted(set(ids) - set(ok))


def select_discard(z: dict[str, float], drop_fraction: float = 0.1, z_max: float | None = None) -> set[str]:
    """R5: discard the HIGH tail (a high FAD means far from the reference). z_max, if set, wins over drop_fraction."""
    if z_max is not None:
        return {i for i, v in z.items() if v > z_max}
    return set(sorted(z, key=z.get, reverse=True)[:round(drop_fraction * len(z))])


def run_fadtk(model: str, baseline, eval_dir, csv, workers: int = 8) -> None:
    """One call of the fadtk command line (not executed in this guide: see the chapter)."""
    shutil.rmtree(Path(baseline) / "stats", ignore_errors=True)   # fadtk caches mu/cov per directory and never invalidates
    subprocess.run(["fadtk", model, str(baseline), str(eval_dir), str(csv), "--indiv", "-w", str(workers)], check=True)


def run(cfg, fad=run_fadtk) -> list[dict]:
    """Vocal gate -> excerpts -> fadtk per model -> combine -> work/gate.jsonl and work/selected.jsonl.
    `fad(model, baseline_dir, eval_dir, csv)` is injectable so that the stage can be tested without fadtk."""
    import soundfile as sf
    root, d = Path(cfg.paths.data_root), cfg.data
    work = root / "work"
    rows, kept = [], {}
    for p in sorted((work / "rec").glob("*.flac")):
        frac = vocal_fraction(np.load(work / "embed" / f"{p.stem}.npz")["p_voice"], d.vocal_prob)
        rows.append({"recording_id": p.stem, "vocal_fraction": frac, "status": "vocal" if frac > d.vocal_max_recording else "ok"})
        if frac <= d.vocal_max_recording:
            kept[p.stem] = p
    pool = work / "fad" / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    for f in pool.glob("*.flac"):                                         # a recording that left the pool must not be scored,
        if f.stem not in kept:                                            # nor may its embeddings enter the baseline statistics
            f.unlink()
            for e in pool.glob(f"embeddings/*/{f.stem}.npy"):
                e.unlink()
    for rid, p in kept.items():
        if not (pool / f"{rid}.flac").exists():
            y, sr = sf.read(p, dtype="int16")
            sf.write(pool / f"{rid}.flac", centre_excerpt(y, d.fad_excerpt_s, sr), sr, format="FLAC", subtype="PCM_16")
    baseline = pool
    if d.fad_reference == "gold":                                         # data/gold.txt: one recording_id per line
        baseline = work / "fad" / "gold"
        shutil.rmtree(baseline, ignore_errors=True)
        baseline.mkdir(parents=True)
        for rid in (root / "gold.txt").read_text().split():
            shutil.copy2(pool / f"{rid}.flac", baseline / f"{rid}.flac")  # FileNotFoundError if a gold recording was gated out
    tag = hashlib.sha1("\n".join(sorted(kept)).encode()).hexdigest()[:8]   # a different pool gets different CSV names
    per_model = {}
    for model in d.fad_models:
        csv = work / "fad" / f"{model}-{tag}.csv"
        if not csv.exists():
            fad(model, baseline, pool, csv)
            shutil.rmtree(pool / "convert", ignore_errors=True)           # fadtk's resampled WAV copies: gigabytes, never reused
        per_model[model] = parse_fadtk_csv(csv.read_text())
    z, _ = combine_scores(per_model, kept)
    gone = select_discard(z, d.fad_drop_fraction, d.fad_z_max)
    for r in rows:
        rid = r["recording_id"]
        if rid in z:
            r["fad_z"], r["status"] = z[rid], "fad_high" if rid in gone else "ok"
        elif rid in kept:
            r["status"] = "fad_failed"                                    # missing from a CSV: never silently kept
    write_jsonl(work / "gate.jsonl", rows)                                # every recording with its verdict
    write_jsonl(work / "selected.jsonl", [r for r in rows if r["status"] == "ok"])
    return rows
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_quality.py` → `5 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/data/quality.py tests/test_data_quality.py && git commit -m "feat: vocal gate and combined per-recording FAD"`

**Not executed:** `run_fadtk` (the real tool). The stage's logic ran with an injected fake.

### Task 6: instrument annotation

**Files:** Create `dcttgen/data/annotate.py` · Test `tests/test_data_annotate.py`

**Interfaces:**
- Produces: `write_sheet(path, recording_ids)`; `parse_cell`; `load_annotations(paths)`; `clip_instruments(row, start, duration, min_overlap=10) -> list[str]`; `agreement(rows_a, rows_b)`; `in_overlap_set(recording_id, percent=10)`.

The workflow for two annotators:

1. Generate one sheet per annotator from the selected recordings. Each row is a recording; each of the six instrument columns takes an empty cell (not heard), `1` (heard), or spans in seconds such as `0-95;240-300` (heard only there). The `other` column is for a loud instrument that is not one of the six.
2. Split the recordings between the annotators, **except** the reproducible 10 % for which `in_overlap_set` is true: both annotate those.
3. Compute `agreement` on the overlap. A kappa below about 0.6 for an instrument means the two do not hear it the same way: discuss, agree on a rule, re-annotate.
4. Resolve the disagreements in a third, adjudicated file and load it last (a later file replaces earlier rows).

A recording-level `1` is inherited by every clip; a span counts for a clip when it overlaps it by at least 10 seconds. A clip with no instrument at all cannot satisfy the contract and is left out of the manifest, with the reason logged.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_data_annotate.py</code> — 67 lines (click to expand)</summary>

```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import tempfile
from pathlib import Path

from dcttgen.data.annotate import (agreement, clip_instruments, cohen_kappa, in_overlap_set, load_annotations,
                                   parse_cell, write_sheet)
from dcttgen.data.vocab import INSTRUMENTS


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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_annotate.py` → `ModuleNotFoundError: No module named 'dcttgen.data.annotate'`
- [ ] **Step 3: implement**

**`dcttgen/data/annotate.py`** — 84 lines

```python
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_annotate.py` → `4 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/data/annotate.py tests/test_data_annotate.py && git commit -m "feat: instrument sheets and annotator agreement"`

### Task 7: captions

**Files:** Create `dcttgen/data/captions.py` · Test `tests/test_data_captions.py`

**Interfaces:**
- Produces: `template_caption(moods, instruments, bpm) -> str`; `tempo_word(bpm)`; `validate_caption(caption, instruments, moods=()) -> list[str]`; `caption_one(row, client, model, attempts, bands)`; `generate(rows, client, cache_path, *, model, attempts=3, workers=8, bands=(70, 110)) -> int`; `cached_captions(rows, cache_path, model, bands) -> {clip_id: caption}`.

The template reproduces the plan's own example word for word (`test_template_reproduces_the_plans_example`). The validator treats GPT output as untrusted data: one ASCII line, at most 60 words, the words "Don ca tai tu", **every** annotated instrument named and **no** other instrument, nothing vocal, at least one annotated mood word, and no `<` or `>`.

The GPT-4o step asks for a JSON object `{"caption": …}` through the Responses API's structured output, validates it, and on failure repeats the request with the validator's complaints appended, up to `data.caption_attempts` times. Every reply is appended to `work/captions.jsonl` immediately, keyed by `clip_id` and by a hash of everything the caption depends on (bpm, tempo word, moods, instruments, model, system prompt) — so a crash loses only the calls in flight, a re-run asks only for what is missing, and re-annotating a clip invalidates its old caption automatically.

- [ ] **Step 1: write the failing test** — the API client is a fake; no network is used.

<details>
<summary><code>tests/test_data_captions.py</code> — 92 lines (click to expand)</summary>

```python
"""Run: pytest -q tests/test_data_captions.py   or   python tests/test_data_captions.py     (no network: the API client is a fake)"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcttgen.data.captions import cached_captions, caption_one, generate, template_caption, tempo_word, validate_caption  # noqa: E402

PLAN_EXAMPLE = "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute"
THREE = ["zither", "two-string fiddle", "moon-shaped lute"]
OK = "A calm Don ca tai tu piece played by zither and monochord"


def test_template_reproduces_the_plans_example():
    assert template_caption(["joyful", "uplifting"], THREE, 120) == PLAN_EXAMPLE
    assert template_caption(["emotional"], ["monochord"], 60) == "An emotional Don ca tai tu piece with slow tempo, performed by monochord"
    assert [tempo_word(b) for b in (69, 70, 109, 110)] == ["slow", "moderate", "moderate", "fast"]
    assert validate_caption(PLAN_EXAMPLE, THREE, ["joyful", "uplifting"]) == []


def test_validator_rejects_each_kind_of_bad_caption():
    inst = ["zither", "monochord"]
    assert validate_caption(OK, inst, ["calm"]) == []
    cases = [("A calm Don ca tai tu piece played by zither", "does not name the annotated instrument monochord"),
             (OK + " and piano", "not one of the annotated instruments"),
             (OK + " and bamboo flute", "not annotated for this clip"),
             (OK + " with a singer", "mentions vocals"),
             (OK.replace("Don ca tai tu ", ""), "must say 'Don ca tai tu'"),
             (OK + " <EOA>", "must not contain < or >"),
             (OK + "\nsecond line", "must be one line"),
             (OK.replace("calm", "cälm"), "plain ASCII"),
             (OK + " word" * 60, "at most 60 words"),
             ("", "empty"), (None, "empty")]
    for text, needle in cases:
        problems = validate_caption(text, inst, ["calm"])
        assert any(needle in p for p in problems), (text, problems)
    assert validate_caption(OK, inst, ["sad"]) == ["uses none of the annotated mood words"]


class FakeClient:
    """Stands in for openai.OpenAI(): .responses.create(**kw) returns the next scripted reply."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.responses = NS(create=self._create)

    def _create(self, **kw):
        self.requests.append(kw)
        return NS(output_text=self.replies.pop(0), usage=NS(input_tokens=100, output_tokens=20))


def reply(caption):
    return json.dumps({"caption": caption})


ROW = {"clip_id": "c0", "bpm": 60, "moods": ["calm"], "instruments": ["zither", "monochord"]}


def test_gpt_step_feeds_complaints_back_and_gives_up_cleanly():
    client = FakeClient([reply(OK + " and piano"), "not json", reply(OK)])
    rec = caption_one(ROW, client, "gpt-4o", 3, (70, 110))
    assert rec["source"] == "gpt" and rec["caption"] == OK and rec["attempts"] == 3
    assert rec["input_tokens"] == 300 and rec["output_tokens"] == 60
    assert "rejected" not in client.requests[0]["input"] and "piano" in client.requests[1]["input"]
    assert "not the requested JSON" in client.requests[2]["input"] and client.requests[0]["model"] == "gpt-4o"
    failed = caption_one(ROW, FakeClient([reply(OK + " and piano")] * 2), "gpt-4o", 2, (70, 110))
    assert failed["source"] == "failed" and failed["caption"] is None and failed["attempts"] == 2 and failed["problems"]


def test_cache_is_resumable_and_never_stale():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "work" / "captions.jsonl"
        rows = [ROW, {**ROW, "clip_id": "c1"}]
        opts = dict(model="gpt-4o", attempts=2, workers=1, bands=(70, 110))
        assert generate(rows, FakeClient([reply(OK), reply(OK + " and piano"), reply(OK + " and piano")]), cache, **opts) == 2
        got = cached_captions(rows, cache, "gpt-4o", (70, 110))
        assert len(got) == 1 and set(got.values()) == {OK}                          # the failed clip has no caption
        assert generate(rows, FakeClient([reply(OK)]), cache, **opts) == 1         # only the failed clip is asked again
        assert cached_captions(rows, cache, "gpt-4o", (70, 110)) == {"c0": OK, "c1": OK}
        assert generate(rows, FakeClient([]), cache, **opts) == 0                  # nothing left to do: no request is made
        changed = [{**ROW, "moods": ["sad"]}, rows[1]]                             # the clip was re-annotated
        assert cached_captions(changed, cache, "gpt-4o", (70, 110)) == {"c1": OK}  # its old caption is not reused
        assert cached_captions(rows, cache, "another-model", (70, 110)) == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_data_captions: {len(tests)} tests passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_captions.py` → `ModuleNotFoundError: No module named 'dcttgen.data.captions'`
- [ ] **Step 3: implement**

**`dcttgen/data/captions.py`** — 144 lines

```python
"""Captions: a deterministic template (free; the pilot uses it) and the plan's GPT-4o step with validation."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dcttgen.data.common import read_jsonl

MAX_WORDS = 60
# Every English word that counts as a mention of one of the six instruments. "gong" alone is read as gong ban.
ALIASES = {
    "moon-shaped lute": ("moon-shaped lute", "moon lute", "lute"),
    "two-string fiddle": ("two-string fiddle", "two-stringed fiddle", "fiddle"),
    "zither": ("zither",),
    "monochord": ("monochord",),
    "bamboo flute": ("bamboo flute", "flute"),
    "gong ban": ("gong ban", "gong"),
}
# Instruments the dataset does not annotate: a caption naming one is a hallucination. Extend as GPT invents new ones.
NOT_ANNOTATED = ("piano", "guitar", "violin", "cello", "harp", "drum", "drums", "percussion", "sitar", "erhu", "pipa",
                 "guzheng", "koto", "saxophone", "trumpet", "organ", "synthesizer", "accordion", "banjo", "ukulele",
                 "oboe", "clarinet", "harmonica", "orchestra", "xylophone", "mandolin", "gamelan")
# The training data is instrumental (see "Vocals"): a caption must not suggest singing.
VOCAL = ("vocal", "vocals", "vocalist", "singer", "singing", "sung", "voice", "voices", "lyrics", "choir", "chanting", "humming")
_OWNER = {a: inst for inst, names in ALIASES.items() for a in names}
_MENTION = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted([*_OWNER, *NOT_ANNOTATED, *VOCAL], key=len, reverse=True))) + r")\b", re.I)


def tempo_word(bpm: float, slow_below: float = 70, fast_from: float = 110) -> str:
    return "slow" if bpm < slow_below else "fast" if bpm >= fast_from else "moderate"


def join_and(items) -> str:
    items = list(items)
    return " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", and " + items[-1]


def template_caption(moods: list[str], instruments: list[str], bpm: float, slow_below: float = 70, fast_from: float = 110) -> str:
    """The plan's own sentence shape, filled from the manifest fields. No period, as in the plan's example."""
    article = "An" if moods[0][0] in "aeiou" else "A"
    return (f"{article} {join_and(moods)} Don ca tai tu piece with {tempo_word(bpm, slow_below, fast_from)} tempo, "
            f"performed by {join_and(instruments)}")


def validate_caption(caption, instruments: list[str], moods: list[str] = ()) -> list[str]:
    """Problems with a caption (empty list = acceptable). Instruments are checked both ways: every annotated
    instrument is named, and no other instrument (or anything vocal) is. Untrusted text: GPT output is data."""
    if not isinstance(caption, str) or not caption.strip():
        return ["the caption is empty"]
    bad = []
    if re.search(r"[\r\n\t]", caption):
        bad.append("must be one line")
    if not caption.isascii():
        bad.append("must be plain ASCII (no accents, no typographic quotes or dashes)")
    if "<" in caption or ">" in caption:
        bad.append("must not contain < or >")                                  # never let text look like a control token
    if len(caption.split()) > MAX_WORDS:
        bad.append(f"must be at most {MAX_WORDS} words")
    if "don ca tai tu" not in caption.lower():
        bad.append("must say 'Don ca tai tu'")
    named = set()
    for m in _MENTION.finditer(caption):                                       # longest alias first: 'two-string fiddle' is one hit
        w = m.group().lower()
        if w in _OWNER:
            named.add(_OWNER[w])
            if _OWNER[w] not in instruments:
                bad.append(f"names {_OWNER[w]}, which is not annotated for this clip")
        elif w in VOCAL:
            bad.append(f"mentions vocals ('{w}'); the recordings are instrumental")
        else:
            bad.append(f"names '{w}', which is not one of the annotated instruments")
    bad += [f"does not name the annotated instrument {i}" for i in instruments if i not in named]
    if moods and not any(re.search(rf"\b{re.escape(m)}\b", caption, re.I) for m in moods):
        bad.append("uses none of the annotated mood words")
    return list(dict.fromkeys(bad))


SYSTEM = ("You write one-sentence English captions for instrumental recordings of Don ca tai tu, a traditional "
          "Southern Vietnamese music form. Use only the facts you are given. Name every listed instrument exactly as "
          "spelled, and no other instrument. Use at least one of the listed moods as a word. Do not mention singing, "
          "voices or lyrics. Do not invent titles, places, people or dates. Write the genre as \"Don ca tai tu\". "
          "Plain ASCII, one line, at most 40 words. Example for tempo slow, moods calm and meditative, instruments X and Y: "
          "\"A calm and meditative Don ca tai tu piece at a slow tempo, played by X and Y.\"")
FORMAT = {"format": {"type": "json_schema", "name": "caption", "strict": True,
                     "schema": {"type": "object", "properties": {"caption": {"type": "string"}},
                                "required": ["caption"], "additionalProperties": False}}}


def user_message(row: dict, bands, problems=()) -> str:
    facts = {"bpm": row["bpm"], "tempo": tempo_word(row["bpm"], *bands), "moods": row["moods"], "instruments": row["instruments"]}
    msg = json.dumps(facts)
    return msg + (f"\nYour previous caption was rejected: {'; '.join(problems)}. Write a new one." if problems else "")


def inputs_hash(row: dict, model: str, bands) -> str:
    """Identifies everything the caption depends on. A cached caption is reused only if this still matches,
    so re-annotating a clip or editing the prompt can never leave a stale caption behind."""
    facts = [row["bpm"], tempo_word(row["bpm"], *bands), row["moods"], row["instruments"], model, SYSTEM]
    return hashlib.sha1(json.dumps(facts).encode()).hexdigest()[:12]


def caption_one(row: dict, client, model: str, attempts: int, bands) -> dict:
    """Ask up to `attempts` times, feeding the validator's complaints back. source is 'gpt' or 'failed'."""
    problems, used = [], {"input_tokens": 0, "output_tokens": 0}
    rec = {"clip_id": row["clip_id"], "inputs": inputs_hash(row, model, bands), "model": model}
    for attempt in range(1, attempts + 1):
        resp = client.responses.create(model=model, instructions=SYSTEM, input=user_message(row, bands, problems),
                                       text=FORMAT, temperature=0.7, max_output_tokens=150)
        used["input_tokens"] += resp.usage.input_tokens
        used["output_tokens"] += resp.usage.output_tokens
        try:
            caption = json.loads(resp.output_text)["caption"].strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            problems = ["the reply was not the requested JSON object"]
            continue
        problems = validate_caption(caption, row["instruments"], row["moods"])
        if not problems:
            return {**rec, "caption": caption, "source": "gpt", "attempts": attempt, **used}
    return {**rec, "caption": None, "source": "failed", "attempts": attempts, "problems": problems, **used}


def generate(rows, client, cache_path, *, model: str, attempts: int = 3, workers: int = 8, bands=(70, 110)) -> int:
    """rows: dicts with clip_id, bpm, moods, instruments. Appends one record per call to cache_path (JSONL; the last
    record of a clip wins), skipping clips whose cached caption is valid for the current inputs. Returns #clips asked."""
    cache_path = Path(cache_path)
    cache = {r["clip_id"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    todo = [r for r in rows if not (cache.get(r["clip_id"], {}).get("source") == "gpt"
                                    and cache[r["clip_id"]]["inputs"] == inputs_hash(r, model, bands))]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex, open(cache_path, "a", encoding="utf-8", newline="\n") as f:
        for fut in as_completed([ex.submit(caption_one, r, client, model, attempts, bands) for r in todo]):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()                                                          # a crash loses at most the calls in flight
    return len(todo)


def cached_captions(rows, cache_path, model: str, bands=(70, 110)) -> dict[str, str]:
    """clip_id -> caption for the cached GPT captions that are still valid for each row's CURRENT fields."""
    if not Path(cache_path).exists():
        return {}
    cache = {r["clip_id"]: r for r in read_jsonl(cache_path)}                # the last record of a clip wins
    return {r["clip_id"]: cache[r["clip_id"]]["caption"] for r in rows
            if cache.get(r["clip_id"], {}).get("source") == "gpt" and cache[r["clip_id"]]["inputs"] == inputs_hash(r, model, bands)}
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_captions.py` → `test_data_captions: 4 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/data/captions.py tests/test_data_captions.py && git commit -m "feat: template and validated GPT-4o captions"`

**Cost of the GPT-4o step (estimate).** One request is roughly 200 input tokens (the instructions plus the facts) and 40 output tokens. For 28,000 clips with a few retries: about 6–7 million input and 1.2–1.5 million output tokens. Multiply by the current per-million prices on OpenAI's pricing page — **Unverified** here; the `input_tokens` and `output_tokens` recorded in `captions.jsonl` give the exact total after the pilot.

### Task 8: the manifest, its validator and the split

**Files:** Create `dcttgen/data/manifest.py` · Test `tests/test_data_manifest.py`

**Interfaces:**
- Consumes: clip rows (Task 3), features (Task 4), instruments (Task 6), captions (Task 7), `Plan`, `plan_sections`, `max_clip_seconds`, `MOODS`.
- Produces: `split_of(group_id, val_percent=2.0) -> "train" | "val"`; `validate_row(row, audio=None, sample_rate=None) -> list[str]`; `validate_manifest(rows, data_root=None, sample_rate=None, probe=probe_flac) -> list[str]`; `build_rows(clips, features, instruments, captions, groups=None, val_percent=2.0) -> (rows, dropped)`; `write_manifests(rows, data_root, sample_rate=None)`; `report(rows) -> dict`; `python -m dcttgen.data.manifest --config C`.

The validator enforces every rule of contract §7.1 and §7.2: the field rules (through `Plan.from_manifest`, so the manifest and the language model can never disagree about what a valid plan is), the mood vocabulary, intro and outro only where the position allows them, unique clip ids, one split per recording, and — when given the data root — that each file exists and is mono, at the right rate, with exactly `duration × rate` samples. `write_manifests` writes nothing if a single rule is broken.

- [ ] **Step 1: write the failing test** — one valid row (the contract's example) and one violation per rule.

<details>
<summary><code>tests/test_data_manifest.py</code> — 117 lines (click to expand)</summary>

```python
"""Run: pytest -q tests/test_data_manifest.py   or   python tests/test_data_manifest.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcttgen.data.common import read_jsonl  # noqa: E402
from dcttgen.data.manifest import build_rows, report, split_of, validate_manifest, validate_row, write_manifests  # noqa: E402

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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_data_manifest.py` → `ModuleNotFoundError: No module named 'dcttgen.data.manifest'`
- [ ] **Step 3: implement**

**`dcttgen/data/manifest.py`** — 172 lines

```python
"""The manifest (contract 7.2): assemble the rows, enforce every rule, split by recording, write the three files, report.

python -m dcttgen.data.manifest --config C      validates manifest/all.jsonl and the audio files; exit code 0 = milestone M1 is met
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from dcttgen.data.common import read_jsonl, write_jsonl
from dcttgen.data.vocab import MOODS
from dcttgen.plan import MIN_CLIP_S, POSITIONS, Plan, max_clip_seconds, plan_sections

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
    from dcttgen.config import load_config      # chapter 04
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_data_manifest.py` → `test_data_manifest: 4 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/data/manifest.py tests/test_data_manifest.py && git commit -m "feat: manifest assembly, validator and leak-free split"`

### Task 9: the pipeline driver

**Files:** Create `dcttgen/data/pipeline.py` · Test `tests/test_data_pipeline.py`

**Interfaces:**
- Consumes: every function above; `load_config` (chapter 04).
- Produces: `python -m dcttgen.data.pipeline <stage> --config C [--workers N]`.

This file is glue with no decisions of its own; the table fixes what each stage reads, calls and writes.

| `<stage>` | Reads | Calls | Writes |
|---|---|---|---|
| `standardise` | `provenance.csv`, `raw/` | `load_provenance` (stop if it reports any problem) → `run_stage(standardise_one)` with `sr=cfg.audio.sample_rate`, `peak`, `top_db`, `min_seconds=30` | `work/rec/<id>.flac`, `work/rec/<id>.json` (marker), `work/failed_standardise.jsonl` |
| `embed` | `work/rec/*.flac` | `embed.embed` for each recording without an `.npz` | `work/embed/<id>.npz` |
| `gate` | `work/rec`, `work/embed` | `quality.run(cfg)` | `work/gate.jsonl`, `work/selected.jsonl` |
| `cut` | `work/selected.jsonl` | `cut_recording(…, slack=cfg.data.cut_slack_s)` for each selected recording | `audio/<clip_id>.flac`, `work/clips.jsonl` |
| `features` | `work/clips.jsonl`, `audio/`, `work/embed` | per clip: `bpm_and_confidence`; `pick_moods(clip_patches(mood, …).mean(0), mood_labels(…))` | `work/features.jsonl`, `work/review_bpm.jsonl` (confidence below the threshold) |
| `sheets` | `work/selected.jsonl` | `write_sheet` for each annotator | `annotations/<annotator>.csv` (never overwrites an existing sheet) |
| `captions` | clips, features, annotations | `captions.generate` with `openai.OpenAI()` | `work/captions.jsonl` |
| `manifest` | all of the above, `provenance.csv` | `clip_instruments`, `template_caption`, `cached_captions` (a valid GPT caption replaces the template), `build_rows`, `write_manifests` | `manifest/*.jsonl`, `work/dropped.jsonl`, `work/report.json` |

- [ ] **Step 1: write the failing test** — `test_pipeline_runs_end_to_end_on_synthetic_recordings`: write three sine-tone recordings (40 s, 200 s, 700 s) and a `provenance.csv` into a temporary `data/`, run `standardise`, then `cut`, `features`, `manifest` with the `embed` and `gate` outputs supplied as files (`p_voice` zeros, uniform `mood`, every recording `ok`) and a one-row-per-recording annotation sheet marking `zither` as `1`. Assert: `validate_manifest(read_jsonl("manifest/all.jsonl"), data_root, 32000) == []`; the 700 s recording yields clips with positions `first, middle, last`; every caption equals `template_caption(...)` of its row; running `manifest` twice gives identical files.
- [ ] **Step 2: run it, expect failure** — `ModuleNotFoundError: No module named 'dcttgen.data.pipeline'`
- [ ] **Step 3: implement** `main(argv=None)` with one small function per row of the table. Print one summary line per stage (`done`, `failed`, `skipped`).
- [ ] **Step 4: run it, expect a pass**
- [ ] **Step 5: commit** — `git add dcttgen/data/pipeline.py tests/test_data_pipeline.py && git commit -m "feat: data pipeline driver"`

**Not written in this guide:** the body of `pipeline.py` and its test. Everything it calls is tested above.

## 4. Running it

```bash
C=configs/bootstrap_k1.yaml,configs/pilot.yaml          # any overlay works: the data stages read only audio.* and data.*

python -m dcttgen.data.pipeline standardise --config $C --workers 8
python -m dcttgen.data.pipeline embed       --config $C      # Linux, Essentia
python -m dcttgen.data.pipeline gate        --config $C      # Linux, fadtk; listen to files near the threshold in work/gate.jsonl
python -m dcttgen.data.pipeline cut         --config $C
python -m dcttgen.data.pipeline features    --config $C --workers 8
python -m dcttgen.data.pipeline sheets      --config $C      # then two people annotate; see Task 6
python -m dcttgen.data.pipeline manifest    --config $C      # template captions: the pilot can stop here
export OPENAI_API_KEY=...                                    # never put the key in a file under version control
python -m dcttgen.data.pipeline captions    --config $C
python -m dcttgen.data.pipeline manifest    --config $C      # again: valid GPT captions replace the templates
python -m dcttgen.data.manifest --config $C                  # exit code 0 = milestone M1
```

**Sizes and times (estimates, not measurements).**

| | Pilot, about 20 h | Full, about 2,000 h |
|---|---|---|
| Clip storage (32 kHz mono 16-bit FLAC; raw PCM is 230 MB per hour, FLAC roughly half) | 2–3 GB | 230–280 GB |
| `work/rec` (a second copy of the same audio) | 2–3 GB | 230–280 GB — delete after `cut` if disk is short |
| Standardise and cut | minutes | CPU-bound and parallel: budget a day on 8 cores, then measure |
| Essentia and fadtk | under an hour on one GPU | measure on the pilot and scale by 100 |
| Annotation | one sheet row per recording, not per clip | the slowest stage: plan people, not machines |
| GPT-4o captions | not needed (template) | see Task 7 |

## 5. What will go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `orphan: x.mp3 has no row in provenance.csv` | a raw file nobody documented | add the row, or remove the file |
| Every standardise test prints `skipped:` | `soundfile`, `soxr`, `librosa` or `ffmpeg` missing | install them; a skip is not a pass |
| `silent: peak below -60 dBFS` | an empty or broken download | re-download; the item is listed in `work/failed_standardise.jsonl` |
| The manifest check reports `expected … exactly duration x rate samples` | clips were produced by another tool, or resampled after cutting | only `cut_recording` writes `audio/` |
| Nearly every recording is `vocal` | the classifier hears instruments as voice (the two-string fiddle is a candidate) | listen to 20 rejected recordings; raise `data.vocal_prob` or `data.vocal_max_recording`; see open question 1 |
| `fad_failed` for many recordings | SoX or ffmpeg not found by fadtk, or files it cannot read | fadtk logs the error per file; fix and re-run — finished models are not recomputed |
| The same tempo for almost every clip, or many at exactly half or double | free rhythm, or octave errors | use `work/review_bpm.jsonl`; consider dropping `bpm` precision to the tempo word |
| GPT captions fail validation repeatedly for one instrument name | GPT paraphrases it ("two-stringed violin") | add the phrase to `ALIASES` or `NOT_ANNOTATED`; the clip keeps its template caption meanwhile |
| `recording X is in both splits` | `group_id` changed between runs | keep `provenance.csv` under version control; the split is a pure function of the group id |

## 6. Open questions for the research team

1. **Vocals.** Don ca tai tu is, literally, playing *and singing*. Dropping every recording with singing may discard most of what is available; separating the voice out (for example with a source-separation model) keeps more data but leaves artefacts. Recommended: run the vocal gate on the pilot, count what survives, listen, then decide. The plan lists vocals as future work, so the choice also shapes that.
2. **"Gong ban".** The plan names six instruments; five have obvious Vietnamese counterparts for annotators. This guide does not guess the sixth. Please give annotators the Vietnamese name and a reference recording for each of the six.
3. **The FAD reference.** With `fad_reference: pool`, a file is judged by its distance from the dataset's own average, so the filter removes *outliers*, not *bad audio* as such. A hand-picked `gold.txt` of 100–200 recordings that experts consider exemplary makes the filter mean what the plan intends.
4. **Are the mood words meaningful for this music?** They come from a model trained on other music. An expert review of 100 pilot clips will tell. If they are not, annotate moods by hand alongside instruments — the sheet format extends naturally.
5. **Position labels (contract D4).** A clip from the middle of a long performance has no `[intro]`. If you prefer the plan's literal three-section layout for every clip, pass `position="whole"` when cutting; nothing else changes.
6. **The dataset size.** Every estimate above assumes the plan's 2,000 hours. If the real corpus is much smaller, the K = 4 codec and the larger backbones are the first things to reconsider (chapters 02 and 03).
