# DcttGen Implementation Guide — 04. Training, Inference, Evaluation and Delivery

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this chapter task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the pieces that make chapters 01–03 one runnable project: environments, repository scaffold, configuration, the one training loop, text-to-music inference, evaluation, and the order in which to do everything.
**Architecture:** a strict YAML config; one `fit()` built on `accelerate` that trains any module whose `forward(batch)` returns a loss dictionary and resumes exactly; inference and evaluation split into stages that can run in different Python environments.
**Tech stack:** `accelerate` 1.15, `torch`, `safetensors`, `PyYAML`, `soundfile`; `stable-audio-metrics` at commit `fd55536` for the three metrics.
**Spec:** plan §3.6, §4, §5, Table 3 · contract: [00 — Overview, Spec and Shared Contracts](00-overview-and-contracts.md)

## Global constraints

- `fit(model, train_loader, val_loader, cfg, run_dir)` calls **`model(batch)`** — the module itself, never a method — so that distributed training sees the forward pass (contract §9).
- Every step count in the config (`warmup_steps`, `max_steps`, `log_every`, `eval_every`, `save_every`) is in **optimiser steps**. Gradient accumulation changes how many batches make a step, nothing else.
- Optimiser: AdamW, `lr 3e-4`, `weight_decay 0.1`, `betas (0.9, 0.999)`, `eps 1e-9` (plan §3.6); linear warm-up, cosine decay to `lr_min 3e-5` (contract D10).
- Trainable parameters and optimiser state are `float32`; half precision is autocast only. `fit` refuses anything else.
- Checkpoints are `runs/<stage>/<name>/step_<7 digits>/` with `model.safetensors` inside (contract §7.4), written atomically.
- A config key that does not exist in `configs/base.yaml` is an error, in an overlay file and in an override alike.
- Nothing under `data/`, `runs/` or `third_party/`, and no key or weight file, goes into git.

## Review focus

| # | Input or condition | Expected behaviour | Test |
|---|---|---|---|
| 1 | A run is killed and restarted, mid-epoch or at an epoch boundary | It continues to the **same weights** as an uninterrupted run | `test_resume_is_exact_across_an_epoch_boundary` |
| 2 | A typo in a config key or an override (`train.lr_mim=1e-5`) | An error naming the key and the nearest real one — never silently ignored | `test_overrides_are_typed_and_strict`, `test_unknown_key_in_an_overlay_file_is_an_error` |
| 3 | The loss becomes `nan` | The run stops with a clear error; the last checkpoint is intact | `test_non_finite_loss_stops_the_run_with_the_last_checkpoint_intact` |
| 4 | A checkpoint of a model with tied weights, or one written from a distributed run | It loads back, strictly | `test_checkpoints_load_back_into_a_tied_model_even_with_a_ddp_prefix` |
| 5 | A prompt that is empty, enormous or not a string; only some of the plan fields given; a duration of 301 | Rejected before any model is loaded | `test_bad_input_is_rejected_before_any_model_is_loaded` |

## 1. Background and design

### 1.1 Four environments, not one

**Verified:** MuCodec's encoder imports `transformers.deepspeed` (`muq_dev/muq_fairseq/models/muq/modules/flash_conformer.py:29`). That module does not exist in `transformers` 4.57.1 — importing it there raises `ModuleNotFoundError` (checked on the authoring machine). The language-model code of chapter 03 was written and tested on 4.57.1 only. So the two cannot share a Python environment as they are, and the project is laid out for that:

| Environment | Runs | Key packages | Platform |
|---|---|---|---|
| `data` | chapter 01 stages | `ffmpeg`, `soundfile`, `soxr`, `librosa`; Essentia (TensorFlow build); `fadtk` 1.1.0; `openai` | Linux for Essentia and fadtk |
| `codec` | `codec.train`, `codec.tokenize`, the audio half of inference | MuCodec's `requirements.txt` — `torch 2.2.0+cu118`, `fairseq 0.12.2`, `diffusers 0.27.2`, `transformers 4.42.4`, `numpy 1.23.5` (**Verified** by reading the file) — plus this package | Linux + CUDA, Python 3.10 |
| `lm` | `lm.train`, the code half of inference | a current `torch`, `transformers 4.57.1`, `accelerate 1.15` | Linux + CUDA |
| `eval` | the `score` stage | `stable-audio-metrics`' `requirements.txt` — `torch 2.1.0`, `tensorflow 2.13.1`, `openl3 0.4.2`, `laion-clap 1.1.4` (**Verified** by reading the file) | Linux + CUDA; its README says GPU only |

The stages hand files to each other (`.npy` codes, `.wav` audio), so no process ever needs two environments. `text_to_music` (contract §9) still exists for the day one environment can import both sides.

**Unverified — check during milestone M0**, each with a one-line command:

- MuCodec's pins install on Python 3.10 (its README names 3.8.12; this package needs 3.10 for its type syntax). `fairseq 0.12.2` depends on old `omegaconf`/`hydra-core` releases that recent `pip` versions are known to reject; if `pip install` fails on their metadata, install with an older `pip` (below 24.1).
- `accelerate 1.15` (needed by `engine.py`) works next to `torch 2.2.0` in the `codec` environment. The engine was tested with `torch 2.9.0` only.
- The exact file layout of the Hugging Face repository `yaoxunxu/mucodec`.

### 1.2 Configuration

One file, `configs/base.yaml`, lists **every** key with its default and a comment. Overlays (`bootstrap_k1.yaml`, `plan_k4.yaml`, `pilot.yaml`) and `--override a.b=value` may only change keys that exist there, and values are coerced to the type of the default — `1e-4` typed without a dot arrives from YAML as a string and is converted. `--config` accepts several overlays separated by commas; later ones win. Relative entries under `paths:` are made absolute against the repository root, so MuCodec's `chdir` (chapter 02) cannot break them.

### 1.3 The training loop

| Concern | What `fit` does | Why |
|---|---|---|
| Learning rate | `lr·(step+1)/warmup` for `step < warmup`, then `lr_min + ½(lr − lr_min)(1 + cos(π·progress))` | contract D10; set on the optimiser directly each step, so there is no scheduler state to save |
| Weight decay | on tensors with 2 or more dimensions; none on biases and norm gains | the usual AdamW practice; the plan is silent |
| Precision | autocast `bf16`; weights and optimiser state stay `float32` | `eps = 1e-9` underflows in half precision; `bf16` weights lose small updates |
| Accumulation | `accelerate`'s accumulate context; gradients are clipped and the step taken on the last micro-batch | effective batch = `batch_size × grad_accum × processes` |
| Validation | the mean of every scalar the model returns, under a fixed forked random stream | comparable between checkpoints; does not disturb the training stream |
| Saving | into `.tmp_step_*`, renamed when complete; `trainer_state.json` holds step, epoch and position in the epoch; old checkpoints are pruned | a crash never leaves a half-written `step_*` |
| Resuming | newest `step_*` → `load_state` → skip the batches already seen in this epoch → restore the random streams as saved | the resumed run reproduces the uninterrupted one exactly |
| Hook | `on_step_end(step)` on the unwrapped module, after each optimiser step | the RF Transformer's moving average |
| Safety | a non-finite loss raises; an empty loader raises; non-`float32` trainable parameters raise | fail loudly, with the last checkpoint intact |

`load_weights(module, checkpoint)` is the strict loader for inference-time code: it understands the tied-weight omission (contract §7.4) and the `module.` prefix of checkpoints written by a distributed run.

### 1.4 Inference in two steps

`prompt_to_codes` (language-model environment) → `[K, 25 × duration]` codes → `codes_to_wav` (codec environment). The prompt is untrusted: it is collapsed to one line and bounded in length before anything is loaded; the plan fields are validated by constructing a `Plan` (chapter 03). **Decision:** the four plan fields are all-or-nothing — give `duration`, `bpm`, `moods` and `instruments`, or none of them and let the model write its plan. Partly constrained plans are not supported.

### 1.5 Evaluation protocol

**Plan §4:** FAD_openl3, KLD_passt and CLAP_score on a 2 % validation split, against baselines trained on the same data.

| Step | What | Detail |
|---|---|---|
| Prompts | the captioned validation clips | `eval.num_prompts: 0` = all; otherwise a seeded subset |
| Generation | one piece per prompt, seed = `eval.seed + index` | `eval.seconds: 0` uses each clip's own plan and duration — the plan's protocol; `30…300` generates pieces of one fixed length with the clip's bpm, moods and instruments (cheaper; the reference is then the clip's centre excerpt of that length) |
| Reference | the validation clips themselves | written as WAV next to the generated files |
| Metrics | `stable-audio-metrics` | the three functions below |

**Verified** by reading `stable-audio-metrics` at commit `fd55536` (not executed — it is GPU-only):

| Metric | Function | Facts |
|---|---|---|
| FD-openl3 | `src/openl3_fd.py`: `openl3_fd(channels, samplingrate, content_type, openl3_hop_size, eval_path, eval_files_extension='.wav', ref_path=None, ref_files_extension='.wav', load_ref_embeddings=None, batching=False)` | `samplingrate` is the bandwidth evaluated, up to 48 kHz; `content_type` is `'music'` or `'env'`; OpenL3's window is 1 s |
| KL-PaSST | `src/passt_kld.py`: `passt_kld(ids, eval_path, eval_files_extension='.wav', ref_path=None, ref_files_extension='.wav', load_ref_probabilities=None, no_ids=[], collect='mean')` | files are paired by id; inputs longer than PaSST's 10 s are split into overlapping windows and pooled |
| CLAP score | `src/clap_score.py`: `clap_score(id2text, audio_path, audio_files_extension='.wav', clap_model='630k-audioset-fusion-best.pt')` | cosine similarity of LAION-CLAP text and audio embeddings |

**Decision — evaluate at 32 kHz mono** (`eval.fd`), the dataset's own format. The settings Stable Audio's published numbers use are different: the reference statistics shipped in the repository are named `…channels2__44100__openl3music__openl3hopsize0.5__batch4`. Numbers computed with our settings are comparable **among the models in this study**, not with numbers copied from other papers.

**How much to trust FD on a 2 % split.** A Fréchet distance compares two covariance matrices and needs far more embedding vectors than embedding dimensions. With the full dataset the split is about 560 clips of several minutes — workable. With the pilot's handful of clips the number is noise: report `n` beside every score, and do not draw conclusions from the pilot's FD.

The results table to fill (the plan's Table 3 values for DcttGen are FAD 1.291, KLD 0.472, CLAP 0.394):

| Model | n | FD-openl3 ↓ | KL-PaSST ↓ | CLAP score ↑ |
|---|---|---|---|---|
| MusicLM | | | | |
| MusicGen-Medium | | | | |
| AudioLDM2-Music | | | | |
| Stable Audio Open | | | | |
| DcttGen (bootstrap, K = 1) | | | | |
| DcttGen (plan, K = 4) | | | | |

## 2. File map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `.gitignore` | the package and what stays out of git |
| `configs/base.yaml`, `bootstrap_k1.yaml`, `plan_k4.yaml`, `pilot.yaml` | every key; the two codec configurations; the small-scale overlay |
| `dcttgen/config.py` | `load_config` |
| `dcttgen/engine.py` | `fit`, `lr_at`, `make_optimizer`, `latest_checkpoint`, `load_weights` |
| `dcttgen/infer.py` | `prompt_to_codes`, `codes_to_wav`, `text_to_music`, the command line |
| `dcttgen/eval/run.py` | the three evaluation stages |
| `tests/test_config.py`, `test_engine.py`, `test_infer.py`, `test_integration.py`, `ddp_check.py` | 23 tests and one two-process check |

## 3. Tasks

### Task 0: repository, environments, third-party code

**Files:** Create `pyproject.toml`, `.gitignore`, `dcttgen/__init__.py` (empty), `dcttgen/eval/__init__.py` (empty)

**`pyproject.toml`** — 29 lines

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "dcttgen"
version = "0.1.0"
description = "DcttGen: text-to-music for Don ca tai tu"
requires-python = ">=3.10"
# Only what every environment needs. transformers is NOT here: the language-model environment uses 4.57.1, the codec environment keeps MuCodec's pin.
dependencies = [
  "torch>=2.2",
  "numpy",
  "pyyaml>=6",
  "accelerate>=1.15,<2",
  "safetensors>=0.4.3",
  "soundfile>=0.10.2",
]

[project.optional-dependencies]
lm = ["transformers==4.57.1"]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["dcttgen*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**`.gitignore`** — 15 lines

```text
# data, runs, weights and secrets never go into git (leading slash: top level only, so dcttgen/data/ stays tracked)
/data/
/runs/
third_party/MuCodec/
third_party/stable-audio-metrics/
.venv*/
*.pt
*.pth
*.ckpt
*.safetensors
.env
*.key
__pycache__/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 1: repository**

```bash
git init
mkdir -p dcttgen/data dcttgen/codec dcttgen/lm dcttgen/eval tests configs third_party
touch dcttgen/__init__.py dcttgen/data/__init__.py dcttgen/codec/__init__.py dcttgen/lm/__init__.py dcttgen/eval/__init__.py
git add pyproject.toml .gitignore dcttgen && git commit -m "chore: scaffold"
```

- [ ] **Step 2: third-party code, pinned** (**Not executed** in this guide)

```bash
git clone https://github.com/xuyaoxun/MuCodec third_party/MuCodec          # includes about 120 MB of test audio
git -C third_party/MuCodec checkout 128f91b
git clone https://github.com/Stability-AI/stable-audio-metrics third_party/stable-audio-metrics
git -C third_party/stable-audio-metrics checkout fd55536

# the three checkpoints MuCodec's readme names; their final places are fixed by its code
huggingface-cli download yaoxunxu/mucodec --local-dir third_party/_mucodec_weights
find third_party/_mucodec_weights -name "*.pt" -o -name "*.pth"          # locate mucodec.pt, muq.pt, audioldm_48k.pth
mkdir -p third_party/MuCodec/ckpt
mv <path>/mucodec.pt       third_party/MuCodec/ckpt/mucodec.pt
mv <path>/muq.pt           third_party/MuCodec/muq_dev/muq.pt
mv <path>/audioldm_48k.pth third_party/MuCodec/tools/audioldm_48k.pth   # also at huggingface.co/haoheliu/audioldm_48k
```

- [ ] **Step 3: environments** (**Not executed**; see the Unverified list in §1.1)

```bash
# lm - also where the CPU test suite runs
python3.10 -m venv .venv-lm && . .venv-lm/bin/activate
pip install torch && pip install -e ".[lm,dev]"

# codec
python3.10 -m venv .venv-codec && . .venv-codec/bin/activate
pip install -r third_party/MuCodec/requirements.txt && pip install -e . scipy

# eval
python3.10 -m venv .venv-eval && . .venv-eval/bin/activate
pip install -r third_party/stable-audio-metrics/requirements.txt && pip install -e . --no-deps && pip install pyyaml soundfile
```

- [ ] **Step 4: import smoke tests** — each must print `ok`

```bash
.venv-lm/bin/python    -c "import dcttgen.lm.model, dcttgen.engine; print('ok')"
.venv-codec/bin/python -c "import fairseq, diffusers, dcttgen.engine, dcttgen.codec.codec; print('ok')"
.venv-eval/bin/python  -c "import openl3, laion_clap, dcttgen.eval.run; print('ok')"
```

### Task 1: configuration

**Files:** Create `configs/base.yaml`, `configs/bootstrap_k1.yaml`, `configs/plan_k4.yaml`, `configs/pilot.yaml`, `dcttgen/config.py` · Test `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path, overrides=()) -> Config` (contract §9); `Config` is a `dict` with attribute access and `to_dict()`; `ROOT`.

**`configs/base.yaml`** — 88 lines

```yaml
# Every key that exists is documented here; overlays and --override may only change keys listed in this file.
paths:
  data_root: data                 # manifest/, audio/, codes/        (relative paths are resolved against the repository root)
  mucodec_root: third_party/MuCodec
  runs: runs
audio:
  sample_rate: 32000              # Plan 3.5 - dataset format, mono
  frame_rate: 25                  # Plan 3.2 - code frames per second, per codebook
codec:
  tag: k4v10000                   # names data/codes/<tag>/
  num_codebooks: 4                # K - Plan 3.2
  codebook_size: 10000            # V - Plan 3.2
  muq_layer: 7                    # MuEncoder layer feeding the RVQ (released default)
  window_frames: 896              # 35.84 s training window - Plan 3.2
  checkpoint: null                # our trained RVQ + RF weights; null = released MuCodec weights (valid only for K=1, V=16384)
data:                             # chapter 01
  peak: 0.8                       # loudest sample of a standardised recording (MuCodec's own encoder limits peaks to 0.8)
  top_db: 50.0                    # head and tail quieter than this below the loudest frame are trimmed
  cut_slack_s: 15                 # how far a cut may move from its even position to reach a quiet second
  vocal_prob: 0.5                 # P(voice) above which a patch counts as sung
  vocal_max_recording: 0.2        # recordings with a larger sung fraction are dropped
  fad_models: [vggish, clap-laion-music, encodec-emb]   # fadtk model names - Plan 3.5: VGGish, CLAP, EnCodec
  fad_reference: pool             # pool = the dataset itself | gold = the recordings listed in data/gold.txt
  fad_excerpt_s: 60               # equal-length centre excerpts make per-file scores comparable
  fad_drop_fraction: 0.1          # R5: the HIGH tail is discarded
  fad_z_max: null                 # if set, discard z > fad_z_max instead of a fixed fraction
  bpm_min_confidence: 0.3         # below this the tempo is flagged for a human check
  tempo_bands: [70, 110]          # bpm < 70 = slow, >= 110 = fast (caption wording)
  caption_model: gpt-4o           # Plan 3.5
  caption_attempts: 3
  val_percent: 2.0                # Plan 4.1
rf:                               # chapter 02
  width: 1024                     # full size: 313.1 M parameters (pilot overlay: 512 / 8 / 8 = 28.8 M)
  depth: 24
  heads: 16
  cond_drop: 0.1                  # how often the code condition is replaced by the null vector in training
  ema_decay: 0.9999
  w_commit: 0.25                  # RVQ loss weights, as in the example at the end of MuCodec's descript_quantize3.py
  w_codebook: 1.0
  norm_batches: 200               # batches used by codec.train --fit-norm
  decode_steps: 16                # Euler steps of Codec.decode; the sweep in chapter 02 decides the final number
  decode_cfg: 1.5                 # guidance scale (the released decoder's value)
  vae_window: 128                 # latent frames per Mel-VAE / HiFi-GAN window (10.24 s)
  vae_hop: 96
lm:
  backbone: Qwen/Qwen2.5-0.5B     # D11
  loss_on_plan: true              # D8
  checkpoint: null
  attn: sdpa                      # attention implementation passed to transformers: sdpa | flash_attention_2 | eager
  grad_checkpointing: false       # trades speed for memory
  max_tokens: 32768               # token budget of one batch: documents x longest document
  max_seconds: null               # drop clips longer than this (pilot runs, small GPUs)
  num_workers: 4
train:                            # Plan 3.6 - the same optimiser for both models. All step counts are OPTIMISER steps.
  lr: 3.0e-4
  lr_min: 3.0e-5                  # D10
  weight_decay: 0.1               # applied to tensors with ndim >= 2 only
  betas: [0.9, 0.999]
  eps: 1.0e-9
  warmup_steps: 500               # Decision: the plan gives no schedule length
  max_steps: 20000                # Decision: the plan gives no length. 20,000 steps x 64 items = about 46 passes over 28,000 clips; set per stage
  batch_size: 1                   # per process, per micro-batch; read by the stage entry points that build the loaders
  num_workers: 4                  # same
  grad_accum: 8                   # micro-batches per optimiser step; effective batch = batch_size * grad_accum * processes
  grad_clip: 1.0                  # global-norm clip; 0 disables. Decision: the plan is silent
  mixed_precision: bf16           # no | bf16 | fp16 - parameters and optimiser state stay float32
  seed: 1234
  log_every: 10
  eval_every: 500
  save_every: 1000
  keep_last: 2                    # newest checkpoints kept; 0 keeps all
infer:                            # dcttgen.infer
  temperature: 1.0                # 0 = greedy (the plan's argmax); Decision: sample, see chapter 03
  top_k: 250                      # Decision: a starting point, tune by ear
  top_p: null
  steps: null                     # null = rf.decode_steps
  cfg_scale: null                 # null = rf.decode_cfg
  dtype: float32                  # dtype of the language model at inference: float32 | bfloat16 | float16
  max_prompt_chars: 500           # about the 128-token caption cap for English text
eval:                             # dcttgen.eval.run
  split: val                      # manifest/<split>.jsonl
  num_prompts: 0                  # 0 = every clip of the split, else a seeded random subset of this size
  seconds: 0                      # 0 = each clip's own plan and duration (the plan's protocol); 30..300 = every piece that long
  plan_mode: given                # given = plan from the manifest row | self = the model writes its own plan (length not controlled)
  seed: 0                         # clip i is generated with seed + i
  fd: {channels: 1, sample_rate: 32000, content_type: music, hop: 0.5, batch: 4}   # openl3; Decision: 32 kHz mono, the dataset's own format
  clap_model: 630k-audioset-fusion-best.pt
  metrics_repo: third_party/stable-audio-metrics
```

**`configs/bootstrap_k1.yaml`** — 2 lines

```yaml
# D1: the released MuCodec weights (works on day one)
codec: {tag: k1v16384, num_codebooks: 1, codebook_size: 16384}
```

**`configs/plan_k4.yaml`** — 2 lines

```yaml
# D1: the plan's codec. After M3 set codec.checkpoint to runs/codec/<name>/step_<N>
codec: {tag: k4v10000, num_codebooks: 4, codebook_size: 10000}
```

**`configs/pilot.yaml`** — 5 lines

```yaml
# About 20 hours of audio: a scale overlay, combine with a codec overlay: --config configs/bootstrap_k1.yaml,configs/pilot.yaml
train: {warmup_steps: 50, max_steps: 1000, grad_accum: 4, eval_every: 100, save_every: 250, keep_last: 2}
rf: {width: 512, depth: 8, heads: 8}
lm: {max_seconds: 120, max_tokens: 12288}
eval: {num_prompts: 8, seconds: 30}
```

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_config.py</code> — 84 lines (click to expand)</summary>

```python
"""Run: pytest -q tests/test_config.py   or   python tests/test_config.py"""
import copy
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcttgen.config import ROOT, load_config  # noqa: E402

BASE = str(ROOT / "configs" / "base.yaml")
OVERLAYS = ROOT / "configs"


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{exc.__name__} not raised")


def test_base_matches_contract_section_6():
    c = load_config(BASE)
    assert (c.audio.sample_rate, c.audio.frame_rate) == (32000, 25)
    assert (c.codec.tag, c.codec.num_codebooks, c.codec.codebook_size) == ("k4v10000", 4, 10000)
    assert (c.codec.muq_layer, c.codec.window_frames, c.codec.checkpoint) == (7, 896, None)
    assert (c.lm.backbone, c.lm.loss_on_plan, c.lm.checkpoint) == ("Qwen/Qwen2.5-0.5B", True, None)
    t = c.train
    assert (t.lr, t.lr_min, t.weight_decay, t.betas, t.eps) == (3.0e-4, 3.0e-5, 0.1, [0.9, 0.999], 1.0e-9)


def test_overlays_merge_over_base_and_compose():
    k1 = load_config(str(OVERLAYS / "bootstrap_k1.yaml"))
    assert (k1.codec.tag, k1.codec.num_codebooks, k1.codec.codebook_size) == ("k1v16384", 1, 16384)
    assert k1.codec.window_frames == 896 and k1.train.lr == 3.0e-4  # untouched keys survive the deep merge
    both = load_config(str(OVERLAYS / "bootstrap_k1.yaml") + "," + str(OVERLAYS / "pilot.yaml"))
    assert both.codec.num_codebooks == 1 and both.train.max_steps == 1000 and both.train.warmup_steps == 50


def test_overrides_are_typed_and_strict():
    c = load_config(BASE, ["train.lr=1e-4", "train.max_steps=300", "train.betas=[0.9, 0.95]", "lm.loss_on_plan=false",
                           "codec.tag=123", "lm.checkpoint=runs/lm/a/step_0000100", "eval.fd.hop=1"])
    assert c.train.lr == 1e-4 and isinstance(c.train.lr, float)  # PyYAML alone would give the string "1e-4"
    assert c.train.max_steps == 300 and isinstance(c.train.max_steps, int)
    assert c.train.betas == [0.9, 0.95] and c.lm.loss_on_plan is False
    assert c.codec.tag == "123" and c.lm.checkpoint == "runs/lm/a/step_0000100"
    assert c.eval.fd.hop == 1.0 and isinstance(c.eval.fd.hop, float)
    assert "did you mean 'train.lr'" in raises(KeyError, load_config, BASE, ["train.lrr=1"])
    assert "unknown config key 'trian'" in raises(KeyError, load_config, BASE, ["trian.lr=1"])
    assert "expected int" in raises(TypeError, load_config, BASE, ["train.max_steps=1.5"])
    assert "expected bool" in raises(TypeError, load_config, BASE, ["lm.loss_on_plan=1"])
    assert "expected float" in raises(TypeError, load_config, BASE, ["train.lr=fast"])
    assert "section.key=value" in raises(ValueError, load_config, BASE, ["train.lr"])


def test_unknown_key_in_an_overlay_file_is_an_error():
    d = Path(tempfile.mkdtemp())
    (d / "typo.yaml").write_text("train: {warmup_step: 5}\n", encoding="utf-8")
    assert "did you mean 'train.warmup_steps'" in raises(KeyError, load_config, str(d / "typo.yaml"))
    (d / "float.yaml").write_text("train: {lr: 1e-4}\n", encoding="utf-8")  # YAML 1.1 parses this as a string
    assert load_config(str(d / "float.yaml")).train.lr == 1e-4


def test_attribute_access_paths_and_copies():
    c = load_config(BASE)
    assert c.train.betas is c["train"]["betas"] and c.paths.runs == str(ROOT / "runs")
    assert all(os.path.isabs(v) for v in c.paths.values())
    elsewhere = str(Path(tempfile.gettempdir()) / "runs_elsewhere")  # absolute on every platform
    assert load_config(BASE, ["paths.runs=" + elsewhere]).paths.runs == elsewhere
    c.train.lr = 5.0  # attribute assignment writes through
    assert c["train"]["lr"] == 5.0
    for clone in (copy.deepcopy(c), pickle.loads(pickle.dumps(c))):  # DataLoader workers pickle the config
        assert clone == c and clone.train.lr == 5.0
    assert type(c.to_dict()["train"]) is dict
    assert raises(AttributeError, lambda: c.nope) == "nope"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{Path(__file__).name}: {len(tests)} passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_config.py` → `ModuleNotFoundError: No module named 'dcttgen.config'`
- [ ] **Step 3: implement**

**`dcttgen/config.py`** — 102 lines

```python
"""Configuration (contract section 9): configs/base.yaml <- overlay file(s) <- "a.b=value" overrides.

Strict on purpose: a key that is not already in base.yaml is an error, so a typo cannot silently do nothing.
"""
import copy
import difflib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # repository root: the directory that holds configs/


class Config(dict):
    """A dict whose items are also attributes: cfg.codec.tag is cfg["codec"]["tag"]."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}


def _wrap(x):
    return Config({k: _wrap(v) for k, v in x.items()}) if isinstance(x, dict) else x


def _coerce(key, old, new):
    """Convert new to the type of old or raise TypeError. PyYAML reads 1e-4 (no dot) as a string; float() fixes that."""
    if old is None or (new is None and isinstance(old, str)):
        return new  # a null default accepts anything; a string key may be reset to null
    if isinstance(old, bool):
        ok = isinstance(new, bool)
    elif isinstance(old, (int, float)):
        try:
            num = None if isinstance(new, bool) else float(new)
        except (TypeError, ValueError):
            num = None
        if num is not None and (isinstance(old, float) or num.is_integer()):
            return num if isinstance(old, float) else int(num)
        ok = False
    else:
        ok = isinstance(new, type(old))
    if not ok:
        raise TypeError(f"{key}: expected {type(old).__name__}, got {new!r}")
    return new


def _merge(base: dict, over: dict, prefix: str = "") -> None:
    for key, val in over.items():
        if key not in base:
            near = difflib.get_close_matches(key, list(base), n=1)
            hint = f" (did you mean {(prefix + near[0])!r}?)" if near else ""
            raise KeyError(f"unknown config key {(prefix + key)!r}{hint}; add it to configs/base.yaml first")
        if isinstance(base[key], dict) and isinstance(val, dict):
            _merge(base[key], val, f"{prefix}{key}.")
        else:
            base[key] = _coerce(f"{prefix}{key}", base[key], val)


def _override(cfg: dict, item: str) -> None:
    dotted, sep, raw = item.partition("=")
    if not sep or not dotted:
        raise ValueError(f"override {item!r} must look like section.key=value")
    parents, leaf = dotted.split(".")[:-1], dotted.split(".")[-1]
    node = cfg
    for p in parents:
        node = node.get(p) if isinstance(node, dict) else None
    old = node.get(leaf) if isinstance(node, dict) else None
    val = yaml.safe_load(raw)
    if isinstance(old, str) and val is not None and not isinstance(val, str):
        val = raw  # codec.tag=123 stays the string "123"
    for p in [leaf, *reversed(parents)]:
        val = {p: val}
    _merge(cfg, val)


def load_config(path: str, overrides: list[str] = ()) -> Config:
    """Deep-merge path over configs/base.yaml, then apply "a.b=value" overrides.

    path may name several overlays separated by commas ("configs/bootstrap_k1.yaml,configs/pilot.yaml"); later ones win.
    Relative entries under paths: become absolute (against the repository root), so a later os.chdir cannot break them.
    """
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    for p in str(path).split(","):
        over = yaml.safe_load(Path(p.strip()).read_text(encoding="utf-8")) or {}
        if not isinstance(over, dict):
            raise ValueError(f"{p}: the top level must be a mapping")
        _merge(cfg, over)
    for item in overrides:
        _override(cfg, item)
    for k, v in cfg["paths"].items():
        if not os.path.isabs(v):
            cfg["paths"][k] = str(ROOT / v)
    return _wrap(cfg)
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_config.py` → `test_config.py: 5 passed`
- [ ] **Step 5: commit** — `git add configs dcttgen/config.py tests/test_config.py && git commit -m "feat: strict typed configuration"`

### Task 2: the training loop

**Files:** Create `dcttgen/engine.py` · Test `tests/test_engine.py`, `tests/ddp_check.py`

**Interfaces:**
- Consumes: `cfg.train.*`; any `nn.Module` whose `forward(batch: dict)` returns a dict with `"loss"`; a `DataLoader` over a map-style dataset whose order depends only on the epoch (a batch sampler with `set_epoch`, as chapter 03's `TokenBudgetSampler`, is supported).
- Produces: `fit(model, train_loader, val_loader, cfg, run_dir)` (contract §9); `lr_at(step, t)`; `make_optimizer(model, t)`; `latest_checkpoint(run_dir)`; `load_weights(module, checkpoint)`. In the run directory: `config.yaml`, `log.jsonl` (one JSON object per line, `split` = `train` or `val`), `step_*/` with `model.safetensors`, `optimizer.bin`, `random_states_<rank>.pkl`, `trainer_state.json`.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_engine.py</code> — 217 lines (click to expand)</summary>

```python
"""Run: pytest -q tests/test_engine.py   or   python tests/test_engine.py   (needs accelerate; CPU only)"""
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcttgen.config import ROOT, load_config  # noqa: E402
from dcttgen.engine import fit, latest_checkpoint, load_weights, lr_at, make_optimizer  # noqa: E402

BASE = str(ROOT / "configs" / "base.yaml")
V, D, T = 17, 8, 9


class Seqs(Dataset):
    """Token sequences obeying next = (cur + 3) % V: learnable by a bigram model, and a pure function of the index."""

    def __init__(self, n=20):
        self.start = torch.arange(n) % V

    def __len__(self):
        return len(self.start)

    def __getitem__(self, i):
        return {"ids": (self.start[i] + 3 * torch.arange(T)) % V}


class TinyLM(nn.Module):
    """Tied embedding and head (like Qwen2.5), dropout (so that RNG restoration matters), an on_step_end hook."""

    def __init__(self, dropout=0.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb, self.mix, self.norm, self.drop = nn.Embedding(V, D), nn.Linear(D, D), nn.LayerNorm(D), nn.Dropout(dropout)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight
        self.calls, self.crash_at, self.nan_at = [], None, None

    def forward(self, batch):
        ids = batch["ids"]
        logits = self.head(self.drop(self.norm(torch.tanh(self.mix(self.emb(ids[:, :-1]))))))
        loss = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1))
        if self.nan_at is not None and len(self.calls) + 1 == self.nan_at:
            loss = loss * float("nan")
        return {"loss": loss, "acc": (logits.argmax(-1) == ids[:, 1:]).float().mean()}

    def on_step_end(self, step):
        self.calls.append(step)
        if step == self.crash_at:
            raise KeyboardInterrupt("simulated crash")


def cfg_for(*overrides):
    return load_config(BASE, ["train.mixed_precision=no", "train.log_every=1", "train.eval_every=1000", "train.save_every=1000",
                              "train.warmup_steps=2", *overrides])


def loader(n=20, batch_size=4, shuffle=True):
    return DataLoader(Seqs(n), batch_size=batch_size, shuffle=shuffle)


def records(run, split="train"):
    lines = (Path(run) / "log.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["split"] == split]


def weights(run, step):
    return load_file(str(Path(run) / f"step_{step:07d}" / "model.safetensors"))


def tmp():
    return Path(tempfile.mkdtemp())


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{exc.__name__} not raised")


def test_lr_boundaries():
    t = cfg_for("train.warmup_steps=10", "train.max_steps=110").train
    close = lambda a, b: abs(a - b) < 1e-12  # noqa: E731
    assert close(lr_at(0, t), 3e-4 / 10) and close(lr_at(9, t), 3e-4) and close(lr_at(10, t), 3e-4)  # warm-up ends at the peak
    assert close(lr_at(60, t), (3e-4 + 3e-5) / 2)  # halfway through the cosine
    assert close(lr_at(110, t), 3e-5) and close(lr_at(10**6, t), 3e-5)  # D10: the floor is lr_min, and it stays there
    assert all(lr_at(s, t) >= lr_at(s + 1, t) for s in range(10, 110))
    assert close(lr_at(0, cfg_for("train.warmup_steps=0").train), 3e-4)


def test_weight_decay_groups():
    m = TinyLM()
    m.mix.bias.requires_grad = False  # a frozen tensor is in no group
    groups = make_optimizer(m, cfg_for().train).param_groups
    decay, none = groups
    assert decay["weight_decay"] == 0.1 and none["weight_decay"] == 0.0
    assert {id(p) for p in decay["params"]} == {id(m.emb.weight), id(m.mix.weight)}  # tied tensor listed once
    assert {id(p) for p in none["params"]} == {id(m.norm.weight), id(m.norm.bias)}  # norm gain and bias, no decay
    assert groups[0]["betas"] == (0.9, 0.999) and groups[0]["eps"] == 1e-9


def test_loss_falls_and_checkpoint_layout():
    run = tmp()
    cfg = cfg_for("train.max_steps=30", "train.grad_accum=1", "train.save_every=10", "train.lr=0.03", "train.lr_min=0.003")
    fit(TinyLM(), loader(), None, cfg, run)
    rec = records(run)
    assert len(rec) == 30 and rec[-1]["loss"] < 0.5 * rec[0]["loss"]  # the model learns next = cur + 3
    assert [r["lr"] for r in rec[:3]] == [lr_at(s, cfg.train) for s in range(3)]  # the logged lr is the schedule
    steps = sorted(p.name for p in Path(run).glob("step_*"))
    assert steps == ["step_0000020", "step_0000030"]  # step_0000010 pruned by keep_last=2
    for name in ("model.safetensors", "optimizer.bin", "trainer_state.json", "random_states_0.pkl"):
        assert (Path(run) / "step_0000030" / name).exists(), name
    assert json.loads((Path(run) / "step_0000030" / "trainer_state.json").read_text())["step"] == 30
    assert not list(Path(run).glob(".tmp_step_*")) and (Path(run) / "config.yaml").exists()
    assert latest_checkpoint(run) == str(Path(run) / "step_0000030")


def test_resume_is_exact_across_an_epoch_boundary():
    # 10 samples, batch 4 -> 3 micro-batches per epoch; grad_accum 2 -> accumulation windows straddle epoch boundaries.
    # save_every=3 writes the checkpoint at the last batch of epoch 1; save_every=2 writes it one batch into epoch 1.
    for save_every, crash_at in ((3, 4), (2, 3)):
        cfg = cfg_for("train.max_steps=6", "train.grad_accum=2", f"train.save_every={save_every}", "train.lr=0.03")
        straight, crashed = tmp(), tmp()
        fit(TinyLM(dropout=0.2), loader(10), None, cfg, straight)
        broken = TinyLM(dropout=0.2)
        broken.crash_at = crash_at  # dies after the checkpoint was written, while the next step is in flight
        assert raises(KeyboardInterrupt, fit, broken, loader(10), None, cfg, crashed) == "simulated crash"
        assert latest_checkpoint(crashed).endswith(f"step_{save_every:07d}")
        fit(TinyLM(dropout=0.2, seed=99), loader(10), None, cfg, crashed)  # new objects, other init: everything comes from disk
        a, b = weights(straight, 6), weights(crashed, 6)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)  # bit-identical, dropout masks included
        assert [r["step"] for r in records(crashed)] == [1, 2, 3, 4, 5, 6]  # no gap, no repeated step
        assert [r["loss"] for r in records(straight)] == [r["loss"] for r in records(crashed)]


def test_accumulation_equals_the_big_batch():
    one, two = tmp(), tmp()
    fit(TinyLM(), loader(8, 8, shuffle=False), None, cfg_for("train.max_steps=3", "train.grad_accum=1", "train.save_every=3"), one)
    fit(TinyLM(), loader(8, 4, shuffle=False), None, cfg_for("train.max_steps=3", "train.grad_accum=2", "train.save_every=3"), two)
    a, b = weights(one, 3), weights(two, 3)
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)  # same update from 2 x 4 as from 1 x 8


def test_on_step_end_counts_optimiser_steps_not_batches():
    m = TinyLM()
    fit(m, loader(), None, cfg_for("train.max_steps=5", "train.grad_accum=2", "train.save_every=5"), tmp())
    assert m.calls == [1, 2, 3, 4, 5]  # the hook runs on the unwrapped module, once per optimiser step


def test_validation_leaves_the_training_stream_alone():
    cfg = cfg_for("train.max_steps=6", "train.grad_accum=1", "train.save_every=6", "train.eval_every=2")
    with_val, without = tmp(), tmp()
    fit(TinyLM(dropout=0.2), loader(10), loader(8, shuffle=False), cfg, with_val)
    fit(TinyLM(dropout=0.2), loader(10), None, cfg, without)
    a, b = weights(with_val, 6), weights(without, 6)
    assert all(torch.equal(a[k], b[k]) for k in a)  # validation drew random numbers, but from a forked stream
    val = records(with_val, "val")
    assert [r["step"] for r in val] == [2, 4, 6] and all("val_loss" in r and "val_acc" in r for r in val)


def test_half_precision_parameters_are_refused():
    msg = raises(TypeError, fit, TinyLM().half(), loader(), None, cfg_for("train.max_steps=1"), tmp())
    assert "float32" in msg
    assert str(torch.tensor(1e-9, dtype=torch.float16).item()) == "0.0"  # why: Adam eps vanishes in float16


def test_non_finite_loss_stops_the_run_with_the_last_checkpoint_intact():
    run, m = tmp(), TinyLM()
    m.nan_at = 7
    msg = raises(FloatingPointError, fit, m, loader(), None, cfg_for("train.max_steps=10", "train.grad_accum=1", "train.save_every=5"), run)
    assert "step 7" in msg and sorted(p.name for p in Path(run).glob("step_*")) == ["step_0000005"]


def test_an_empty_loader_is_an_error_not_an_endless_loop():
    assert "no batches" in raises(RuntimeError, fit, TinyLM(), DataLoader(Seqs(0), batch_size=4), None, cfg_for("train.max_steps=2"), tmp())


def test_checkpoints_load_back_into_a_tied_model_even_with_a_ddp_prefix():
    run = tmp()
    fit(TinyLM(), loader(), None, cfg_for("train.max_steps=3", "train.grad_accum=1", "train.save_every=3"), run)
    ckpt = Path(latest_checkpoint(run))
    saved = load_file(str(ckpt / "model.safetensors"))
    assert "head.weight" not in saved and "emb.weight" in saved  # save_state stores a tied tensor once ...
    assert "head.weight" in raises(RuntimeError, TinyLM().load_state_dict, saved)  # ... so a plain load_state_dict fails
    fresh = TinyLM(seed=5)
    load_weights(fresh, str(ckpt))
    assert fresh.head.weight is fresh.emb.weight and torch.equal(fresh.emb.weight, saved["emb.weight"])
    prefixed = tmp()  # what an 8-GPU run writes if accelerate saves the DDP wrapper: every key starts with "module."
    save_file({"module." + k: v for k, v in saved.items()}, str(prefixed / "model.safetensors"), metadata={"format": "pt"})
    other = TinyLM(seed=6)
    load_weights(other, str(prefixed))
    assert torch.equal(other.mix.weight, fresh.mix.weight) and not (prefixed / "model.unwrapped.safetensors").exists()


def test_overfit_one_batch():
    batch = {"ids": torch.stack([Seqs()[i]["ids"] for i in range(4)])}
    run = tmp()
    fit(TinyLM(), DataLoader([{"ids": r} for r in batch["ids"]], batch_size=4), None,
        cfg_for("train.max_steps=150", "train.grad_accum=1", "train.save_every=150", "train.lr=0.05", "train.lr_min=0.01"), run)
    rec = records(run)
    assert rec[0]["loss"] > 2.0 and rec[-1]["loss"] < 0.05 and rec[-1]["acc"] == 1.0  # ln(17) = 2.83 -> memorised


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{Path(__file__).name}: {len(tests)} passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_engine.py` → `ModuleNotFoundError: No module named 'dcttgen.engine'`
- [ ] **Step 3: implement**

**`dcttgen/engine.py`** — 213 lines

```python
"""The one training loop (contract section 9). fit() trains any nn.Module whose forward(batch: dict) returns a dict with "loss".

All step counts are OPTIMISER steps; gradient accumulation only changes how many micro-batches make one step.
"""
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, GradientAccumulationPlugin, set_seed

_STEP_DIR = re.compile(r"step_(\d{7})$")


def lr_at(step: int, t) -> float:
    """D10. Learning rate of the optimiser step about to run; step = optimiser steps already taken (0-based)."""
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps  # linear warm-up: lr/warmup_steps ... lr
    progress = min(1.0, (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps))
    return t.lr_min + 0.5 * (t.lr - t.lr_min) * (1.0 + math.cos(math.pi * progress))  # cosine: lr -> lr_min


def make_optimizer(model, t) -> torch.optim.AdamW:
    """AdamW with the values of plan section 3.6. Weight decay on matrices and embeddings (ndim >= 2), none on biases and norm gains."""
    params = [p for p in model.parameters() if p.requires_grad]  # model.parameters() lists a tied tensor once
    groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": t.weight_decay},
              {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=t.lr, betas=tuple(t.betas), eps=t.eps)


def latest_checkpoint(run_dir) -> str | None:
    found = [(int(m.group(1)), p) for p in Path(run_dir).glob("step_*") if (m := _STEP_DIR.match(p.name))]
    return str(max(found)[1]) if found else None


def load_weights(module: torch.nn.Module, checkpoint: str) -> None:
    """Load <checkpoint>/model.safetensors (contract 7.4) into an unwrapped module, strictly.

    Use this and not module.load_state_dict(load_file(...)): save_state drops duplicate tied tensors (lm_head = embed_tokens),
    and a checkpoint written from a DistributedDataParallel model carries a "module." prefix on every key.
    """
    from safetensors.torch import load_file, load_model, save_file

    path = Path(checkpoint) / "model.safetensors"
    state = load_file(str(path))
    if state and all(k.startswith("module.") for k in state):
        tmp = path.with_name("model.unwrapped.safetensors")
        save_file({k[len("module."):]: v for k, v in state.items()}, str(tmp), metadata={"format": "pt"})
        path = tmp
    try:
        load_model(module, str(path), strict=True)  # knows which missing keys are tied aliases
    finally:
        if path.name == "model.unwrapped.safetensors":
            path.unlink()


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def _set_rng_state(s) -> None:
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2])
    if s[3] is not None:
        torch.cuda.set_rng_state_all(s[3])


def _add(sums: dict, out: dict, device) -> None:
    for k, v in out.items():
        v = torch.as_tensor(v, dtype=torch.float32, device=device).detach().mean()
        sums[k] = sums.get(k, 0.0) + v


def _mean(acc: Accelerator, sums: dict, n: int) -> dict:
    """Mean over n micro-batches and over processes, with one collective call."""
    keys = sorted(sums)
    vec = acc.reduce(torch.stack([torch.as_tensor(sums[k]) for k in keys]) / n, reduction="mean")
    return dict(zip(keys, vec.tolist()))


def _log(acc: Accelerator, path: Path, rec: dict) -> None:
    if acc.is_main_process:
        line = json.dumps(rec)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def evaluate(acc: Accelerator, model, loader, seed: int) -> dict:
    """Mean of every scalar the model returns. A fixed RNG stream makes the numbers comparable between checkpoints
    (the RF loss draws random times and noise) and, being forked, leaves the training RNG stream untouched."""
    model.eval()
    sums, n = {}, 0
    devices = [acc.device.index] if acc.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in loader:
            _add(sums, model(batch), acc.device)
            n += 1
    model.train()
    # ponytail: mean of per-batch means, not token-weighted; even_batches may repeat a few samples - return a count and weight if it matters
    return {"val_" + k: v for k, v in _mean(acc, sums, n).items()} if n else {}


def _save(acc: Accelerator, run: Path, step: int, epoch: int, seen: int, keep_last: int) -> None:
    """Write run/step_<7 digits>/ atomically: into a temporary directory first, renamed once complete (a crash never
    leaves a half-written step_* directory for the next resume to pick up). Every process takes part in save_state."""
    final, tmp = run / f"step_{step:07d}", run / f".tmp_step_{step:07d}"
    acc.wait_for_everyone()
    acc.save_state(str(tmp))  # model.safetensors, optimizer.bin, random_states_<rank>.pkl (+ scaler.pt for fp16)
    acc.wait_for_everyone()
    if acc.is_main_process:
        (tmp / "trainer_state.json").write_text(json.dumps({"step": step, "epoch": epoch, "batch_in_epoch": seen}))
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        for old in sorted(run.glob("step_*"))[:-keep_last]:  # keep_last = 0: the slice is empty, everything is kept
            shutil.rmtree(old)
    acc.wait_for_everyone()


def fit(model: torch.nn.Module, train_loader, val_loader, cfg, run_dir: str) -> None:
    """Train model for cfg.train.max_steps optimiser steps; resumes from the newest step_* directory in run_dir.

    train_loader: a torch DataLoader over a map-style dataset whose order depends only on the epoch (fit calls set_epoch).
    val_loader: the same kind, or None.
    """
    t, run = cfg.train, Path(run_dir)
    bad = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:  # eps = 1e-9 underflows to 0 in float16, and bf16 weights lose small updates: keep float32 master weights
        raise TypeError(f"trainable parameters must be float32 ({bad[0]} is not); set train.mixed_precision for half precision")
    acc = Accelerator(
        mixed_precision=t.mixed_precision,  # explicit argument beats ACCELERATE_MIXED_PRECISION from accelerate launch
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=t.grad_accum, sync_with_dataloader=False),
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=t.seed),
    )
    set_seed(t.seed, device_specific=True)
    opt = make_optimizer(model, t)
    model, opt, train_loader = acc.prepare(model, opt, train_loader)
    if val_loader is not None:
        val_loader = acc.prepare(val_loader)
    model.train()

    if acc.is_main_process:
        run.mkdir(parents=True, exist_ok=True)
        for stale in run.glob(".tmp_step_*"):
            shutil.rmtree(stale, ignore_errors=True)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    acc.wait_for_everyone()
    step = epoch = seen = 0  # optimiser steps done; epoch; micro-batches already consumed in this epoch
    ckpt = latest_checkpoint(run)
    if ckpt:
        acc.load_state(ckpt)  # model, optimizer, RNG streams of every process, GradScaler
        s = json.loads((Path(ckpt) / "trainer_state.json").read_text())
        step, epoch, seen = s["step"], s["epoch"], s["batch_in_epoch"]
        acc.print(f"resumed from {ckpt}")
    rng = _rng_state() if ckpt and seen else None  # the RNG exactly as it was when the checkpoint was written

    log, sums, n_micro, t_last, s_last = run / "log.jsonl", {}, 0, time.time(), step
    while step < t.max_steps:
        train_loader.set_epoch(epoch)
        start = seen
        for batch in (acc.skip_first_batches(train_loader, seen) if seen else train_loader):
            if rng is not None:  # first batch after a resume: starting this iterator drew a worker seed from the torch RNG,
                _set_rng_state(rng)  # which the uninterrupted run did not draw here, so put the RNG back as saved
                rng = None
            seen += 1
            with acc.accumulate(model):  # no gradient all-reduce on the non-final micro-batches
                out = model(batch)  # always the wrapped module itself, never a method: DDP and autocast hook forward()
                acc.backward(out["loss"])  # divides by grad_accum
                if acc.sync_gradients:  # last micro-batch of the window: this call performs the optimiser step
                    gnorm = acc.clip_grad_norm_(model.parameters(), t.grad_clip or float("inf"))
                    for g in opt.param_groups:
                        g["lr"] = lr_at(step, t)
                opt.step()  # no-ops until sync_gradients
                opt.zero_grad()
            _add(sums, out, acc.device)
            n_micro += 1
            if not acc.sync_gradients:
                continue
            step += 1
            hook = getattr(acc.unwrap_model(model), "on_step_end", None)
            if hook:
                hook(step)  # e.g. the EMA update of the RF Transformer
            last = step >= t.max_steps
            if last or step % t.log_every == 0 or step % t.eval_every == 0 or step % t.save_every == 0:
                rec = {"split": "train", "step": step, "epoch": epoch, "lr": lr_at(step - 1, t), "grad_norm": float(gnorm),
                       "seconds_per_step": (time.time() - t_last) / (step - s_last), **_mean(acc, sums, n_micro)}
                loss = rec["loss"]
                if not math.isfinite(loss):  # every process sees the same reduced value, so all raise together
                    raise FloatingPointError(f"loss is {loss} at step {step}; the last checkpoint is intact")
                _log(acc, log, rec)
                sums, n_micro, t_last, s_last = {}, 0, time.time(), step
            if val_loader is not None and (last or step % t.eval_every == 0):
                _log(acc, log, {"split": "val", "step": step, **evaluate(acc, model, val_loader, t.seed)})
            if last or step % t.save_every == 0:
                _save(acc, run, step, epoch, seen, t.keep_last)
            if last:
                break
        else:  # the loader ran dry: next epoch
            if start == 0 and seen == 0:
                raise RuntimeError("train_loader yielded no batches")
            if rng is not None:  # the checkpoint was written at the last batch of its epoch: nothing was left to skip
                _set_rng_state(rng)
                rng = None
            epoch, seen = epoch + 1, 0
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_engine.py` → `test_engine.py: 12 passed`
- [ ] **Step 5: the two-process check, on Linux**

<details>
<summary><code>tests/ddp_check.py</code> — 70 lines (click to expand)</summary>

```python
"""Two-process check of fit() on CPU (gloo), not part of the pytest suite:  python tests/ddp_check.py"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_engine import TinyLM, cfg_for, loader  # noqa: E402

from dcttgen.engine import fit, latest_checkpoint  # noqa: E402


class Stop(Exception):
    pass


def run(model, cfg, run_dir, stop_at=None):
    hook = model.on_step_end

    def on_step_end(step):
        hook(step)
        if step == stop_at:
            raise Stop

    model.on_step_end = on_step_end
    try:
        fit(model, loader(10), None, cfg, run_dir)
    except Stop:
        pass


def worker(rank, world, store, shared):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT="29533", ACCELERATE_USE_CPU="true", USE_LIBUV="0")
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=world)
    straight, crashed = Path(shared) / "straight", Path(shared) / "crashed"
    cfg = cfg_for("train.max_steps=6", "train.grad_accum=2", "train.save_every=2", "train.lr=0.03")
    a = TinyLM(dropout=0.2)
    run(a, cfg, str(straight))
    gathered = [torch.zeros_like(a.emb.weight) for _ in range(world)]
    dist.all_gather(gathered, a.emb.weight.detach().clone())
    assert torch.equal(gathered[0], gathered[1]), "ranks drifted apart: gradients were not synchronised"
    run(TinyLM(dropout=0.2), cfg, str(crashed), stop_at=3)  # every rank stops after step 3; the checkpoint is at step 2
    b = TinyLM(dropout=0.2, seed=99)
    run(b, cfg, str(crashed))
    assert torch.equal(a.emb.weight, b.emb.weight), "the resumed 2-process run differs from the uninterrupted one"
    if rank == 0:
        ckpt = Path(latest_checkpoint(straight))
        files = sorted(p.name for p in ckpt.iterdir())
        print("files:", files)
        print("safetensors keys:", sorted(load_file(str(ckpt / "model.safetensors"))))
        steps = [json.loads(line)["step"] for line in (straight / "log.jsonl").read_text().splitlines()]
        assert "random_states_0.pkl" in files and "random_states_1.pkl" in files  # one RNG stream per process
        assert steps == [1, 2, 3, 4, 5, 6]  # logged once, by the main process only
        print("DDP CHECK OK")


if __name__ == "__main__":
    shared = Path(tempfile.gettempdir()) / "dcttgen_ddp_check"
    shutil.rmtree(shared, ignore_errors=True)
    shared.mkdir()
    mp.spawn(worker, args=(2, (shared / "store").as_uri(), str(shared)), nprocs=2)
```

</details>

Run `python tests/ddp_check.py`; it must end with `DDP CHECK OK`. It starts two CPU processes, checks that their weights stay identical (gradients are synchronised), that a crashed two-process run resumes to the same weights, that each process saved its own random state, and that only the main process logged.
**Not executed:** on the authoring machine (Windows) it stops at `RuntimeError: makeDeviceForHostname(): unsupported gloo device` before reaching `fit`. **Everything about `fit` on more than one process is therefore Unverified** until this check passes on Linux. Do it in milestone M0, before any multi-GPU run.

- [ ] **Step 6: commit** — `git add dcttgen/engine.py tests/test_engine.py tests/ddp_check.py && git commit -m "feat: one training loop with exact resume"`

### Task 3: inference

**Files:** Create `dcttgen/infer.py` · Test `tests/test_infer.py` (first three tests)

**Interfaces:**
- Consumes: `generate_codes`, `load_lm`, `Vocab` (chapter 03); `Codec` (chapter 02); `Plan`, `plan_sections`; `cfg.infer.*`.
- Produces: `prompt_to_codes(prompt, cfg, *, duration, bpm, moods, instruments, seed) -> (Plan, Long[K, T])`; `codes_to_wav(codes, cfg, *, seed) -> (Float[N], sample_rate)`; `text_to_music(prompt, cfg, …) -> (Float[N], sample_rate)` (contract §9); `python -m dcttgen.infer … --stage all|codes|audio`.

- [ ] **Step 1: write the failing test** — the language model and the codec are replaced by fakes with the contract's signatures. The test file imports chapter 03's `dcttgen/lm/generate.py` (to replace `generate_codes` in it), so chapter 03 must be in place.

<details>
<summary><code>tests/test_infer.py</code> — 149 lines (click to expand)</summary>

```python
"""End-to-end smoke tests with fake components: a prompt in, a waveform of the right length out; the evaluation runner's control flow.
Run: pytest -q tests/test_infer.py   or   python tests/test_infer.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import dcttgen.infer as infer
import dcttgen.lm.generate as lm_generate
from dcttgen.config import ROOT, load_config
from dcttgen.eval.run import centre, eval_plan, make_audio, make_codes, score, select
from dcttgen.plan import Plan, plan_sections

K, V = 4, 10000
CALLS = []


class FakeCodec:
    sample_rate = 32000

    def decode(self, codes, *, steps=None, cfg_scale=None, seed=None):       # Long[B, K, T] -> Float[B, T * 1280]
        CALLS.append(("decode", tuple(codes.shape), steps, cfg_scale, seed))
        return torch.zeros(codes.shape[0], codes.shape[2] * 1280)


def fake_generate_codes(model, vocab, caption, plan=None, *, temperature=1.0, top_k=None, top_p=None, seed=None):
    CALLS.append(("generate", caption, plan, temperature, top_k, seed))
    plan = plan or Plan(90, 60, plan_sections(60), ["calm"], ["zither"])     # what a model-written plan would look like
    return plan, torch.zeros(K, 25 * plan.duration, dtype=torch.long)


def setup():
    CALLS.clear()
    infer.load_lm_side = lambda cfg: ("vocab", "model")
    infer.load_codec = lambda cfg: FakeCodec()
    lm_generate.generate_codes = fake_generate_codes
    return load_config(str(ROOT / "configs" / "plan_k4.yaml"))


def test_prompt_with_a_full_plan_gives_exactly_that_many_seconds():
    cfg = setup()
    wav, sr = infer.text_to_music("  A joyful piece\nfor zither ", cfg, duration=150, bpm=80, moods=["joyful"], instruments=["zither"], seed=7)
    assert sr == 32000 and wav.shape == (150 * 32000,)
    kind, caption, plan, temperature, top_k, seed = CALLS[0]
    assert caption == "A joyful piece for zither" and plan.sections == [("intro", 30), ("main", 90), ("outro", 30)] and seed == 7
    assert (temperature, top_k) == (cfg.infer.temperature, cfg.infer.top_k)
    assert CALLS[1] == ("decode", (1, K, 3750), cfg.infer.steps, cfg.infer.cfg_scale, 7)


def test_prompt_alone_lets_the_model_plan_and_the_two_steps_compose():
    cfg = setup()
    wav, _ = infer.text_to_music("A calm piece", cfg)
    assert CALLS[0][2] is None and wav.shape == (60 * 32000,)
    plan, codes = infer.prompt_to_codes("A calm piece", cfg, seed=1)         # step 1 alone: the language-model environment
    assert plan.duration == 60 and codes.shape == (K, 1500)
    wav, sr = infer.codes_to_wav(codes, cfg, seed=1)                         # step 2 alone: the codec environment
    assert wav.shape == (60 * 32000,) and sr == 32000


def test_bad_input_is_rejected_before_any_model_is_loaded():
    cfg = setup()
    infer.load_lm_side = infer.load_codec = lambda cfg: (_ for _ in ()).throw(AssertionError("a model was loaded"))
    full = dict(duration=150, bpm=80, moods=["joyful"], instruments=["zither"])
    for prompt, fields in [("", full), ("   \n ", full), ("x" * (cfg.infer.max_prompt_chars + 1), full), ("ok", {"duration": 150}),
                           ("ok", {**full, "duration": 301}), ("ok", {**full, "duration": 29}), ("ok", {**full, "bpm": 500}),
                           ("ok", {**full, "instruments": ["piano"]}), ("ok", {**full, "moods": []})]:
        try:
            infer.text_to_music(prompt, cfg, **fields)
            raise AssertionError(f"accepted {prompt[:10]!r} {fields}")
        except ValueError:
            pass
    try:
        infer.text_to_music(None, cfg)
        raise AssertionError("accepted a prompt that is not a string")
    except TypeError:
        pass


def _rows():
    mk = lambda i, dur, cap: {"clip_id": f"c{i}", "recording_id": f"r{i}", "position": "whole", "audio": f"audio/c{i}.flac", "duration": dur,
                              "bpm": 80, "moods": ["calm"], "instruments": ["zither"], "sections": [list(s) for s in plan_sections(dur)],
                              "caption": cap, "split": "val"}
    return [mk(0, 150, "caption zero"), mk(1, 60, None), mk(2, 90, "caption two"), mk(3, 40, "caption three")]


def test_eval_selection_and_plans():
    rows = _rows()
    assert [r["clip_id"] for r in select(rows, 0, 0)] == ["c0", "c2", "c3"]                 # clips without a caption cannot be prompts
    assert select(rows, 2, 5) == select(rows, 2, 5) and len(select(rows, 2, 5)) == 2
    assert eval_plan(rows[0], 0).sections == [("intro", 30), ("main", 90), ("outro", 30)]
    assert eval_plan(rows[0], 30).duration == 30 and eval_plan(rows[0], 30).sections == [("intro", 6), ("main", 18), ("outro", 6)]
    assert centre(list(range(100)), 4, 10) == list(range(30, 70)) and centre(list(range(100)), 0, 10) == list(range(100))


def test_eval_stages_are_resumable_and_score_reports_three_numbers():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "manifest").mkdir()
        (root / "manifest" / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
        cfg = load_config(str(ROOT / "configs" / "plan_k4.yaml"), ["eval.seconds=30", f"paths.data_root={d}"])
        seen, decoded, files = [], [], {}

        def to_codes(prompt, cfg, *, seed, **fields):
            seen.append((prompt, seed, fields.get("duration")))
            return None, torch.full((K, 25 * fields["duration"]), 7)

        def to_wav(codes, cfg, *, seed):
            decoded.append((tuple(codes.shape), int(codes.max()), seed))
            return torch.zeros(codes.shape[1] * 1280), 32000

        read = lambda path: (torch.zeros(150 * 32000), 32000)
        write = lambda path, wav, sr: (files.__setitem__(path.name + "@" + path.parent.name, len(wav)), path.write_bytes(b""))
        out = root / "eval"
        assert make_codes(cfg, out, to_codes) == 3                                                  # stage 1: language-model environment
        assert seen == [("caption zero", 0, 30), ("caption two", 1, 30), ("caption three", 2, 30)]  # seed = eval.seed + index
        assert np.load(out / "codes" / "c0.npy").dtype == np.int16 and np.load(out / "codes" / "c0.npy").shape == (K, 750)
        assert json.loads((out / "prompts.json").read_text()) == {"c0": "caption zero", "c2": "caption two", "c3": "caption three"}
        assert make_audio(cfg, out, to_wav, read, write) == 3                                       # stage 2: codec environment
        assert decoded == [((K, 750), 7, 0), ((K, 750), 7, 1), ((K, 750), 7, 2)]
        assert files == {f"c{i}.wav@{kind}": 30 * 32000 for i in (0, 2, 3) for kind in ("gen", "ref")}   # references cut to the same length
        (out / "codes" / "c2.npy").unlink()
        (out / "gen" / "c2.wav").unlink()
        assert make_codes(cfg, out, to_codes) == 1 and seen[-1] == ("caption two", 1, 30)           # only the missing piece, with its own seed
        assert make_audio(cfg, out, to_wav, read, write) == 1 and decoded[-1][2] == 1
        got = {}
        fd = lambda **kw: got.setdefault("fd", kw) and 1.5
        kld = lambda **kw: got.setdefault("kld", kw) and 0.5
        clap = lambda id2text, path, clap_model: got.setdefault("clap", (id2text, clap_model)) and 0.3
        res = score(cfg, out, (fd, kld, clap))                                                     # stage 3: metrics environment
        assert res == {"n": 3, "fd_openl3": 1.5, "kl_passt": 0.5, "clap_score": 0.3} == json.loads((out / "metrics.json").read_text())
        assert got["fd"]["samplingrate"] == cfg.eval.fd.sample_rate and got["fd"]["eval_path"].endswith("gen") and got["fd"]["ref_path"].endswith("ref")
        assert got["kld"]["ids"] == ["c0", "c2", "c3"] and got["clap"][1] == cfg.eval.clap_model
        (out / "ref" / "c0.wav").unlink()
        try:
            score(cfg, out, (fd, kld, clap))
            raise AssertionError("scored an incomplete set")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_infer: {len(tests)} tests passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_infer.py` → `ModuleNotFoundError: No module named 'dcttgen.infer'`
- [ ] **Step 3: implement**

**`dcttgen/infer.py`** — 133 lines

```python
"""python -m dcttgen.infer --config C --prompt "..." --out out.wav [--duration S --bpm N --moods a,b --instruments "x,y"] [--seed N]
python -m dcttgen.infer --config C --prompt "..." --out piece.npy --stage codes     (language-model environment)
python -m dcttgen.infer --config C --codes piece.npy --out out.wav --stage audio    (codec environment)

Text -> music in two steps: prompt_to_codes (chapter 03's generate_codes) and codes_to_wav (chapter 02's Codec.decode).
The steps can run in different Python environments, because MuCodec and the language model need different library versions."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dcttgen.plan import Plan, plan_sections

_LOADED: dict = {}


def _once(key, make):
    if key not in _LOADED:
        _LOADED[key] = make()
    return _LOADED[key]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lm_side(cfg):
    """(vocab, model), loaded once per process. Needs the language-model environment."""
    def make():
        from dcttgen.lm.model import load_lm
        from dcttgen.lm.vocab import Vocab
        if not cfg.lm.checkpoint:
            raise ValueError("lm.checkpoint is not set: there is no trained language model to generate with")
        vocab = Vocab.build(cfg)
        return vocab, load_lm(cfg, vocab).to(_device(), dtype=getattr(torch, cfg.infer.dtype)).eval()
    return _once(("lm", id(cfg)), make)


def load_codec(cfg):
    """The codec with its decoder, loaded once per process. Needs the codec environment."""
    def make():
        from dcttgen.codec.codec import Codec
        return Codec.load(cfg, _device(), encoder=False)
    return _once(("codec", id(cfg)), make)


def clean_prompt(prompt, max_chars: int) -> str:
    """The prompt is untrusted text: one line, bounded length. (Control-token strings in it are harmless: Vocab.enc
    tokenises them as ordinary text.)"""
    if not isinstance(prompt, str):
        raise TypeError("the prompt must be a string")
    prompt = " ".join(prompt.split())                # collapses new lines, tabs and runs of spaces
    if not prompt:
        raise ValueError("the prompt is empty")
    if len(prompt) > max_chars:
        raise ValueError(f"the prompt has {len(prompt)} characters; the limit is {max_chars}")
    return prompt


def make_plan(duration, bpm, moods, instruments) -> Plan | None:
    """All four fields -> a Plan (which validates them); none -> None (the model writes the plan); anything else is an error."""
    given = [x is not None for x in (duration, bpm, moods, instruments)]
    if not any(given):
        return None
    if not all(given):
        raise ValueError("give all of duration, bpm, moods and instruments, or none of them (the model then writes the plan)")
    return Plan(bpm, duration, plan_sections(duration), list(moods), list(instruments))


def prompt_to_codes(prompt: str, cfg, *, duration: int | None = None, bpm: int | None = None, moods: list[str] | None = None,
                    instruments: list[str] | None = None, seed: int | None = None) -> tuple[Plan, torch.Tensor]:
    """-> (the plan that was used, codes Long[K, 25 * plan.duration])."""
    caption = clean_prompt(prompt, cfg.infer.max_prompt_chars)
    plan = make_plan(duration, bpm, moods, instruments)          # everything is validated before any model is loaded
    from dcttgen.lm import generate as lm_generate               # imported here so that the codec environment never imports it
    vocab, model = load_lm_side(cfg)
    i = cfg.infer
    return lm_generate.generate_codes(model, vocab, caption, plan, temperature=i.temperature, top_k=i.top_k, top_p=i.top_p, seed=seed)


def codes_to_wav(codes: torch.Tensor, cfg, *, seed: int | None = None) -> tuple[torch.Tensor, int]:
    """codes Long[K, T] -> (wav Float[T * sample_rate // 25] in [-1, 1], sample_rate)."""
    codec, i = load_codec(cfg), cfg.infer
    return codec.decode(codes[None], steps=i.steps, cfg_scale=i.cfg_scale, seed=seed)[0], codec.sample_rate


def text_to_music(prompt: str, cfg, *, duration: int | None = None, bpm: int | None = None, moods: list[str] | None = None,
                  instruments: list[str] | None = None, seed: int | None = None) -> tuple[torch.Tensor, int]:
    """Both steps in one process (contract 9): -> (wav Float[N], sample_rate). Needs one environment that can import both sides."""
    _, codes = prompt_to_codes(prompt, cfg, duration=duration, bpm=bpm, moods=moods, instruments=instruments, seed=seed)
    return codes_to_wav(codes, cfg, seed=seed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("all", "codes", "audio"), default="all")
    ap.add_argument("--prompt")
    ap.add_argument("--codes", help="a .npy file written by --stage codes")
    ap.add_argument("--duration", type=int)
    ap.add_argument("--bpm", type=int)
    ap.add_argument("--moods", help="comma-separated")
    ap.add_argument("--instruments", help="comma-separated")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    args = ap.parse_args(argv)
    from dcttgen.config import load_config
    cfg = load_config(args.config, args.override)
    items = lambda s: None if s is None else [x.strip() for x in s.split(",") if x.strip()]
    if args.stage == "audio":
        if not args.codes:
            ap.error("--stage audio needs --codes")
        codes = torch.from_numpy(np.load(args.codes).astype(np.int64))
    else:
        if not args.prompt:
            ap.error("--prompt is required")
        plan, codes = prompt_to_codes(args.prompt, cfg, duration=args.duration, bpm=args.bpm, moods=items(args.moods),
                                      instruments=items(args.instruments), seed=args.seed)
        print("plan:", plan.to_text())
        if args.stage == "codes":
            np.save(args.out, codes.cpu().numpy().astype(np.int16))
            return
    import soundfile as sf
    wav, sr = codes_to_wav(codes, cfg, seed=args.seed)
    sf.write(args.out, wav.numpy(), sr, subtype="PCM_16")
    print(f"wrote {args.out}: {len(wav) / sr:.0f} s at {sr} Hz")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run it, expect a pass** — after Task 4: `python tests/test_infer.py` → `test_infer: 5 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/infer.py tests/test_infer.py && git commit -m "feat: two-step text-to-music inference"`

### Task 4: evaluation

**Files:** Create `dcttgen/eval/run.py` · Test `tests/test_infer.py` (last two tests)

**Interfaces:**
- Consumes: `prompt_to_codes`, `codes_to_wav`; `manifest/<split>.jsonl`; `cfg.eval.*`; `third_party/stable-audio-metrics`.
- Produces: `select`, `eval_plan`, `centre`; `make_codes(cfg, out, to_codes)`; `make_audio(cfg, out, to_wav, read, write)`; `score(cfg, out, metrics=None) -> {"n", "fd_openl3", "kl_passt", "clap_score"}`; `python -m dcttgen.eval.run --config C --out DIR --stage codes|audio|score`. In `DIR`: `codes/<clip_id>.npy`, `prompts.json`, `gen/<clip_id>.wav`, `ref/<clip_id>.wav`, `metrics.json`.

- [ ] **Step 1:** the tests are the last two functions of `tests/test_infer.py` above.
- [ ] **Step 2: run it, expect failure** — `ModuleNotFoundError: No module named 'dcttgen.eval.run'`
- [ ] **Step 3: implement**

**`dcttgen/eval/run.py`** — 143 lines

```python
"""python -m dcttgen.eval.run --config C --out runs/eval/NAME --stage codes|audio|score [--override a.b=value ...]

codes (language-model environment): one code matrix per validation prompt -> OUT/codes/<clip_id>.npy, and OUT/prompts.json
audio (codec environment):          OUT/gen/<clip_id>.wav from those codes, and the reference clips -> OUT/ref/<clip_id>.wav
score (metrics environment):        FD-openl3, KL-PaSST and CLAP score through stable-audio-metrics -> OUT/metrics.json
Every stage is resumable: finished clips are skipped."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from dcttgen.plan import Plan, plan_sections


def select(rows: list[dict], n: int, seed: int) -> list[dict]:
    """The captioned clips, in a fixed order; n > 0 keeps a seeded random subset of that size."""
    rows = sorted((r for r in rows if r.get("caption")), key=lambda r: r["clip_id"])
    return rows if not n or n >= len(rows) else sorted(random.Random(seed).sample(rows, n), key=lambda r: r["clip_id"])


def eval_plan(row: dict, seconds: int) -> Plan:
    """seconds = 0: the clip's own plan (the research plan's protocol). Otherwise a whole piece of that length with the
    clip's bpm, moods and instruments."""
    if not seconds:
        return Plan.from_manifest(row)
    return Plan(row["bpm"], seconds, plan_sections(seconds), row["moods"], row["instruments"])


def centre(wav, seconds: int, sr: int):
    """The middle `seconds` of a reference clip (the whole clip if seconds = 0 or the clip is shorter)."""
    n = seconds * sr
    if not seconds or len(wav) <= n:
        return wav
    a = (len(wav) - n) // 2
    return wav[a:a + n]


def _prompts(cfg) -> list[dict]:
    e, root = cfg.eval, Path(cfg.paths.data_root)
    text = (root / "manifest" / f"{e.split}.jsonl").read_text(encoding="utf-8")
    rows = select([json.loads(line) for line in text.splitlines() if line.strip()], e.num_prompts, e.seed)
    if not rows:
        raise ValueError(f"manifest/{e.split}.jsonl has no captioned clips")
    return rows


def make_codes(cfg, out, to_codes) -> int:
    """to_codes(prompt, cfg, seed=..., **plan fields) -> (plan, codes Long[K, T]) is infer.prompt_to_codes. Returns the number made."""
    e, out, made = cfg.eval, Path(out), 0
    rows = _prompts(cfg)
    (out / "codes").mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        path = out / "codes" / f"{row['clip_id']}.npy"
        if path.exists():
            continue
        fields = {}
        if e.plan_mode == "given":
            p = eval_plan(row, e.seconds)
            fields = dict(duration=p.duration, bpm=p.bpm, moods=p.moods, instruments=p.instruments)
        _, codes = to_codes(row["caption"], cfg, seed=e.seed + i, **fields)      # seed + index: reproducible, and unchanged by a resume
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(codes).astype(np.int16))
        os.replace(tmp, path)
        made += 1
    (out / "prompts.json").write_text(json.dumps({r["clip_id"]: r["caption"] for r in rows}, indent=1), encoding="utf-8")
    return made


def make_audio(cfg, out, to_wav, read, write) -> int:
    """to_wav(codes, cfg, seed=...) -> (wav, sr) is infer.codes_to_wav; read(path) -> (wav, sr); write(path, wav, sr)."""
    import torch
    e, root, out, made = cfg.eval, Path(cfg.paths.data_root), Path(out), 0
    (out / "gen").mkdir(parents=True, exist_ok=True)
    (out / "ref").mkdir(exist_ok=True)
    for i, row in enumerate(_prompts(cfg)):
        gen, ref = out / "gen" / f"{row['clip_id']}.wav", out / "ref" / f"{row['clip_id']}.wav"
        if not ref.exists():
            wav, sr = read(root / row["audio"])
            write(ref, centre(wav, e.seconds, sr), sr)
        if gen.exists():
            continue
        codes = torch.from_numpy(np.load(out / "codes" / f"{row['clip_id']}.npy").astype(np.int64))
        wav, sr = to_wav(codes, cfg, seed=e.seed + i)
        write(gen, wav, sr)
        made += 1
    return made


def score(cfg, out, metrics=None) -> dict:
    """metrics = (openl3_fd, passt_kld, clap_score); by default they are imported from the stable-audio-metrics clone."""
    e, out = cfg.eval, Path(out)
    prompts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    missing = [i for i in prompts if not (out / "gen" / f"{i}.wav").exists() or not (out / "ref" / f"{i}.wav").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prompts have no generated or reference file (first: {missing[:3]}); finish the audio stage")
    if metrics is None:
        sys.path.insert(0, str(Path(e.metrics_repo).resolve()))
        from src.clap_score import clap_score
        from src.openl3_fd import openl3_fd
        from src.passt_kld import passt_kld
        metrics = (openl3_fd, passt_kld, clap_score)
    fd, kld, clap = metrics
    gen, ref, f = str(out / "gen"), str(out / "ref"), e.fd
    res = {"n": len(prompts),
           "fd_openl3": float(fd(channels=f.channels, samplingrate=f.sample_rate, content_type=f.content_type, openl3_hop_size=f.hop,
                                 eval_path=gen, ref_path=ref, batching=f.batch)),
           "kl_passt": float(kld(ids=sorted(prompts), eval_path=gen, ref_path=ref, collect="mean")),
           "clap_score": float(clap(prompts, gen, clap_model=e.clap_model))}
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=("codes", "audio", "score"))
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    args = ap.parse_args(argv)
    from dcttgen.config import load_config
    cfg = load_config(args.config, args.override)
    if args.stage == "score":
        print(json.dumps(score(cfg, args.out), indent=1))
    elif args.stage == "codes":
        from dcttgen.infer import prompt_to_codes
        print(f"generated {make_codes(cfg, args.out, prompt_to_codes)} code files in {args.out}/codes")
    else:
        import soundfile as sf
        from dcttgen.infer import codes_to_wav
        read = lambda path: sf.read(str(path), dtype="float32")
        write = lambda path, wav, sr: sf.write(str(path), wav.numpy() if hasattr(wav, "numpy") else wav, sr, subtype="PCM_16")
        print(f"decoded {make_audio(cfg, args.out, codes_to_wav, read, write)} pieces into {args.out}/gen")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_infer.py` → `test_infer: 5 tests passed`
- [ ] **Step 5: commit** — `git add dcttgen/eval tests/test_infer.py && git commit -m "feat: staged evaluation runner"`

**Not executed:** the real metric functions. `score` was run with stand-ins that record their arguments; the argument names are the ones in the library's source (§1.5).

### Task 5: the cross-chapter check

**Files:** Test `tests/test_integration.py`

Run this once chapters 03 and 04 are both in place. It drives the real `dcttgen.lm.train` through the real `load_config` and `fit` on CPU — tiny random Qwen2, synthetic dataset — then resumes, fine-tunes from the pre-training checkpoint, reloads the result and generates codes. It is the only test that exercises the seams between the chapters: the batch sampler under `accelerate`, per-level losses reaching the log, the tied-weight checkpoint, the codec guard, and deterministic generation from a reloaded model.

<details>
<summary><code>tests/test_integration.py</code> — 57 lines (click to expand)</summary>

```python
"""Cross-chapter check on CPU: the real language-model package (chapter 03) driven by the real engine and config loader (chapter 04),
then reloaded from its checkpoint and used to generate codes that a codec could decode. Tiny random Qwen2, synthetic data.
Run: pytest -q tests/test_integration.py   or   python tests/test_integration.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from lm_testkit import backbone_dir, write_dataset

from dcttgen.config import ROOT, load_config
from dcttgen.lm.generate import generate_codes
from dcttgen.lm.model import load_lm
from dcttgen.lm.train import main as train_lm
from dcttgen.lm.vocab import Vocab
from dcttgen.plan import Plan, plan_sections


def test_train_checkpoint_resume_reload_generate():
    with tempfile.TemporaryDirectory() as d:
        write_dataset(Path(d) / "data", caption_none=(1,))                     # 6 clips, 1 code frame per second, K = 2, V = 16
        over = [f"paths.data_root={d}/data", f"paths.runs={d}/runs", "codec.tag=k2v16", "codec.num_codebooks=2", "codec.codebook_size=16",
                "audio.frame_rate=1", f"lm.backbone={backbone_dir()}", "lm.max_tokens=600", "lm.num_workers=0", "train.mixed_precision=no",
                "train.warmup_steps=2", "train.grad_accum=1", "train.log_every=1", "train.eval_every=3", "train.save_every=3", "train.lr=1e-3"]
        args = ["--config", str(ROOT / "configs" / "plan_k4.yaml"), "--name", "t"]
        flat = lambda o: [x for item in o for x in ("--override", item)]
        train_lm(args + ["--phase", "pretrain"] + flat(over + ["train.max_steps=6"]))
        run = Path(d) / "runs" / "lm_pretrain" / "t"
        assert sorted(p.name for p in run.glob("step_*")) == ["step_0000003", "step_0000006"]
        assert (run / "vocab.json").is_file() and (run / "tokenizer").is_dir() and (run / "config.yaml").is_file()
        log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in log if r["split"] == "train"] == [1, 2, 3, 4, 5, 6] and [r["step"] for r in log if r["split"] == "val"] == [3, 6]
        first = log[0]                                                           # per-level losses reach the log; at initialisation each is ln(V)
        assert abs(first["ce_k0"] - 2.7726) < 1e-3 and abs(first["ce_k1"] - 2.7726) < 1e-3 and abs(first["ce_text"] - 11.93) < 0.05
        assert {"val_ce_k0", "val_ce_k1", "val_loss"} <= set(log[-1])
        train_lm(args + ["--phase", "pretrain"] + flat(over + ["train.max_steps=8"]))                                 # resumes at step 6
        log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in log if r["split"] == "train"] == [1, 2, 3, 4, 5, 6, 7, 8] and (run / "step_0000008").is_dir()
        # fine-tuning starts from the pre-training checkpoint (lm.checkpoint) and sees captions and plans
        train_lm(args + ["--phase", "finetune"] + flat(over + ["train.max_steps=2", "train.save_every=2", f"lm.checkpoint={run / 'step_0000008'}"]))
        fine = Path(d) / "runs" / "lm_finetune" / "t" / "step_0000002"
        cfg = load_config(str(ROOT / "configs" / "plan_k4.yaml"), over + [f"lm.checkpoint={fine}"])
        vocab = Vocab.build(cfg)
        model = load_lm(cfg, vocab)                                              # tied weights: lm_head is absent from the file
        assert model.head_weight is model.lm.get_input_embeddings().weight
        plan = Plan(80, 40, plan_sections(40), ["calm"], ["zither"])
        used, codes = generate_codes(model, vocab, "A calm piece for zither", plan, temperature=1.0, top_k=8, seed=3)
        assert used == plan and codes.shape == (2, 40) and codes.dtype == torch.long and 0 <= codes.min() and codes.max() < 16
        assert torch.equal(codes, generate_codes(model, vocab, "A calm piece for zither", plan, temperature=1.0, top_k=8, seed=3)[1])


if __name__ == "__main__":
    test_train_checkpoint_resume_reload_generate()
    print("test_integration: 1 test passed")
```

</details>

- [ ] **Step 1: run it** — `python tests/test_integration.py` → `test_integration: 1 test passed`
- [ ] **Step 2: run the whole suite** — `python -m pytest -q` → `131 passed` (about a minute on a laptop CPU, with `soundfile`, `soxr`, `librosa`, `scipy`, `accelerate` and `ffmpeg` present)
- [ ] **Step 3: commit** — `git add tests/test_integration.py && git commit -m "test: cross-chapter training and generation"`

## 4. Running it

### 4.1 The runbook

`E:` names the environment. Durations are not given: nothing could be timed on the authoring machine. Each training command logs `seconds_per_step`; multiply by `train.max_steps` after the first hundred steps.

| Milestone | Commands | Produces |
|---|---|---|
| **M0** environment | Task 0. `E:lm` `python -m pytest -q`. `E:lm` on Linux `python tests/ddp_check.py`. `E:codec` the reconstruction snippet of chapter 02 §4 on three clips | a green suite; `DDP CHECK OK`; three reconstructions to listen to |
| **M1** pilot data (≈ 20 h) | `E:data` the commands of chapter 01 §4, with `C=configs/bootstrap_k1.yaml,configs/pilot.yaml` | `data/audio/`, `data/manifest/`; `python -m dcttgen.data.manifest --config $C` exits 0 |
| **M2** bootstrap, end to end | `E:codec` `python -m dcttgen.codec.tokenize --config $C` → `E:lm` `python -m dcttgen.lm.train --config $C --phase pretrain` → `E:lm` `python -m dcttgen.lm.train --config $C --phase finetune --override lm.checkpoint=runs/lm_pretrain/<name>/step_<N>` → the two inference commands below | a wav file of the requested length from a text prompt |
| **M3** codec (plan path) | `E:codec` `python -m dcttgen.codec.train --config configs/plan_k4.yaml --fit-norm`, then `accelerate launch -m dcttgen.codec.train --config configs/plan_k4.yaml --override train.batch_size=4 --override train.grad_accum=2` ; the sweep of chapter 02 §4 | `runs/codec/k4v10000/step_*`; the steps-versus-quality table |
| **M4** full model | `E:codec` tokenise with `--override codec.checkpoint=…` (8 shards) → `E:lm` `accelerate launch -m dcttgen.lm.train --config configs/plan_k4.yaml --phase pretrain`, then `--phase finetune` | `runs/lm_finetune/<name>/step_*` |
| **M5** evaluation | the three evaluation commands below, once per model | `runs/eval/<name>/metrics.json`; the table of §1.5 |

```bash
# inference in two steps (M2 onwards). CKPT = runs/lm_finetune/<name>/step_<N>
# E:lm
python -m dcttgen.infer --config $C --override lm.checkpoint=$CKPT --stage codes --out piece.npy --seed 1 \
    --prompt "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute" \
    --duration 150 --bpm 80 --moods uplifting,joyful --instruments "zither,two-string fiddle,moon-shaped lute"
# E:codec
python -m dcttgen.infer --config $C --stage audio --codes piece.npy --out piece.wav --seed 1

# evaluation in three steps (M5)
python -m dcttgen.eval.run --config $C --override lm.checkpoint=$CKPT --out runs/eval/dcttgen --stage codes     # E:lm
python -m dcttgen.eval.run --config $C --out runs/eval/dcttgen --stage audio                                    # E:codec
python -m dcttgen.eval.run --config $C --out runs/eval/dcttgen --stage score                                    # E:eval
```

On 8 GPUs, prefix a training command with `accelerate launch --multi_gpu --num_processes 8 -m` in place of `python -m` (**Verified** flags: `accelerate launch --help` lists `-m/--module`, `--multi_gpu`, `--num_processes`). The effective batch is then `batch_size × grad_accum × 8`; only the main process logs and writes `trainer_state.json`.

For a baseline, put its generated files, named `<clip_id>.wav`, into `runs/eval/<baseline>/gen/`, copy `prompts.json` and `ref/` from the DcttGen evaluation directory, and run only the `score` stage.

### 4.2 Baselines (plan §4.1)

Pointers and facts read from each project's own documentation; none was run.

| Baseline | Repository | Training on your own data | Note |
|---|---|---|---|
| MusicGen-Medium | `facebookresearch/audiocraft` | `dora run solver=musicgen/musicgen_base_32khz model/lm/model_scale=medium continue_from=//pretrained/facebook/musicgen-medium` (`docs/MUSICGEN.md`); the dataset format is in `docs/DATASETS.md` | fine-tuning is documented; 32 kHz mono matches our data |
| Stable Audio Open | `Stability-AI/stable-audio-tools` | `python3 ./train.py --dataset-config … --model-config … --name …`; a dataset of type `audio_dir`, with a `custom_metadata_module` that returns the prompt for each file (`docs/datasets.md`); a pre-trained checkpoint can be passed to `train.py` (README) | the exact fine-tuning flag is **Unverified**: read the README's fine-tuning section |
| AudioLDM 2 | `haoheliu/AudioLDM-training-finetuning` | `python3 audioldm_train/train/latent_diffusion.py -c <config.yaml>`; the README has sections on fine-tuning the pre-trained model and on training with your own dataset | whether the AudioLDM2-**Music** checkpoint is covered is **Unverified** |
| MusicLM | `lucidrains/musiclm-pytorch` | no pre-trained weights: its README states that MuLaN must be trained first, then the AudioLM-based stages | by far the largest effort — three models from scratch on 2,000 h. Decide early whether this baseline is worth it |

### 4.3 Ablations that test the plan's own hypotheses (plan §1.3)

| Hypothesis | Vary | Measure |
|---|---|---|
| Coarse-to-fine helps | codebook-major order (as built) against frame-interleaved order — change `flatten_c2f`, `unflatten_c2f`, `build_schedule` | `ce_k0…ce_k3` on validation; FD and KL |
| The plan metadata helps (Chain-of-Thought) | fine-tune with and without the `<PLAN>` span in the document | CLAP score; whether generated pieces follow the requested tempo and length |
| More codebooks help | `bootstrap_k1` against `plan_k4`, same language model | FD and KL; reconstruction FD of the codec alone |
| Rectified flow enables few steps | the sweep of chapter 02 §4 | FD against step count; seconds per minute of audio |
| Qwen2.5 as backbone | pre-trained initialisation against random initialisation of the same architecture | validation loss at equal steps |

### 4.4 Compute ladder

Every entry is an expectation, **Unverified**; replace it with the output of `python -m dcttgen.lm.train … --probe` and the first hundred logged steps.

| Hardware | Realistic scope |
|---|---|
| CPU laptop | the whole test suite; editing; reading logs |
| Free notebook GPU (about 16 GB) | M0 reconstruction; M2 with K = 1, clips of at most 60 s, `lm.grad_checkpointing: true` — a functional demo, not a good model |
| 1 × 24 GB | M1–M2 on the pilot as configured in `pilot.yaml` (K = 1, clips of at most 120 s, 28.8 M-parameter RF) |
| 1 × 80 GB | full-length documents with the 0.5B backbone; the full-size RF at small batch |
| 8 × H100 80 GB (the plan) | M3–M5 at full scale |

### 4.5 Risk register

| # | Risk | Mitigation |
|---|---|---|
| 1 | The released MuEncoder, trained on the Million Song Dataset, represents Don ca tai tu instruments poorly | M0: listen to three reconstructions before building anything else |
| 2 | Far less than 2,000 h of rights-cleared, instrumental audio exists | Count after the vocal gate on the pilot (chapter 01, open question 1); scale K and the backbone to the data |
| 3 | Whole-section coarse-to-fine is too long-range to learn (6,000 positions between levels) | Watch `ce_k1…ce_k3`; the interleaved ablation is three small functions |
| 4 | A 4 × 10,000 RVQ does not fill on a small corpus | Per-level perplexity is logged; start with K = 1 |
| 5 | Dependency conflicts between MuCodec and the language model | Verified and designed for: separate environments, file hand-offs |
| 6 | Multi-GPU training has never run | `tests/ddp_check.py` on Linux in M0 |
| 7 | FD on a small validation set is unstable | Report `n`; judge the pilot by ear and by validation loss |
| 8 | Baselines cost more than DcttGen itself (MusicLM from scratch) | Decide the baseline set before M3 |
| 9 | Tempo and mood labels are unreliable for this music | Expert review of 100 pilot clips (chapter 01) |
| 10 | The codec weights are CC-BY-NC | Fine for research; any public demo or release inherits the restriction |

## 5. What will go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `unknown config key 'train.lr_mim' (did you mean 'train.lr_min'?)` | a typo | fix it; add genuinely new keys to `configs/base.yaml` first |
| `train.max_steps: expected int, got 'ten'` | an override of the wrong type | values are coerced to the default's type |
| `trainable parameters must be float32` | the model was loaded in half precision | load in `float32`; set `train.mixed_precision` for autocast |
| `loss is nan at step N; the last checkpoint is intact` | divergence | lower `train.lr`; check the data; resume from the checkpoint |
| A resumed run repeats or skips data | the loader's order depends on something other than the epoch | give the sampler a `set_epoch` and make its order a function of the epoch only |
| The learning rate decays in an eighth of the expected time | `max_steps` was counted in batches, not optimiser steps | all step counts are optimiser steps |
| `ModuleNotFoundError: transformers.deepspeed` | MuCodec imported in the `lm` environment | run codec work in the `codec` environment (§1.1) |
| `lm.checkpoint is not set` at inference | no trained model was given | `--override lm.checkpoint=runs/lm_finetune/<name>/step_<N>` |
| `N prompts have no generated or reference file` at scoring | the `audio` stage did not finish | re-run it; it resumes |
| The FD value changes a lot between runs on the pilot | too few validation clips | expected; see §1.5 |

## 6. Open questions for the research team

1. **What hardware is really available, and when?** The plan lists 8 × H100. Everything is built to start on one GPU, but M3 and M4 at full scale need the cluster; knowing the date sets the order of work.
2. **The length of training.** The plan gives no step or epoch count. `train.max_steps: 20000` is a placeholder; the validation curves of the pilot are the basis for choosing it.
3. **The evaluation length.** Generating full-length pieces for about 560 validation prompts is roughly 15 million sampled tokens. If that is too slow, evaluate a fixed 30 or 60 s (`eval.seconds`) and say so in the paper.
4. **Comparability of Table 3.** With our evaluation settings (32 kHz mono) the numbers are comparable among the models trained here, not with published Stable Audio numbers. To claim the latter, evaluate with `channels: 2, sample_rate: 44100` and the library's defaults.
5. **Which baselines.** MusicLM from scratch is a project of its own (§4.2). Is it required?
6. **One learning rate for everything?** The plan states the same optimiser for all models. Fine-tuning a pre-trained language model at 3e-4 is unusually aggressive; contract D10 keeps it and notes the alternative reading.
