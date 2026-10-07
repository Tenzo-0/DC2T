# DC2T Implementation Guide — 00. Overview, Spec and Shared Contracts

> Read this chapter first. Chapters 01–04 are separate plans that can each be built and tested on their own; this chapter holds everything two or more of them depend on. **If a chapter and this file disagree, this file wins.**

**Source spec:** [`Research-Plan-1.pdf`](../../Research-Plan-1.pdf) — *Integration of Deep Learning in Automatic Music Generation, Aiming at Preserving and Developing Don ca tai tu*. Called "the plan" below; "§3.3.2" means that section of the plan.

Every claim in this guide carries one of four labels, so you always know how much to trust it:

| Label | Meaning |
|---|---|
| **Plan** | Stated in the research plan (section cited). |
| **Verified** | Checked against a primary source: a file and line in a pinned repository, or a command that was actually run. |
| **Decision** | An engineering choice made by this guide because the plan is silent or ambiguous. The reason and the way to change it are given. |
| **Unverified** | Could not be checked. Confirm it yourself before relying on it. |

---

## 1. What we are building

```
 TRAINING-DATA PATH                               GENERATION PATH
 audio, 32 kHz mono                               prompt text (+ optional bpm / duration / moods / instruments)
   |                                                |
   v  MuEncoder (frozen)                            v  Qwen2.5 tokenizer + 5 special tokens
 features [1024, T] at 25 Hz                      <INST> ... <PLAN> ...          (the conditions)
   |                                                |
   v  RVQ: K codebooks x V codes                    v  Autoregressive Transformer (Qwen2.5), next-token prediction
 codes [K, T]  ---- LM is trained on these --->   audio tokens, section by section, coarse-to-fine
   |                                                |  unflatten
   |                                                v
   |  RF is trained on (codes -> latents)         codes [K, T]
   v                                                |
 Mel-VAE latents [16, T/2, 32] at 12.5 Hz  <----   v  Rectified Flow Transformer (few-step Euler ODE)
                                                  Mel-VAE latents
                                                    v  Mel-VAE decoder (frozen)  -> mel, 256 bins, 100 frames/s
                                                    v  HiFi-GAN (frozen)         -> waveform 48 kHz -> 32 kHz
```

The plan names four components (§3.1): an audio tokenizer, a text tokenizer, an autoregressive (AR) transformer and a rectified flow (RF) transformer. **We train two things**: the AR language model (chapter 03) and the RF Transformer, together with its RVQ quantiser (chapter 02). Everything else is a frozen pre-trained network.

| Chapter | Builds | Trains anything? |
|---|---|---|
| [01 — Data pipeline](01-data-pipeline.md) | Raw recordings → clean 32 kHz clips + a manifest with bpm, moods, instruments, sections, caption | No |
| [02 — Codec and Rectified Flow](02-codec-and-rectified-flow.md) | `Codec.encode` (audio → codes) and `Codec.decode` (codes → audio); the RF Transformer | RVQ + RF Transformer |
| [03 — Autoregressive language model](03-ar-language-model.md) | Vocabulary, Music Chain-of-Thought sequences, Qwen2.5 training, constrained decoding | AR LM (2 phases) |
| [04 — Training, inference, evaluation](04-training-evaluation-and-delivery.md) | Environment, configs, the shared training engine, text→music CLI, FAD / KLD / CLAP evaluation, roadmap | No (runs the others) |

---

## 2. The spec, as written in the plan

Equations below were re-typed from the PDF pages (the PDF's math fonts do not survive text extraction).

### 2.1 Tokenization (§3.2)

| Item | Value |
|---|---|
| Text tokenizer | Qwen2.5 tokenizer, byte-level BPE, 151,643 regular tokens |
| Added special tokens | `<EOD>` end of document · `<SOA>` start of an audio segment · `<EOA>` end of an audio segment · `<INST>` start of the instruction · `<PLAN>` start of the metadata |
| Text-format items | Instructions, metadata, structure annotations and segment boundary signals are plain text tokenized with BPE |
| Audio tokenizer | MuCodec: 32 kHz audio → discrete tokens at 25 Hz |
| MuEncoder | 13 stacked Conformer blocks |
| RVQ | 4 layers, codebook size 10,000 each |
| Reconstruction chain | flow matching → pre-trained Mel-VAE decoder → pre-trained HiFi-GAN |
| MuCodec training window | 35.84 s, sampling rate ≥ 32 kHz |

### 2.2 Autoregressive transformer and Music Chain-of-Thought (§3.3)

- Backbone: Qwen2.5 (grouped-query attention, SwiGLU, RoPE, QKV bias, pre-norm RMSNorm). These are properties of the stock model — nothing to implement.
- A composition is cut into three sections: **intro** (≤ 30 s), **main** (≤ 240 s), **outro** (≤ 30 s).
- One training document:

  `D = Instruct ⊙ Metadata ⊙ Audio Segments ⊙ <EOD>`   (⊙ = concatenation)

  - Instruct: `<INST> A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute`
  - Metadata: `<PLAN> bpm: 80; duration: 150; Sections: [intro] 30, [main] 90, [outro] 30 ; moods: uplifting, joyful; instruments: zither, two-string fiddle, moon-shaped lute`
  - Each audio segment: `Seg = [start_of_seg] ⊙ s ⊙ <SOA> ⊙ a ⊙ <EOA> ⊙ [end_of_seg]`, with `s ∈ {[intro], [main], [outro]}` and `a` the segment's audio tokens.
- Figure 3 shows the same arrangement: `<INST>` instruction tokens, `<PLAN>` metadata tokens, then three `<SOA>` audio `<EOA>` blocks, then `<EOD>`.
- **Coarse-to-fine flattened RVQ** (Figure 4): all tokens of codebook 0, then all of codebook 1, … then codebook 3:

  `x = ( x_1^(0), …, x_T^(0),  x_1^(1), …, x_T^(1),  …,  x_1^(3), …, x_T^(3) )`

- Next-token prediction, with `c = {c_inst, c_meta}` the instruction and metadata conditions:

  `p(x) = ∏_{i=1}^{4T} p(x_i | x_<i, c)`    `x̂_i = argmax p(x_i | x_<i, c)`    `L_AR = − Σ_{i=1}^{4T} log p(x_i | x_<i, c)`

### 2.3 Rectified Flow Transformer (§3.4)

- Replaces MuCodec's flow-matching model. Input condition `c`: the discretised MuEncoder representation (the RVQ codes). Output: Mel-VAE features.
- Forward process: `z_t = (1 − t)·x_0 + t·ε`, with `t ∈ [0, 1]`, `x_0` a data sample, `ε ~ N(0, I)`. So **t = 0 is data and t = 1 is noise**.
- Objective: `L = E_{t, ε} ‖ v_θ(z_t, t; c) − u_t(z_t | ε, c) ‖²`. The network output is the velocity field directly. For the straight path above the target is `u_t = dz_t/dt = ε − x_0` (this follows from the forward process; the plan does not print it).
- Figure 2: MuEncoder (an intermediate Conformer block) → RVQ (4 quantisers) → "recovered embedding" + random noise → RF Transformer → "denoised embedding" → Mel-VAE decoder → mel-spectrogram → HiFi-GAN.

### 2.4 Data (§3.5)

Recordings from online sources, musicians and experts → resampled to **32 kHz, mono** → quality filter using Fréchet Audio Distance computed from **VGGish, CLAP and EnCodec** embeddings, averaged into one score per file → `librosa` for bpm, duration, sampling rate → `Essentia` for moods → **manual** annotation of six instruments (moon-shaped lute, two-string fiddle, zither, monochord, bamboo flute, gong ban) → captions written by the **GPT-4o API** from those fields. Result: about 28,000 clips, about 2,000 hours (average 257 s per clip).

### 2.5 Training and inference (§3.6)

- Hardware: 8 × NVIDIA H100 80 GB.
- AR LM, two phases: **pre-training** on audio tokens only; **fine-tuning** on text description + metadata + audio tokens.
- RF Transformer: trained on pairs of (MuEncoder codes, Mel-VAE features).
- Optimiser for all models: AdamW, base learning rate 3 × 10⁻⁴, weight decay 0.1, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁹; linear warm-up then cosine annealing "with a maximum learning rate of 3 × 10⁻⁵".
- Decoding rule: only audio-range vocabulary tokens are allowed until `<EOA>` is predicted.

### 2.6 Evaluation (§4)

Train/validation split 98 % / 2 %. Baselines: MusicLM (lucidrains implementation), MusicGen-Medium, AudioLDM2-Music, Stable Audio Open, all trained on the same dataset. Metrics: **FAD_openl3**, **KLD_passt**, **CLAP_score**. The plan's Table 3 row for DC2T: FAD 1.291, KLD 0.472, CLAP 0.394.

---

## 3. What the public building blocks really are

Checked on 2026-10-06 against `github.com/xuyaoxun/MuCodec` at commit `128f91b` (2024-11-22, branch `master`) and the Hugging Face files of `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B`. File references are to that MuCodec commit.

### 3.1 MuCodec, as released — Verified

| Fact | Evidence |
|---|---|
| Only the **0.35 kbps** model is public; inference code only, **no training code** | `readme.md` |
| RVQ: **1 codebook × 16,384 codes**, `codebook_dim=32`, input 1024-d | `model.py:190` |
| I/O is **48 kHz stereo**. MuEncoder ("MuQ", loaded through `fairseq`) runs at **24 kHz** on the mean of left and right | `generate.py:24`, `model.py:188`, `model.py:237-238`, `muq_dev/test.py` |
| Encoder features come from layer `layer_num=7` (index 6 of `layer_results`), 1024-d, **25 frames/s** | `generate.py:23`, `generate.py:224`, `model.py:239-240`, `generate.py:85` |
| Mel-VAE and HiFi-GAN are the **AudioLDM 48 kHz** checkpoint: STFT 2048 / hop 480 / win 2048, 256 mel bins, 20–24,000 Hz; VAE latent has 16 channels | `tools/get_melvaehifigan48k.py:1472-1510` |
| Latent layout for stereo is `[B, 32, L, 32]` = 2 audio channels × 16 latent channels, `L` frames, 32 frequency bins. **Two code frames map to one latent frame** (latents run at 12.5 Hz) | `generate.py:113`, `generate.py:191`, `model.py:306-307` |
| Flow model: PixArt-style `Transformer2DModel`, 24 layers, 22 heads × 72 = 1584 wide, patch 2, **no cross-attention**; input channels 96 = noisy latent (32) + in-context latent (32) + code condition (32) | `configs/models/transformer2D.json`, `model.py:142` |
| Its probability path is **already a straight line**: `x_t = (1 − (1 − σ_min)·t)·ε + t·x_1`, σ_min = 1e-4, with t = 0 noise and t = 1 data (the opposite direction to the plan's convention) | `model.py:129`, `model.py:89` |
| Sampling: Euler, 20 steps by default (50 in `sound2sound`), classifier-free guidance 1.5 | `generate.py:111`, `generate.py:171`, `generate.py:217`, `model.py:130-146` |
| Decoding windows: 1024 code frames (40.96 s), hop 768 frames, the last 128 latent frames of one window are fed as in-context latents to the next | `generate.py:143-178` |
| Three checkpoints are needed: `mucodec.pt`, `muq.pt` (Hugging Face `yaoxunxu/mucodec`) and `audioldm_48k.pth` (`haoheliu/audioldm_48k`) | `readme.md` |
| Licence: code MIT; weights `mucodec.pt` and `muq.pt` are **CC-BY-NC 4.0** (non-commercial) | `readme.md`, `LICENSE_weights` |
| `model.py` resolves `muq_dev/muq_fairseq` relative to the **current working directory** | `model.py:181-186` |

### 3.2 Qwen2.5 — Verified by loading the tokenizer and configs

| Fact | Value |
|---|---|
| Tokenizer class / size | `Qwen2TokenizerFast`; `vocab_size` = 151,643 regular tokens; `len(tokenizer)` = **151,665** (22 control tokens occupy ids 151,643–151,664) |
| Model embedding rows | `config.vocab_size` = 151,936 (rows 151,665–151,935 exist but are unused) |
| Ids after adding our five tokens in the order below | `<EOD>` 151,665 · `<SOA>` 151,666 · `<EOA>` 151,667 · `<INST>` 151,668 · `<PLAN>` 151,669 → `len(tokenizer)` = **151,670** |
| End-of-text / padding | `<|endoftext|>` = 151,643 is both `eos_token` and `pad_token`; there is no BOS; `encode()` adds nothing by default |
| Structure strings tokenised on their own | `[start_of_seg]` → 4 tokens, `[end_of_seg]` → 4, `[intro]` → 3, `[main]` → 3, `[outro]` → 3. The split depends on preceding whitespace (`" [intro]"` gives different ids), so these strings are always tokenised alone |
| Plan's own examples | Instruct line = 29 tokens, Metadata line = 59 tokens |
| Qwen2.5-0.5B | 24 layers, hidden 896, 14 heads / 2 KV heads, FFN 4864, `max_position_embeddings` 32,768, tied input/output embeddings |
| Qwen2.5-1.5B | 28 layers, hidden 1536, 12 heads / 2 KV heads, FFN 8960, `max_position_embeddings` 131,072, tied embeddings |

### 3.3 Where the plan and reality differ

| # | The plan says | What is true | What this guide does |
|---|---|---|---|
| R1 | MuCodec with 4 RVQ layers × 10,000 codes | That 1.35 kbps configuration was never released. Public weights are 1 × 16,384 | Decision D1: two codec configurations |
| R2 | MuCodec compresses 32 kHz audio | Released stack is 48 kHz stereo in and out; the encoder itself runs at 24 kHz | Decision D2 |
| R3 | Training examples are 35.84 s | The released *inference* code uses 40.96 s windows | Our RF trains on 35.84 s (896 frames); chapter 02 handles long-form decoding |
| R4 | Flow matching follows curved paths; rectified flow straightens them | MuCodec's path is already linear (OT flow matching). What differs in the SD3 recipe is the timestep sampling, the parameterisation details and the network | Implement the plan's RF objective exactly; treat "few-step" as a claim to **measure** (steps-versus-quality sweep in chapter 02) |
| R5 | "Samples with low FAD scores were discarded" | Lower FAD means closer to the reference, i.e. better | Discard the **high**-FAD tail (chapter 01) |
| R6 | Base learning rate 3e-4, "maximum" 3e-5 | The two numbers contradict each other | Decision D10 |
| R7 | The loss sums over audio tokens only | The decoding rule needs `<EOA>` to be predictable, so at least that token must be trained too | Decision D8 |
| R8 | Qwen2.5 size is not stated | 0.5B to 72B exist | Decision D11 |
| R9 | One framework | MuCodec's encoder imports `transformers.deepspeed`, a module that no longer exists in the `transformers` release the language model uses (**Verified**: `ModuleNotFoundError` in 4.57.1). The two cannot share a Python environment as they are | Decision D13: separate environments, files handed between stages (chapter 04 §1.1) |

---

## 4. Decisions shared by all chapters

Chapter-specific decisions live in the chapters. These ones cross chapter boundaries.

**D1 — The codec configuration is data, not code.** `K` (codebooks) and `V` (codebook size) are read from config everywhere; nothing hard-codes 4 or 10,000.
- `configs/bootstrap_k1.yaml`: K = 1, V = 16,384 — the released MuCodec weights. Works on day one. Coarse-to-fine is trivial here (one level).
- `configs/plan_k4.yaml`: K = 4, V = 10,000 — the plan. Requires training the RVQ and the RF Transformer ourselves (chapter 02).
- Build the bootstrap path end to end first, then switch. Reason: it proves the data pipeline, the LM and the decoding logic before any money is spent on codec training.

**D2 — Sample rates.** Dataset audio is 32 kHz mono, as in the plan. `Codec` resamples internally (24 kHz for the encoder, 48 kHz for the Mel-VAE) and returns 32 kHz. To get full-band output later, change `audio.sample_rate` to 48000 and rebuild the dataset; no code changes.

**D3 — Mono latents.** Our audio is mono, so Mel-VAE latents are `[16, L, 32]` (the released stereo layout is two of these stacked on the channel axis).

**D4 — Clips are whole seconds, 30–300 s, and know where they came from.** Every clip is trimmed to an integer number of seconds so that `T = 25 × duration` exactly and section boundaries fall on frame boundaries. Recordings longer than a clip are split, and each clip records its `position` in the source recording: `whole`, `first`, `middle` or `last`. Only `whole`/`first` clips have an intro; only `whole`/`last` clips have an outro. Reason: with a 257 s average against a 300 s cap, most clips are pieces of longer recordings; labelling the middle of a performance `[intro]` would teach the model nothing. *Plan-literal alternative:* pass `position="whole"` for every clip.

**D5 — Section lengths.** `intro = outro = min(30, duration // 5)`, `main` = the rest, `main ≤ 240`. This reproduces the plan's own example (150 s → 30 / 90 / 30). The manifest stores the resulting numbers per clip, so expert-annotated boundaries can replace the rule later without touching any code.

**D6 — Only five tokens are added to the tokenizer.** Structure labels and segment markers stay plain text (§3.2), are tokenised once at start-up, and are concatenated as id lists.

**D7 — Audio tokens are arithmetic, not strings.** `id = audio_offset + k·V + code`, where `audio_offset = len(tokenizer)` after adding the five special tokens (151,670). The embedding matrix is resized to `audio_offset + K·V`. Each codebook level has its own id range, which is what lets the model and the decoder tell levels apart. Reason: 40,000 added string tokens would slow the tokenizer for no benefit.

**D8 — Which positions are trained.** The loss covers every token after the Instruct span: the metadata, the structure tokens, the audio tokens, `<EOA>` and `<EOD>`. `lm.loss_on_plan: false` additionally masks the metadata, which is the literal reading of `c = {c_inst, c_meta}`. Reason for the default: it costs nothing (≈ 60 tokens against ≈ 25,000) and lets the model write its own plan from a bare prompt — the "thought" in Chain-of-Thought — while still accepting a user-supplied plan.

**D9 — Decoding is schedule-constrained.** Once the `<PLAN>` text is fixed, the length of every section is known, so the allowed id range of every future position is known before generation starts. This is stricter than the plan's rule (audio range until `<EOA>`) and guarantees a rectangular `[K, T]` code matrix for the RF decoder. Chapter 03 owns the implementation.

**D10 — Learning rate.** Linear warm-up to `lr = 3e-4`, cosine annealing down to `lr_min = 3e-5` (reading the plan's "maximum 3e-5" as the floor). Each stage config may override both. The other reading — a 3e-5 peak for LM fine-tuning — is one line in a config.

**D11 — Backbone.** `Qwen/Qwen2.5-0.5B` (base, not Instruct) for bring-up; `Qwen/Qwen2.5-1.5B` is the main candidate for the full run. Pre-trained weights are used as initialisation. Reason: ≈ 2,000 h is ≈ 720 M audio tokens — far too little data for the 7B-class models.

**D12 — Pre-training documents** are fine-tuning documents without the Instruct and Metadata parts: `Seg … Seg ⊙ <EOD>`. One builder, one flag.

**D13 — Platform and environments.** Anything that trains or touches MuCodec runs on **Linux + CUDA, Python 3.10**. Windows is for editing and for the CPU-only unit tests. Because of R9 there are four Python environments — `data`, `codec`, `lm`, `eval` — and every stage hands the next one files (`.npy` codes, `.wav` audio), so no process needs two of them. Chapter 04 owns the recipe.

---

## 5. Repository layout and who owns what

```
DC2T/
├── Research-Plan-1.pdf
├── pyproject.toml                 # 04
├── configs/                       # 04 — base.yaml, bootstrap_k1.yaml, plan_k4.yaml, pilot.yaml
├── third_party/MuCodec/           # 04 sets it up: git clone pinned to 128f91b + three checkpoints
├── dc2t/
│   ├── config.py                  # 04 — load_config
│   ├── engine.py                  # 04 — fit(): the one training loop
│   ├── plan.py                    # 03 — plan_sections, Plan (code fixed in §8.3 below)
│   ├── infer.py                   # 04 — text → wav
│   ├── data/                      # 01 — standardise, filter, features, captions, manifest
│   ├── codec/                     # 02 — codec.py (Codec), rf.py (RF Transformer), train.py, tokenize.py
│   ├── lm/                        # 03 — vocab.py, sequence.py, model.py, data.py, train.py, generate.py
│   └── eval/                      # 04 — run.py
├── tests/                         # every chapter adds tests/test_<module>.py
├── data/                          # not in git — see §7
└── runs/                          # not in git — checkpoints and logs
```

Command names used across chapters:

| Command | Owner | Purpose |
|---|---|---|
| `python -m dc2t.data.pipeline <stage> --config C` | 01 | data pipeline stages (chapter 01 lists them) |
| `python -m dc2t.data.manifest --config C` | 01 | validate the manifest and the audio files; exit code 0 = milestone M1 |
| `python -m dc2t.codec.train --config C [--fit-norm]` | 02 | measure latent statistics, then train RVQ + RF Transformer |
| `python -m dc2t.codec.tokenize --config C [--shard i/n]` | 02 | write `data/codes/<tag>/<clip_id>.npy` for every clip |
| `python -m dc2t.lm.train --config C --phase pretrain\|finetune [--probe]` | 03 | train the AR LM; `--probe` measures one step |
| `python -m dc2t.infer --config C --stage codes\|audio\|all …` | 04 | prompt → codes (`lm` environment), codes → wav (`codec` environment) |
| `python -m dc2t.eval.run --config C --out DIR --stage codes\|audio\|score` | 04 | generate for the validation prompts; FAD / KLD / CLAP |

`C` may be several overlay files separated by commas (`configs/bootstrap_k1.yaml,configs/pilot.yaml`). Multi-GPU runs replace `python -m` with `accelerate launch --multi_gpu --num_processes N -m`.

---

## 6. Shared configuration

`configs/base.yaml` holds the keys below. Each chapter adds keys under its own section (`data:`, `rf:`, `lm:`, `train:`, `infer:`, `eval:`); **the complete file, with every key and its default, is in chapter 04, Task 1.** Code reads values as attributes: `cfg.codec.num_codebooks`. A key that is not in `base.yaml` is an error.

```yaml
paths:
  data_root: data                 # manifest/, audio/, codes/
  mucodec_root: third_party/MuCodec
  runs: runs
audio:
  sample_rate: 32000              # Plan §3.5 — dataset format, mono
  frame_rate: 25                  # Plan §3.2 — code frames per second, per codebook
codec:
  tag: k4v10000                   # names data/codes/<tag>/
  num_codebooks: 4                # K — Plan §3.2
  codebook_size: 10000            # V — Plan §3.2
  muq_layer: 7                    # MuEncoder layer feeding the RVQ (released default)
  window_frames: 896              # 35.84 s training window — Plan §3.2
  checkpoint: null                # our trained RVQ + RF weights; null = released MuCodec weights (valid only for K=1, V=16384)
lm:
  backbone: Qwen/Qwen2.5-0.5B     # D11
  loss_on_plan: true              # D8
  checkpoint: null
train:                            # Plan §3.6 — the same optimiser for both models
  lr: 3.0e-4
  lr_min: 3.0e-5                  # D10
  weight_decay: 0.1
  betas: [0.9, 0.999]
  eps: 1.0e-9
```

`configs/bootstrap_k1.yaml` overrides `codec: {tag: k1v16384, num_codebooks: 1, codebook_size: 16384}`.

Numbers that follow from the config (plan configuration):

| Quantity | Value |
|---|---|
| Samples per code frame | 1,280 at 32 kHz · 960 at 24 kHz · 1,920 at 48 kHz |
| Audio tokens per second | K × 25 = **100** |
| Longest clip | 300 s → T = 7,500 frames → **30,000** audio tokens |
| Text and structure overhead per document | ≈ 130–300 tokens (caption and plan are each capped at 128 tokens) |
| Longest document | ≈ 30,300 tokens (fits Qwen2.5-0.5B's 32,768) |
| Vocabulary size | 151,670 + 4 × 10,000 = **191,670** (bootstrap: 151,670 + 16,384 = 168,054) |
| RF training window | 896 code frames = 448 latent frames = 35.84 s |

---

## 7. On-disk data contracts

All paths are relative to `paths.data_root`.

### 7.1 Audio — written by chapter 01

`audio/<clip_id>.flac` — FLAC, **mono, 32,000 Hz, exactly `duration × 32000` samples**.

### 7.2 Manifest — written by chapter 01, read by 02, 03, 04

`manifest/all.jsonl`, plus `manifest/train.jsonl` and `manifest/val.jsonl`. One JSON object per line:

```json
{"clip_id": "yt_3fA9c_00", "recording_id": "yt_3fA9c", "position": "first",
 "audio": "audio/yt_3fA9c_00.flac", "duration": 270, "bpm": 80,
 "moods": ["uplifting", "joyful"],
 "instruments": ["zither", "two-string fiddle", "moon-shaped lute"],
 "sections": [["intro", 30], ["main", 240]],
 "caption": "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute",
 "split": "train"}
```

| Field | Type | Rule |
|---|---|---|
| `clip_id` | str | unique; `[A-Za-z0-9_]+`; also the file stem |
| `recording_id` | str | the source recording; all its clips share one `split` |
| `position` | str | `whole` \| `first` \| `middle` \| `last` (D4) |
| `duration` | int | seconds, `30 ≤ duration ≤ max_clip_seconds(position)` (§8.3) |
| `bpm` | int | 30–240 |
| `moods` | list[str] | 1–3 lowercase words from chapter 01's mood vocabulary |
| `instruments` | list[str] | non-empty subset of: `moon-shaped lute`, `two-string fiddle`, `zither`, `monochord`, `bamboo flute`, `gong ban` |
| `sections` | list[[str, int]] | labels from `intro`, `main`, `outro` in that order, each at most once; seconds ≥ 1; **sum equals `duration`**; intro, outro ≤ 30; main ≤ 240 |
| `caption` | str \| null | English, one line. `null` is allowed only for clips used in LM pre-training |
| `split` | str | `train` \| `val` (98 % / 2 %, split by `recording_id`) |

Chapter 01 may add fields (quality scores, source, licence). Readers must ignore fields they do not know.

### 7.3 Codes — written by chapter 02, read by chapter 03

`codes/<codec.tag>/<clip_id>.npy` — `int16`, shape **`[K, 25 × duration]`**, values in `[0, V)`. The tag directory keeps token sets from different codecs apart; never mix them. `codes/<codec.tag>/meta.json` records a fingerprint of the RVQ that wrote the tokens; a codec with another RVQ refuses to tokenise into, or decode from, that directory.

### 7.4 Checkpoints — written by the engine (chapter 04)

`runs/<stage>/<name>/step_<7-digit step>/` holds the full training state; the model weights are `model.safetensors` inside it. `codec.checkpoint` and `lm.checkpoint` point at such a directory.

**Verified** (accelerate 1.15.0, transformers 4.57.1, torch 2.9.0, tiny tied-embedding Qwen2 model on CPU): `Accelerator.save_state` writes `model.safetensors`, `optimizer.bin` and `random_states_0.pkl` (plus scheduler and gradient-scaler files when those are in use). **Tied tensors are stored once**: for the language model, `lm_head.weight` is absent from the file because it is the same tensor as the input embedding. Two consequences:

- Resuming with `Accelerator.load_state` works unchanged, and the tie survives.
- Loading the file into a bare module with `load_state_dict(strict=True)` fails with `Missing key(s): "…lm_head.weight"`. Inference-time loaders therefore use `safetensors.torch.load_model(module, path)` — strict, but aware that a missing tied alias is fine — or `dc2t.engine.load_weights`, which additionally strips the `module.` prefix of checkpoints written by a distributed run. Never a bare `strict=False`, which would also hide a genuinely wrong checkpoint.

---

## 8. Token and sequence contract

Owned by chapter 03; consumed by 02 (only `K`, `V`) and 04.

### 8.1 Vocabulary layout

```
[0, 151643)                    Qwen2.5 regular BPE tokens
[151643, 151665)               Qwen2.5 control tokens (151643 = <|endoftext|>, used as padding)
151665 <EOD>  151666 <SOA>  151667 <EOA>  151668 <INST>  151669 <PLAN>
[audio_offset + k·V, audio_offset + (k+1)·V)    codebook k,  k = 0 … K−1,  audio_offset = 151670
```

The five tokens are added in exactly that order. `audio_offset` is computed as `len(tokenizer)` and **asserted** to equal 151,670, never typed in as a literal elsewhere.

### 8.2 Coarse-to-fine flattening

Applied **per section** to that section's slice of the clip's code matrix.

```python
def flatten_c2f(codes, audio_offset, V):      # codes: Long[K, T]  ->  Long[K*T]
    K, T = codes.shape
    return (codes + audio_offset + torch.arange(K).unsqueeze(1) * V).reshape(-1)   # row-major = codebook-major

def unflatten_c2f(ids, audio_offset, K, V):   # ids: Long[K*T]  ->  Long[K, T]
    codes = ids.reshape(K, -1) - audio_offset - torch.arange(K).unsqueeze(1) * V
    assert ((codes >= 0) & (codes < V)).all()
    return codes
```

A section of `s` seconds is the frame slice `codes[:, 25·start : 25·(start + s)]`, where `start` is the sum of the earlier sections' seconds.

### 8.3 Sections and the plan text

This code is canonical. Chapter 03 places it in `dc2t/plan.py`; chapter 01 calls it.

```python
EDGE_MAX_S, MAIN_MAX_S, MIN_CLIP_S = 30, 240, 30          # Plan §3.3.2; MIN_CLIP_S is Decision D4
POSITIONS = ("whole", "first", "middle", "last")

def max_clip_seconds(position: str = "whole") -> int:
    return MAIN_MAX_S + EDGE_MAX_S * (position in ("whole", "first")) + EDGE_MAX_S * (position in ("whole", "last"))

def plan_sections(duration: int, position: str = "whole") -> list[tuple[str, int]]:
    if position not in POSITIONS:
        raise ValueError(f"unknown position {position!r}")
    if not MIN_CLIP_S <= duration <= max_clip_seconds(position):
        raise ValueError(f"duration {duration}s is outside [{MIN_CLIP_S}, {max_clip_seconds(position)}] for position={position}")
    edge = min(EDGE_MAX_S, duration // 5)
    intro = edge if position in ("whole", "first") else 0
    outro = edge if position in ("whole", "last") else 0
    parts = [("intro", intro), ("main", duration - intro - outro), ("outro", outro)]
    return [(name, sec) for name, sec in parts if sec > 0]
```

| Call | Result |
|---|---|
| `plan_sections(150)` | `[("intro", 30), ("main", 90), ("outro", 30)]` — the plan's example |
| `plan_sections(300)` | `[("intro", 30), ("main", 240), ("outro", 30)]` |
| `plan_sections(30)` | `[("intro", 6), ("main", 18), ("outro", 6)]` |
| `plan_sections(270, "first")` | `[("intro", 30), ("main", 240)]` |
| `plan_sections(240, "middle")` | `[("main", 240)]` |
| `plan_sections(100, "last")` | `[("main", 80), ("outro", 20)]` |
| `plan_sections(300, "first")` | `ValueError` (longest `first` clip is 270 s) |

**Plan text** — the string that follows `<PLAN> `. Fields in this order, lower-case keys, `; ` between fields, `, ` inside lists:

```
bpm: 80; duration: 150; sections: [intro] 30, [main] 90, [outro] 30; moods: uplifting, joyful; instruments: zither, two-string fiddle, moon-shaped lute
```

(The plan's printed example capitalises `Sections` and has a stray space before one semicolon; both are normalised here.) Only the sections that exist are listed, and `duration` is their sum.

### 8.4 Documents

With `enc(text) = tokenizer.encode(text, add_special_tokens=False)` applied to **untrusted text in a way that cannot produce our special tokens** (chapter 03 specifies how), and `SEG_START`, `SEG_END`, `LABEL[s]` the id lists of `[start_of_seg]`, `[end_of_seg]`, `[intro]` / `[main]` / `[outro]` tokenised alone:

```
fine-tuning document =
    [<INST>] + enc(" " + caption)
  + [<PLAN>] + enc(" " + plan_text)
  + for each (label, seconds) in sections:
        SEG_START + LABEL[label] + [<SOA>] + flatten_c2f(section codes) + [<EOA>] + SEG_END
  + [<EOD>]

pre-training document = the same without the first two lines          (D12)
```

Nothing else is inserted — no spaces, no BOS. Padding uses `<|endoftext|>` with an attention mask.

**Labels:** a copy of the ids with `-100` on the Instruct span (`<INST>` and the caption), on padding, and — only when `lm.loss_on_plan` is false — on the Metadata span (D8).

### 8.5 Decoding schedule (D9)

Given the prompt `[<INST>] caption [<PLAN>] plan_text`, the rest of the document has a fixed layout. For each section of `s` seconds, with `T = 25·s`:

```
SEG_START, LABEL[label], <SOA>           forced
T tokens from codebook 0's id range
T tokens from codebook 1's id range
…
T tokens from codebook K−1's id range
<EOA>, SEG_END                           forced
```

then `<EOD>`. The generated audio ids of each section are unflattened to `[K, T]` and the sections are concatenated along time into the clip's `[K, 25·duration]` matrix.

---

## 9. Code interfaces between chapters

Signatures only; bodies belong to the owning chapter. Shapes: `B` batch, `K` codebooks, `T` code frames (25 Hz), `L` latent frames (12.5 Hz), `N` samples, `S` sequence length.

The chapters add to these without changing them: `Vocab` also carries `frame_rate`, `codec_tag` and the structure id lists, and has `enc`, `save` and `check_checkpoint` (chapter 03); `Codec` has `fingerprint()`, and its `encoder` / `decoder` flags only skip *our* weights because the released stack always loads whole (chapter 02); `infer.py` exposes the two halves of `text_to_music` as `prompt_to_codes` and `codes_to_wav` so that they can run in different environments, and `engine.py` exposes `load_weights` (chapter 04).

```python
# dc2t/config.py — chapter 04
def load_config(path: str, overrides: list[str] = ()) -> "Config": ...
#   deep-merges `path` over configs/base.yaml, then applies "a.b=value" overrides; attribute access (cfg.codec.tag)

# dc2t/plan.py — chapter 03 (code in §8.3)
def max_clip_seconds(position: str = "whole") -> int: ...
def plan_sections(duration: int, position: str = "whole") -> list[tuple[str, int]]: ...
@dataclass
class Plan:
    bpm: int; duration: int; sections: list[tuple[str, int]]; moods: list[str]; instruments: list[str]
    def to_text(self) -> str: ...                  # §8.3 format, without the "<PLAN> " prefix
    @staticmethod
    def from_text(text: str) -> "Plan": ...        # raises ValueError if malformed or violating §7.2
    @staticmethod
    def from_manifest(row: dict) -> "Plan": ...

# dc2t/codec/codec.py — chapter 02. The ONLY module allowed to import from third_party/MuCodec.
class Codec:
    num_codebooks: int            # K
    codebook_size: int            # V
    frame_rate: int               # 25
    sample_rate: int              # cfg.audio.sample_rate
    @classmethod
    def load(cls, cfg, device, *, encoder: bool = True, decoder: bool = True) -> "Codec": ...
    def encode(self, wav: Tensor) -> Tensor: ...
    #   wav Float[B, N], mono, at sample_rate, N a multiple of sample_rate // 25  ->  codes Long[B, K, N * 25 // sample_rate]
    def decode(self, codes: Tensor, *, steps: int | None = None, cfg_scale: float | None = None,
               seed: int | None = None) -> Tensor: ...
    #   codes Long[B, K, T], any T >= 1  ->  wav Float[B, T * sample_rate // 25], values in [-1, 1]

# dc2t/lm/ — chapter 03
class Vocab:                                        # vocab.py
    tokenizer: "PreTrainedTokenizerFast"
    eod: int; soa: int; eoa: int; inst: int; plan: int; pad: int
    audio_offset: int; K: int; V: int; size: int    # size = audio_offset + K * V
    @classmethod
    def build(cls, cfg) -> "Vocab": ...             # asserts the §8.1 layout
def build_document(vocab: Vocab, codes: Tensor, sections: list[tuple[str, int]],
                   caption: str | None, plan: Plan | None, *, loss_on_plan: bool = True
                   ) -> tuple[Tensor, Tensor]: ...  # sequence.py — (input_ids Long[S], labels Long[S]); codes is Long[K, T]
def load_lm(cfg, vocab: Vocab, checkpoint: str | None = None) -> "nn.Module": ...   # model.py — forward(batch) returns the loss dict fit() expects
def generate_codes(model, vocab: Vocab, caption: str, plan: Plan | None = None, *,
                   temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                   seed: int | None = None) -> tuple[Plan, Tensor]: ...             # generate.py
#   returns the plan actually used and codes Long[K, 25 * plan.duration]; plan=None lets the model write the plan

# dc2t/engine.py — chapter 04
def fit(model: "nn.Module", train_loader, val_loader, cfg, run_dir: str) -> None: ...
#   Contract with every trainable model:
#     model(batch: dict) -> dict                its forward(); must contain "loss" (scalar Tensor with grad); other entries are scalar logs.
#                                               fit() always calls the module itself, never a method, so that DistributedDataParallel
#                                               sees the forward pass and keeps gradients in sync across GPUs.
#     model.on_step_end(step: int) -> None      optional; called on the unwrapped module after each optimiser step (e.g. EMA update)
#   fit() owns: AdamW + warm-up/cosine from cfg.train, mixed precision, gradient accumulation and clipping,
#   multi-GPU through accelerate, validation, logging, saving and resuming (§7.4).

# dc2t/infer.py — chapter 04
def text_to_music(prompt: str, cfg, *, duration: int | None = None, bpm: int | None = None,
                  moods: list[str] | None = None, instruments: list[str] | None = None,
                  seed: int | None = None) -> tuple[Tensor, int]: ...     # (wav Float[N], sample_rate)
```

---

## 10. Build order

Stages depend on each other in one direction only: the RVQ must be trained and frozen **before** the dataset is tokenised, and the dataset must be tokenised before the LM is trained.

| Milestone | What is done | Chapters | Done when |
|---|---|---|---|
| **M0** Environment | The four environments; the CPU test suite; the two-process training check on Linux; MuCodec cloned and loaded; three Don ca tai tu clips reconstructed with the released codec | 04, 02 | The suite is green, `tests/ddp_check.py` prints `DDP CHECK OK`, and you have listened to the reconstructions and know how well the public codec handles these instruments |
| **M1** Pilot data | ≈ 20 hours through the full data pipeline | 01 | `manifest/train.jsonl` and `val.jsonl` pass the validator |
| **M2** Bootstrap, end to end | `bootstrap_k1` config: tokenise → LM pre-train → LM fine-tune (Qwen2.5-0.5B) → generate → decode with the released decoder | 02, 03, 04 | A text prompt produces a wav file of the requested length |
| **M3** Codec stage | Train RVQ (4 × 10,000) + RF Transformer; reconstruction metrics; steps-versus-quality sweep | 02 | Reconstructions from our decoder are at least as good as the released one on validation clips |
| **M4** Full model | Re-tokenise with the trained RVQ; train the LM with K = 4 coarse-to-fine on the full dataset | 02, 03 | Generated pieces follow their plan and sound coherent over minutes |
| **M5** Evaluation | Generate from validation captions; FAD_openl3, KLD_passt, CLAP_score; baselines | 04 | A filled-in version of the plan's Table 3 |

---

## 11. Conventions

- **Reuse before writing.** Hugging Face `transformers` for Qwen2.5; MuCodec's own modules for the RVQ, Mel-VAE and HiFi-GAN; `accelerate` for multi-GPU; published tools for metrics. Do not re-implement what these already do.
- **No speculative structure.** No base classes with one subclass, no registries, no options nobody asked for. A deliberate shortcut with a known ceiling gets a comment: `# ponytail: <the ceiling> — <the upgrade path>`.
- **Every non-trivial module leaves one runnable check** in `tests/test_<module>.py`: plain functions with bare `assert`, no fixtures, no network, no checkpoints, CPU only. `pytest -q` for the whole suite stays under a minute. Tests that need `third_party/MuCodec` or a GPU skip themselves when it is absent.
- **Never hard-code** `K`, `V`, the sample rate, the frame rate or token ids; read them from `cfg` or `Vocab`.
- **Tensors are documented by shape and dtype** in a comment or docstring at every function boundary, using the letters in §9.
- **Secrets** (API keys) come from environment variables and are never written to config files, logs or the manifest.
