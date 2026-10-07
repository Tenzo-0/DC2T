# DC2T Implementation Guide

How to build DC2T — the text-to-music system for Don ca tai tu described in [`Research-Plan-1.pdf`](../../Research-Plan-1.pdf) — from an empty repository to a filled-in results table.

## The chapters

| # | Chapter | What you build | Tasks | Tests |
|---|---|---|---|---|
| 00 | [Overview, Spec and Shared Contracts](00-overview-and-contracts.md) | Nothing — read it first. The plan's spec, what the public building blocks really are, and every name, shape and file format two chapters share | — | — |
| 01 | [Data Pipeline](01-data-pipeline.md) | Raw recordings → clean 32 kHz clips + a manifest with bpm, moods, instruments, sections, caption | 9 | 33 |
| 02 | [Codec and Rectified Flow](02-codec-and-rectified-flow.md) | Audio ↔ codes: the wrapper around the released MuCodec, our RVQ + Rectified Flow Transformer, dataset tokenisation | 4 | 30 |
| 03 | [Autoregressive Language Model](03-ar-language-model.md) | Vocabulary, Music Chain-of-Thought documents, Qwen2.5 with a restricted loss, schedule-constrained decoding | 7 | 45 |
| 04 | [Training, Inference, Evaluation and Delivery](04-training-evaluation-and-delivery.md) | Environments, config, the one training loop, text → music, FAD / KLD / CLAP, the runbook | 6 | 23 |

Each chapter is a plan you can execute task by task: write the test, watch it fail, implement, watch it pass, commit. Source files are printed in full; test files are in collapsed blocks — click to open them.

## State of the repository

The reference implementation printed in the chapters is **already in this repository**: `dc2t/` (the package), `tests/` (131 tests), `configs/`, `pyproject.toml` and `.gitignore`. For those files the chapters are the explanation, not work to redo — their "expect failure" steps describe how the code was built and will not fail now.

To see the suite pass on a machine that has only `uv` (this is the command that was run here; it installs nothing into the repository):

```
uv run --no-project --with pytest --with accelerate --with scipy --with soundfile --with librosa --with soxr --with torch==2.9.0 --with transformers==4.57.1 --with pyyaml --with numpy --with safetensors python -m pytest -q
```

(One line, so it works in PowerShell and in bash alike. Run it from the repository root.)

In an environment built as in chapter 04, Task 0, it is simply `python -m pytest -q`.

**What is still to do, in order:** milestone M0 (the environments, and the checks that could not run on the authoring machine — listed at the end of this page); chapter 01's Task 9 (the pipeline driver, specified but not written); then milestones M1–M5.

The files in the repository are the source of truth from here on. Each chapter holds a copy of its files as they were on 2026-10-06; when you change a file, its copy in the chapter goes stale.

## The order of work

| Milestone | What | Done when |
|---|---|---|
| **M0** | Environments, the test suite, and three clips reconstructed with the released codec | The suite is green and **you have listened** to how the public codec handles these instruments |
| **M1** | About 20 hours through the data pipeline | `python -m dc2t.data.manifest --config …` exits 0 |
| **M2** | End to end with the released codec (K = 1): tokenise → pre-train → fine-tune → generate | A text prompt produces a wav of the requested length |
| **M3** | Train our own RVQ (4 × 10,000) and Rectified Flow Transformer | Our reconstructions are at least as good as the released decoder's |
| **M4** | Full model with coarse-to-fine K = 4 on the full dataset | Pieces follow their plan and stay coherent over minutes |
| **M5** | Evaluation against baselines | The table of the plan's §4.3, filled in |

Build chapter 04's Task 0–2 and chapter 03's Task 1 first (everything else imports them); after that the chapters are independent. The exact commands for every milestone are in [chapter 04, §4.1](04-training-evaluation-and-delivery.md).

## Seven things the plan does not tell you

These were found by checking the plan against the real code and model files. Details and evidence are in [chapter 00, §3](00-overview-and-contracts.md).

1. **The MuCodec of the plan was never released.** The plan uses 4 codebooks × 10,000 codes. The public weights are 1 codebook × 16,384, with inference code only. The guide therefore has two paths: start with the released codec (works on day one), and train the plan's codec yourself (chapter 02).
2. **MuCodec is a 48 kHz stereo system** whose encoder runs at 24 kHz, not the 32 kHz the plan states. The guide keeps 32 kHz mono data, as the plan asks, and resamples inside the codec.
3. **MuCodec's flow matching is already a straight path.** So "rectified flow" does not straighten anything by itself; what changes is the timestep sampling and the network. Few-step sampling is a claim to *measure* — chapter 02 gives the experiment.
4. **MuCodec and the language model cannot share a Python environment** (a module MuCodec imports was removed from current `transformers`). Inference and evaluation are therefore built as stages that pass files.
5. **"Samples with low FAD scores were discarded" is backwards**: low FAD is good. The pipeline discards the high tail.
6. **The learning rate is stated twice, inconsistently** (base 3e-4, "maximum" 3e-5). The guide reads it as 3e-4 decaying to 3e-5.
7. **Computing the loss the obvious way needs 23 GB per document.** Chapter 03 scores each audio token only against its own codebook's 10,000 rows (1.2 GB), and proves the two are equivalent.

Also worth fixing in the plan itself: Table 1 gives MusicCaps as "55,210 hours" and LP-MusicCaps-MSD as "15,419,310 hours" — those are seconds (about 15 hours and 4,283 hours).

## Decisions that are yours to confirm

The guide makes a default choice wherever the plan is silent or ambiguous, and says how to change it. These are the ones that shape the research:

| Question | The guide's default | Where |
|---|---|---|
| Recordings with singing | Dropped by a voice classifier; no source separation | [01, open question 1](01-data-pipeline.md) |
| A clip from the middle of a long performance | Labelled `[main]` only — no fake `[intro]` | [00, D4](00-overview-and-contracts.md) |
| Section lengths | intro = outro = min(30, duration / 5) | [00, D5](00-overview-and-contracts.md) |
| Does the model write its own plan? | Yes: the plan text is trained (`lm.loss_on_plan: true`) | [00, D8](00-overview-and-contracts.md) |
| Decoding | The plan fixes the length of every section, so every position's allowed tokens are known in advance — stricter than the plan's rule | [00, D9](00-overview-and-contracts.md), [03, §1.5](03-ar-language-model.md) |
| Sampling | Sampling with top-k, not the plan's argmax | [03, §1.5](03-ar-language-model.md) |
| Qwen2.5 size | 0.5B to bring up, 1.5B for the full run | [00, D11](00-overview-and-contracts.md) |
| RF Transformer conditioning | Added to the input frame by frame, not SD3's two-stream layout | [02, §1.3](02-codec-and-rectified-flow.md) |
| Evaluation settings | 32 kHz mono — comparable within this study, not with published numbers | [04, §1.5](04-training-evaluation-and-delivery.md) |
| Which baselines | All four are documented; MusicLM means training three models from scratch | [04, §4.2](04-training-evaluation-and-delivery.md) |

Every chapter ends with its own list of open questions.

## What was verified, and what was not

**Ran, and passes** — on a Windows laptop, CPU only, Python 3.12/3.13, `torch 2.9.0`, `transformers 4.57.1`, `accelerate 1.15.0`:

- All **131 tests** in the guide, in one tree (`python -m pytest -q`, about a minute) — and again from inside this repository after the files were placed here. The code blocks in the chapters are spliced from those exact files, not retyped.
- A cross-chapter run with the real modules on a tiny random Qwen2: train → checkpoint → resume → fine-tune from the pre-training checkpoint → reload → generate. The first logged losses are `ln V` per codebook and `ln 151,670` for text, as theory predicts.
- The audio stages (ffmpeg decoding, resampling, trimming, cutting, tempo) on synthetic audio.
- The Qwen2.5 tokenizer layout and the five special-token ids, with the real tokenizer.

**Checked by reading the source, not by running it:** the released MuCodec (commit `128f91b`), `fadtk` 1.1.0, `stable-audio-metrics` (commit `fd55536`), the Essentia model metadata, the OpenAI SDK's parameter names, the MuCodec and SD3 papers.

**Not run at all — verify these first, in milestone M0:**

| What | Why it could not run here | How to check |
|---|---|---|
| Anything on a GPU; every memory and speed figure | No GPU | `python -m dc2t.lm.train … --probe`; `seconds_per_step` in `log.jsonl` |
| Training on more than one process | The two-process check needs Linux | `python tests/ddp_check.py` → `DDP CHECK OK` |
| The real MuCodec: loading, encoding, decoding | Its checkpoints and dependencies are Linux/GPU | The reconstruction snippet in chapter 02 §4 |
| Installing the four environments | Linux-only packages | The import smoke tests in chapter 04, Task 0 |
| Essentia, `fadtk`, the GPT-4o API, the three metrics | Linux-only, GPU-only or paid | Run each on the 20-hour pilot |
| The data pipeline's command-line driver | Specified (chapter 01, Task 9) but not written | Its test is described there |

Anything marked **Unverified** in a chapter is in this category: a stated expectation, not a measured fact.
