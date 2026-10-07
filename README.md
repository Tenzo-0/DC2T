# DcttGen

Text-to-music generation for **Đờn ca tài tử**, the traditional chamber music of Southern Vietnam (UNESCO Intangible Cultural Heritage, 2013).

A text prompt goes in like *"A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute"* and a piece of up to five minutes comes out.

> **Status: research code, not a trained model.** The repository contains the implementation and its test suite. No weights are published, and nothing here has been trained or run on a GPU yet. See [what is verified](#what-is-verified).

## How it works

```
prompt (+ optional bpm, duration, moods, instruments)
   |  Qwen2.5 tokenizer + 5 special tokens
   v
<INST> caption <PLAN> plan                       the "Music Chain-of-Thought": the plan fixes every section's length
   |  autoregressive language model (Qwen2.5)
   v
audio codes [K, 25 x seconds]                    intro / main / outro, coarse-to-fine over K codebooks
   |  Rectified Flow Transformer
   v
Mel-VAE latents  ->  Mel-VAE decoder  ->  HiFi-GAN  ->  waveform
```

The audio tokenizer builds on [MuCodec](https://github.com/xuyaoxun/MuCodec). Two configurations are supported: the released MuCodec weights (1 codebook), and a 4-codebook quantiser with our own Rectified Flow decoder, which this code trains.

## Repository layout

| Path | Contents |
|---|---|
| [`docs/guide/`](docs/guide/README.md) | The implementation guide: the spec, every design decision, and a task-by-task build plan. **Start here.** |
| `dcttgen/data/` | Raw recordings to clean 32 kHz clips and a validated manifest |
| `dcttgen/codec/` | Audio to codes and back; Rectified Flow Transformer; tokenisation |
| `dcttgen/lm/` | Vocabulary, documents, the language model and its loss, constrained decoding |
| `dcttgen/engine.py` | The training loop, with exact resume |
| `dcttgen/infer.py`, `dcttgen/eval/` | Text to music; FAD / KL / CLAP evaluation |
| `configs/` | Every setting, with the two codec configurations and a small-scale overlay |
| `tests/` | 131 CPU tests |
| `Research-Plan-1.pdf` | The research plan this implements |

## Third-party components

MuCodec's code is MIT-licensed; its released weights are **CC-BY-NC 4.0** (non-commercial). Anything trained on top of those weights inherits that restriction. This repository contains none of them; the guide explains how to fetch them.

## Authors

Tan Duc Nguyen and Nhat Bao Ha — Le Hong Phong High School for the Gifted, Ho Chi Minh City, Vietnam.
