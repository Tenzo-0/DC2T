# DC2T

**Text-to-music generation for Đờn ca tài tử**, the traditional chamber music of Southern Vietnam, recognised by UNESCO in 2013 as an Intangible Cultural Heritage of Humanity.

DC2T takes a text description such as

> A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute

and generates a long-form instrumental piece in that style, up to five minutes. It is the code for the research project *Integration of Deep Learning in Automatic Music Generation, Aiming at Preserving and Developing Don ca tai tu*.

**Authors:** Tan Duc Nguyen and Nhat Bao Ha, Le Hong Phong High School for the Gifted, Ho Chi Minh City, Vietnam.

## How it works

```
text prompt ──► Qwen 2.5 tokenizer ──┐
                                     ├──► Autoregressive LM ──► audio codes ──► Rectified Flow ──► Mel-VAE ──► HiFi-GAN ──► waveform
audio ──► MuCodec encoder + RVQ ─────┘    (Qwen 2.5)                            Transformer
          (training only)
```

| Component | What it does | Code |
|---|---|---|
| Text tokenizer | Qwen 2.5 byte-level BPE, extended with five special tokens: `<INST>`, `<PLAN>`, `<SOA>`, `<EOA>`, `<EOD>`. Section labels such as `[intro]` stay plain text | [dc2t/lm/vocab.py](dc2t/lm/vocab.py) |
| Audio tokenizer | A wrapper around [MuCodec](https://github.com/xuyaoxun/MuCodec): 32 kHz mono audio to discrete codes at 25 frames per second, and back. Two configurations: the released weights (1 codebook of 16,384), or our own quantiser (4 codebooks of 10,000) | [dc2t/codec/codec.py](dc2t/codec/codec.py) |
| Rectified Flow Transformer | Turns codes into Mel-VAE latents; replaces MuCodec's flow-matching decoder in the 4-codebook configuration | [dc2t/codec/rf.py](dc2t/codec/rf.py), [train.py](dc2t/codec/train.py) |
| Autoregressive language model | Pre-trained Qwen 2.5 with the embedding matrix enlarged by the audio codes, trained with next-token prediction | [dc2t/lm/model.py](dc2t/lm/model.py) |
| Documents and decoding | Build the training sequences for both stages; decode under a schedule derived from the plan | [sequence.py](dc2t/lm/sequence.py), [generate.py](dc2t/lm/generate.py) |
| Data pipeline | Raw recordings to clean clips and a validated manifest | [dc2t/data/](dc2t/data/) |
| Trainer | One training loop on Hugging Face Accelerate, with checkpointing and exact resume | [dc2t/engine.py](dc2t/engine.py) |

### Music Chain-of-Thought

Đờn ca tài tử is largely improvised, so a piece cannot be cut into verses and choruses the way pop songs can. DC2T instead splits every piece into three sections, **intro**, **main** and **outro**, and has the model write a plan before it writes any audio. One training example is a single token sequence:

```
<INST> {description} <PLAN> bpm: 80; duration: 150; sections: [intro] 30, [main] 90, [outro] 30; moods: ...; instruments: ...
[start_of_seg][intro]<SOA> {audio tokens} <EOA>[end_of_seg]
[start_of_seg][main]<SOA>  {audio tokens} <EOA>[end_of_seg]
[start_of_seg][outro]<SOA> {audio tokens} <EOA>[end_of_seg]
<EOD>
```

The loss is not computed on the instruction, so the model learns to produce the plan and the audio but not the prompt. Within each section the codebooks are written coarse to fine: all tokens of the first codebook, then all of the second, and so on. Because the plan fixes the length of every section, the decoder knows in advance which tokens are allowed at each position.

## Repository layout

```
├── dc2t/
│   ├── plan.py          # sections and the plan text
│   ├── config.py        # load_config
│   ├── engine.py        # fit(): the training loop
│   ├── infer.py         # text to music
│   ├── data/            # standardise, cut, quality filter, features, captions, manifest
│   ├── codec/           # Codec, Rectified Flow Transformer, training, tokenisation
│   ├── lm/              # vocabulary, documents, model, dataset, training, decoding
│   └── eval/            # FAD / KL / CLAP runner
├── configs/             # base.yaml and three overlays
├── tests/
└── docs/guide/          # the implementation guide
```

The [implementation guide](docs/guide/README.md) explains every design decision and gives the commands for each milestone. Read it before a real run.

## Setup

You need Linux, Python 3.10 and a CUDA GPU. MuCodec and the language model need different library versions, so they run in separate environments; the recipe is in [guide chapter 04](docs/guide/04-training-evaluation-and-delivery.md).

1. Clone MuCodec into `third_party/MuCodec` at commit `128f91b`.
2. Download the MuCodec weights from <https://huggingface.co/yaoxunxu/mucodec>. They are not in this repository. Put them here:

   | File | Location |
   |---|---|
   | `mucodec.pt` | `third_party/MuCodec/ckpt/` |
   | `muq.pt` | `third_party/MuCodec/muq_dev/` |
   | `audioldm_48k.pth` | `third_party/MuCodec/tools/` |

Run every command from the repository root.

## Data

Training reads three things under `data/`:

```
data/
├── audio/<clip_id>.flac            # mono, 32 kHz, a whole number of seconds (30 to 300)
├── manifest/{all,train,val}.jsonl  # one JSON object per clip
└── codes/<codec tag>/<clip_id>.npy # written by the tokenisation step
```

One manifest row supplies the instruction and the plan:

```json
{"clip_id": "yt_3fA9c_00", "recording_id": "yt_3fA9c", "position": "first",
 "audio": "audio/yt_3fA9c_00.flac", "duration": 270, "bpm": 80,
 "moods": ["uplifting", "joyful"],
 "instruments": ["zither", "two-string fiddle", "moon-shaped lute"],
 "sections": [["intro", 30], ["main", 240]],
 "caption": "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute",
 "split": "train"}
```

- `python -m dc2t.data.manifest --config <config>` checks every row and every audio file, and exits with an error if anything is wrong.
- Clips without a caption are used for pre-training only.
- The stages that build these files from raw recordings are in [dc2t/data/](dc2t/data/) and described in [guide chapter 01](docs/guide/01-data-pipeline.md).

The Đờn ca tài tử dataset itself is not included in this repository.

## Training

`<config>` is one or more overlay files separated by commas. `configs/bootstrap_k1.yaml` uses the released MuCodec weights; `configs/plan_k4.yaml` uses our 4-codebook quantiser; add `configs/pilot.yaml` for a small run.

```bash
# 1. (4-codebook configuration only) train the quantiser and the Rectified Flow Transformer
python -m dc2t.codec.train --config configs/plan_k4.yaml --fit-norm
python -m dc2t.codec.train --config configs/plan_k4.yaml

# 2. tokenise the dataset
python -m dc2t.codec.tokenize --config <config>

# 3. language model, stage 1: audio tokens only
python -m dc2t.lm.train --config <config> --phase pretrain

# 4. language model, stage 2: description, plan and audio
python -m dc2t.lm.train --config <config> --phase finetune --override lm.checkpoint=runs/lm_pretrain/<name>/step_<N>
```

Checkpoints and `log.jsonl` are written to `runs/<stage>/<name>/`. Re-running a command resumes from the newest checkpoint. For several GPUs, replace `python -m` with `accelerate launch --multi_gpu --num_processes 8 -m`.

Main settings, all in [configs/base.yaml](configs/base.yaml):

| Key | Meaning |
|---|---|
| `codec.num_codebooks`, `codec.codebook_size` | 4 × 10,000 (plan) or 1 × 16,384 (released weights) |
| `codec.checkpoint` | our trained quantiser and decoder; `null` uses the released MuCodec weights |
| `lm.backbone` | Hugging Face id of the Qwen 2.5 model, `Qwen/Qwen2.5-0.5B` by default |
| `lm.checkpoint` | a `step_*` folder to start from or to generate with |
| `lm.max_tokens`, `lm.max_seconds` | tokens per batch; longest clip used |
| `train.lr`, `lr_min`, `max_steps`, `grad_accum`, `save_every` | optimisation and checkpointing |

Any key can be changed on the command line with `--override section.key=value`. A key that is not in `base.yaml` is an error.

## Generating

In two steps, because the language model and MuCodec run in different environments:

```bash
# language-model environment: prompt to codes
python -m dc2t.infer --config <config> --override lm.checkpoint=runs/lm_finetune/<name>/step_<N> \
    --stage codes --out piece.npy --seed 1 \
    --prompt "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute" \
    --duration 150 --bpm 80 --moods uplifting,joyful --instruments "zither,two-string fiddle,moon-shaped lute"

# codec environment: codes to audio
python -m dc2t.infer --config <config> --stage audio --codes piece.npy --out piece.wav --seed 1
```

Give all four of `--duration`, `--bpm`, `--moods` and `--instruments`, or none of them. With none, the model writes its own plan from the prompt.

## Acknowledgements and licences

- **MuCodec** (Xu et al., 2024, [arXiv:2409.13216](https://arxiv.org/abs/2409.13216)). DC2T uses its encoder, Mel-VAE and vocoder unmodified, from a separate clone. Its code is under the MIT licence. Its weights are under CC BY-NC 4.0, so they, and anything trained on top of them, may not be used commercially.
- **Qwen 2.5** (Yang et al., 2024) for the tokenizer and the language model.
- **Scaling Rectified Flow Transformers** (Esser et al., 2024) for the decoder's training recipe.
- **YuE** (Yuan et al., 2025) and **MusiCoT** (Lam et al., 2025) for the segment-level chain-of-thought and coarse-to-fine token ordering that our Music Chain-of-Thought builds on.
