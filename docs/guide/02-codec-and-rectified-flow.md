# DC2T Implementation Guide — 02. Codec and Rectified Flow

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this chapter task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a `Codec` class that turns audio into `[K, T]` codes and codes back into audio, in two configurations — the released MuCodec weights (K = 1) and our own RVQ + Rectified Flow Transformer (K = 4) — plus the training of the second and the job that tokenises the dataset.
**Architecture:** one module (`codec.py`) is the only place that touches MuCodec's code; it keeps the released MuEncoder, Mel-VAE and HiFi-GAN frozen. A pure-PyTorch module (`rf.py`) holds the Rectified Flow Transformer, its loss and its sampler. A training module wires RVQ + RF to the shared engine.
**Tech stack:** Python 3.10, PyTorch, `scipy` (resampling), MuCodec at commit `128f91b` with its own pinned dependencies (`fairseq 0.12.2`, `diffusers 0.27.2`, `torch 2.2.0`) — the codec environment of chapter 04.
**Spec:** plan §2.2, §3.1, §3.2 (audio), §3.4, §3.6; Figures 1 and 2 · contract: [00 — Overview, Spec and Shared Contracts](00-overview-and-contracts.md)

## Global constraints

- `Codec` has exactly the interface of contract [§9](00-overview-and-contracts.md#9-code-interfaces-between-chapters): `encode` returns exactly `N × 25 / sample_rate` frames; `decode` accepts any `T ≥ 1` and returns exactly `T × sample_rate / 25` samples in `[-1, 1]`.
- `dc2t/codec/codec.py` is the **only** module that imports from `third_party/MuCodec`.
- **Time convention (plan §3.4): `t = 0` is data, `t = 1` is noise.** MuCodec's released sampler runs the other way. Never mix the two.
- `K`, `V`, the sample rate, the frame rate and the window length come from config. The Mel-VAE's own constants (48 kHz, 3,840 samples per latent frame, 16 × 32 values per latent frame) are properties of the frozen checkpoint and live as named constants.
- **The RVQ is frozen for the lifetime of a tokenised dataset.** Retraining it means a new `codec.tag`, re-tokenising, and retraining the language model. A fingerprint check enforces this.
- Tests: CPU only, random weights, tiny dimensions. The released stack is replaced by a fake with the same shapes *and the same quirks*.

## Review focus

| # | Input or condition | Expected behaviour | Test |
|---|---|---|---|
| 1 | Tokens in `data/codes/<tag>/` were written by a different RVQ than the one now loaded | Refusal with an explanation, at tokenise time and at decode time | `test_tag_directory_refuses_tokens_from_another_rvq`, `test_tokenize_writes_contract_files_is_resumable_and_refuses_another_rvq` |
| 2 | Odd `T`, `T = 1`, `T` shorter than a window, `T = 7,500` | Exact output length every time | `test_plan_decode_exact_length_for_odd_short_and_long_T`, `test_released_decode_gives_exact_length_in_range_for_every_T` |
| 3 | A sign or time-direction error in the loss or the sampler | Caught without training anything | `test_oracle_loss_is_zero_and_sign_is_pinned`, `test_euler_oracle_lands_on_x0_for_any_step_count` |
| 4 | The condition is shifted by one code frame relative to the latents | Reconstruction error rises several-fold: alignment matters and is tested | `test_pair_frames_alignment`, `test_tiny_model_learns_the_condition_and_alignment_matters` |
| 5 | Long-form decoding in windows | Identical to one-shot decoding when the model is local; no seam at window edges | `test_windowed_sampler_is_exact_through_windows`, `test_vae_windows_reproduce_one_shot_decode`, `test_blend_weights_partition_unity_and_edges` |

## 1. Background and design

### 1.1 What the plan says, and what exists

**Plan §3.2, §3.4, Figure 2:** MuEncoder → RVQ (4 quantisers × 10,000 codes) → "recovered embedding" + noise → Rectified Flow Transformer → Mel-VAE features → Mel-VAE decoder → HiFi-GAN.

**Verified** (contract §3.1): only the 1 × 16,384 model is public, with inference code only. So this chapter builds two paths behind one class (contract D1):

| | Bootstrap path | Plan path |
|---|---|---|
| Config | `configs/bootstrap_k1.yaml` | `configs/plan_k4.yaml` |
| RVQ | released, 1 × 16,384 | **trained by us**, K × V from config |
| Decoder | released flow-matching transformer | **our Rectified Flow Transformer** |
| MuEncoder, Mel-VAE, HiFi-GAN | released, frozen | released, frozen |
| `codec.checkpoint` | `null` | a `runs/codec/<name>/step_*` directory |
| Needs training | no | yes — Task 3 |

### 1.2 How the released MuCodec works

Everything below was read from the pinned clone and is cited by file and line; none of it was executed (the checkpoints and MuCodec's dependencies are not on the authoring machine).

| Fact | Source |
|---|---|
| **Encode.** `sound2code` takes stereo 48 kHz, rescales peaks above 0.8, cuts the audio into windows of 40.96 s **plus 480 samples**, and for short inputs repeats the audio until a window is full. It returns `int(len / 48000 × 25) + 1` frames — one more than `T` for whole-second clips | `generate.py:74-108`, `:206-214` |
| The encoder input is the mean of left and right, resampled to 24 kHz; the features are layer `layer_num − 1` of the MuEncoder's `layer_results`, `Float[B, 1024, T]` | `model.py:236-242`, `generate.py:23` |
| **RVQ.** `ResidualVectorQuantize.forward(z)` returns a 6-tuple `(z_q, codes, latents, commitment_loss, codebook_loss, n_quantizers)`; `from_codes(codes)` returns `(z_q, z_p, codes)`. The example at the end of the file weights the losses `0.25 × commitment + 1.0 × codebook` | `libs/rvq/descript_quantize3.py:149-229`, `:231-251`, `:296` |
| **Decode.** `code2sound` embeds the codes (`from_codes` → `Linear(1024, 512)`), folds **two code frames into one latent frame**, and samples latents `[B, 32, L, 32]` with Euler steps | `model.py:304-307`, `generate.py:111-204` |
| Its guidance scale is hard-coded to 1.5 inside the two calls, whatever the argument says | `generate.py:171`, `:178` |
| It decodes in windows of 1,024 code frames with hop 768, feeding the last 128 latent frames of one window to the next as context, and cross-fades the waveforms | `generate.py:143-203` |
| `load_main_model=False` cannot work: the constructor asserts that a model config path is given | `model.py:167` |
| The checkpoint is loaded with `strict=False`, so a wrong or truncated file loads silently | `generate.py:43` |
| `sound2code` is decorated with CUDA autocast; the code prints on every call | `generate.py:72-73`, `:86` |
| Paths such as `muq_dev/muq_fairseq` are resolved against the current working directory; the top-level module names are `model`, `models`, `tools`, `libs`, `generate` | `model.py:181-186` |
| Mel-VAE: one latent frame = 8 mel frames = 3,840 samples at 48 kHz; the latent is the posterior sample times `scale_factor`; `decode_first_stage` divides by it again | `tools/get_melvaehifigan48k.py:984-1009`, `:1478`, `:1516` |

Two further facts were **derived** by replaying the index arithmetic of `generate.py` in a stand-alone script (no third-party code was run), and should be confirmed during milestone M0:

- `code2sound` fails for any `T ≥ 1024` with `(T − 128) % 768 = 0` (for example T = 3,200, a 128 s clip): its padding branch leaves a last window shorter than 1,024 frames. The wrapper appends one frame for those lengths and trims the output.
- Each encoder window is 480 samples (10 ms) longer than the 1,024 frames it yields, so `sound2code` drifts by 10 ms per 40.96 s window.

From the MuCodec paper (arXiv 2409.13216, §III, read as text): training used fixed 35.84 s segments on 8 × A100 40 GB with batch size 4; the compared models were trained for 20,000 steps and the best row for 200,000; inference used 50 flow-matching steps. The paper's table lists the `4 × 10000` configuration at 25 Hz as 1.33 kbps.

### 1.3 The Rectified Flow Transformer

**Token.** One token per latent frame: the 16 channels × 32 mel bins of a Mel-VAE latent frame are one 512-value vector. A 35.84 s window is 448 tokens; a five-minute clip is 3,750.
**Decision.** MuCodec patches the latent as a 2-D image with 2 × 2 patches over (frames × mel bins), which is 8 tokens per latent frame; one token per frame is 8× cheaper and makes the sequence one-dimensional, which is what makes long-form decoding simple (§1.5). If reconstruction lacks high-frequency detail, splitting each frame into several tokens along the mel axis is the first thing to try.

**Network** (all in `rf.py`):

| Part | Choice | Reason |
|---|---|---|
| Blocks | pre-norm transformer, width 1024, depth 24, 16 heads, MLP ratio 4 | as deep as the released decoder (24 layers) and narrower (1024 against 1584); `313,123,328` parameters (pilot: 512 / 8 / 8 = `28,774,400`) — exact, `test_param_counts_pilot_and_full` |
| Position | rotary embedding on queries and keys | sequences of any length; nothing to resize |
| Timestep | sinusoidal embedding of `1000·t` → MLP → shift/scale/gate per block ("adaLN-single", as PixArt and the released decoder) | one shared timestep MLP |
| Attention stability | RMSNorm on queries and keys (QK-normalisation, SD3 §5.3.2) | keeps attention logits bounded under half precision |
| Initialisation | the modulation and the output layer start at zero | every block is the identity and the output is 0 at step 0, so the loss starts at exactly 2.0 (§1.4) |
| Condition | `Linear(2 × 1024 → width)` of the two code-frame embeddings of each latent frame, **added to the input token** | see below |

**Decision — the condition is added at the input, not fed through a second stream.** SD3's two-stream MM-DiT exists because a text prompt is not aligned with image patches. Our condition is aligned frame for frame with the target: latent frame `i` is described by code frames `2i` and `2i+1`. Adding it to the token is the direct way to use that alignment, and is what the released decoder does with channel concatenation (`model.py:142`). What this chapter takes from "Scaling Rectified Flow Transformers" (the plan's citation) is the training recipe, not the two-stream layout.

**Guidance.** In training the condition is replaced by a learned null vector for a fraction `rf.cond_drop` of the examples; at decode time `v = v_uncond + s·(v_cond − v_uncond)` with `s = rf.decode_cfg`.

### 1.4 The objective, exactly

Plan §3.4 (contract §2.3), with `x0` the **standardised** latent:

```
z_t = (1 - t) * x0 + t * eps          eps ~ N(0, I),  t in (0, 1)
u   = eps - x0                        the velocity dz_t/dt of that straight path
L   = mean( (v_theta(z_t, t; c) - u)^2 )
```

- **The plan's remark "`w_t = t / (1 − t)`"** is the SD3 paper's way of writing this same loss as a weighted noise-prediction loss (SD3 §3, "Flow Trajectories"). It is not an extra factor: implement the plain mean-squared error above and add no weight. `test_weight_identity` checks the algebra numerically.
- **Timestep sampling — Decision:** `t = sigmoid(n)`, `n ~ N(0, 1)` — SD3's `rf/lognorm(0.00, 1.00)`, the variant that paper found consistently best (SD3 §3.1, "Logit-Normal Sampling"). Uniform `t` is the plain rectified-flow choice; switch with one line in `sample_t`.
- **Why the loss starts at 2.0:** with a zero output, `L = E|eps − x0|² = 1 + 1` when `x0` has unit variance. A first logged loss far from 2 means the latents are not standardised. `test_zero_init_output_and_loss_two` and the trainer test both pin this.
- **Standardisation.** `LatentNorm` holds a mean and a standard deviation per (channel, mel bin), as MuCodec's `Feature2DProcessor` does (`model.py:37-80`). They are measured once before training (`--fit-norm`) and saved with the weights.

**What "rectified flow" adds to what MuCodec already does (R4).** MuCodec's path `x_t = (1 − (1 − σ)t)·ε + t·x_1` is already a straight line between noise and data (`model.py:129`); the plan's statement that flow matching follows curved trajectories does not apply to it. The differences in this implementation are the logit-normal timestep density, the network, and the time direction. The technique in the rectified-flow paper (arXiv 2209.03003) that actually straightens *sampling* trajectories is **reflow**: retraining on (noise, generated sample) pairs produced by the trained model. It is not part of the plan and not implemented here. **Treat "few-step" as a claim to measure** — §4 has the sweep — and consider reflow if the sweep disappoints.

### 1.5 Sampling and long-form decoding

Euler steps on a uniform grid from `t = 1` to `t = 0`: `z ← z − v(z, t)/steps`, evaluated at `t = 1, 1 − 1/steps, …, 1/steps` — never at `t = 0`, and nothing divides by `t`.

**Decision — one shared state, overlapping windows, averaged velocities.** Training windows are 448 latent frames; a five-minute clip is 3,750. At every Euler step the sampler runs the network on windows of 448 frames with hop 224 over the *whole* noisy sequence and averages the predicted velocities with sine weights. Because all windows read and write one shared state, there is nothing to stitch afterwards and no in-context channel to train. `plan_windows` guarantees every frame is covered; the last window is shifted back to end exactly at the last frame.

The frozen Mel-VAE decoder and HiFi-GAN are run the same way on windows of `rf.vae_window = 128` latent frames (10.24 s, the VAE's own training length) with hop `rf.vae_hop = 96`, cross-faded with the same sine weights.

**Odd `T`.** Two code frames make one latent frame, so an odd `T` repeats its last code frame (`pair_frames`); the waveform is trimmed to exactly `T × 1920` samples at 48 kHz before resampling.

**Encoding on the plan path.** Windows of `codec.window_frames = 896` frames, non-overlapping; the last window is shifted back to end at `T` and only its new frames are kept, so every encoder call sees a full window as in training. A clip shorter than one window is zero-padded.

### 1.6 Decisions in this chapter

| | Decision | Reason | How to change |
|---|---|---|---|
| 2.1 | One token per latent frame | 8× fewer tokens; 1-D sequence | `x_in`, `out` in `RFTransformer` |
| 2.2 | Condition added at the input; null vector for guidance | the condition is frame-aligned | `RFTransformer.forward` |
| 2.3 | Logit-normal timesteps | SD3 §3.1 | `sample_t` |
| 2.4 | Shared-state windowed sampling | no seams, no in-context training | `sample`, `blended_velocity` |
| 2.5 | RVQ and RF are trained **jointly**; the RF loss reaches the RVQ through the condition | the codes should carry what the decoder needs | `CodecTrainer.forward` |
| 2.6 | An exponential moving average of the RF weights is kept and used for decoding | standard for flow and diffusion decoders | `rf.ema_decay` |
| 2.7 | Features and latents are computed on the fly from audio, not cached | caching latents for 2,000 h needs about 2,000 × 3600 × 12.5 × 512 × 2 bytes ≈ 92 GB at half precision, and fixes the crop positions | `WindowDataset` |
| 2.8 | A 16-hex fingerprint of the RVQ weights is written to `data/codes/<tag>/meta.json` | a mismatched decoder must refuse to run | `rvq_fingerprint` |

## 2. File map

| File | Responsibility |
|---|---|
| `dc2t/codec/rf.py` | `RFTransformer`, `LatentNorm`, `pair_frames`, `sample_t`, `rf_loss`, `plan_windows`, `sample` |
| `dc2t/codec/codec.py` | `Released` (everything third-party), `Codec`, exact-length helpers, fingerprint and tag-directory guard |
| `dc2t/codec/train.py` | `CodecTrainer`, `fit_latent_norm`, `WindowDataset`, entry point |
| `dc2t/codec/tokenize.py` | the tokenisation job |
| `tests/test_codec_rf.py`, `test_codec_codec.py`, `test_codec_train.py` | 30 tests |

## 3. Tasks

### Task 1: the Rectified Flow Transformer, its loss and its sampler

**Files:** Create `dc2t/codec/__init__.py` (empty), `dc2t/codec/rf.py` · Test `tests/test_codec_rf.py`

**Interfaces:**
- Consumes: `cfg.rf.width`, `cfg.rf.depth`, `cfg.rf.heads` (through `RFConfig.from_cfg`).
- Produces: `RFTransformer(RFConfig)` with `forward(z Float[B,16,L,32], t Float[B], cond Float[B,L,2·1024], cond_drop Bool[B] | None) -> Float[B,16,L,32]`; `pair_frames(Float[B,T,D]) -> Float[B,ceil(T/2),2D]`; `LatentNorm`; `sample_t(n)`; `rf_loss(model, x0, cond, t, eps, cond_drop)`; `plan_windows(n, win, hop)`; `sine_window(w)`; `sample(v_fn, cond, *, steps, win, cfg_scale=1.0, noise=None, generator=None)`.

The oracle test is worth understanding before reading the code. For a fixed target `x0` the true velocity field is `v(z, t) = (z − x0) / t`. With that oracle in place of the network, the loss must be exactly zero, and one Euler step from `t = 1` must land exactly on `x0` — for any step count. A flipped sign or a reversed time axis fails both, with no training involved.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_codec_rf.py</code> — 191 lines (click to expand)</summary>

```python
"""CPU tests for dc2t.codec.rf (no third_party, no checkpoints).
Run:  pytest -q tests/test_codec_rf.py     or     python tests/test_codec_rf.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from dc2t.codec.rf import (LAT_C, LAT_F, LatentNorm, RFConfig, RFTransformer, blended_velocity, noisy, pair_frames,
                              plan_windows, rf_loss, sample, sample_t, sine_window)


def _x0(B=2, L=8, seed=0):
    return torch.randn(B, LAT_C, L, LAT_F, generator=torch.Generator().manual_seed(seed))


def _oracle(x0):  # exact velocity field for one fixed target: v(z, t) = (z - x0) / t   (plan convention)
    return lambda z, t, c, d: (z - x0.repeat(z.shape[0] // x0.shape[0], 1, 1, 1)) / t.view(-1, 1, 1, 1)  # repeat: CFG doubles the batch


def _tokens(x0):  # latent [B, C, L, F] -> condition-shaped tokens [B, L, C*F]; lets an oracle read its target from cond
    return x0.permute(0, 2, 1, 3).reshape(x0.shape[0], x0.shape[2], -1)


def _oracle_from_cond(z, t, c, d):
    N, C, w, F_ = z.shape
    return (z - c.reshape(N, w, C, F_).permute(0, 2, 1, 3)) / t.view(-1, 1, 1, 1)


def test_noise_path_and_velocity_target():
    x0, eps = _x0(), _x0(seed=1)
    assert torch.equal(noisy(x0, eps, torch.zeros(2)), x0)  # t = 0 is data
    assert torch.equal(noisy(x0, eps, torch.ones(2)), eps)  # t = 1 is noise
    t, h = torch.tensor([0.3, 0.7]), 1e-2
    slope = (noisy(x0, eps, t + h) - noisy(x0, eps, t)) / h  # the path is a straight line, so dz/dt = eps - x0 exactly
    assert torch.allclose(slope, eps - x0, atol=1e-3)


def test_oracle_loss_is_zero_and_sign_is_pinned():
    x0, eps = _x0(), _x0(seed=1)
    t = sample_t(2, generator=torch.Generator().manual_seed(0))
    assert rf_loss(_oracle(x0), x0, None, t, eps).item() < 1e-9
    flipped = lambda z, tt, c, d: -_oracle(x0)(z, tt, c, d)
    assert rf_loss(flipped, x0, None, t, eps).item() > 1.0  # wrong sign (or t running the other way) is caught


def test_weight_identity():
    """Plan 3.4 'w_t = t/(1-t)': the plain velocity MSE equals the eps-prediction loss with that weight. Nothing to add to the loss."""
    g = torch.Generator().manual_seed(0)
    x0, eps, v = (torch.randn(1000, generator=g) for _ in range(3))
    t = torch.rand(1000, generator=g) * 0.98 + 0.01
    z = (1 - t) * x0 + t * eps
    eps_theta = z + (1 - t) * v  # SD3 Eq. 12: eps_theta = -2/(lambda'_t b_t) * (v - a'_t/a_t z) with a = 1-t, b = t
    assert torch.allclose((v - (eps - x0)) ** 2, (eps_theta - eps) ** 2 / (1 - t) ** 2, rtol=1e-4, atol=1e-6)
    lam_prime = -2 / (t * (1 - t))  # d/dt log(a^2 / b^2)
    w = -0.5 * lam_prime * t**2  # SD3: w_t = -1/2 lambda'_t b_t^2
    assert torch.allclose(w, t / (1 - t), rtol=1e-5)
    assert torch.allclose(-0.5 * w * lam_prime, 1 / (1 - t) ** 2, rtol=1e-5)  # the weight L_w puts on ||eps_theta - eps||^2


def test_logit_normal_timesteps():
    t = sample_t(200_000, generator=torch.Generator().manual_seed(0))
    assert 0 < t.min() and t.max() < 1
    assert abs(t.median().item() - 0.5) < 0.01
    u = torch.logit(t)
    assert abs(u.mean().item()) < 0.01 and abs(u.std().item() - 1.0) < 0.01
    t2 = sample_t(200_000, mean=0.5, std=0.6, generator=torch.Generator().manual_seed(0))
    assert abs(torch.logit(t2).mean().item() - 0.5) < 0.01


def test_pair_frames_alignment():
    c = torch.randn(2, 7, 5)  # odd T = 7
    p = pair_frames(c)
    assert p.shape == (2, 4, 10)
    for i in range(3):  # code frames 2i and 2i+1 feed latent frame i
        assert torch.equal(p[:, i, :5], c[:, 2 * i]) and torch.equal(p[:, i, 5:], c[:, 2 * i + 1])
    assert torch.equal(p[:, 3, :5], c[:, 6]) and torch.equal(p[:, 3, 5:], c[:, 6])  # odd T repeats the last frame
    assert pair_frames(torch.randn(1, 1, 5)).shape == (1, 1, 10)  # T = 1
    assert pair_frames(torch.randn(1, 896, 5)).shape == (1, 448, 10)  # one training window


def test_zero_init_output_and_loss_two():
    m = RFTransformer(RFConfig(width=32, depth=2, heads=2, cond_dim=4))
    x0, eps = _x0(4, 8), _x0(4, 8, seed=1)
    t, cond = sample_t(4, generator=torch.Generator().manual_seed(0)), torch.randn(4, 8, 8)
    assert m(noisy(x0, eps, t), t, cond).abs().max() == 0  # zero-init head: output 0 at step 0
    assert abs(rf_loss(m, x0, cond, t, eps).item() - 2.0) < 0.15  # E|eps - x0|^2 = 2 for unit-variance data: the "healthy at init" value
    m(noisy(x0, eps, t), t, cond, torch.tensor([True, False, True, False])).shape == x0.shape


def test_latent_norm_round_trip():
    n = LatentNorm()
    n.mean.copy_(torch.randn(LAT_C, 1, LAT_F))
    n.std.copy_(torch.rand(LAT_C, 1, LAT_F) + 0.5)
    x = _x0()
    assert torch.allclose(n.denormalize(n.normalize(x)), x, atol=1e-5)


def test_param_counts_pilot_and_full():
    def count(c):
        with torch.device("meta"):
            return sum(p.numel() for p in RFTransformer(c).parameters())

    assert count(RFConfig(width=512, depth=8, heads=8)) == PILOT
    assert count(RFConfig(width=1024, depth=24, heads=16)) == FULL


def test_euler_oracle_lands_on_x0_for_any_step_count():
    x0, cond = _x0(2, 8), torch.zeros(2, 8, 4)
    for steps in (1, 2, 5):
        for cfg in (1.0, 2.0):  # identical conditional / unconditional fields: guidance cannot change the answer
            out = sample(_oracle(x0), cond, steps=steps, win=8, cfg_scale=cfg, generator=torch.Generator().manual_seed(0))
            assert torch.allclose(out, x0, atol=1e-5), (steps, cfg)


def test_plan_windows_cover_every_frame():
    for n in (1, 2, 3, 447, 448, 449, 672, 673, 1000, 3750):
        w = min(448, n)
        starts = plan_windows(n, 448, 224)
        assert starts[0] == 0 and starts[-1] + w == n, (n, starts)
        assert all(b - a <= 224 for a, b in zip(starts, starts[1:])) and starts == sorted(set(starts))
        covered = torch.zeros(n, dtype=torch.long)
        for s in starts:
            covered[s : s + w] += 1
        assert covered.min() >= 1
    assert plan_windows(3750, 448, 224) == [i * 224 for i in range(15)] + [3302]  # T = 7500 code frames: 16 windows
    assert plan_windows(1, 448, 224) == [0]  # T = 1 or 2 code frames: one window of one latent frame


def test_windowed_sampler_is_exact_through_windows():
    for L in (1, 2, 5, 8, 9, 23, 100):  # win = 8, hop = 4: single window, exact fit, ragged tail, many windows
        x0 = _x0(2, L, seed=L)
        out = sample(_oracle_from_cond, _tokens(x0), steps=3, win=8, generator=torch.Generator().manual_seed(1))
        assert torch.allclose(out, x0, atol=1e-5), L
    x0 = _x0(1, 3750)  # a 300 s clip: T = 7500 code frames, 16 windows of 448
    out = sample(_oracle_from_cond, _tokens(x0), steps=2, win=448, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(out, x0, atol=1e-4)


def test_blend_weights_partition_unity_and_edges():
    z, cond = torch.zeros(1, LAT_C, 10, LAT_F), torch.arange(10.0).view(1, 10, 1)
    const = lambda zz, t, c, d: torch.full_like(zz, 3.0)
    assert torch.allclose(blended_velocity(const, z, 0.5, cond, [0, 2], 8, 1.0), torch.full_like(z, 3.0))
    by_start = lambda zz, t, c, d: torch.where(c[:, :1, :1] == 0, 1.0, 2.0).view(-1, 1, 1, 1).expand_as(zz)  # window at 0 says 1, at 2 says 2
    v = blended_velocity(by_start, z, 0.5, cond, [0, 2], 8, 1.0)
    w = sine_window(8)
    assert torch.allclose(v[0, 0, 0, 0], torch.tensor(1.0))  # frame 0 only in window 0 (its weight is tiny but positive)
    assert torch.allclose(v[0, 0, 9, 0], torch.tensor(2.0))  # last frame only in window 1
    expect = (w[5] * 1 + w[3] * 2) / (w[5] + w[3])  # frame 5: position 5 in window 0, position 3 in window 1
    assert torch.allclose(v[0, 0, 5, 0], expect)


def _toy(n, seed, Dc=4, Tc=8):  # a learnable task: the latent is a fixed linear map of the paired code embeddings
    codes = torch.randn(n, Tc, Dc, generator=torch.Generator().manual_seed(seed))
    cond = pair_frames(codes)
    wmap = torch.randn(2 * Dc, LAT_C * LAT_F, generator=torch.Generator().manual_seed(123)) / (2 * Dc) ** 0.5
    return codes, cond, (cond @ wmap).view(n, Tc // 2, LAT_C, LAT_F).permute(0, 2, 1, 3).contiguous()


def test_tiny_model_learns_the_condition_and_alignment_matters():
    """width 512 on purpose: the head must be able to pass the 512 noise values per frame (v ~ z at t ~ 1); width 64 plateaus at loss ~ 0.9."""
    torch.manual_seed(0)
    m = RFTransformer(RFConfig(width=512, depth=1, heads=8, cond_dim=4))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    g, hist = torch.Generator().manual_seed(0), []
    for step in range(300):
        _, cond, x0 = _toy(16, 1000 + step)
        loss = rf_loss(m, x0, cond, sample_t(16, generator=g), torch.randn(x0.shape, generator=g))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
    assert abs(hist[0] - 2.0) < 0.3 and sum(hist[-50:]) / 50 < 0.4, (hist[0], sum(hist[-50:]) / 50)
    codes, cond, x0 = _toy(8, 7)  # held-out conditions
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(5))
    v_fn = lambda z, t, c, d: m(z, t, c, d)
    ok = (sample(v_fn, cond, steps=16, win=4, noise=noise) - x0).pow(2).mean().item()
    bad = (sample(v_fn, pair_frames(codes.roll(1, 1)), steps=16, win=4, noise=noise) - x0).pow(2).mean().item()
    assert ok < 0.2 and bad > 3 * ok, (ok, bad)  # unit-variance targets: 1.0 is "ignore the condition"; a one-code-frame shift hurts


PILOT, FULL = 28_774_400, 313_123_328  # exact counts: width 512 / depth 8 / heads 8 and width 1024 / depth 24 / heads 16

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_codec_rf: {len(tests)} tests passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_codec_rf.py` → `ModuleNotFoundError: No module named 'dc2t.codec.rf'`
- [ ] **Step 3: implement**

**`dc2t/codec/rf.py`** — 225 lines

```python
"""Rectified-flow transformer for Mel-VAE latents (plan 3.4). Pure PyTorch: no third_party import, CPU-testable.

Plan time convention: z_t = (1 - t) * x0 + t * eps,  t = 0 is DATA, t = 1 is NOISE, velocity target u = eps - x0.
(MuCodec's released sampler runs the other way: t = 0 noise, t = 1 data. Never mix the two.)
Shapes: B batch, C = 16 latent channels, L latent frames (12.5 Hz), F = 32 mel bins, T = 2 * L code frames (25 Hz).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LAT_C, LAT_F = 16, 32  # Mel-VAE latent of the released AudioLDM-48k VAE: z_channels 16, 256 mel bins / 8 = 32
FRAMES_PER_LATENT = 2  # 25 Hz code frames per 12.5 Hz latent frame


@dataclass(frozen=True)
class RFConfig:
    width: int = 1024
    depth: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    cond_dim: int = 1024  # width of one code-frame embedding = RVQ input_dim

    @staticmethod
    def from_cfg(rf) -> "RFConfig":
        return RFConfig(width=rf.width, depth=rf.depth, heads=rf.heads)


# ---------------------------------------------------------------- condition alignment
def pair_frames(cond: Tensor) -> Tensor:
    """Float[B, T, D] code-frame embeddings (25 Hz) -> Float[B, ceil(T/2), 2D]: latent frame i gets code frames 2i, 2i+1.
    An odd T repeats the last frame so that T = 1 still gives one latent frame."""
    if cond.shape[1] % 2:
        cond = torch.cat([cond, cond[:, -1:]], 1)
    B, T, D = cond.shape
    return cond.reshape(B, T // FRAMES_PER_LATENT, FRAMES_PER_LATENT * D)


# ---------------------------------------------------------------- network
def timestep_embedding(t: Tensor, dim: int = 256, max_period: float = 10000.0) -> Tensor:
    """Float[B] in [0, 1] -> Float[B, dim]; sinusoidal in 1000 * t, [cos, sin] order (as the released flow model)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class RMSNorm(nn.Module):  # torch.nn.RMSNorm needs torch >= 2.4; MuCodec pins 2.2
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.w


def rope_tables(L: int, hd: int, device) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    ang = torch.arange(L, device=device, dtype=torch.float32)[:, None] * inv[None]
    return torch.cos(ang), torch.sin(ang)  # each [L, hd/2]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # x [B, H, L, hd], rotate-half convention
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)


class Block(nn.Module):
    """Pre-norm block with PixArt-style adaLN-single: the timestep MLP is shared, each block adds its own table."""

    def __init__(self, D: int, H: int, mlp_ratio: int):
        super().__init__()
        self.H, self.hd = H, D // H
        self.n1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)  # QK-norm (SD3 sec. 5.3.2) keeps bf16 attention stable
        self.n2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(D, mlp_ratio * D), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * D, D))
        self.table = nn.Parameter(torch.zeros(6, D))  # shift/scale/gate for attention and MLP; zeros => block starts as identity

    def forward(self, x: Tensor, mod: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, L, D = x.shape
        s1, c1, g1, s2, c2, g2 = (self.table[None] + mod).unbind(1)  # each [B, D]
        h = self.n1(x) * (1 + c1[:, None]) + s1[:, None]
        q, k, v = self.qkv(h).view(B, L, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)  # each [B, H, L, hd]
        q, k = apply_rope(self.qn(q), cos, sin), apply_rope(self.kn(k), cos, sin)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        x = x + g1[:, None] * self.proj(a)
        h = self.n2(x) * (1 + c2[:, None]) + s2[:, None]
        return x + g2[:, None] * self.mlp(h)


class RFTransformer(nn.Module):
    """v_theta(z_t, t; c). One token per latent frame (16 x 32 = 512 values); the code condition is added at the input.

    z Float[B, 16, L, 32] noisy latent, t Float[B] in [0, 1], cond Float[B, L, 2 * cond_dim] (see pair_frames),
    cond_drop Bool[B] | None (True = replace the condition by the learned null vector) -> v Float[B, 16, L, 32].
    """

    def __init__(self, c: RFConfig):
        super().__init__()
        D, H = c.width, c.heads
        assert D % H == 0 and (D // H) % 2 == 0, "width must split into an even head size"
        self.hd = D // H
        self.x_in = nn.Linear(LAT_C * LAT_F, D)
        self.c_in = nn.Linear(FRAMES_PER_LATENT * c.cond_dim, D)
        self.null = nn.Parameter(torch.zeros(D))
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.t_mod = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        self.blocks = nn.ModuleList(Block(D, H, c.mlp_ratio) for _ in range(c.depth))
        self.final_table = nn.Parameter(torch.zeros(2, D))
        self.final_norm = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(D, LAT_C * LAT_F)
        for m in (self.t_mod[1], self.out):  # zero-init: every block is the identity and the output is 0 at step 0
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z: Tensor, t: Tensor, cond: Tensor, cond_drop: Tensor | None = None) -> Tensor:
        B, C, L, F_ = z.shape
        h = self.x_in(z.permute(0, 2, 1, 3).reshape(B, L, C * F_))
        c = self.c_in(cond)
        if cond_drop is not None:
            c = torch.where(cond_drop.view(B, 1, 1), self.null.to(c.dtype), c)
        h = h + c
        emb = self.t_mlp(timestep_embedding(t))
        mod = self.t_mod(emb).view(B, 6, -1)
        cos, sin = rope_tables(L, self.hd, h.device)
        for blk in self.blocks:
            h = blk(h, mod, cos, sin)
        shift, scale = (self.final_table[None] + emb[:, None]).unbind(1)
        h = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]
        return self.out(h).view(B, L, C, F_).permute(0, 2, 1, 3)


class LatentNorm(nn.Module):
    """Per-(channel, mel-bin) standardisation of Mel-VAE latents, as MuCodec's Feature2DProcessor (model.py:37-80)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(LAT_C, 1, LAT_F))
        self.register_buffer("std", torch.ones(LAT_C, 1, LAT_F))

    def normalize(self, x: Tensor) -> Tensor:  # x [B, 16, L, 32]
        return (x - self.mean) / self.std

    def denormalize(self, z: Tensor) -> Tensor:
        return z * self.std + self.mean


# ---------------------------------------------------------------- objective (plan 3.4, SD3 sec. 3.1)
def sample_t(n: int, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None) -> Tensor:
    """Logit-normal timesteps, SD3's rf/lognorm(m, s): t = sigmoid(N(m, s)). Float[n] in (0, 1)."""
    return torch.sigmoid(torch.randn(n, generator=generator) * std + mean)


def noisy(x0: Tensor, eps: Tensor, t: Tensor) -> Tensor:  # z_t = (1 - t) x0 + t eps
    t = t.view(-1, 1, 1, 1).to(x0.dtype)
    return (1 - t) * x0 + t * eps


def rf_loss_per_sample(model, x0: Tensor, cond: Tensor, t: Tensor, eps: Tensor, cond_drop: Tensor | None = None) -> Tensor:
    """Plain velocity MSE per example, Float[B]. x0 Float[B,16,L,32] is already standardised."""
    v = model(noisy(x0, eps, t), t, cond, cond_drop)
    return (v.float() - (eps - x0)).pow(2).mean((1, 2, 3))


def rf_loss(model, x0: Tensor, cond: Tensor, t: Tensor, eps: Tensor, cond_drop: Tensor | None = None) -> Tensor:
    """Scalar training loss. Uniform t with this loss IS the weighting w_t = t / (1 - t) of plan 3.4 (test_weight_identity);
    sampling t from a density pi(t) multiplies that weight by pi(t). Add NO explicit weight on top."""
    return rf_loss_per_sample(model, x0, cond, t, eps, cond_drop).mean()


# ---------------------------------------------------------------- sampler and long-form windows
def plan_windows(n: int, win: int, hop: int) -> list[int]:
    """Start frames of windows of length min(win, n) that cover [0, n); consecutive starts differ by <= hop."""
    w = min(win, n)
    starts = list(range(0, n - w + 1, hop))
    if starts[-1] + w < n:
        starts.append(n - w)
    return starts


def sine_window(w: int) -> Tensor:  # strictly positive, so edge frames covered by one window still get weight
    return torch.sin(math.pi * (torch.arange(w, dtype=torch.float32) + 0.5) / w)


def blended_velocity(v_fn, z: Tensor, t: float, cond: Tensor, starts: list[int], w: int, cfg_scale: float) -> Tensor:
    """Run every window on the SHARED state z and average the velocities with sine weights (MultiDiffusion, Eq. 4 closed form)."""
    B, C, L, F_ = z.shape
    n = len(starts)
    zs = torch.cat([z[:, :, s : s + w] for s in starts])  # [n*B, C, w, F]
    cs = torch.cat([cond[:, s : s + w] for s in starts])  # [n*B, w, Dc]
    tt = torch.full((n * B,), t, device=z.device)
    if cfg_scale == 1.0:
        v = v_fn(zs, tt, cs, None).float()
    else:
        drop = torch.cat([torch.zeros(n * B, dtype=torch.bool), torch.ones(n * B, dtype=torch.bool)]).to(z.device)
        vc, vu = v_fn(torch.cat([zs, zs]), torch.cat([tt, tt]), torch.cat([cs, cs]), drop).float().chunk(2)
        v = vu + cfg_scale * (vc - vu)  # classifier-free guidance; scale 1 = conditional only
    v = v.view(n, B, C, w, F_)
    wt = sine_window(w).to(z).view(1, 1, w, 1)
    num, den = torch.zeros_like(z), torch.zeros(1, 1, L, 1, device=z.device, dtype=z.dtype)
    for i, s in enumerate(starts):
        num[:, :, s : s + w] += v[i] * wt
        den[:, :, s : s + w] += wt
    return num / den


@torch.no_grad()
def sample(v_fn, cond: Tensor, *, steps: int, win: int, cfg_scale: float = 1.0, noise: Tensor | None = None,
           generator: torch.Generator | None = None) -> Tensor:
    """Euler ODE from t = 1 (noise) to t = 0 (data) on a uniform grid; the model is never evaluated at t = 0.
    cond Float[B, L, Dc] (paired frames); win = longest window in latent frames. Returns x0 Float[B, 16, L, 32]."""
    B, L, _ = cond.shape
    w = min(win, L)
    starts = plan_windows(L, w, max(w // 2, 1))
    z = noise if noise is not None else torch.randn(B, LAT_C, L, LAT_F, generator=generator).to(cond.device)
    for i in range(steps):
        z = z - blended_velocity(v_fn, z, 1.0 - i / steps, cond, starts, w, cfg_scale) / steps  # z + (t_next - t) v
    return z
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_codec_rf.py` → `test_codec_rf: 13 tests passed` (about 20 s on a laptop CPU; one test trains a small model for 300 steps)
- [ ] **Step 5: commit** — `git add dc2t/codec tests/test_codec_rf.py && git commit -m "feat: rectified flow transformer, loss and windowed sampler"`

### Task 2: the `Codec` class

**Files:** Create `dc2t/codec/codec.py` · Test `tests/test_codec_codec.py`

**Interfaces:**
- Consumes: `rf.py`; `cfg.paths.mucodec_root`, `cfg.paths.data_root`, `cfg.audio.sample_rate`, `cfg.audio.frame_rate`, `cfg.codec.*`, `cfg.rf.decode_steps`, `cfg.rf.decode_cfg`, `cfg.rf.vae_window`, `cfg.rf.vae_hop`.
- Produces: `Codec.load(cfg, device, *, encoder=True, decoder=True)`, `Codec.encode(wav)`, `Codec.decode(codes, *, steps=None, cfg_scale=None, seed=None)`, `Codec.fingerprint()` (contract §9); `Released` with `features(x48)`, `latents(x48)`, `decode_window(z)`, `new_rvq(K, V)`; `to_rate`, `tile_windows`, `rvq_fingerprint`, `claim_tag_dir`, `check_tag_dir`; constants `VAE_SR = 48000`, `VAE_FRAME = 3840`.
- Checkpoint layout it reads (written by Task 3 through the engine): keys prefixed `rvq.`, `ema.` and `norm.` inside `<codec.checkpoint>/model.safetensors`.

What the wrapper absorbs, each with its line in the code: the working-directory and module-name assumptions (`_in_mucodec`); the extra frame of `sound2code`; mono 32 kHz against stereo 48 kHz; the crash lengths of `code2sound`; its use of the global random generator (`fork_rng`, so a seed is honoured without disturbing the caller); its printing; the silent `strict=False` load (checked through the normalisation statistics).

Note on `encoder=` / `decoder=`: the released stack always loads all three checkpoints (its own flag is broken, §1.2). The flags only skip **our** weights: `decoder=False` skips the RF Transformer (tokenising), and on the plan path the released flow transformer is deleted to free memory.

- [ ] **Step 1: write the failing test** — the fake `Released` reproduces the quirks: one frame too many from `sound2code`, an assertion on the crash lengths, output occasionally one sample short and outside `[-1, 1]`.

<details>
<summary><code>tests/test_codec_codec.py</code> — 202 lines (click to expand)</summary>

```python
"""CPU tests for dc2t.codec.codec: exact lengths, windows, fingerprint. The released stack is replaced by a fake that has the
same shapes and the same quirks (T+1 frames, stereo output, one sample short, a crash length). No third_party, no checkpoints.
Run:  pytest -q tests/test_codec_codec.py     or     python tests/test_codec_codec.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from dc2t.codec.codec import VAE_FRAME, Codec, claim_tag_dir, rvq_fingerprint, tile_windows, to_rate
from dc2t.codec.rf import LatentNorm, RFConfig, RFTransformer

ns = types.SimpleNamespace
K4, V4 = 4, 50  # a tiny plan-path codebook; nothing in the code may assume 4 x 10000


def make_cfg(checkpoint=None, K=1, V=16384, root=".", sr=32000):
    return ns(paths=ns(data_root=root, mucodec_root="unused"), audio=ns(sample_rate=sr, frame_rate=25),
              codec=ns(tag="t", num_codebooks=K, codebook_size=V, muq_layer=7, window_frames=896, checkpoint=checkpoint),
              rf=ns(decode_steps=2, decode_cfg=1.0, vae_window=128, vae_hop=96, width=32, depth=1, heads=2))


class FakeRVQ(nn.Module):  # same interface as ResidualVectorQuantize: forward -> 6-tuple, from_codes -> (z_q, z_p, codes)
    def __init__(self, K=K4, V=V4, seed=0):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(K, V, 1024, generator=torch.Generator().manual_seed(seed)), requires_grad=False)
        self.register_buffer("stale_counter", torch.zeros(V))

    def forward(self, z):  # z [B, 1024, T]; codes depend on the features only, level k reads channel k
        codes = torch.stack([(z[:, k] * 1000).round().long().abs() % self.emb.shape[1] for k in range(self.emb.shape[0])], 1)
        return self.from_codes(codes)[0], codes, None, torch.zeros(()), torch.zeros(()), None

    def from_codes(self, codes):  # [B, K, T] -> [B, 1024, T]
        return sum(self.emb[k][codes[:, k]] for k in range(codes.shape[1])).transpose(1, 2), None, codes


class FakeReleased:
    def __init__(self, rvq=None):
        self.calls, self.rvq = [], rvq

    def sound2code(self, stereo):  # [2, N48] -> [1, 1, T + 1]: the release returns one frame too many (generate.py:85)
        T = stereo.shape[1] // 1920
        return (torch.arange(T + 1) % 16384).view(1, 1, -1)

    def code2sound(self, codes, steps):
        T = codes.shape[-1]
        self.calls.append(T)
        assert not (T >= 1024 and (T - 128) % 768 == 0), f"the released code2sound crashes on T={T}"
        t = torch.linspace(-1.5, 1.5, T * 1920 - (T % 7 == 0))  # sometimes 1 sample short; exceeds [-1, 1]
        return torch.stack([t, t])

    def features(self, x48):  # local, position-independent: channel c of frame t = c-th scaled frame mean
        B, N = x48.shape
        m = x48.view(B, N // 1920, 1920).mean(-1)
        return m[:, None, :] * torch.arange(1, 1025).view(1, -1, 1) / 100

    def decode_window(self, z):  # a purely local map, so windows must reproduce the one-shot result exactly
        return z.mean(dim=(1, 3)).repeat_interleave(VAE_FRAME, -1)


def released_codec(sr=32000):
    return Codec(make_cfg(sr=sr), "cpu", FakeReleased())


def plan_codec(seed=0):
    cfg = make_cfg(checkpoint="ckpt", K=K4, V=V4)
    rf = RFTransformer(RFConfig(width=32, depth=1, heads=2)).eval()
    for p in rf.parameters():  # un-zero the head so the output depends on the latents
        torch.nn.init.normal_(p, std=0.02) if p.ndim > 1 else None
    return Codec(cfg, "cpu", FakeReleased(), rvq=FakeRVQ(seed=seed).eval(), rf=rf, norm=LatentNorm())


def test_to_rate_exact_lengths():
    for T in (1, 2, 3, 897, 7500):
        up = to_rate(torch.randn(T * 1280), 32000, 48000)
        assert up.shape == (T * 1920,) and to_rate(up, 48000, 32000).shape == (T * 1280,)
    assert to_rate(torch.randn(2, 4410), 44100, 48000).shape == (2, 4800)  # any rate that is a multiple of 25 works


def test_tile_windows_keep_every_frame_once():
    for T in (1, 5, 895, 896, 897, 1791, 1792, 1793, 7500):
        win = tile_windows(T, 896)
        assert all(stop - start == 896 for start, stop, _ in win) or len(win) == 1  # every encoder call sees one full window
        kept = [f for start, stop, keep in win for f in range(keep, stop)]
        assert kept == list(range(T)), T


def test_released_encode_gives_exactly_T_frames():
    c = released_codec()
    for T in (1, 25, 750, 7500):
        codes = c.encode(torch.zeros(2, T * 1280))
        assert codes.shape == (2, 1, T) and codes.dtype == torch.long
        assert torch.equal(codes[0, 0], torch.arange(T))  # frame t is frame t: the extra frame is cut from the END


def test_encode_rejects_bad_input():
    c = released_codec()
    for bad in (torch.zeros(1280), torch.zeros(1, 1281), torch.zeros(1, 0), torch.full((1, 1280), float("nan"))):
        try:
            c.encode(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {tuple(bad.shape)}")


def test_released_decode_gives_exact_length_in_range_for_every_T():
    c = released_codec()
    for T in (1, 2, 3, 7, 25, 1023, 1024, 1664, 3200, 7500):  # 7 and 3200 hit the short-by-one-sample and the crash cases
        wav = c.decode(torch.zeros(2, 1, T, dtype=torch.long), seed=1)
        assert wav.shape == (2, T * 1280) and wav.abs().max() <= 1.0, T
    assert 3201 in c.rel.calls and 3200 not in c.rel.calls  # the wrapper never lets T = 128 s reach code2sound


def test_released_decode_is_exact_at_48k_too():
    """D2: audio.sample_rate may become 48000 later. Then no resampler hides code2sound's one-sample shortfall (T = 7 is short)."""
    c = released_codec(sr=48000)
    for T in (1, 7, 14, 3200):
        assert c.decode(torch.zeros(1, 1, T, dtype=torch.long)).shape == (1, T * 1920), T
    assert c.encode(torch.zeros(1, 5 * 1920)).shape == (1, 1, 5)


def test_released_decode_rejects_other_guidance_and_bad_codes():
    c = released_codec()
    ok = torch.zeros(1, 1, 5, dtype=torch.long)
    assert c.decode(ok, cfg_scale=1.5).shape == (1, 6400)
    for args in (dict(codes=ok, cfg_scale=2.0), dict(codes=torch.zeros(1, 2, 5, dtype=torch.long)),
                 dict(codes=torch.full((1, 1, 5), 16384)), dict(codes=torch.zeros(1, 1, 0, dtype=torch.long))):
        try:
            c.decode(**args)
        except (NotImplementedError, ValueError):
            continue
        raise AssertionError(args)


def test_plan_encode_windows_equal_one_shot_encode():
    c = plan_codec()
    for T in (1, 5, 896, 897, 1800, 2689):
        wav = torch.randn(1, T * 1280, generator=torch.Generator().manual_seed(T)) * 0.1
        x48 = to_rate(wav, 32000, 48000)
        want = c.rvq(c.rel.features(x48))[1]  # whole clip in one call (the fake is local, so this is the reference)
        got = c.encode(wav)
        assert got.shape == (1, K4, T) and torch.equal(got, want), T
        assert got.min() >= 0 and got.max() < V4


def test_plan_decode_exact_length_for_odd_short_and_long_T():
    c = plan_codec()
    for T in (1, 2, 3, 897, 1500, 7500):
        codes = torch.randint(0, V4, (1, K4, T), generator=torch.Generator().manual_seed(T))
        wav = c.decode(codes, steps=1 if T == 7500 else 2, seed=0)
        assert wav.shape == (1, T * 1280) and wav.abs().max() <= 1.0, T


def test_plan_decode_is_deterministic_for_a_seed():
    c = plan_codec()
    codes = torch.randint(0, V4, (1, K4, 40))
    a, b, d = c.decode(codes, seed=3), c.decode(codes, seed=3), c.decode(codes, seed=4)
    assert torch.equal(a, b) and not torch.equal(a, d)


def test_vae_windows_reproduce_one_shot_decode():
    c = plan_codec()
    for L in (1, 2, 127, 128, 129, 200, 1000, 3750):
        lat = torch.randn(1, 16, L, 32, generator=torch.Generator().manual_seed(L))
        assert torch.allclose(c._decode_latents(lat), c.rel.decode_window(lat)[0], atol=1e-5), L  # no seams, any length


def test_fingerprint_changes_with_weights_not_with_counters():
    a, b = FakeRVQ(seed=0), FakeRVQ(seed=1)
    f = lambda m: rvq_fingerprint(m, K4, V4, 7, 896)
    assert f(a) == f(FakeRVQ(seed=0)) and f(a) != f(b)
    a.stale_counter += 5
    assert f(a) == f(FakeRVQ(seed=0))  # training bookkeeping is not part of the identity
    assert rvq_fingerprint(a, K4, V4, 6, 896) != f(a)  # nor is the encoder layer


def test_tag_directory_refuses_tokens_from_another_rvq():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d) / "codes" / "k4v50"
        mine = {"fingerprint": "aaaa", "K": K4}
        claim_tag_dir(d, mine)
        claim_tag_dir(d, mine)  # same tokeniser: fine (resume)
        assert json.loads((d / "meta.json").read_text())["fingerprint"] == "aaaa"
        try:
            claim_tag_dir(d, {"fingerprint": "bbbb", "K": K4})
        except RuntimeError as e:
            assert "codec.tag" in str(e)
        else:
            raise AssertionError("a different RVQ was allowed to write into an existing tag")
        assert [p.name for p in d.iterdir()] == ["meta.json"]  # no temp files left behind


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_codec_codec: {len(tests)} tests passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_codec_codec.py` → `ModuleNotFoundError: No module named 'dc2t.codec.codec'`
- [ ] **Step 3: implement**

**`dc2t/codec/codec.py`** — 311 lines

```python
"""Codec: waveform <-> RVQ codes (contract section 9). The ONLY module that imports from third_party/MuCodec.

Two configurations (decision D1), one class:
  released: K = 1, V = 16384, the public MuCodec weights, wrapped (encode = sound2code, decode = code2sound);
  plan:     K = 4, V = 10000, our RVQ + rectified-flow transformer, with the released MuEncoder / Mel-VAE / HiFi-GAN frozen.
Shapes: B batch, N samples at cfg.audio.sample_rate, T = N * 25 / sample_rate code frames, L = ceil(T / 2) latent frames.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch import Tensor

from dc2t.codec.rf import (FRAMES_PER_LATENT, LatentNorm, RFConfig, RFTransformer, pair_frames, plan_windows, sample,
                              sine_window)

VAE_SR = 48000  # Mel-VAE / HiFi-GAN rate (tools/get_melvaehifigan48k.py:1472)
VAE_FRAME = 3840  # samples at 48 kHz per latent frame = STFT hop 480 (:1478) x 8 mel frames per latent frame (ch_mult has 4 levels, :1516)
RELEASED_WINDOW = 1024  # code frames per encoder window in the released sound2code (generate.py:84)


# ---------------------------------------------------------------- small helpers
@contextlib.contextmanager
def quiet():  # the released code prints on every call (RVQ usage, output_len, ...)
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def _in_mucodec(root: Path):
    """The released code resolves muq_dev/... against the CWD (model.py:181-186) and uses top-level names
    (model, models, tools, libs, generate): put the repo first on sys.path and chdir into it, then undo both."""
    old = os.getcwd()
    sys.path.insert(0, str(root))
    os.chdir(root)  # not thread-safe; Python 3.10 has no contextlib.chdir
    try:
        yield
    finally:
        os.chdir(old)
        sys.path.remove(str(root))


def to_rate(x: Tensor, orig: int, new: int) -> Tensor:
    """Float[..., N] -> Float[..., N * new / orig] on the CPU (scipy polyphase). The length is exact whenever
    N * new / orig is an integer, which holds for every N = T * rate / 25 and every rate that is a multiple of 25."""
    x = x.detach().cpu().float()
    if orig == new:
        return x
    g = math.gcd(orig, new)
    return torch.from_numpy(np.ascontiguousarray(resample_poly(x.numpy(), new // g, orig // g, axis=-1), dtype=np.float32))


def tile_windows(T: int, W: int) -> list[tuple[int, int, int]]:
    """(start, stop, keep_from) frame ranges the encoder sees so that every call gets exactly W frames (the training shape).
    The frames [keep_from, stop) of each window are kept; the kept ranges tile [0, T) once. T < W: one short window."""
    if T <= W:
        return [(0, T, 0)]
    starts = list(range(0, T - W + 1, W))
    if starts[-1] + W < T:
        starts.append(T - W)  # last window overlaps the previous one; only its new frames are kept
    out, kept = [], 0
    for s in starts:
        out.append((s, s + W, kept))
        kept = s + W
    return out


# ---------------------------------------------------------------- which RVQ made these tokens
def rvq_fingerprint(rvq: torch.nn.Module, K: int, V: int, layer: int, window: int) -> str:
    """16 hex chars identifying the tokeniser: shapes + every RVQ weight (not the training-time stale counters)."""
    h = hashlib.sha256(f"K={K};V={V};layer={layer};window={window}".encode())
    for k, v in sorted(rvq.state_dict().items()):
        if not k.endswith("stale_counter"):
            h.update(k.encode())
            h.update(v.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def check_tag_dir(d: Path, fp: str) -> None:
    """Refuse to use a tag directory that another tokeniser wrote (FileNotFoundError: it was never tokenised)."""
    old = json.loads((d / "meta.json").read_text())
    if old["fingerprint"] != fp:
        raise RuntimeError(f"{d} holds tokens from tokeniser {old['fingerprint']} but this codec is {fp}: "
                           "retraining the RVQ means a NEW codec.tag, re-tokenising and retraining the language model")


def claim_tag_dir(d: Path, meta: dict) -> None:
    """Create d/meta.json atomically (shards race here), or verify that the existing one came from the same tokeniser."""
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".meta.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    try:
        os.link(tmp, d / "meta.json")  # fails if it exists: exclusive and complete-or-absent
    except FileExistsError:
        pass
    finally:
        tmp.unlink()
    check_tag_dir(d, meta["fingerprint"])


# ---------------------------------------------------------------- the released stack (everything third-party is here)
class Released:
    def __init__(self, mu, torch_tools, rvq_cls):
        self.mu, self.tt, self.rvq_cls = mu, torch_tools, rvq_cls
        self.device = torch.device(mu.device)

    @classmethod
    def load(cls, cfg, device) -> "Released":
        root = Path(cfg.paths.mucodec_root).resolve()
        files = [root / "ckpt" / "mucodec.pt", root / "muq_dev" / "muq.pt", root / "tools" / "audioldm_48k.pth"]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"MuCodec checkpoints missing (chapter 04, Task 0 fetches them): {missing}")
        with _in_mucodec(root), quiet():
            import generate  # third_party/MuCodec/generate.py
            import tools.torch_tools as torch_tools
            from libs.rvq.descript_quantize3 import ResidualVectorQuantize

            # load_main_model=False is broken in the release (model.py:167 asserts), so always load everything.
            # strict=False inside (generate.py:43): a wrong checkpoint would load silently, hence the check below.
            mu = generate.MuCodec(model_path=str(files[0]), layer_num=cfg.codec.muq_layer, load_main_model=True, device=str(device))
        if mu.model.normfeat.counts.item() <= 0:
            raise RuntimeError("mucodec.pt did not load (normalisation statistics are empty): wrong or truncated file")
        return cls(mu, torch_tools, ResidualVectorQuantize)

    def new_rvq(self, K: int, V: int) -> torch.nn.Module:  # same hyper-parameters as model.py:190, except K and V
        return self.rvq_cls(input_dim=1024, n_codebooks=K, codebook_size=V, codebook_dim=32, quantizer_dropout=0.0, stale_tolerance=200)

    @property
    def rvq(self) -> torch.nn.Module:  # the released 1 x 16384 quantiser
        return self.mu.model.rvq_muencoder_emb

    # -- released path
    def sound2code(self, stereo48: Tensor) -> Tensor:  # [2, N48] -> [1, 1, >= N48/1920 frames]; one clip per call (generate.py:82)
        return self.mu.sound2code(stereo48)

    def code2sound(self, codes: Tensor, steps: int) -> Tensor:  # [1, 1, T] -> [2, ~T * 1920] stereo on the CPU; guidance is fixed at 1.5
        return self.mu.code2sound(codes, num_steps=steps, disable_progress=True)

    # -- plan path: the frozen MuEncoder / Mel-VAE / HiFi-GAN
    @torch.no_grad()
    def features(self, x48: Tensor) -> Tensor:  # Float[B, N48] -> Float[B, 1024, N48 // 1920]  (MuEncoder block `layer`, fp32)
        x = self.mu.preprocess_audio(x48[:, None])[:, 0]  # peak limit 0.8 (generate.py:206-214), as the released encoder input
        return self.mu.model.extract_muencoder_embeds(x, x, self.mu.layer_num)  # model.py:236-242

    @torch.no_grad()
    def latents(self, x48: Tensor, sample: bool = True) -> Tensor:  # Float[B, N48] -> Float[B, 16, N48 // 3840, 32]
        mel, _, _ = self.tt.wav_to_fbank2(x48.float(), -1, fn_STFT=self.mu.stft)  # [B, N48 // 480 + 1, 256]; clips to [-1, 1]
        vae, out = self.mu.vae, []
        for i in range(mel.shape[0]):  # one at a time: the VAE's mid-block attention is quadratic in (frames x 32)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                post = vae.encode_first_stage(mel[i : i + 1, None])
            out.append(vae.get_first_stage_encoding(post, use_mode=not sample).float())  # posterior sample x scale_factor
        return torch.cat(out)

    @torch.no_grad()
    def decode_window(self, z: Tensor) -> Tensor:  # Float[B, 16, W, 32] (scaled latent space) -> Float[B, 3840 * W] at 48 kHz
        vae = self.mu.vae
        mel = vae.decode_first_stage(z)  # [B, 1, 8W, 256]; divides by scale_factor (get_melvaehifigan48k.py:984-992)
        return vae.vocoder(mel.squeeze(1).permute(0, 2, 1)).squeeze(1)  # HiFi-GAN [B, 256, 8W] -> [B, 1, 480 * 8W], tanh output


def load_rvq(cfg, rel: Released, checkpoint: str | Path) -> torch.nn.Module:
    """Build the K x V RVQ and load the `rvq.*` weights of an engine checkpoint directory (strict)."""
    from safetensors.torch import load_file

    sd = {k.removeprefix("module."): v for k, v in load_file(str(Path(checkpoint) / "model.safetensors")).items()}
    w = {k[4:]: v for k, v in sd.items() if k.startswith("rvq.")}
    if not w:
        raise KeyError(f"{checkpoint}/model.safetensors has no 'rvq.' weights")
    rvq = rel.new_rvq(cfg.codec.num_codebooks, cfg.codec.codebook_size)
    rvq.load_state_dict(w)
    return rvq.eval().requires_grad_(False)


# ---------------------------------------------------------------- the public class
class Codec:
    def __init__(self, cfg, device, released: Released, *, rvq=None, rf=None, norm=None):
        self.cfg, self.device, self.rel = cfg, torch.device(device), released
        self.num_codebooks, self.codebook_size = cfg.codec.num_codebooks, cfg.codec.codebook_size
        self.frame_rate, self.sample_rate = cfg.audio.frame_rate, cfg.audio.sample_rate
        self.spf = self.sample_rate // self.frame_rate  # samples per code frame (1280 at 32 kHz)
        self.window = cfg.codec.window_frames
        self.released = cfg.codec.checkpoint is None
        self.rvq, self.rf, self.norm = rvq, rf, norm

    @classmethod
    def load(cls, cfg, device, *, encoder: bool = True, decoder: bool = True) -> "Codec":
        """The flags only skip OUR rectified-flow weights (decoder=False); the released MuCodec always loads all three checkpoints."""
        K, V = cfg.codec.num_codebooks, cfg.codec.codebook_size
        rel = Released.load(cfg, device)
        if cfg.codec.checkpoint is None:
            if (K, V) != (1, 16384):
                raise ValueError(f"the released weights are 1 x 16384 but the config asks for {K} x {V}: "
                                 "train the codec (python -m dc2t.codec.train) and set codec.checkpoint")
            codec = cls(cfg, device, rel)
        else:
            rvq = load_rvq(cfg, rel, cfg.codec.checkpoint).to(device)
            rf = norm = None
            if decoder:
                from safetensors.torch import load_file

                sd = {k.removeprefix("module."): v for k, v in load_file(str(Path(cfg.codec.checkpoint) / "model.safetensors")).items()}
                rf = RFTransformer(RFConfig.from_cfg(cfg.rf))
                rf.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("ema.")})  # decode with the EMA weights
                norm = LatentNorm()
                norm.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("norm.")})
                rf, norm = rf.to(device).eval(), norm.to(device)
            del rel.mu.model.cfm_wrapper  # frees ~3 GB: the released flow transformer is unused on this path
            codec = cls(cfg, device, rel, rvq=rvq, rf=rf, norm=norm)
        tag_dir = Path(cfg.paths.data_root) / "codes" / cfg.codec.tag
        if (tag_dir / "meta.json").exists():
            check_tag_dir(tag_dir, codec.fingerprint())  # a decoder with a different RVQ refuses to run
        return codec

    def fingerprint(self) -> str:
        rvq = self.rel.rvq if self.released else self.rvq
        return rvq_fingerprint(rvq, self.num_codebooks, self.codebook_size, self.cfg.codec.muq_layer,
                               RELEASED_WINDOW if self.released else self.window)

    # ------------------------------------------------------------ encode
    def encode(self, wav: Tensor) -> Tensor:
        """Float[B, N] mono at sample_rate, N a multiple of sample_rate // 25 -> Long[B, K, N * 25 // sample_rate] on the CPU."""
        if wav.ndim != 2 or wav.shape[1] == 0 or wav.shape[1] % self.spf:
            raise ValueError(f"wav must be [B, N] with N a non-zero multiple of {self.spf} samples, got {tuple(wav.shape)}")
        if not torch.isfinite(wav).all():
            raise ValueError("wav contains NaN or inf")
        T = wav.shape[1] // self.spf
        x48 = to_rate(wav, self.sample_rate, VAE_SR)  # [B, T * 1920]
        one = self._encode_released if self.released else self._encode_plan
        return torch.stack([one(x48[b], T) for b in range(wav.shape[0])])

    def _encode_released(self, x: Tensor, T: int) -> Tensor:
        with quiet():  # sound2code repeats short audio, encodes 40.96 s + 480-sample windows, returns T or T + 1 frames
            c = self.rel.sound2code(torch.stack([x, x]).to(self.device))  # mono -> "stereo"
        assert c.shape[-1] >= T, (c.shape, T)
        return c[0, :, :T].cpu()

    def _encode_plan(self, x: Tensor, T: int) -> Tensor:
        spf48, W, out = VAE_SR // self.frame_rate, self.window, []
        for start, stop, keep in tile_windows(T, W):
            seg = x[start * spf48 : stop * spf48]
            seg = F.pad(seg, (0, W * spf48 - seg.numel()))  # a clip shorter than one window is zero-padded, as in training
            with quiet(), torch.autocast(self.device.type, enabled=False):
                _, c, *_ = self.rvq(self.rel.features(seg[None].to(self.device)).float())  # [1, K, W]
            out.append(c[0, :, keep - start : stop - start])
        return torch.cat(out, -1).cpu()

    # ------------------------------------------------------------ decode
    def decode(self, codes: Tensor, *, steps: int | None = None, cfg_scale: float | None = None, seed: int | None = None) -> Tensor:
        """Long[B, K, T] (any T >= 1) -> Float[B, T * sample_rate // 25] in [-1, 1] on the CPU."""
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks or codes.shape[2] < 1:
            raise ValueError(f"codes must be [B, {self.num_codebooks}, T >= 1], got {tuple(codes.shape)}")
        if codes.min() < 0 or codes.max() >= self.codebook_size:
            raise ValueError(f"codes must lie in [0, {self.codebook_size})")
        one = self._decode_released if self.released else self._decode_plan
        return torch.stack([one(codes[b : b + 1].long(), codes.shape[2], steps, cfg_scale, seed) for b in range(codes.shape[0])])

    def _finish(self, wav48: Tensor, T: int) -> Tensor:  # mono 48 kHz -> exactly T * spf samples in [-1, 1]
        n48 = T * (VAE_SR // self.frame_rate)
        wav48 = wav48[:n48] if wav48.numel() >= n48 else F.pad(wav48, (0, n48 - wav48.numel()))  # code2sound's int() can be 1 short
        wav = to_rate(wav48, VAE_SR, self.sample_rate)
        assert wav.numel() == T * self.spf, (wav.numel(), T, self.spf)
        return wav.clamp(-1, 1)

    def _decode_released(self, c: Tensor, T: int, steps, cfg_scale, seed) -> Tensor:
        if cfg_scale not in (None, 1.5):
            raise NotImplementedError("the released decoder hard-codes guidance 1.5 (generate.py:171, 178)")
        if T >= RELEASED_WINDOW and (T - 128) % 768 == 0:  # code2sound crashes on these lengths (T = 3200 = 128 s is one)
            c = torch.cat([c, c[..., -1:]], -1)  # one extra frame sidesteps its padding branch; _finish trims it again
        devs = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs), quiet():  # code2sound draws from the global RNG
            if seed is not None:
                torch.manual_seed(seed)
            stereo = self.rel.code2sound(c.to(self.device), steps or 20)  # [2, N] CPU; 20 = the release's default
        return self._finish(stereo.float().mean(0), T)

    @torch.no_grad()
    def _decode_plan(self, c: Tensor, T: int, steps, cfg_scale, seed) -> Tensor:
        rf = self.cfg.rf
        zq = self.rvq.from_codes(c.to(self.device))[0]  # [1, 1024, T] recovered embedding
        cond = pair_frames(zq.transpose(1, 2).float())  # [1, L, 2048]: code frames 2i, 2i+1 -> latent frame i
        gen = None if seed is None else torch.Generator().manual_seed(seed)
        cuda = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=cuda):
            x0 = sample(self.rf, cond, steps=steps or rf.decode_steps, win=self.window // FRAMES_PER_LATENT,
                        cfg_scale=rf.decode_cfg if cfg_scale is None else cfg_scale, generator=gen)
        return self._finish(self._decode_latents(self.norm.denormalize(x0)), T)

    def _decode_latents(self, lat: Tensor) -> Tensor:
        """Float[1, 16, L, 32] -> Float[3840 * L]: Mel-VAE + HiFi-GAN in windows of cfg.rf.vae_window latent frames
        (the VAE's training length), blended with the same sine weights as the sampler so no window edge is audible."""
        L = lat.shape[2]
        w = min(self.cfg.rf.vae_window, L)
        out, den, win = torch.zeros(L * VAE_FRAME), torch.zeros(L * VAE_FRAME), sine_window(w * VAE_FRAME)
        for s in plan_windows(L, w, self.cfg.rf.vae_hop):
            y = self.rel.decode_window(lat[:, :, s : s + w])[0].float().cpu()
            out[s * VAE_FRAME : (s + w) * VAE_FRAME] += y * win
            den[s * VAE_FRAME : (s + w) * VAE_FRAME] += win
        return out / den
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_codec_codec.py` → `test_codec_codec: 13 tests passed`
- [ ] **Step 5: commit** — `git add dc2t/codec/codec.py tests/test_codec_codec.py && git commit -m "feat: Codec over the released MuCodec stack and our RF decoder"`

**Not executed:** `Released` itself, against the real checkpoints. Its first real run is milestone M0 (§4).

### Task 3: training the RVQ and the RF Transformer

**Files:** Create `dc2t/codec/train.py` · Test `tests/test_codec_train.py` (first three tests)

**Interfaces:**
- Consumes: `fit(model, train_loader, val_loader, cfg, run_dir)` and `load_config` (chapter 04); `Released`; `cfg.rf.cond_drop`, `ema_decay`, `w_commit`, `w_codebook`, `norm_batches`; `cfg.train.batch_size`, `num_workers`; `manifest/{train,val}.jsonl` and `audio/*.flac` (contract §7).
- Produces: `CodecTrainer(cfg, rvq, released)` — `forward(batch) -> {"loss", "rf", "commit", "codebook", "ppl_k0", …}` and `on_step_end(step)` (the engine's contract); `fit_latent_norm`; `WindowDataset`; `python -m dc2t.codec.train --config C [--fit-norm] [--name N] [--override …]`. Run directory `runs/codec/<name>/` with `latent_norm.pt` and the engine's `step_*/`.

What is trained and what is not:

| Module | State | In the checkpoint |
|---|---|---|
| MuEncoder, Mel-VAE, HiFi-GAN | frozen, held outside the module tree | no |
| `rvq` | trained (commitment 0.25 + codebook 1.0 + the RF loss through the straight-through estimator) | yes, prefix `rvq.` |
| `rf` | trained | yes, prefix `rf.` |
| `ema` | moving average of `rf`, updated in `on_step_end` | yes, prefix `ema.` — **decoding uses this copy** |
| `norm` | buffers measured by `--fit-norm` | yes, prefix `norm.` |

- [ ] **Step 1: write the failing test** — a small frozen stand-in whose latents have mean 3 and standard deviation 2 (so the statistics step is observable), and a tiny trainable RVQ with the released 6-tuple.

<details>
<summary><code>tests/test_codec_train.py</code> — 150 lines (click to expand)</summary>

```python
"""CPU tests for the codec training module and the tokenisation job. The frozen MuCodec stack and the RVQ are small stand-ins
with the released shapes and return signatures. Run: pytest -q tests/test_codec_train.py   or   python tests/test_codec_train.py"""
import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dc2t.codec.tokenize import tokenize
from dc2t.codec.train import CodecTrainer, fit_latent_norm

ns = types.SimpleNamespace
K, V, W = 2, 8, 8  # codebooks, codebook size, code frames per window


def make_cfg(root=".", **rf):
    return ns(paths=ns(data_root=str(root)), audio=ns(sample_rate=32000, frame_rate=25),
              codec=ns(tag="k2v8", num_codebooks=K, codebook_size=V, window_frames=W),
              rf=ns(**{**dict(width=32, depth=1, heads=2, cond_drop=0.5, ema_decay=0.9, w_commit=0.25, w_codebook=1.0), **rf}))


class Frozen:  # stands in for codec.Released: features Float[B, 1024, T], latents Float[B, 16, T / 2, 32] with mean 3 and std 2
    device = torch.device("cpu")

    def features(self, x48):
        B, N = x48.shape
        return x48.view(B, N // 1920, 1920).mean(-1)[:, None, :] * torch.arange(1, 1025).view(1, -1, 1)

    def latents(self, x48):
        B, N = x48.shape
        return 3.0 + 2.0 * x48.view(B, N // 3840, 3840)[:, :, :512].reshape(B, N // 3840, 16, 32).permute(0, 2, 1, 3)


class TinyRVQ(nn.Module):  # a trainable residual quantiser with the 6-tuple that MuCodec's ResidualVectorQuantize.forward returns
    def __init__(self):
        super().__init__()
        self.inp, self.out, self.books = nn.Linear(1024, 4), nn.Linear(4, 1024), nn.Parameter(torch.randn(K, V, 4))

    def forward(self, z):  # z Float[B, 1024, T]
        e = self.inp(z.transpose(1, 2))
        res, q, codes, commit, codebook = e, 0, [], 0, 0
        for k in range(K):
            idx = torch.cdist(res, self.books[k].expand(len(z), -1, -1)).argmin(-1)
            qk = self.books[k][idx]
            commit, codebook = commit + F.mse_loss(res, qk.detach()), codebook + F.mse_loss(qk, res.detach())
            q, res, codes = q + res + (qk - res).detach(), res - qk.detach(), codes + [idx]
        return self.out(q).transpose(1, 2), torch.stack(codes, 1), None, commit, codebook, None


def batch(B=4, seed=0):
    return {"wav48": torch.randn(B, W * 1920, generator=torch.Generator().manual_seed(seed))}


def test_forward_trains_rvq_and_rf_and_leaves_the_frozen_stack_out():
    torch.manual_seed(0)
    m = CodecTrainer(make_cfg(), TinyRVQ(), Frozen())
    fit_latent_norm(m.norm, Frozen(), [batch(8, s) for s in range(20)], batches=20)
    assert abs(m.norm.mean.mean() - 3) < 0.05 and abs(m.norm.std.mean() - 2) < 0.05          # the statistics of Frozen.latents
    out = m(batch())
    assert out["loss"].ndim == 0 and out["loss"].requires_grad and set(out) == {"loss", "rf", "commit", "codebook", "ppl_k0", "ppl_k1"}
    assert abs(out["rf"] - 2.0) < 0.2            # zero-initialised output, unit-variance target: E|eps - x0|^2 = 2 exactly when the latents are standardised
    assert all(1.0 <= out[f"ppl_k{k}"] <= V for k in range(K))
    out["loss"].backward()
    grads = {n.split(".")[0] for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads == {"rvq", "rf"}                # the RF loss reaches the RVQ through the condition; the EMA copy gets no gradient
    assert {k.split(".")[0] for k in m.state_dict()} == {"rvq", "rf", "ema", "norm"}        # the prefixes Codec.load reads; no frozen weights
    assert all(not p.requires_grad for p in m.ema.parameters())
    m.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        a = m(batch())["loss"]
        torch.manual_seed(1)
        assert torch.equal(a, m(batch())["loss"])                                              # validation is repeatable under a seed


def test_ema_follows_the_trained_weights():
    m = CodecTrainer(make_cfg(), TinyRVQ(), Frozen())
    with torch.no_grad():
        m.rf.out.weight.fill_(1.0)
    m.on_step_end(1)
    assert torch.allclose(m.ema.out.weight, torch.full_like(m.ema.out.weight, 0.1))            # (1 - 0.9) of the way
    m.on_step_end(2)
    assert torch.allclose(m.ema.out.weight, torch.full_like(m.ema.out.weight, 0.19))


def test_a_few_steps_reduce_the_loss():
    torch.manual_seed(0)
    m = CodecTrainer(make_cfg(width=64, cond_drop=0.0), TinyRVQ(), Frozen())
    fit_latent_norm(m.norm, Frozen(), [batch(8, s) for s in range(4)], batches=4)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-3)
    hist = []
    for step in range(60):
        loss = m(batch(8, step))["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        m.on_step_end(step + 1)
        hist.append(loss.item())
    assert sum(hist[-10:]) / 10 < 0.9 * sum(hist[:10]) / 10, (hist[:3], hist[-3:])


class FakeCodec:
    num_codebooks, codebook_size, frame_rate, sample_rate = K, V, 25, 32000

    def __init__(self, fp="aaaa", bad=False):
        self.fp, self.bad, self.calls = fp, bad, 0

    def fingerprint(self):
        return self.fp

    def encode(self, wav):  # Float[1, N] -> Long[1, K, N / 1280]
        self.calls += 1
        T = wav.shape[1] // 1280 - self.bad
        return (torch.arange(K * T).view(1, K, T) + int(wav.abs().sum() > 0)) % V


def test_tokenize_writes_contract_files_is_resumable_and_refuses_another_rvq():
    rows = [{"clip_id": f"c{i}", "duration": 30 + i, "audio": f"audio/c{i}.flac"} for i in range(5)]
    read = lambda row: torch.zeros(row["duration"] * 32000)
    with tempfile.TemporaryDirectory() as d:
        cfg, codec = make_cfg(d), FakeCodec()
        assert tokenize(cfg, codec, rows, read, (0, 2)) == (3, 0) and tokenize(cfg, codec, rows, read, (1, 2)) == (2, 0)   # two shards
        out = Path(d) / "codes" / "k2v8"
        assert sorted(p.name for p in out.glob("*.npy")) == [f"c{i}.npy" for i in range(5)] and not list(out.glob(".*.tmp"))
        for row in rows:
            c = np.load(out / f"{row['clip_id']}.npy")
            assert c.dtype == np.int16 and c.shape == (K, 25 * row["duration"]) and c.min() >= 0 and c.max() < V
        assert json.loads((out / "meta.json").read_text())["fingerprint"] == "aaaa"
        assert tokenize(cfg, codec, rows, read) == (0, 5) and codec.calls == 5                       # a re-run encodes nothing
        for bad_call in (lambda: tokenize(cfg, FakeCodec("bbbb"), rows, read),                      # tokens of another RVQ live here
                         lambda: tokenize(make_cfg(Path(d) / "x"), FakeCodec(bad=True), rows, read),  # the codec returned T - 1 frames
                         lambda: tokenize(make_cfg(Path(d) / "y"), codec, rows, lambda row: torch.zeros(7))):   # the audio has the wrong length
            try:
                bad_call()
                raise AssertionError("expected a refusal")
            except (RuntimeError, ValueError):
                pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_codec_train: {len(tests)} tests passed")
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_codec_train.py` → `ModuleNotFoundError: No module named 'dc2t.codec.train'`
- [ ] **Step 3: implement**

**`dc2t/codec/train.py`** — 133 lines

```python
"""python -m dc2t.codec.train --config C [--override a.b=value ...] [--name N] [--fit-norm]

Trains the K x V RVQ and the Rectified Flow Transformer together, on frozen MuEncoder features and frozen Mel-VAE latents
(plan 3.4, 3.6). Run it once with --fit-norm (one process), then without it (any number of processes, through fit())."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dc2t.codec.codec import VAE_SR, Released, to_rate
from dc2t.codec.rf import LatentNorm, RFConfig, RFTransformer, pair_frames, rf_loss, sample_t


class CodecTrainer(nn.Module):
    """forward(batch) -> loss dict (contract 9). Its state-dict prefixes are the ones Codec.load reads: rvq., rf., ema., norm.
    The frozen MuEncoder / Mel-VAE stack sits in a plain list, so it is not saved, not optimised and not wrapped by DDP."""

    def __init__(self, cfg, rvq: nn.Module, released):
        super().__init__()
        self.rvq = rvq                                              # trainable: MuCodec's ResidualVectorQuantize with K and V from config
        self.rf = RFTransformer(RFConfig.from_cfg(cfg.rf))          # trainable
        self.ema = copy.deepcopy(self.rf).requires_grad_(False)     # what Codec.decode uses
        self.norm = LatentNorm()                                    # buffers, filled by fit_latent_norm before training
        self._frozen = [released]
        r = cfg.rf
        self.V, self.cond_drop, self.ema_decay, self.w_commit, self.w_codebook = cfg.codec.codebook_size, r.cond_drop, r.ema_decay, r.w_commit, r.w_codebook

    def forward(self, batch: dict) -> dict:
        """batch["wav48"]: Float[B, W * 1920], mono 48 kHz windows of W = codec.window_frames code frames."""
        released, wav = self._frozen[0], batch["wav48"]
        with torch.no_grad():
            feats = released.features(wav).float()                      # Float[B, 1024, W]      frozen MuEncoder
            x0 = self.norm.normalize(released.latents(wav).float())     # Float[B, 16, W/2, 32]  frozen Mel-VAE, standardised
        zq, codes, _, commit, codebook, *_ = self.rvq(feats)            # zq Float[B, 1024, W] (straight-through), codes Long[B, K, W]
        cond = pair_frames(zq.transpose(1, 2))                          # Float[B, W/2, 2048]: code frames 2i, 2i+1 -> latent frame i
        B = x0.shape[0]
        drop = torch.rand(B, device=x0.device) < self.cond_drop if self.training else None   # teach the null condition (guidance)
        rf = rf_loss(self.rf, x0, cond, sample_t(B).to(x0.device), torch.randn_like(x0), drop)
        out = {"loss": rf + self.w_commit * commit + self.w_codebook * codebook,
               "rf": rf.detach(), "commit": commit.detach(), "codebook": codebook.detach()}
        for k in range(codes.shape[1]):                                 # per-level perplexity inside this batch: a collapsing codebook shows here first
            p = torch.bincount(codes[:, k].reshape(-1), minlength=self.V).float()
            p = p / p.sum()
            out[f"ppl_k{k}"] = torch.exp(-(p * p.clamp_min(1e-12).log()).sum())
        return out

    @torch.no_grad()
    def on_step_end(self, step: int) -> None:                           # fit() calls this after every optimiser step
        for e, p in zip(self.ema.parameters(), self.rf.parameters()):
            e.lerp_(p, 1.0 - self.ema_decay)


@torch.no_grad()
def fit_latent_norm(norm: LatentNorm, released, loader, batches: int) -> None:
    """Per-(channel, mel-bin) mean and std of the Mel-VAE latents over `batches` batches (MuCodec's Feature2DProcessor)."""
    s, s2, n = 0.0, 0.0, 0
    for _, batch in zip(range(batches), loader):
        x = released.latents(batch["wav48"].to(released.device)).float().cpu()      # Float[B, 16, L, 32]
        s, s2, n = s + x.mean((0, 2)), s2 + x.pow(2).mean((0, 2)), n + 1           # each Float[16, 32]
    if n == 0:
        raise ValueError("the loader gave no batches")
    mean = s / n
    norm.mean.copy_(mean[:, None])
    norm.std.copy_((s2 / n - mean ** 2).clamp_min(1e-8).sqrt()[:, None])


class WindowDataset(Dataset):
    """One window of codec.window_frames code frames per clip, as mono 48 kHz audio: {"wav48": Float[W * 1920]}.
    The window starts on the code-frame grid, at a random frame (fixed=True: frame 0, for validation and statistics).
    A clip shorter than the window is zero-padded at the end; Codec.encode pads the same way."""

    def __init__(self, cfg, split: str, fixed: bool = False):
        self.root = Path(cfg.paths.data_root)
        text = (self.root / "manifest" / f"{split}.jsonl").read_text(encoding="utf-8")
        self.rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.sr, self.fr, self.W, self.fixed = cfg.audio.sample_rate, cfg.audio.frame_rate, cfg.codec.window_frames, fixed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        import soundfile as sf
        row, spf = self.rows[i], self.sr // self.fr
        T = row["duration"] * self.fr
        start = 0 if self.fixed or T <= self.W else int(torch.randint(0, T - self.W + 1, ()))
        y, sr = sf.read(str(self.root / row["audio"]), start=start * spf, frames=min(self.W, T) * spf, dtype="float32")
        if sr != self.sr or y.ndim != 1:
            raise ValueError(f"{row['audio']}: expected mono {self.sr} Hz, got {sr} Hz with shape {y.shape}")
        y = F.pad(torch.from_numpy(y), (0, self.W * spf - len(y)))
        return {"wav48": to_rate(y, self.sr, VAE_SR)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--name", help="run directory name; default codec.tag")
    ap.add_argument("--fit-norm", action="store_true", help="compute the latent statistics, write latent_norm.pt and exit")
    args = ap.parse_args(argv)
    from dc2t.config import load_config      # chapter 04
    from dc2t.engine import fit              # chapter 04
    cfg = load_config(args.config, args.override)
    t = cfg.train
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    released = Released.load(cfg, device)
    del released.mu.model.cfm_wrapper           # the released flow transformer is not used on this path
    model = CodecTrainer(cfg, released.new_rvq(cfg.codec.num_codebooks, cfg.codec.codebook_size), released)
    run_dir = Path(cfg.paths.runs) / "codec" / (args.name or cfg.codec.tag)
    stats = run_dir / "latent_norm.pt"
    loader = lambda split, fixed: DataLoader(WindowDataset(cfg, split, fixed), batch_size=t.batch_size, shuffle=not fixed,
                                             num_workers=t.num_workers, drop_last=not fixed)
    if args.fit_norm:
        run_dir.mkdir(parents=True, exist_ok=True)
        fit_latent_norm(model.norm, released, loader("train", True), cfg.rf.norm_batches)
        torch.save(model.norm.state_dict(), stats)
        print(f"wrote {stats}: mean {model.norm.mean.mean():.3f}, std {model.norm.std.mean():.3f}")
        return
    if not stats.is_file():
        raise SystemExit(f"{stats} is missing: run this command once with --fit-norm before training")
    model.norm.load_state_dict(torch.load(stats))
    fit(model, loader("train", False), loader("val", True), cfg, str(run_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_codec_train.py` → `test_codec_train: 4 tests passed` (after Task 4)
- [ ] **Step 5: commit** — `git add dc2t/codec/train.py tests/test_codec_train.py && git commit -m "feat: joint RVQ and rectified flow training module"`

**Not executed:** `WindowDataset` (needs `soundfile` and real clips) and `main` (needs MuCodec and a GPU). Both were compiled; the module's `forward`, the statistics and the EMA ran on CPU.
**Unverified:** on several GPUs each process picks its device from the `LOCAL_RANK` environment variable, which `accelerate launch` is expected to set. Confirm on the first multi-GPU run that the processes do not all load MuCodec onto GPU 0 (`nvidia-smi`).

### Task 4: tokenising the dataset

**Files:** Create `dc2t/codec/tokenize.py` · Test `tests/test_codec_train.py` (last test)

**Interfaces:**
- Consumes: `Codec.encode`, `Codec.fingerprint`, `claim_tag_dir`; `manifest/all.jsonl`, `audio/*.flac`.
- Produces: `tokenize(cfg, codec, rows, read_audio, shard=(0, 1)) -> (written, skipped)`; `python -m dc2t.codec.tokenize --config C [--shard i/n]`; files `codes/<tag>/<clip_id>.npy` (`int16`, `[K, 25 × duration]`) and `codes/<tag>/meta.json`.

- [ ] **Step 1:** the test is the last function of `tests/test_codec_train.py` above.
- [ ] **Step 2: run it, expect failure** — `ModuleNotFoundError: No module named 'dc2t.codec.tokenize'`
- [ ] **Step 3: implement**

**`dc2t/codec/tokenize.py`** — 69 lines

```python
"""python -m dc2t.codec.tokenize --config C [--override a.b=value ...] [--shard i/n]

Writes data/codes/<codec.tag>/<clip_id>.npy (contract 7.3) for every clip of manifest/all.jsonl.
Safe to re-run (finished clips are skipped) and to run as n shards at once, one per GPU: --shard 0/8 ... --shard 7/8."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dc2t.codec.codec import claim_tag_dir


def tokenize(cfg, codec, rows, read_audio, shard: tuple[int, int] = (0, 1)) -> tuple[int, int]:
    """rows: manifest rows. read_audio(row) -> Float[duration * sample_rate] mono. Returns (written, skipped)."""
    K, V, fr, sr = codec.num_codebooks, codec.codebook_size, codec.frame_rate, codec.sample_rate
    if V > 2 ** 15:
        raise ValueError("codes are stored as int16 (contract 7.3): codebook_size must not exceed 32768")
    d = Path(cfg.paths.data_root) / "codes" / cfg.codec.tag
    claim_tag_dir(d, {"fingerprint": codec.fingerprint(), "tag": cfg.codec.tag, "K": K, "V": V})   # refuses another RVQ's directory
    written = skipped = 0
    for i, row in enumerate(rows):
        if i % shard[1] != shard[0]:
            continue
        out = d / f"{row['clip_id']}.npy"
        if out.exists():
            skipped += 1
            continue
        wav = read_audio(row)
        if wav.ndim != 1 or wav.numel() != row["duration"] * sr:
            raise ValueError(f"{row['clip_id']}: {tuple(wav.shape)} samples, expected ({row['duration'] * sr},)")
        codes = codec.encode(wav[None])[0]                                       # Long[K, T]
        if tuple(codes.shape) != (K, row["duration"] * fr) or int(codes.min()) < 0 or int(codes.max()) >= V:
            raise ValueError(f"{row['clip_id']}: the codec returned {tuple(codes.shape)}, expected {(K, row['duration'] * fr)} with values in [0, {V})")
        tmp = d / f".{row['clip_id']}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:                                               # a crash never leaves a half-written .npy
            np.save(f, codes.cpu().numpy().astype(np.int16))
        os.replace(tmp, out)
        written += 1
    return written, skipped


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip starting at i")
    args = ap.parse_args(argv)
    import soundfile as sf
    from dc2t.codec.codec import Codec
    from dc2t.config import load_config      # chapter 04
    cfg = load_config(args.config, args.override)
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit("--shard must be i/n with 0 <= i < n")
    root = Path(cfg.paths.data_root)
    rows = [json.loads(line) for line in (root / "manifest" / "all.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    codec = Codec.load(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"), decoder=False)
    read = lambda row: torch.from_numpy(sf.read(str(root / row["audio"]), dtype="float32")[0])
    written, skipped = tokenize(cfg, codec, rows, read, (i, n))
    print(f"shard {i}/{n}: wrote {written}, skipped {skipped} existing")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_codec_train.py` → `test_codec_train: 4 tests passed`
- [ ] **Step 5: commit** — `git add dc2t/codec/tokenize.py && git commit -m "feat: resumable, shardable dataset tokenisation"`

## 4. Running it

All commands run in the codec environment (chapter 04), from the repository root.

```bash
# M0 - does the released codec handle Don ca tai tu?  Reconstruct three clips and listen.   (not executed in this guide)
python - <<'EOF'
import soundfile as sf, torch
from dc2t.config import load_config
from dc2t.codec.codec import Codec
cfg = load_config("configs/bootstrap_k1.yaml")
codec = Codec.load(cfg, "cuda")
for name in ["clip_a", "clip_b", "clip_c"]:                      # 32 kHz mono FLAC files of whole seconds
    wav, sr = sf.read(f"data/audio/{name}.flac", dtype="float32")
    codes = codec.encode(torch.from_numpy(wav)[None])            # Long[1, 1, 25 * seconds]
    sf.write(f"recon_{name}.wav", codec.decode(codes, seed=0)[0].numpy(), sr)
EOF

# M2 - bootstrap: tokenise with the released RVQ (one shard per GPU)
python -m dc2t.codec.tokenize --config configs/bootstrap_k1.yaml --shard 0/1

# M3 - plan path: statistics once, then joint RVQ + RF training
python -m dc2t.codec.train --config configs/plan_k4.yaml --fit-norm
accelerate launch -m dc2t.codec.train --config configs/plan_k4.yaml

# M4 - freeze: point the config at the final checkpoint, then tokenise with OUR RVQ
python -m dc2t.codec.tokenize --config configs/plan_k4.yaml \
    --override codec.checkpoint=runs/codec/k4v10000/step_0020000 --shard 0/8      # ... --shard 7/8
```

**What to watch in `runs/codec/<name>/log.jsonl`:**

| Signal | Healthy | Unhealthy |
|---|---|---|
| `rf` at step 1 | about 2.0 | far from 2: `--fit-norm` was skipped or measured on other data |
| `rf` over time | falls well below 1.0 (1.0 = the model ignores the condition) | stuck near 1.0: the condition is not reaching the network |
| `ppl_k0 … ppl_k3` | each a large fraction of the tokens in a batch, slowly rising | one level near 1: that codebook has collapsed |
| `val_rf` | follows `rf` | rises while `rf` falls: overfitting on a small dataset |

**Validating the codec (the acceptance test of M3).** On validation clips, reconstruct with (a) the released decoder at K = 1 and (b) our decoder, then compare with the evaluation tools of chapter 04 (FD-openl3 of reconstructions against the originals) and by listening.

**The steps-versus-quality sweep (tests R4 / the plan's "few-step" claim).** One table, same 50 validation clips, same seeds:

| Decoder | Steps | Guidance | FD-openl3 vs originals | Seconds per minute of audio |
|---|---|---|---|---|
| released flow matching (K = 1) | 20 (its default), 50 | 1.5 (fixed) | | |
| ours (K = 4) | 1, 2, 4, 8, 16, 32 | 1.0 and `rf.decode_cfg` | | |

Set `rf.decode_steps` to the smallest step count whose row is within noise of the 32-step row. For a like-for-like comparison of the two *objectives*, also train our RF on the frozen released 1 × 16,384 RVQ and add that row.

**Cost.** Nothing here could be measured on the authoring machine. The MuCodec paper reports usable reconstructions after 20,000 steps and better ones after 200,000, on 8 × A100 40 GB with batch size 4; take that as the order of magnitude for the full run (`train.max_steps` defaults to 20,000). For the pilot use `configs/pilot.yaml` (RF 28.8 M parameters). The first 100 steps of any run give `seconds_per_step` in `log.jsonl`; multiply by `train.max_steps` before committing GPU time.

## 5. What will go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `MuCodec checkpoints missing` | the three files are not where MuCodec expects them | chapter 04, Task 0 |
| `mucodec.pt did not load (normalisation statistics are empty)` | wrong or truncated checkpoint; MuCodec loads it with `strict=False` | re-download; compare the file size |
| `ModuleNotFoundError: transformers.deepspeed` when MuCodec loads | the environment has a recent `transformers`; MuCodec's encoder imports a module removed from it (**Verified**: absent in 4.57.1) | use the codec environment with MuCodec's pins (chapter 04) |
| `the released weights are 1 x 16384 but the config asks for 4 x 10000` | `plan_k4` config without `codec.checkpoint` | train the codec first, or use `bootstrap_k1` |
| `…holds tokens from tokeniser X but this codec is Y` | the RVQ changed after the dataset was tokenised | new `codec.tag`, re-tokenise, retrain the LM — or restore the old checkpoint |
| `rf` loss starts near 1.3 or 5 instead of 2 | latent statistics missing or stale | re-run `--fit-norm` on this dataset |
| A codebook's perplexity collapses | too few distinct frames per batch, or the commitment weight too low | larger batch; the RVQ module already re-seeds stale codes (`stale_tolerance=200`) |
| Clicks every 10 s in decoded audio | the VAE windows are not cross-faded, or `rf.vae_hop` ≥ `rf.vae_window` | keep `vae_hop < vae_window` |
| Decoded audio is quieter or louder than the input | MuCodec rescales peaks above 0.8 before encoding | standardise clips to a 0.8 peak (chapter 01 does) |
| All ranks load MuCodec on GPU 0 | `LOCAL_RANK` not set by the launcher | see the Unverified note in Task 3 |

## 6. Open questions for the research team

1. **How good is the released codec on these instruments?** It was trained on the Million Song Dataset. M0 answers this by ear in an hour, before anything else is built. If the released MuEncoder itself misses the timbre of the monochord or the zither, no RVQ or decoder trained on top of it will recover it.
2. **Rectified flow versus flow matching.** As built, the difference from MuCodec's decoder is the timestep density and the network, not the straightness of the path (§1.4). Should the paper claim few-step sampling, the sweep in §4 is the evidence; reflow is the technique to add if it is needed.
3. **K = 4 × 10,000 needs the full dataset.** MuCodec trained its quantiser on a large corpus; a 20-hour pilot will not fill 40,000 codes. Recommended: validate the whole pipeline with the released K = 1 codec, and train the K = 4 codec only once most of the data exists.
4. **32 kHz or 48 kHz.** The Mel-VAE and vocoder are 48 kHz; with 32 kHz data the top 8 kHz of the output band is empty (contract D2). Changing `audio.sample_rate` to 48000 and rebuilding the dataset gives full-band output with no code change.
5. **Licence.** `mucodec.pt` and `muq.pt` are CC-BY-NC 4.0. Anything built on them inherits the non-commercial restriction.
