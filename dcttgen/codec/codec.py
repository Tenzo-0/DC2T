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

from dcttgen.codec.rf import (FRAMES_PER_LATENT, LatentNorm, RFConfig, RFTransformer, pair_frames, plan_windows, sample,
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
                                 "train the codec (python -m dcttgen.codec.train) and set codec.checkpoint")
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
