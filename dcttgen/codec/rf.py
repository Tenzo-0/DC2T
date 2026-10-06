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
