"""CPU tests for dcttgen.codec.rf (no third_party, no checkpoints).
Run:  pytest -q tests/test_codec_rf.py     or     python tests/test_codec_rf.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from dcttgen.codec.rf import (LAT_C, LAT_F, LatentNorm, RFConfig, RFTransformer, blended_velocity, noisy, pair_frames,
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
