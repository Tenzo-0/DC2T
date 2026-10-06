"""CPU tests for dcttgen.codec.codec: exact lengths, windows, fingerprint. The released stack is replaced by a fake that has the
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

from dcttgen.codec.codec import VAE_FRAME, Codec, claim_tag_dir, rvq_fingerprint, tile_windows, to_rate
from dcttgen.codec.rf import LatentNorm, RFConfig, RFTransformer

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
