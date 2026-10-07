"""End-to-end smoke tests with fake components: a prompt in, a waveform of the right length out; the evaluation runner's control flow.
Run: pytest -q tests/test_infer.py   or   python tests/test_infer.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import dc2t.infer as infer
import dc2t.lm.generate as lm_generate
from dc2t.config import ROOT, load_config
from dc2t.eval.run import centre, eval_plan, make_audio, make_codes, score, select
from dc2t.plan import Plan, plan_sections

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
