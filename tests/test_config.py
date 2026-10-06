"""Run: pytest -q tests/test_config.py   or   python tests/test_config.py"""
import copy
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcttgen.config import ROOT, load_config  # noqa: E402

BASE = str(ROOT / "configs" / "base.yaml")
OVERLAYS = ROOT / "configs"


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{exc.__name__} not raised")


def test_base_matches_contract_section_6():
    c = load_config(BASE)
    assert (c.audio.sample_rate, c.audio.frame_rate) == (32000, 25)
    assert (c.codec.tag, c.codec.num_codebooks, c.codec.codebook_size) == ("k4v10000", 4, 10000)
    assert (c.codec.muq_layer, c.codec.window_frames, c.codec.checkpoint) == (7, 896, None)
    assert (c.lm.backbone, c.lm.loss_on_plan, c.lm.checkpoint) == ("Qwen/Qwen2.5-0.5B", True, None)
    t = c.train
    assert (t.lr, t.lr_min, t.weight_decay, t.betas, t.eps) == (3.0e-4, 3.0e-5, 0.1, [0.9, 0.999], 1.0e-9)


def test_overlays_merge_over_base_and_compose():
    k1 = load_config(str(OVERLAYS / "bootstrap_k1.yaml"))
    assert (k1.codec.tag, k1.codec.num_codebooks, k1.codec.codebook_size) == ("k1v16384", 1, 16384)
    assert k1.codec.window_frames == 896 and k1.train.lr == 3.0e-4  # untouched keys survive the deep merge
    both = load_config(str(OVERLAYS / "bootstrap_k1.yaml") + "," + str(OVERLAYS / "pilot.yaml"))
    assert both.codec.num_codebooks == 1 and both.train.max_steps == 1000 and both.train.warmup_steps == 50


def test_overrides_are_typed_and_strict():
    c = load_config(BASE, ["train.lr=1e-4", "train.max_steps=300", "train.betas=[0.9, 0.95]", "lm.loss_on_plan=false",
                           "codec.tag=123", "lm.checkpoint=runs/lm/a/step_0000100", "eval.fd.hop=1"])
    assert c.train.lr == 1e-4 and isinstance(c.train.lr, float)  # PyYAML alone would give the string "1e-4"
    assert c.train.max_steps == 300 and isinstance(c.train.max_steps, int)
    assert c.train.betas == [0.9, 0.95] and c.lm.loss_on_plan is False
    assert c.codec.tag == "123" and c.lm.checkpoint == "runs/lm/a/step_0000100"
    assert c.eval.fd.hop == 1.0 and isinstance(c.eval.fd.hop, float)
    assert "did you mean 'train.lr'" in raises(KeyError, load_config, BASE, ["train.lrr=1"])
    assert "unknown config key 'trian'" in raises(KeyError, load_config, BASE, ["trian.lr=1"])
    assert "expected int" in raises(TypeError, load_config, BASE, ["train.max_steps=1.5"])
    assert "expected bool" in raises(TypeError, load_config, BASE, ["lm.loss_on_plan=1"])
    assert "expected float" in raises(TypeError, load_config, BASE, ["train.lr=fast"])
    assert "section.key=value" in raises(ValueError, load_config, BASE, ["train.lr"])


def test_unknown_key_in_an_overlay_file_is_an_error():
    d = Path(tempfile.mkdtemp())
    (d / "typo.yaml").write_text("train: {warmup_step: 5}\n", encoding="utf-8")
    assert "did you mean 'train.warmup_steps'" in raises(KeyError, load_config, str(d / "typo.yaml"))
    (d / "float.yaml").write_text("train: {lr: 1e-4}\n", encoding="utf-8")  # YAML 1.1 parses this as a string
    assert load_config(str(d / "float.yaml")).train.lr == 1e-4


def test_attribute_access_paths_and_copies():
    c = load_config(BASE)
    assert c.train.betas is c["train"]["betas"] and c.paths.runs == str(ROOT / "runs")
    assert all(os.path.isabs(v) for v in c.paths.values())
    elsewhere = str(Path(tempfile.gettempdir()) / "runs_elsewhere")  # absolute on every platform
    assert load_config(BASE, ["paths.runs=" + elsewhere]).paths.runs == elsewhere
    c.train.lr = 5.0  # attribute assignment writes through
    assert c["train"]["lr"] == 5.0
    for clone in (copy.deepcopy(c), pickle.loads(pickle.dumps(c))):  # DataLoader workers pickle the config
        assert clone == c and clone.train.lr == 5.0
    assert type(c.to_dict()["train"]) is dict
    assert raises(AttributeError, lambda: c.nope) == "nope"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{Path(__file__).name}: {len(tests)} passed")
