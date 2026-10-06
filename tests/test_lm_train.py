import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
from unittest import mock

import torch
from lm_testkit import get_vocab, make_cfg, random_rows_model, run_all, write_dataset
from safetensors.torch import load_file, save_file

from dcttgen.lm import train as T


def fake_chapter_04(root, runs, calls):
    """Stand-ins for dcttgen.config.load_config and dcttgen.engine.fit, which chapter 04 owns."""
    def load_config(path, overrides):
        cfg = make_cfg(2, 16, 1, data_root=str(root))
        cfg.paths.runs = str(runs)
        for o in overrides:                                                    # "a.b=value"
            key, value = o.split("=", 1)
            *parents, leaf = key.split(".")
            obj = cfg
            for p in parents:
                obj = getattr(obj, p)
            try:
                value = json.loads(value)
            except ValueError:
                pass
            setattr(obj, leaf, value)
        return cfg

    def fit(model, train, val, cfg, run_dir):
        calls.append(dict(train=len(train.dataset), val=len(val.dataset), start=model.head_weight.detach().clone(), losses=[]))
        opt, _ = torch.optim.AdamW(model.parameters(), lr=1e-2), model.train()
        for epoch in range(3):
            train.batch_sampler.set_epoch(epoch)
            for batch in train:
                opt.zero_grad()
                loss = model(batch)["loss"]
                loss.backward()
                opt.step()
                calls[-1]["losses"].append(loss.item())
        step = pathlib.Path(run_dir) / "step_0000003"
        step.mkdir(parents=True)
        save_file({k: t.contiguous() for k, t in model.state_dict().items() if k != "lm.lm_head.weight"}, str(step / "model.safetensors"))

    return {"dcttgen.config": types.SimpleNamespace(load_config=load_config), "dcttgen.engine": types.SimpleNamespace(fit=fit)}


def test_main_runs_both_phases_and_the_weights_carry_over():
    with tempfile.TemporaryDirectory() as d:
        root, runs, calls = pathlib.Path(d) / "data", pathlib.Path(d) / "runs", []
        write_dataset(root, caption_none=(1,))
        with mock.patch.dict(sys.modules, fake_chapter_04(root, runs, calls)):
            T.main(["--config", "x.yaml", "--phase", "pretrain", "--name", "t"])
            pre = runs / "lm_pretrain" / "t"
            assert (pre / "vocab.json").is_file() and (pre / "tokenizer" / "tokenizer.json").is_file() and (pre / "step_0000003" / "model.safetensors").is_file()
            T.main(["--config", "x.yaml", "--phase", "finetune", "--name", "t", "--override", f"lm.checkpoint={pre / 'step_0000003'}"])
        a, b = calls
        assert (a["train"], a["val"], b["train"], b["val"]) == (6, 2, 5, 1)                  # fine-tuning leaves out the clip without a caption
        assert a["losses"][-1] < a["losses"][0]                                              # the loop trains the model it was given
        assert torch.equal(b["start"], load_file(str(pre / "step_0000003" / "model.safetensors"))["lm.model.embed_tokens.weight"])   # fine-tuning starts from it
        assert (runs / "lm_finetune" / "t" / "vocab.json").is_file()


def test_main_refuses_documents_longer_than_the_backbone_positions():
    with tempfile.TemporaryDirectory() as d:
        root, runs = pathlib.Path(d) / "data", pathlib.Path(d) / "runs"
        write_dataset(root, n=1, fr=25, clips=[("whole", 300)])                              # 300 s at 25 Hz, K = 2: 15,000+ tokens; the tiny backbone has 4,096 positions
        with mock.patch.dict(sys.modules, fake_chapter_04(root, runs, [])):
            try:
                T.main(["--config", "x.yaml", "--phase", "pretrain", "--override", "audio.frame_rate=25"])
            except SystemExit as e:
                assert "positions only 4096" in str(e) and "lm.max_seconds" in str(e)
            else:
                raise AssertionError("a document longer than the backbone's positions was accepted")


def test_probe_reports_speed_on_a_synthetic_document():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        T.probe(make_cfg(2, 16, 1), get_vocab(2, 16, 1), random_rows_model(2, 16, 1), seconds=30, steps=1)
    assert "parameters" in out.getvalue() and "tokens/s" in out.getvalue() and "peak GPU memory" in out.getvalue(), out.getvalue()


if __name__ == "__main__":
    run_all(globals())
