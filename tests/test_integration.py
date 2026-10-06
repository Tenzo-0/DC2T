"""Cross-chapter check on CPU: the real language-model package (chapter 03) driven by the real engine and config loader (chapter 04),
then reloaded from its checkpoint and used to generate codes that a codec could decode. Tiny random Qwen2, synthetic data.
Run: pytest -q tests/test_integration.py   or   python tests/test_integration.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from lm_testkit import backbone_dir, write_dataset

from dcttgen.config import ROOT, load_config
from dcttgen.lm.generate import generate_codes
from dcttgen.lm.model import load_lm
from dcttgen.lm.train import main as train_lm
from dcttgen.lm.vocab import Vocab
from dcttgen.plan import Plan, plan_sections


def test_train_checkpoint_resume_reload_generate():
    with tempfile.TemporaryDirectory() as d:
        write_dataset(Path(d) / "data", caption_none=(1,))                     # 6 clips, 1 code frame per second, K = 2, V = 16
        over = [f"paths.data_root={d}/data", f"paths.runs={d}/runs", "codec.tag=k2v16", "codec.num_codebooks=2", "codec.codebook_size=16",
                "audio.frame_rate=1", f"lm.backbone={backbone_dir()}", "lm.max_tokens=600", "lm.num_workers=0", "train.mixed_precision=no",
                "train.warmup_steps=2", "train.grad_accum=1", "train.log_every=1", "train.eval_every=3", "train.save_every=3", "train.lr=1e-3"]
        args = ["--config", str(ROOT / "configs" / "plan_k4.yaml"), "--name", "t"]
        flat = lambda o: [x for item in o for x in ("--override", item)]
        train_lm(args + ["--phase", "pretrain"] + flat(over + ["train.max_steps=6"]))
        run = Path(d) / "runs" / "lm_pretrain" / "t"
        assert sorted(p.name for p in run.glob("step_*")) == ["step_0000003", "step_0000006"]
        assert (run / "vocab.json").is_file() and (run / "tokenizer").is_dir() and (run / "config.yaml").is_file()
        log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in log if r["split"] == "train"] == [1, 2, 3, 4, 5, 6] and [r["step"] for r in log if r["split"] == "val"] == [3, 6]
        first = log[0]                                                           # per-level losses reach the log; at initialisation each is ln(V)
        assert abs(first["ce_k0"] - 2.7726) < 1e-3 and abs(first["ce_k1"] - 2.7726) < 1e-3 and abs(first["ce_text"] - 11.93) < 0.05
        assert {"val_ce_k0", "val_ce_k1", "val_loss"} <= set(log[-1])
        train_lm(args + ["--phase", "pretrain"] + flat(over + ["train.max_steps=8"]))                                 # resumes at step 6
        log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in log if r["split"] == "train"] == [1, 2, 3, 4, 5, 6, 7, 8] and (run / "step_0000008").is_dir()
        # fine-tuning starts from the pre-training checkpoint (lm.checkpoint) and sees captions and plans
        train_lm(args + ["--phase", "finetune"] + flat(over + ["train.max_steps=2", "train.save_every=2", f"lm.checkpoint={run / 'step_0000008'}"]))
        fine = Path(d) / "runs" / "lm_finetune" / "t" / "step_0000002"
        cfg = load_config(str(ROOT / "configs" / "plan_k4.yaml"), over + [f"lm.checkpoint={fine}"])
        vocab = Vocab.build(cfg)
        model = load_lm(cfg, vocab)                                              # tied weights: lm_head is absent from the file
        assert model.head_weight is model.lm.get_input_embeddings().weight
        plan = Plan(80, 40, plan_sections(40), ["calm"], ["zither"])
        used, codes = generate_codes(model, vocab, "A calm piece for zither", plan, temperature=1.0, top_k=8, seed=3)
        assert used == plan and codes.shape == (2, 40) and codes.dtype == torch.long and 0 <= codes.min() and codes.max() < 16
        assert torch.equal(codes, generate_codes(model, vocab, "A calm piece for zither", plan, temperature=1.0, top_k=8, seed=3)[1])


if __name__ == "__main__":
    test_train_checkpoint_resume_reload_generate()
    print("test_integration: 1 test passed")
