"""Shared helpers for the LM tests: a config namespace, the real Qwen2.5 tokenizer (from the Hugging Face cache) and a tiny random Qwen2."""
import atexit
import json
import pathlib
import shutil
import sys
import tempfile
from functools import lru_cache
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))     # lets `python tests/test_x.py` run without installing the package

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from dc2t.lm.sequence import build_document
from dc2t.lm.vocab import Vocab
from dc2t.plan import Plan, plan_sections

CAPTION = "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute"


@lru_cache(maxsize=None)
def backbone_dir() -> str:
    """A random Qwen2 (hidden 32, 2 layers, the real 151,936 embedding rows) saved next to the real Qwen2.5 tokenizer files.
    `from_pretrained(backbone_dir())` then behaves like `from_pretrained("Qwen/Qwen2.5-0.5B")`, without any weights."""
    d = tempfile.mkdtemp(prefix="tiny_qwen_")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=4096, rope_theta=1e6, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    return d


def make_cfg(K=2, V=16, frame_rate=25, data_root="data", **lm):
    return NS(paths=NS(data_root=data_root, runs="runs"), audio=NS(sample_rate=32000, frame_rate=frame_rate),
              codec=NS(tag=f"k{K}v{V}", num_codebooks=K, codebook_size=V),
              lm=NS(**{**dict(backbone=backbone_dir(), loss_on_plan=True, checkpoint=None, attn="sdpa", grad_checkpointing=False,
                              max_tokens=4096, max_seconds=None, num_workers=0, temperature=1.0, top_k=None, top_p=None), **lm}))


@lru_cache(maxsize=None)
def get_vocab(K=2, V=16, frame_rate=25) -> Vocab:
    return Vocab.build(make_cfg(K, V, frame_rate))


def random_rows_model(K=2, V=16, frame_rate=1, seed=1):
    """load_lm, then random new rows. At initialisation every row of a codebook is the same, which would make most tests vacuous."""
    from dc2t.lm.model import load_lm
    v = get_vocab(K, V, frame_rate)
    model = load_lm(make_cfg(K, V, frame_rate), v)
    torch.manual_seed(seed)
    with torch.no_grad():
        model.head_weight[v.eod:] += 0.5 * torch.randn_like(model.head_weight[v.eod:])
    return model


def make_plan(duration, position="whole", **kw):
    return Plan(**{**dict(bpm=80, duration=duration, sections=plan_sections(duration, position), moods=["uplifting", "joyful"],
                          instruments=["zither", "two-string fiddle", "moon-shaped lute"]), **kw})


def make_doc(v, duration=150, position="whole", fine=True, loss_on_plan=True, caption=CAPTION, seed=0):
    """-> (codes Long[K, T], plan, (input_ids, labels)) for random codes."""
    p = make_plan(duration, position)
    codes = torch.randint(0, v.V, (v.K, duration * v.frame_rate), generator=torch.Generator().manual_seed(seed))
    return codes, p, build_document(v, codes, p.sections, caption if fine else None, p if fine else None, loss_on_plan=loss_on_plan)


def raises(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return True
    return False


CLIPS = [("whole", 150), ("first", 270), ("middle", 120), ("last", 90), ("whole", 30), ("first", 60)]      # (position, seconds)


def write_dataset(root, caption_none=(), n=6, K=2, V=16, fr=1, clips=CLIPS):
    """A synthetic data/ tree in the contract 7.2 / 7.3 formats: manifest/{train,val}.jsonl and codes/k{K}v{V}/<clip_id>.npy (int16 [K, fr * duration])."""
    root = pathlib.Path(root)
    (root / "manifest").mkdir(parents=True)
    (root / "codes" / f"k{K}v{V}").mkdir(parents=True)
    rng, rows = np.random.default_rng(0), []
    for i in range(n):
        position, duration = clips[i % len(clips)]
        rows.append({"clip_id": f"c{i}", "recording_id": f"r{i // 2}", "position": position, "audio": f"audio/c{i}.flac", "duration": duration, "bpm": 80,
                     "moods": ["calm"], "instruments": ["zither"], "sections": [list(s) for s in plan_sections(duration, position)],
                     "caption": None if i in caption_none else "A calm piece for zither", "split": "train"})
        np.save(root / "codes" / f"k{K}v{V}" / f"c{i}.npy", rng.integers(0, V, (K, duration * fr)).astype(np.int16))
    for name, part in (("train", rows), ("val", rows[:2])):                  # val reuses two clips: the loaders only need a readable file
        (root / "manifest" / f"{name}.jsonl").write_text("".join(json.dumps({**r, "split": name}) + "\n" for r in part), encoding="utf-8")
    return rows


def run_all(ns):
    tests = [f for n, f in sorted(ns.items()) if n.startswith("test_") and callable(f)]
    for f in tests:
        f()
    print(f"{pathlib.Path(ns['__file__']).stem}: {len(tests)} passed")
