import pathlib
import random
import tempfile

import numpy as np
import torch
from lm_testkit import get_vocab, make_cfg, random_rows_model, raises, run_all, write_dataset

from dc2t.lm.data import ClipDataset, TokenBudgetSampler, collate
from dc2t.lm.sequence import parse_document
from dc2t.lm.train import make_loader

K, V, FR = 2, 16, 1                                  # one frame per second keeps the documents small; the code reads it from cfg


def cfg_for(root, **lm):
    return make_cfg(K, V, FR, data_root=str(root), **lm)


def test_documents_match_the_manifest_and_the_saved_codes_in_both_phases():
    v = get_vocab(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        rows = write_dataset(d)
        for phase in ("pretrain", "finetune"):
            ds = ClipDataset(cfg_for(d), v, "train", phase)
            assert len(ds) == len(rows)
            for i, row in enumerate(rows):
                item = ds[i]
                out = parse_document(v, item["input_ids"])
                saved = torch.from_numpy(np.load(f"{d}/codes/k{K}v{V}/{row['clip_id']}.npy").astype(np.int64))
                assert torch.equal(out["codes"], saved) and out["sections"] == [tuple(s) for s in row["sections"]]
                assert (out["caption"] is not None) == (phase == "finetune") and len(item["input_ids"]) == ds.lengths[i]
                assert item["input_ids"].dtype == item["labels"].dtype == torch.long


def test_finetuning_skips_clips_without_a_caption_pretraining_keeps_them():
    v = get_vocab(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        write_dataset(d, caption_none=(1, 4))
        assert len(ClipDataset(cfg_for(d), v, "train", "pretrain")) == 6
        assert [r["clip_id"] for r in ClipDataset(cfg_for(d), v, "train", "finetune").rows] == ["c0", "c2", "c3", "c5"]
        assert [r["clip_id"] for r in ClipDataset(cfg_for(d, max_seconds=100), v, "train", "pretrain").rows] == ["c3", "c4", "c5"]   # max_seconds drops 150, 270, 120 s
        assert raises(ClipDataset, cfg_for(d), v, "train", "pretrainx") and raises(ClipDataset, cfg_for(d, max_seconds=5), v, "train", "pretrain")


def test_codes_of_another_shape_dtype_or_range_are_refused():           # a stale or foreign codes/<tag>/ directory must not train silently
    v = get_vocab(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        write_dataset(d)
        ds, path = ClipDataset(cfg_for(d), v, "train", "pretrain"), pathlib.Path(d, "codes", f"k{K}v{V}", "c0.npy")
        good = np.load(path)
        for name, bad in (("one codebook", good[:1]), ("one frame short", good[:, :-1]), ("float", good.astype(np.float32)),
                          ("code >= V", good + V), ("negative code", good - V)):
            np.save(path, bad)
            assert raises(ds.__getitem__, 0), name
        np.save(path, good)
        assert ds[0]["input_ids"].numel() == ds.lengths[0]
        path.unlink()
        try:
            ClipDataset(cfg_for(d), v, "train", "pretrain")
        except FileNotFoundError as e:
            assert "c0" in str(e) and "tokenize" in str(e)
        else:
            raise AssertionError("missing codes were accepted")


def test_collate_pads_input_labels_and_mask():
    a = {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([-100, 6, 7])}
    b = {"input_ids": torch.tensor([8, 9]), "labels": torch.tensor([8, 9])}
    out = collate([a, b], pad_id=151643)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 9, 151643]] and out["labels"].tolist() == [[-100, 6, 7], [8, 9, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_token_budget_sampler():
    rnd = random.Random(0)
    lengths = [rnd.randint(24000, 30300) if rnd.random() < 0.7 else rnd.randint(3000, 24000) for _ in range(400)]   # most clips are near the 300 s cap
    s = TokenBudgetSampler(lengths, 32768, shuffle=True, seed=0)
    epoch0 = list(s)
    assert sorted(i for b in epoch0 for i in b) == list(range(400)) and len(epoch0) == len(s)                  # every document exactly once
    assert all(len(b) * max(lengths[i] for i in b) <= 32768 or len(b) == 1 for b in epoch0)                    # the budget, padding included
    padded, real = sum(len(b) * max(lengths[i] for i in b) for b in epoch0), sum(lengths)
    assert padded / real < 1.1, padded / real                                                                  # sorting keeps the padding below 10 %
    assert list(s) == epoch0                                                                                   # same (seed, epoch): same order
    s.set_epoch(1)
    epoch1 = list(s)
    assert epoch1 != epoch0 and sorted(map(tuple, epoch1)) == sorted(map(tuple, epoch0)) and len(epoch1) == len(epoch0)   # new order, same batches
    assert [i for b in TokenBudgetSampler(lengths, 32768, shuffle=False) for i in b] == sorted(range(400), key=lambda i: (lengths[i], i))
    assert TokenBudgetSampler([40000], 32768, shuffle=False).batches == [[0]]                                   # a document over the budget goes alone


def test_loader_batches_are_accepted_by_the_model():
    v, model = get_vocab(K, V, FR), random_rows_model(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        write_dataset(d)
        loader = make_loader(cfg_for(d, max_tokens=700), v, "train", "finetune", shuffle=True)
        batches = list(loader)
        assert sum(b["input_ids"].shape[0] for b in batches) == 6 and len(batches) == len(loader)
        assert all(b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape for b in batches)
        assert all(torch.isfinite(model(b)["loss"]) for b in batches)


if __name__ == "__main__":
    run_all(globals())
