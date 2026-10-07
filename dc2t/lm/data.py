"""Manifest rows + codes/<tag>/<clip_id>.npy -> documents -> padded batches of about `lm.max_tokens` tokens."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dc2t.lm.sequence import build_document, doc_length
from dc2t.lm.vocab import Vocab
from dc2t.plan import Plan

PHASES = ("pretrain", "finetune")


class ClipDataset(Dataset):
    """One item per manifest row of `split`: {"input_ids": Long[S], "labels": Long[S]} (contract 8.4).
    pretrain: every clip, audio only (D12). finetune: only clips that have a caption, with caption and plan."""

    def __init__(self, cfg, vocab: Vocab, split: str, phase: str):
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        root = Path(cfg.paths.data_root)
        self.vocab, self.fine, self.loss_on_plan = vocab, phase == "finetune", cfg.lm.loss_on_plan
        self.codes_dir = root / "codes" / cfg.codec.tag
        rows = [json.loads(line) for line in (root / "manifest" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if self.fine:
            rows = [r for r in rows if r.get("caption") is not None]
        if cfg.lm.max_seconds:
            rows = [r for r in rows if r["duration"] <= cfg.lm.max_seconds]
        if not rows:
            raise ValueError(f"no {split} clips for phase {phase} in {root / 'manifest'}")
        missing = [r["clip_id"] for r in rows if not (self.codes_dir / f"{r['clip_id']}.npy").is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} clips have no codes in {self.codes_dir} (first: {missing[:3]}); run dc2t.codec.tokenize first")
        self.rows = rows
        self.plans = [Plan.from_manifest(r) for r in rows]
        self.captions = [r["caption"] if self.fine else None for r in rows]
        self.lengths = [doc_length(vocab, p.sections, c, p if self.fine else None) for p, c in zip(self.plans, self.captions)]   # tokens, no I/O

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row, plan, v = self.rows[i], self.plans[i], self.vocab
        codes = np.load(self.codes_dir / f"{row['clip_id']}.npy")
        if codes.shape != (v.K, row["duration"] * v.frame_rate) or codes.dtype.kind not in "iu":
            raise ValueError(f"{row['clip_id']}: codes are {codes.dtype}{codes.shape}, expected integers of shape {(v.K, row['duration'] * v.frame_rate)}; "
                             f"were they written with another codec or frame rate?")
        ids, labels = build_document(v, torch.from_numpy(codes.astype(np.int64)), plan.sections, self.captions[i], plan if self.fine else None,
                                     loss_on_plan=self.loss_on_plan)                 # also checks 0 <= code < V
        return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pads to the longest document: input_ids Long[B, S] (pad id), attention_mask Long[B, S] (0 on padding), labels Long[B, S] (-100 on padding)."""
    B, S = len(batch), max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": torch.full((B, S), pad_id), "attention_mask": torch.zeros(B, S, dtype=torch.long), "labels": torch.full((B, S), -100)}
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        out["input_ids"][i, :n], out["labels"][i, :n], out["attention_mask"][i, :n] = b["input_ids"], b["labels"], 1
    return out


class TokenBudgetSampler:
    """Batches of document indices with len(batch) * longest <= max_tokens (a single longer document goes alone).
    Documents are sorted by length before they are cut into batches, so padding is a few tokens per document.
    # ponytail: the composition of the batches is fixed, only their order is shuffled per epoch; re-bucket with a jittered sort key if that matters."""

    def __init__(self, lengths: list[int], max_tokens: int, shuffle: bool, seed: int = 0):
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.batches, cur = [], []
        for i in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
            if cur and (len(cur) + 1) * lengths[i] > max_tokens:       # ascending order: lengths[i] is the longest in the batch
                self.batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            self.batches.append(cur)

    def set_epoch(self, epoch: int) -> None:            # the engine calls this at the start of every epoch (and when it resumes)
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return iter([self.batches[j] for j in order])
