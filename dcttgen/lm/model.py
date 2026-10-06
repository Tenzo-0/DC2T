"""The language model: stock Qwen2.5 with an enlarged embedding matrix, and a loss that scores each target only against the rows it may take (D9)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_model
from torch import nn
from transformers import AutoModelForCausalLM

from dcttgen.lm.vocab import Vocab


def resize_for_audio(lm, vocab: Vocab) -> None:
    """Rows [0, vocab.eod) keep their pre-trained values. Every row from vocab.eod on (Qwen's 271 unused rows, our 5 special tokens and the
    K*V audio rows) is created by one resize call, so all of them start from the same distribution: the mean of the old rows."""
    lm.resize_token_embeddings(vocab.eod, mean_resizing=True)         # shrink: drops the unused rows 151,665 .. 151,935
    lm.resize_token_embeddings(vocab.size, mean_resizing=True)        # grow: new rows ~ N(mean of old rows, 1e-9 * their covariance)
    emb = lm.get_input_embeddings().weight
    assert emb.shape[0] == lm.config.vocab_size == vocab.size
    assert not lm.config.tie_word_embeddings or lm.get_output_embeddings().weight is emb


def restricted_loss(h, tgt, weight, audio_offset: int, K: int, V: int) -> dict:
    """h Float[N, d]: the states that predict tgt Long[N] (no -100 left); weight Float[size, d]: the output matrix.
    A target in codebook k is scored against the V rows of codebook k only, any other target against the rows [0, audio_offset).
    That equals the cross-entropy of the full softmax with every other column masked to -inf (tests/test_lm_model.py)."""
    names = ["text"] + [f"k{k}" for k in range(K)]
    ranges = [(0, audio_offset)] + [(audio_offset + k * V, audio_offset + (k + 1) * V) for k in range(K)]
    total, seen, logs = 0.0 * h.sum(), 0, {}
    for name, (lo, hi) in zip(names, ranges):
        sel = (tgt >= lo) & (tgt < hi)
        n = int(sel.sum())
        if n:
            ce = F.cross_entropy(F.linear(h[sel], weight[lo:hi]).float(), tgt[sel] - lo, reduction="sum")   # Float[n, hi - lo]: 10,000 columns, not 191,670
            total, seen, logs[f"ce_{name}"] = total + ce, seen + n, (ce / n).detach()
    if seen != len(tgt):
        raise ValueError(f"{len(tgt) - seen} targets lie outside every range")
    return {"loss": total / max(seen, 1), "tokens": torch.tensor(float(seen)), **logs}


class MusicLM(nn.Module):
    """Qwen2ForCausalLM plus the audio vocabulary. forward(batch) -> loss dict, as fit() expects (contract 9)."""

    def __init__(self, lm, vocab: Vocab):
        super().__init__()
        self.lm = lm
        self.layout = (vocab.audio_offset, vocab.K, vocab.V)

    @property
    def backbone(self):                     # Qwen2Model; its output has already passed the final RMSNorm
        return self.lm.model

    @property
    def head_weight(self):                  # Float[size, d]; the same tensor as the input embeddings when the weights are tied
        return self.lm.get_output_embeddings().weight

    def forward(self, batch: dict) -> dict:
        """batch: input_ids, attention_mask, labels, all Long[B, S]; labels is -100 where nothing is learned."""
        h = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state   # Float[B, S, d]
        tgt = batch["labels"][:, 1:]                                    # the state at position i predicts token i + 1
        keep = tgt != -100
        return restricted_loss(h[:, :-1][keep], tgt[keep], self.head_weight, *self.layout)


def load_lm(cfg, vocab: Vocab, checkpoint: str | None = None) -> MusicLM:
    """Stock Qwen weights (or a step_* directory written by the engine) behind the contract-9 `forward(batch)`."""
    checkpoint = checkpoint or cfg.lm.checkpoint
    lm = AutoModelForCausalLM.from_pretrained(cfg.lm.backbone, dtype=torch.float32, attn_implementation=cfg.lm.attn)   # fp32 master weights
    resize_for_audio(lm, vocab)
    lm.config.use_cache = False                                         # generate_codes asks for the cache explicitly
    if cfg.lm.grad_checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = MusicLM(lm, vocab)
    if checkpoint:
        vocab.check_checkpoint(checkpoint)
        load_model(model, str(Path(checkpoint) / "model.safetensors"))  # tolerates the tied lm_head.weight that the engine leaves out
    return model
