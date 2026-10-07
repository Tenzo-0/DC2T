"""Vocabulary layout, contract 8.1: Qwen2.5 tokens, five added special tokens, then K * V audio ids."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

from dc2t.plan import SECTION_ORDER

SPECIALS = ("<EOD>", "<SOA>", "<EOA>", "<INST>", "<PLAN>")             # added in exactly this order
LAYOUT = {"<EOD>": 151665, "<SOA>": 151666, "<EOA>": 151667, "<INST>": 151668, "<PLAN>": 151669}
AUDIO_OFFSET, PAD = 151670, 151643          # contract 8.1: asserted in build() and nowhere else; the rest of the code reads Vocab


@dataclass
class Vocab:
    tokenizer: object                       # Qwen2TokenizerFast with the five specials added
    eod: int
    soa: int
    eoa: int
    inst: int
    plan: int
    pad: int
    audio_offset: int
    K: int                                  # codebooks; the audio id of (level k, code c) is audio_offset + k * V + c
    V: int
    frame_rate: int                         # code frames per second, cfg.audio.frame_rate; a section of s seconds has s * frame_rate frames
    codec_tag: str                          # cfg.codec.tag: which RVQ produced the codes this model is trained on
    seg_start: list[int]                    # "[start_of_seg]" tokenised alone
    seg_end: list[int]                      # "[end_of_seg]"
    label: dict[str, list[int]]             # "[intro]", "[main]", "[outro]"

    @property
    def size(self) -> int:                  # rows of the embedding matrix
        return self.audio_offset + self.K * self.V

    @classmethod
    def build(cls, cfg) -> "Vocab":
        tok = AutoTokenizer.from_pretrained(cfg.lm.backbone)
        tok.add_special_tokens({"additional_special_tokens": list(SPECIALS)})
        ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIALS}
        assert ids == LAYOUT and len(tok) == AUDIO_OFFSET, f"token layout {ids}, len {len(tok)} differs from contract 8.1"
        assert tok.pad_token_id == tok.eos_token_id == PAD, "expected <|endoftext|> to be both pad and eos"
        v = cls(tok, ids["<EOD>"], ids["<SOA>"], ids["<EOA>"], ids["<INST>"], ids["<PLAN>"], tok.pad_token_id, len(tok),
                cfg.codec.num_codebooks, cfg.codec.codebook_size, cfg.audio.frame_rate, cfg.codec.tag, [], [], {})
        v.seg_start, v.seg_end = v.enc("[start_of_seg]"), v.enc("[end_of_seg]")
        v.label = {s: v.enc(f"[{s}]") for s in SECTION_ORDER}
        return v

    def enc(self, text: str) -> list[int]:
        """Untrusted text -> ids. `split_special_tokens=True` makes '<EOA>', '<|endoftext|>' ... ordinary text."""
        ids = self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
        if ids and max(ids) >= self.tokenizer.vocab_size:       # ids from 151,643 up are control tokens (Qwen's 22 and ours)
            raise ValueError("the tokenizer turned text into a control token; refusing")   # belt and braces against a transformers change
        return ids

    def _meta(self) -> dict:
        return {"codec_tag": self.codec_tag, "K": self.K, "V": self.V, "audio_offset": self.audio_offset, "frame_rate": self.frame_rate}

    def save(self, run_dir) -> None:
        """Written once per run, next to the step_* checkpoint directories (contract 7.4)."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(run_dir / "tokenizer")
        (run_dir / "vocab.json").write_text(json.dumps(self._meta(), indent=2), encoding="utf-8")

    def check_checkpoint(self, checkpoint) -> None:
        """Refuse a checkpoint trained on another codec: shapes alone cannot tell two RVQs of the same K and V apart."""
        f = Path(checkpoint).parent / "vocab.json"
        if not f.is_file():
            raise ValueError(f"{f} is missing; a checkpoint is only usable next to the vocab.json that lm.train writes")
        saved = json.loads(f.read_text(encoding="utf-8"))
        if saved != self._meta():
            raise ValueError(f"checkpoint {checkpoint} was trained with {saved}, but the config now says {self._meta()}")
