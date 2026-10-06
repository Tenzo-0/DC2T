"""Documents, contract 8.2 and 8.4: flatten the code matrix, build a training document, parse one back."""
from __future__ import annotations

import torch

from dcttgen.lm.vocab import Vocab
from dcttgen.plan import Plan, validate_sections

CAPTION_MAX_TOKENS = PLAN_MAX_TOKENS = 128          # contract 6: caption and plan text are each capped at 128 tokens


def flatten_c2f(codes, audio_offset, V):            # codes: Long[K, T]  ->  Long[K*T]      (contract 8.2)
    K, T = codes.shape
    return (codes + audio_offset + torch.arange(K, device=codes.device).unsqueeze(1) * V).reshape(-1)   # row-major = codebook-major


def unflatten_c2f(ids, audio_offset, K, V):         # ids: Long[K*T]  ->  Long[K, T]        (contract 8.2; ValueError, not assert, so -O keeps it)
    codes = ids.reshape(K, -1) - audio_offset - torch.arange(K, device=ids.device).unsqueeze(1) * V
    if not ((codes >= 0) & (codes < V)).all():
        raise ValueError("audio ids outside their codebook range")
    return codes


def instruct_ids(vocab: Vocab, caption: str) -> list[int]:     # [<INST>] + enc(" " + caption), caption cut at 128 tokens
    return [vocab.inst] + vocab.enc(" " + caption)[:CAPTION_MAX_TOKENS]


def metadata_ids(vocab: Vocab, plan_text: str) -> list[int]:   # [<PLAN>] + enc(" " + plan_text); a plan over the cap is a bug, so no truncation
    ids = vocab.enc(" " + plan_text)
    if len(ids) > PLAN_MAX_TOKENS:
        raise ValueError(f"plan text is {len(ids)} tokens, over the cap of {PLAN_MAX_TOKENS}")
    return [vocab.plan] + ids


def build_document(vocab: Vocab, codes, sections, caption, plan, *, loss_on_plan: bool = True):
    """codes Long[K, T], T = frame_rate * sum(seconds)  ->  (input_ids Long[S], labels Long[S]). Contract 8.4 and D8, D12.
    caption and plan are both given (fine-tuning) or both None (pre-training)."""
    total = validate_sections(sections)
    K, T = codes.shape
    if K != vocab.K or T != total * vocab.frame_rate:
        raise ValueError(f"codes are {tuple(codes.shape)}, expected ({vocab.K}, {total * vocab.frame_rate}) for sections {list(sections)}")
    if int(codes.min()) < 0 or int(codes.max()) >= vocab.V:
        raise ValueError(f"codes must lie in [0, {vocab.V})")
    if (caption is None) != (plan is None):
        raise ValueError("caption and plan go together: both for fine-tuning, neither for pre-training")
    head, n_masked = [], 0
    if plan is not None:
        if [tuple(s) for s in plan.sections] != [tuple(s) for s in sections]:
            raise ValueError("plan.sections differs from sections")
        inst, meta = instruct_ids(vocab, caption), metadata_ids(vocab, plan.to_text())
        head, n_masked = inst + meta, len(inst) + (0 if loss_on_plan else len(meta))     # D8: Instruct is never trained; Metadata optionally
    parts, start = [torch.tensor(head, dtype=torch.long)], 0
    for label, sec in sections:
        n = sec * vocab.frame_rate
        parts += [torch.tensor(vocab.seg_start + vocab.label[label] + [vocab.soa]),
                  flatten_c2f(codes[:, start:start + n].long(), vocab.audio_offset, vocab.V),
                  torch.tensor([vocab.eoa] + vocab.seg_end)]
        start += n
    ids = torch.cat(parts + [torch.tensor([vocab.eod])])
    labels = ids.clone()
    labels[:n_masked] = -100
    return ids, labels


def doc_length(vocab: Vocab, sections, caption=None, plan=None) -> int:
    """len(build_document(...)[0]) without needing the codes; the batch sampler uses it."""
    n = 1                                                                      # <EOD>
    if plan is not None:
        n += len(instruct_ids(vocab, caption)) + len(metadata_ids(vocab, plan.to_text()))
    for label, sec in sections:
        n += len(vocab.seg_start) + len(vocab.label[label]) + 1 + vocab.K * sec * vocab.frame_rate + 1 + len(vocab.seg_end)
    return n


def _find(seq: list, sub: list, start: int) -> int:
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    raise ValueError(f"{sub} not found after position {start}")


def parse_document(vocab: Vocab, ids) -> dict:
    """Inverse of build_document for an unpadded document (also validates the output of the decoder).
    -> {"caption": str | None, "plan_text": str | None, "sections": [(label, seconds)], "codes": Long[K, T]}"""
    ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
    pos, caption, plan_text = 0, None, None
    if ids and ids[0] == vocab.inst:
        p = ids.index(vocab.plan)
        s = _find(ids, vocab.seg_start, p + 1)
        caption, plan_text = vocab.tokenizer.decode(ids[1:p])[1:], vocab.tokenizer.decode(ids[p + 1:s])[1:]   # [1:] drops the " " we put in front
        pos = s
    sections, chunks = [], []
    while pos < len(ids) and ids[pos] != vocab.eod:
        if ids[pos:pos + len(vocab.seg_start)] != vocab.seg_start:
            raise ValueError(f"expected [start_of_seg] at {pos}")
        pos += len(vocab.seg_start)
        names = [n for n, l in vocab.label.items() if ids[pos:pos + len(l)] == l]
        if len(names) != 1 or ids[pos + len(vocab.label[names[0]]):][:1] != [vocab.soa]:
            raise ValueError(f"expected a section label and <SOA> at {pos}")
        pos += len(vocab.label[names[0]]) + 1
        end = _find(ids, [vocab.eoa], pos)
        if (end - pos) % (vocab.K * vocab.frame_rate):
            raise ValueError(f"{end - pos} audio ids is not a whole number of seconds of {vocab.K} codebooks")
        chunks.append(unflatten_c2f(torch.tensor(ids[pos:end]), vocab.audio_offset, vocab.K, vocab.V))
        sections.append((names[0], chunks[-1].shape[1] // vocab.frame_rate))
        if ids[end + 1:end + 1 + len(vocab.seg_end)] != vocab.seg_end:
            raise ValueError(f"expected [end_of_seg] after <EOA> at {end}")
        pos = end + 1 + len(vocab.seg_end)
    if pos != len(ids) - 1 or not chunks:
        raise ValueError("a document is one or more segments followed by <EOD> and nothing else")
    return {"caption": caption, "plan_text": plan_text, "sections": sections, "codes": torch.cat(chunks, dim=1)}
