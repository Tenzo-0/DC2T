"""Decoding, D9 and contract 8.5: every position is sampled from the id range the plan allows; forced tokens are fed, not sampled.
We do not call `model.generate`: its defaults stop after 2,048 new tokens and its processors know nothing about a schedule."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from dcttgen.lm.sequence import PLAN_MAX_TOKENS, instruct_ids, metadata_ids, parse_document
from dcttgen.lm.vocab import Vocab
from dcttgen.plan import Plan

PLAN_ATTEMPTS = 8                                  # how often the model may write an invalid plan before generate_codes gives up


class Step(NamedTuple):
    lo: int                                        # n consecutive positions, each taken from the ids [lo, hi) ...
    hi: int                                        # ... and hi - lo == 1 means the token is forced
    n: int


def build_schedule(vocab: Vocab, sections) -> list[Step]:
    """Everything that follows `[<INST>] caption [<PLAN>] plan_text` (contract 8.5)."""
    forced = lambda ids: [Step(i, i + 1, 1) for i in ids]
    steps: list[Step] = []
    for label, sec in sections:
        steps += forced(vocab.seg_start + vocab.label[label] + [vocab.soa])
        steps += [Step(vocab.audio_offset + k * vocab.V, vocab.audio_offset + (k + 1) * vocab.V, sec * vocab.frame_rate) for k in range(vocab.K)]
        steps += forced([vocab.eoa] + vocab.seg_end)
    return steps + forced([vocab.eod])


def range_logits(h, weight, lo: int, hi: int):
    """The logits processor: h Float[d], weight Float[size, d] -> Float[hi - lo]. Identical to masking the full logits
    to -inf outside [lo, hi), but computes 10,000 of the 191,670 dot products."""
    return F.linear(h, weight[lo:hi])


def sample(logits, temperature: float, top_k, top_p, gen):
    """logits Float[n] over the allowed ids -> Long[] index into them. temperature 0 is argmax (the plan's literal rule)."""
    if temperature == 0:
        return logits.argmax()
    logits = logits.float() / temperature
    if top_k and top_k < logits.numel():
        logits = logits.masked_fill(logits < logits.topk(top_k).values[-1], float("-inf"))
    if top_p and 0.0 < top_p < 1.0:                                 # smallest set of ids whose probability reaches top_p
        probs, order = logits.softmax(-1).sort(descending=True)
        drop = torch.zeros_like(probs, dtype=torch.bool).scatter(0, order, probs.cumsum(-1) - probs >= top_p)
        logits = logits.masked_fill(drop, float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1, generator=gen)[0]


def _feed(model, ids, cache):
    """ids Long[1, m] appended to the key-value cache -> (hidden state of the last position Float[d], cache)."""
    out = model.backbone(input_ids=ids, past_key_values=cache, use_cache=True)
    return out.last_hidden_state[0, -1], out.past_key_values


def _run(model, vocab: Vocab, prompt: list[int], schedule, temperature, top_k, top_p, gen):
    """Prefill the prompt, then walk the schedule. -> Long[sum(n)]: every id after the prompt, forced ones included."""
    dev = model.head_weight.device
    h, cache = _feed(model, torch.tensor([prompt], device=dev), None)
    out, pending = [], []                                           # pending: forced ids not yet shown to the model
    for lo, hi, n in schedule:
        for _ in range(n):
            if hi - lo == 1:
                pending.append(lo)
                out.append(torch.tensor([lo], device=dev))
                continue
            if pending:                                             # a run of forced tokens costs one forward pass
                h, cache = _feed(model, torch.tensor([pending], device=dev), cache)
                pending = []
            tok = lo + sample(range_logits(h, model.head_weight, lo, hi), temperature, top_k, top_p, gen)
            out.append(tok.reshape(1))
            h, cache = _feed(model, tok.reshape(1, 1), cache)
    return torch.cat(out)


def _write_plan(model, vocab: Vocab, caption: str, temperature, top_k, top_p, gen) -> Plan:
    """Self-planned mode: sample the plan text after `<PLAN>` until the model starts the first segment ('[start_of_seg]')."""
    dev, text_hi = model.head_weight.device, vocab.tokenizer.vocab_size       # only ordinary text ids: no control token can end up in the plan
    prompt, last = instruct_ids(vocab, caption) + [vocab.plan], None
    for _ in range(1 if temperature == 0 else PLAN_ATTEMPTS):              # greedy decoding would repeat itself
        h, cache = _feed(model, torch.tensor([prompt], device=dev), None)
        text = []
        while len(text) < PLAN_MAX_TOKENS + len(vocab.seg_start) and text[-len(vocab.seg_start):] != vocab.seg_start:
            tok = sample(range_logits(h, model.head_weight, 0, text_hi), temperature, top_k, top_p, gen)
            text.append(int(tok))
            h, cache = _feed(model, tok.reshape(1, 1), cache)
        if text[-len(vocab.seg_start):] != vocab.seg_start:
            last = f"no [start_of_seg] within {len(text)} tokens"
            continue
        last = vocab.tokenizer.decode(text[:-len(vocab.seg_start)])
        try:
            return Plan.from_text(last[1:] if last.startswith(" ") else last)
        except ValueError:
            continue
    raise ValueError(f"the model did not write a valid plan in {PLAN_ATTEMPTS} attempts; the last one was {last!r}")


@torch.inference_mode()
def generate_codes(model, vocab: Vocab, caption: str, plan: Plan | None = None, *, temperature: float = 1.0,
                   top_k: int | None = None, top_p: float | None = None, seed: int | None = None) -> tuple[Plan, torch.Tensor]:
    """-> (the plan actually used, codes Long[K, frame_rate * plan.duration] on the CPU). plan=None lets the model write the plan."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0 (0 = greedy)")
    model.eval()
    gen = None if seed is None else torch.Generator(device=model.head_weight.device).manual_seed(seed)
    if plan is None:
        plan = _write_plan(model, vocab, caption, temperature, top_k, top_p, gen)
    prompt = instruct_ids(vocab, caption) + metadata_ids(vocab, plan.to_text())          # a model-written plan is re-encoded canonically
    schedule = build_schedule(vocab, plan.sections)
    if len(prompt) + sum(s.n for s in schedule) > model.lm.config.max_position_embeddings:
        raise ValueError(f"{len(prompt) + sum(s.n for s in schedule)} positions exceed max_position_embeddings of this backbone")
    ids = _run(model, vocab, prompt, schedule, temperature, top_k, top_p, gen)
    return plan, parse_document(vocab, ids)["codes"]                                      # the parser re-checks the layout and every range
