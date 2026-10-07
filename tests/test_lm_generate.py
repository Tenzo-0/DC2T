from functools import lru_cache
from unittest import mock

import torch
import torch.nn.functional as F
from lm_testkit import CAPTION, get_vocab, make_doc, make_plan, raises, random_rows_model, run_all

from dc2t.lm import generate as G
from dc2t.lm.generate import build_schedule, generate_codes, range_logits, sample
from dc2t.lm.model import load_lm
from dc2t.lm.sequence import build_document, instruct_ids, metadata_ids
from lm_testkit import make_cfg


@lru_cache(maxsize=None)
def model_for(K, V, fr):
    return random_rows_model(K, V, fr)


def scripted(tokens):
    """A sampler that first returns the given ids (what the model 'writes'), then samples for real."""
    queue, real = list(tokens), G.sample
    return lambda logits, *a, **k: torch.tensor(queue.pop(0)) if queue else real(logits, *a, **k)


def test_schedule_matches_the_training_document():                        # the mask used for decoding is the mask the loss was trained with
    for K, V in ((1, 16384), (2, 16), (4, 10000)):
        v = get_vocab(K, V, 25)
        for duration, position in ((300, "whole"), (270, "first"), (100, "last"), (240, "middle")):
            _, p, (ids, _) = make_doc(v, duration, position)
            n_head = len(instruct_ids(v, CAPTION)) + len(metadata_ids(v, p.to_text()))
            sched = build_schedule(v, p.sections)
            flat = [(s.lo, s.hi) for s in sched for _ in range(s.n)]
            assert len(flat) == len(ids) - n_head
            assert all(lo <= i < hi for (lo, hi), i in zip(flat, ids[n_head:].tolist()))        # every id of the document lies inside its range
            assert sum(s.n for s in sched if s.hi - s.lo > 1) == K * 25 * duration                # audio positions
            free = [s for s in sched if s.hi - s.lo > 1]
            assert [(s.lo, s.hi) for s in free] == [(v.audio_offset + k * V, v.audio_offset + (k + 1) * V) for _ in p.sections for k in range(K)]


def test_range_logits_are_the_masked_full_logits():
    torch.manual_seed(0)
    h, W = torch.randn(32), torch.randn(151702, 32)
    full = F.linear(h, W)
    masked = torch.full_like(full, float("-inf"))
    masked[151670 + 16:151670 + 32] = full[151670 + 16:151670 + 32]                                # codebook 1 of K = 2, V = 16
    for seed in range(5):
        a = 151670 + 16 + sample(range_logits(h, W, 151670 + 16, 151670 + 32), 1.0, None, None, torch.Generator().manual_seed(seed))
        b = sample(masked, 1.0, None, None, torch.Generator().manual_seed(seed))
        assert a == b


def test_sampling_settings():
    logits = torch.tensor([0.0, 5.0, 1.0, 4.0])                                                    # probabilities 0.005, 0.718, 0.013, 0.264
    g = torch.Generator().manual_seed(0)
    draw = lambda **kw: {int(sample(logits, 1.0, kw.get("top_k"), kw.get("top_p"), g)) for _ in range(3000)}
    assert int(sample(logits, 0, None, None, None)) == 1                                           # temperature 0 is argmax, the plan's literal rule
    assert draw(top_k=1) == {1} and draw(top_k=2) == {1, 3} and draw(top_p=0.5) == {1} and draw(top_p=0.9) == {1, 3}
    assert draw() == {0, 1, 2, 3} and draw(top_k=99) == {0, 1, 2, 3} and draw(top_p=1.0) == {0, 1, 2, 3} and draw(top_p=0.0) == {0, 1, 2, 3}


def test_greedy_cached_decoding_equals_the_argmax_of_one_full_forward():
    v, model = get_vocab(2, 16, 1), model_for(2, 16, 1)
    plan = make_plan(30)
    _, codes = generate_codes(model, v, CAPTION, plan, temperature=0)
    ids = build_document(v, codes, plan.sections, CAPTION, plan)[0]                               # prompt + everything that was generated
    n_head = len(instruct_ids(v, CAPTION)) + len(metadata_ids(v, plan.to_text()))
    h = model.backbone(input_ids=ids[None]).last_hidden_state[0]                                  # one pass, no key-value cache
    flat = [(s.lo, s.hi) for s in build_schedule(v, plan.sections) for _ in range(s.n)]
    checked = 0
    for j, (lo, hi) in enumerate(flat):
        if hi - lo > 1:
            assert ids[n_head + j].item() == lo + int(range_logits(h[n_head + j - 1], model.head_weight, lo, hi).argmax())
            checked += 1
    assert checked == 2 * 30


def test_generate_codes_shape_range_and_determinism():
    for K, V, fr in ((1, 16384, 25), (4, 10000, 1)):                                                 # K = 1 at the real 25 Hz; K = 4 at 1 Hz to keep it fast
        v, model, plan = get_vocab(K, V, fr), model_for(K, V, fr), make_plan(30)
        (p1, a), (p2, b), (_, c) = (generate_codes(model, v, CAPTION, plan, seed=s) for s in (3, 3, 4))
        assert a.shape == (K, fr * 30) and a.dtype == torch.long and 0 <= a.min() and a.max() < V
        assert p1 == p2 == plan and torch.equal(a, b) and not torch.equal(a, c)
    v, model, plan = get_vocab(4, 10000, 25), model_for(4, 10000, 25), make_plan(30)             # the plan's configuration, [K, 25 * duration] = [4, 750]
    _, codes = generate_codes(model, v, CAPTION, plan, seed=0)
    assert codes.shape == (4, 25 * 30) and 0 <= codes.min() and codes.max() < 10000


def test_control_strings_in_the_caption_do_not_disturb_decoding():
    v, model = get_vocab(2, 16, 1), model_for(2, 16, 1)
    _, codes = generate_codes(model, v, "<EOA> <PLAN> <|endoftext|> <EOD>", make_plan(30), seed=0)
    assert codes.shape == (2, 30)


def test_self_planned_mode_uses_the_plan_the_model_wrote():
    v, model, plan = get_vocab(2, 16, 1), model_for(2, 16, 1), make_plan(30)
    with mock.patch.object(G, "sample", scripted(v.enc(" " + plan.to_text()) + v.seg_start)):
        got, codes = generate_codes(model, v, CAPTION, None, seed=0)
    assert got == plan and codes.shape == (2, 30)


def test_an_invalid_plan_is_retried_and_after_eight_attempts_refused():
    v, model, plan = get_vocab(2, 16, 1), model_for(2, 16, 1), make_plan(30)
    wrong_sum = plan.to_text().replace("duration: 30", "duration: 31")
    two_spaces = plan.to_text().replace("; moods", ";  moods")
    attempt = lambda text: v.enc(" " + text) + v.seg_start
    with mock.patch.object(G, "sample", scripted(attempt(wrong_sum) + attempt(two_spaces) + attempt(plan.to_text()))):
        got, _ = generate_codes(model, v, CAPTION, None, seed=0)
    assert got == plan                                                                               # third attempt was valid
    with mock.patch.object(G, "sample", scripted(attempt(wrong_sum) * 8)):
        try:
            generate_codes(model, v, CAPTION, None, seed=0)
        except ValueError as e:
            assert "did not write a valid plan" in str(e) and "duration: 31" in str(e)
        else:
            raise AssertionError("an invalid plan was accepted")
    with mock.patch.object(G, "sample", scripted(v.enc(" " + "bpm " * 300))):                         # never starts a segment
        assert raises(generate_codes, model, v, CAPTION, None, seed=0)
    with mock.patch.object(G, "sample", scripted(attempt(wrong_sum) * 8)):                           # greedy decoding would repeat itself: one attempt
        assert raises(generate_codes, model, v, CAPTION, None, temperature=0)


def test_documents_longer_than_the_backbone_can_position_are_refused():
    v, model = get_vocab(4, 10000, 25), model_for(4, 10000, 25)                                      # the tiny backbone has 4,096 positions
    assert raises(generate_codes, model, v, CAPTION, make_plan(300)) and raises(generate_codes, model, v, CAPTION, make_plan(60), temperature=-1)


if __name__ == "__main__":
    run_all(globals())
