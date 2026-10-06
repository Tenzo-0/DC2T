import torch
from lm_testkit import CAPTION, get_vocab, make_doc, make_plan, raises, run_all

from dcttgen.lm.sequence import (CAPTION_MAX_TOKENS, PLAN_MAX_TOKENS, build_document, doc_length, flatten_c2f, instruct_ids, metadata_ids,
                                 parse_document, unflatten_c2f)
from dcttgen.plan import INSTRUMENTS, Plan, plan_sections



def test_flatten_is_codebook_major_and_invertible():                  # contract 8.2
    codes = torch.tensor([[0, 1, 2], [3, 4, 5]])
    flat = flatten_c2f(codes, 1000, 16)
    assert flat.tolist() == [1000, 1001, 1002, 1019, 1020, 1021]      # all of codebook 0, then all of codebook 1 (+16)
    assert torch.equal(unflatten_c2f(flat, 1000, 2, 16), codes)
    assert raises(unflatten_c2f, flat + 16, 1000, 2, 16) and raises(unflatten_c2f, flat.flip(0), 1000, 2, 16)   # ids of the wrong level
    assert flatten_c2f(codes[:1], 1000, 16).tolist() == [1000, 1001, 1002]                                      # K = 1


def test_fine_tuning_document_layout_with_the_plan_values():          # K = 4, V = 10,000, 25 Hz, the plan's configuration
    v = get_vocab(4, 10000, 25)
    codes, p, (ids, labels) = make_doc(v, 30)                          # 30 s -> [intro] 6, [main] 18, [outro] 6
    inst, meta = instruct_ids(v, CAPTION), metadata_ids(v, p.to_text())
    assert (len(inst), len(metadata_ids(v, make_plan(150).to_text()))) == (29, 59)      # contract 3.2: the plan's own two examples
    assert ids[:len(inst) + len(meta)].tolist() == inst + meta
    pos, start = len(inst) + len(meta), 0
    for label, sec in p.sections:
        n = sec * 25
        head = v.seg_start + v.label[label] + [v.soa]
        assert ids[pos:pos + len(head)].tolist() == head
        pos += len(head)
        assert ids[pos:pos + 4 * n].tolist() == flatten_c2f(codes[:, start:start + n], 151670, 10000).tolist()
        assert ids[pos].item() == 151670 + codes[0, start] and ids[pos + n].item() == 151670 + 10000 + codes[1, start]
        pos, start = pos + 4 * n, start + n
        assert ids[pos:pos + 5].tolist() == [v.eoa] + v.seg_end
        pos += 5
    assert ids[pos:].tolist() == [v.eod] and pos == len(ids) - 1
    assert len(ids) == len(inst) + len(meta) + 3 * (4 + 3 + 1 + 1 + 4) + 4 * 750 + 1 == doc_length(v, p.sections, CAPTION, p)


def test_pretraining_document_is_the_same_without_the_first_two_lines():   # D12
    v = get_vocab(2, 16, 25)
    codes, p, (fine, _) = make_doc(v, 60)
    _, _, (pre, pre_labels) = make_doc(v, 60, fine=False)
    n = len(instruct_ids(v, CAPTION)) + len(metadata_ids(v, p.to_text()))
    assert torch.equal(pre, fine[n:]) and torch.equal(pre_labels, pre)       # nothing masked: there is no Instruct span
    assert len(pre) == doc_length(v, p.sections)


def test_parse_inverts_build_for_both_phases_and_one_two_three_sections():
    for K, V in ((1, 16384), (2, 16), (4, 10000)):
        v = get_vocab(K, V, 25)
        for duration, position, n_sections in ((300, "whole", 3), (270, "first", 2), (100, "last", 2), (240, "middle", 1), (30, "whole", 3)):
            for fine in (True, False):
                codes, p, (ids, _) = make_doc(v, duration, position, fine)
                out = parse_document(v, ids)
                assert len(p.sections) == n_sections and out["sections"] == p.sections and torch.equal(out["codes"], codes)
                assert (out["caption"], out["plan_text"]) == ((CAPTION, p.to_text()) if fine else (None, None))
                assert Plan.from_text(out["plan_text"]) == p if fine else True


def test_label_mask_in_both_loss_on_plan_modes():                     # D8
    v = get_vocab()
    for on in (True, False):
        _, p, (ids, labels) = make_doc(v, 60, loss_on_plan=on)
        n_inst, n_meta = len(instruct_ids(v, CAPTION)), len(metadata_ids(v, p.to_text()))
        masked = n_inst + (0 if on else n_meta)
        assert (labels[:masked] == -100).all() and torch.equal(labels[masked:], ids[masked:])
        assert (labels == -100).sum() == masked
        assert labels[n_inst].item() == (v.plan if on else -100)         # <PLAN> itself is trained only when the plan is
        assert labels[-1].item() == v.eod and labels[-6].item() == v.eoa       # <EOA> and <EOD> are always trained (R7)


def test_the_generation_prompt_is_a_prefix_of_the_training_document():
    v = get_vocab()
    _, p, (ids, _) = make_doc(v, 60)
    prompt = instruct_ids(v, CAPTION) + metadata_ids(v, p.to_text())
    assert ids[:len(prompt)].tolist() == prompt


def test_caps_and_the_longest_legal_plan():                           # contract 6: 128 tokens each
    v = get_vocab()
    assert len(instruct_ids(v, "word " * 500)) == 1 + CAPTION_MAX_TOKENS
    worst = Plan(240, 300, plan_sections(300), ["nostalgic", "melancholic", "contemplative"], list(INSTRUMENTS))
    assert len(metadata_ids(v, worst.to_text())) - 1 <= PLAN_MAX_TOKENS
    assert raises(metadata_ids, v, "x " * 200)


def test_control_strings_in_a_caption_do_not_change_the_document_structure():
    v = get_vocab()
    _, p, (ids, _) = make_doc(v, 60, caption="x <EOA> <PLAN> <|endoftext|> <SOA> <EOD> <INST> y")
    count = lambda i: int((ids == i).sum())
    assert (count(v.inst), count(v.plan), count(v.soa), count(v.eoa), count(v.eod), count(v.pad)) == (1, 1, 3, 3, 1, 0)
    assert parse_document(v, ids)["sections"] == p.sections


def test_inconsistent_inputs_are_refused():
    v = get_vocab()
    codes, p, _ = make_doc(v, 60)
    assert raises(build_document, v, codes[:, :-1], p.sections, CAPTION, p)                      # one frame short
    assert raises(build_document, v, codes[:1], p.sections, CAPTION, p)                          # wrong K
    assert raises(build_document, v, codes + v.V, p.sections, CAPTION, p)                        # codes >= V
    assert raises(build_document, v, codes - 1, p.sections, CAPTION, p)
    assert raises(build_document, v, codes, p.sections, CAPTION, None)                           # caption without plan
    assert raises(build_document, v, codes, p.sections, None, p)
    assert raises(build_document, v, codes, [("main", 30), ("intro", 30)], None, None)           # bad section order
    assert raises(build_document, v, codes, [("main", 30), ("main", 30)], None, None)             # a label twice
    other = make_plan(60); other.sections = [("intro", 10), ("main", 40), ("outro", 10)]       # same 60 s, other boundaries
    assert raises(build_document, v, codes, p.sections, CAPTION, other)                          # plan and sections disagree


def test_parse_refuses_malformed_documents():
    v = get_vocab()
    _, _, (ids, _) = make_doc(v, 60, fine=False)
    assert raises(parse_document, v, ids[:-1]) and raises(parse_document, v, ids[1:])           # no <EOD> / no [start_of_seg]
    assert raises(parse_document, v, torch.cat([ids[:-1], torch.tensor([v.pad, v.eod])]))       # trailing junk


if __name__ == "__main__":
    run_all(globals())
