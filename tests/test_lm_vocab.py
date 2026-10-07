import dataclasses
import tempfile
from unittest import mock

from lm_testkit import get_vocab, make_cfg, run_all

from transformers import AutoTokenizer

from dc2t.lm import vocab as vocab_mod
from dc2t.lm.vocab import Vocab

CONTROL_STRINGS = ["<EOA>", "<PLAN>", "<EOD>", "<SOA>", "<INST>", "<|endoftext|>", "<|im_start|>", "a<EOA>b <PLAN> c<|endoftext|>"]
CAPTIONS = ["A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute",
            "", " ", "  leading", "trailing ", "line1\nline2", "\ttab", "Đờọn ca tài tử miền Nam", "emoji \U0001f3b5 中文", "120 bpm 3/4"]


def test_layout_matches_contract_8_1():
    v = get_vocab(4, 10000, 25)
    assert (v.eod, v.soa, v.eoa, v.inst, v.plan, v.pad) == (151665, 151666, 151667, 151668, 151669, 151643)
    assert v.tokenizer.convert_ids_to_tokens(list(range(151665, 151670))) == ["<EOD>", "<SOA>", "<EOA>", "<INST>", "<PLAN>"]
    assert v.audio_offset == len(v.tokenizer) == 151670 and v.size == 191670 and (v.K, v.V, v.frame_rate) == (4, 10000, 25)
    assert get_vocab(1, 16384, 25).size == 168054                                   # bootstrap configuration
    assert [len(v.seg_start), len(v.seg_end)] + [len(v.label[s]) for s in ("intro", "main", "outro")] == [4, 4, 3, 3, 3]   # contract 3.2
    assert all(i < 151643 for ids in [v.seg_start, v.seg_end, *v.label.values()] for i in ids)


def test_a_different_order_of_the_special_tokens_is_refused():
    with mock.patch.object(vocab_mod, "SPECIALS", vocab_mod.SPECIALS[::-1]):
        try:
            Vocab.build(make_cfg())
        except AssertionError as e:
            assert "contract 8.1" in str(e)
        else:
            raise AssertionError("a wrong layout was accepted")


def test_untrusted_text_never_becomes_control_tokens():
    v = get_vocab()
    for text in CONTROL_STRINGS:
        ids = v.enc(text)
        assert max(ids) < 151643 and v.tokenizer.decode(ids) == text, text           # plain text, and nothing is lost
        assert max(v.tokenizer.encode(text, add_special_tokens=False)) >= 151643      # the default call WOULD make control ids: the test is not vacuous
    class Leaky:                                                                      # a tokenizer that ignores split_special_tokens
        vocab_size = 151643
        def encode(self, text, add_special_tokens, split_special_tokens):
            return [151667]
    try:
        dataclasses.replace(v, tokenizer=Leaky()).enc("hi")
    except ValueError as e:
        assert "control token" in str(e)
    else:
        raise AssertionError("a control id from text was accepted")


def test_inst_plus_encoded_caption_equals_one_call():                                # contract 8.4: [<INST>] + enc(" " + caption)
    v = get_vocab()
    for c in CAPTIONS:
        assert [v.inst] + v.enc(" " + c) == v.tokenizer.encode("<INST> " + c, add_special_tokens=False), repr(c)


def test_tokenizer_and_layout_are_saved_with_the_run():
    v = get_vocab()
    with tempfile.TemporaryDirectory() as d:
        run = f"{d}/lm_pretrain/x"
        v.save(run)
        tok = AutoTokenizer.from_pretrained(f"{run}/tokenizer")
        assert len(tok) == 151670 and tok.convert_tokens_to_ids("<PLAN>") == 151669
        v.check_checkpoint(f"{run}/step_0000100")                                     # same codec: accepted
        for other in (get_vocab(3, 16), get_vocab(2, 32), dataclasses.replace(v, codec_tag="another_rvq"), dataclasses.replace(v, frame_rate=50)):
            try:
                other.check_checkpoint(f"{run}/step_0000100")
            except ValueError as e:
                assert "was trained with" in str(e)
            else:
                raise AssertionError("a checkpoint of another codec was accepted")
        try:
            v.check_checkpoint(f"{d}/elsewhere/step_0000100")
        except ValueError as e:
            assert "vocab.json" in str(e)
        else:
            raise AssertionError("a checkpoint without vocab.json was accepted")


if __name__ == "__main__":
    run_all(globals())
