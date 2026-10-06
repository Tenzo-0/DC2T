"""Run: pytest -q tests/test_data_captions.py   or   python tests/test_data_captions.py     (no network: the API client is a fake)"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcttgen.data.captions import cached_captions, caption_one, generate, template_caption, tempo_word, validate_caption  # noqa: E402

PLAN_EXAMPLE = "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute"
THREE = ["zither", "two-string fiddle", "moon-shaped lute"]
OK = "A calm Don ca tai tu piece played by zither and monochord"


def test_template_reproduces_the_plans_example():
    assert template_caption(["joyful", "uplifting"], THREE, 120) == PLAN_EXAMPLE
    assert template_caption(["emotional"], ["monochord"], 60) == "An emotional Don ca tai tu piece with slow tempo, performed by monochord"
    assert [tempo_word(b) for b in (69, 70, 109, 110)] == ["slow", "moderate", "moderate", "fast"]
    assert validate_caption(PLAN_EXAMPLE, THREE, ["joyful", "uplifting"]) == []


def test_validator_rejects_each_kind_of_bad_caption():
    inst = ["zither", "monochord"]
    assert validate_caption(OK, inst, ["calm"]) == []
    cases = [("A calm Don ca tai tu piece played by zither", "does not name the annotated instrument monochord"),
             (OK + " and piano", "not one of the annotated instruments"),
             (OK + " and bamboo flute", "not annotated for this clip"),
             (OK + " with a singer", "mentions vocals"),
             (OK.replace("Don ca tai tu ", ""), "must say 'Don ca tai tu'"),
             (OK + " <EOA>", "must not contain < or >"),
             (OK + "\nsecond line", "must be one line"),
             (OK.replace("calm", "cälm"), "plain ASCII"),
             (OK + " word" * 60, "at most 60 words"),
             ("", "empty"), (None, "empty")]
    for text, needle in cases:
        problems = validate_caption(text, inst, ["calm"])
        assert any(needle in p for p in problems), (text, problems)
    assert validate_caption(OK, inst, ["sad"]) == ["uses none of the annotated mood words"]


class FakeClient:
    """Stands in for openai.OpenAI(): .responses.create(**kw) returns the next scripted reply."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.responses = NS(create=self._create)

    def _create(self, **kw):
        self.requests.append(kw)
        return NS(output_text=self.replies.pop(0), usage=NS(input_tokens=100, output_tokens=20))


def reply(caption):
    return json.dumps({"caption": caption})


ROW = {"clip_id": "c0", "bpm": 60, "moods": ["calm"], "instruments": ["zither", "monochord"]}


def test_gpt_step_feeds_complaints_back_and_gives_up_cleanly():
    client = FakeClient([reply(OK + " and piano"), "not json", reply(OK)])
    rec = caption_one(ROW, client, "gpt-4o", 3, (70, 110))
    assert rec["source"] == "gpt" and rec["caption"] == OK and rec["attempts"] == 3
    assert rec["input_tokens"] == 300 and rec["output_tokens"] == 60
    assert "rejected" not in client.requests[0]["input"] and "piano" in client.requests[1]["input"]
    assert "not the requested JSON" in client.requests[2]["input"] and client.requests[0]["model"] == "gpt-4o"
    failed = caption_one(ROW, FakeClient([reply(OK + " and piano")] * 2), "gpt-4o", 2, (70, 110))
    assert failed["source"] == "failed" and failed["caption"] is None and failed["attempts"] == 2 and failed["problems"]


def test_cache_is_resumable_and_never_stale():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "work" / "captions.jsonl"
        rows = [ROW, {**ROW, "clip_id": "c1"}]
        opts = dict(model="gpt-4o", attempts=2, workers=1, bands=(70, 110))
        assert generate(rows, FakeClient([reply(OK), reply(OK + " and piano"), reply(OK + " and piano")]), cache, **opts) == 2
        got = cached_captions(rows, cache, "gpt-4o", (70, 110))
        assert len(got) == 1 and set(got.values()) == {OK}                          # the failed clip has no caption
        assert generate(rows, FakeClient([reply(OK)]), cache, **opts) == 1         # only the failed clip is asked again
        assert cached_captions(rows, cache, "gpt-4o", (70, 110)) == {"c0": OK, "c1": OK}
        assert generate(rows, FakeClient([]), cache, **opts) == 0                  # nothing left to do: no request is made
        changed = [{**ROW, "moods": ["sad"]}, rows[1]]                             # the clip was re-annotated
        assert cached_captions(changed, cache, "gpt-4o", (70, 110)) == {"c1": OK}  # its old caption is not reused
        assert cached_captions(rows, cache, "another-model", (70, 110)) == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"test_data_captions: {len(tests)} tests passed")
