import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_plan.py` run without installing the package

from dcttgen.plan import MIN_CLIP_S, POSITIONS, Plan, max_clip_seconds, plan_sections, validate_sections

EXAMPLE = "bpm: 80; duration: 150; sections: [intro] 30, [main] 90, [outro] 30; moods: uplifting, joyful; instruments: zither, two-string fiddle, moon-shaped lute"
ROW = {"clip_id": "yt_3fA9c_00", "recording_id": "yt_3fA9c", "position": "first", "audio": "audio/yt_3fA9c_00.flac", "duration": 270, "bpm": 80,
       "moods": ["uplifting", "joyful"], "instruments": ["zither", "two-string fiddle", "moon-shaped lute"],
       "sections": [["intro", 30], ["main", 240]], "caption": "x", "split": "train", "licence": "unknown-extra-field"}


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def rejects(fragment, fn, *args):                                  # ValueError whose message names the violated rule
    try:
        fn(*args)
    except ValueError as e:
        assert fragment in str(e), (fragment, str(e))
        return True
    raise AssertionError(f"accepted: {args!r}")


def plan(**kw):
    base = dict(bpm=80, duration=150, sections=[("intro", 30), ("main", 90), ("outro", 30)], moods=["uplifting", "joyful"],
                instruments=["zither", "two-string fiddle", "moon-shaped lute"])
    return Plan(**{**base, **kw})


def test_contract_table_of_plan_sections():                      # contract 8.3, every row
    assert plan_sections(150) == [("intro", 30), ("main", 90), ("outro", 30)]
    assert plan_sections(300) == [("intro", 30), ("main", 240), ("outro", 30)]
    assert plan_sections(30) == [("intro", 6), ("main", 18), ("outro", 6)]
    assert plan_sections(270, "first") == [("intro", 30), ("main", 240)]
    assert plan_sections(240, "middle") == [("main", 240)]
    assert plan_sections(100, "last") == [("main", 80), ("outro", 20)]
    assert raises(plan_sections, 300, "first") and raises(plan_sections, 29) and raises(plan_sections, 100, "nowhere")
    assert [max_clip_seconds(p) for p in POSITIONS] == [300, 270, 240, 270]


def test_every_legal_clip_has_a_valid_plan():                    # the manifest can never contain a row Plan refuses
    for position in POSITIONS:
        for d in range(MIN_CLIP_S, max_clip_seconds(position) + 1):
            secs = plan_sections(d, position)
            assert validate_sections(secs) == d
            p = Plan(120, d, secs, ["calm"], ["zither"])
            assert Plan.from_text(p.to_text()) == p


def test_text_is_the_contract_string_and_round_trips():
    assert plan().to_text() == EXAMPLE
    assert Plan.from_text(EXAMPLE) == plan()
    two = plan(duration=270, sections=[("intro", 30), ("main", 240)])
    assert Plan.from_text(two.to_text()) == two and "[outro]" not in two.to_text()


def test_malformed_or_invalid_plans_are_rejected():              # one violation per rule of contract 7.2 / 8.3
    bad_text = [
        (EXAMPLE.replace("bpm: 80", "bpm: 29"), "bpm must be"), (EXAMPLE.replace("bpm: 80", "bpm: 241"), "bpm must be"),
        (EXAMPLE.replace("bpm: 80", "bpm: 080"), "canonical"),
        (EXAMPLE.replace("duration: 150", "duration: 149"), "must equal the sum"),
        (EXAMPLE.replace("[intro] 30, [main] 90, [outro] 30", "[main] 90, [intro] 30, [outro] 30"), "sections must follow"),
        (EXAMPLE.replace("[outro] 30", "[main] 30"), "sections must follow"),
        (EXAMPLE.replace("[intro] 30", "[intro] 31").replace("150", "151"), "[intro] must be"),
        (EXAMPLE.replace("[main] 90", "[main] 241").replace("duration: 150", "duration: 301"), "[main] must be"),
        (EXAMPLE.replace("[main] 90", "[main] 0").replace("duration: 150", "duration: 60"), "[main] must be"),
        (EXAMPLE.replace("[intro] 30, [main] 90, [outro] 30", "[main] 20").replace("duration: 150", "duration: 20"), ">= 30"),
        (EXAMPLE.replace("zither", "guitar"), "instruments must be"), (EXAMPLE.replace("instruments: zither", "instruments: zither, zither"), "instruments must be"),
        (EXAMPLE.replace("uplifting, joyful", "a, b, c, d"), "moods must be"), (EXAMPLE.replace("moods: uplifting, joyful", "moods: "), "malformed"),
        (EXAMPLE.replace("uplifting", "Uplifting"), "moods must be"), (EXAMPLE.replace("uplifting, joyful", "uplifting,joyful"), "moods must be"),
        (EXAMPLE + " ", "instruments must be"), (EXAMPLE + "\n", "instruments must be"), (" " + EXAMPLE, "malformed"),
        (EXAMPLE + "; tempo: fast", "malformed"),
        (EXAMPLE.replace("bpm: 80", "bpm: ٨٠"), "malformed"),                   # Arabic-Indic digits
        ("bpm: 80; duration: 150; Sections: [intro] 30, [main] 90, [outro] 30 ; moods: uplifting, joyful; instruments: zither", "malformed"),   # the plan's printed spelling
        ("", "malformed"), ("hello", "malformed"),
    ]
    for t, why in bad_text:
        assert rejects(why, Plan.from_text, t), t
    assert rejects("malformed", Plan.from_text, None)
    for kw, why in ((dict(bpm=80.0), "bpm must be"), (dict(bpm=True), "bpm must be"), (dict(duration="150"), "must equal the sum"),
                    (dict(sections=[]), "non-empty"), (dict(sections=[("intro", 30.0), ("main", 90), ("outro", 30)]), "[intro] must be"),
                    (dict(moods=[]), "moods must be"), (dict(instruments=["gong"]), "instruments must be")):
        assert rejects(why, lambda: plan(**kw)), kw


def test_from_manifest_reads_a_contract_row_and_ignores_unknown_fields():
    p = Plan.from_manifest(ROW)
    assert p == Plan(80, 270, [("intro", 30), ("main", 240)], ["uplifting", "joyful"], ["zither", "two-string fiddle", "moon-shaped lute"])
    assert raises(Plan.from_manifest, {**ROW, "bpm": 80.5}) and raises(Plan.from_manifest, {k: v for k, v in ROW.items() if k != "bpm"})
    assert raises(Plan.from_manifest, {**ROW, "sections": [["main", 240]]})                 # sum != duration


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for f in tests:
        f()
    print(f"test_plan: {len(tests)} passed")
