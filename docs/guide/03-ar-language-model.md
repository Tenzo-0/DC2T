# DcttGen Implementation Guide — 03. Autoregressive Language Model

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this chapter task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a Qwen2.5 model that turns `<INST> caption <PLAN> plan` into a `[K, 25 × duration]` matrix of audio codes, with its training entry point for both phases.
**Architecture:** stock Qwen2.5 from `transformers` with the embedding matrix enlarged by `K × V` audio rows; documents are built by concatenating id lists; the loss scores every target only against the rows it is allowed to be; decoding walks a schedule derived from the plan.
**Tech stack:** Python 3.10+, `torch`, `transformers` 4.57.1 (the version every test here ran on), `safetensors`, `numpy`. No other dependency.
**Spec:** plan §2.3, §2.4, §3.2 (text), §3.3, §3.6; Figures 3 and 4 · contract: [00 — Overview, Spec and Shared Contracts](00-overview-and-contracts.md)

## Global constraints

- The vocabulary layout, the document layout, the label mask and the decoding schedule are fixed by contract [§8](00-overview-and-contracts.md#8-token-and-sequence-contract). This chapter implements them; it does not change them.
- `K`, `V`, the frame rate and every token id are read from `Vocab`. The literals `151670` and `151643` appear once, in the assertion inside `Vocab.build`.
- Structure strings are tokenised alone, once, and concatenated as id lists — never as text (contract §3.2).
- Caption and plan text are capped at 128 tokens each. A caption over the cap is truncated; a plan over the cap is a bug and raises.
- The model's built-in loss (`labels=`) and `model.generate()` are **not** used; §1.3 and §1.5 say why.
- Trainable parameters stay `float32`; half precision comes from the engine's autocast (chapter 04).
- Tests: plain `assert`, CPU only, tiny random Qwen2 (hidden 32, 2 layers) with the **real** Qwen2.5 tokenizer and the real 151,936 embedding rows. They need the tokenizer files of `Qwen/Qwen2.5-0.5B` in the Hugging Face cache (a few megabytes, fetched once; no model weights).

## Review focus

The inputs most likely to hurt a real user that a naive test suite would miss. Each is pinned by a named test.

| # | Input or condition | Expected behaviour | Test |
|---|---|---|---|
| 1 | A caption containing the literal text `<EOA>`, `<PLAN>` or `<|endoftext|>` | It is tokenised as ordinary characters; the document structure and the decoded codes are unaffected | `test_untrusted_text_never_becomes_control_tokens`, `test_control_strings_in_a_caption_do_not_change_the_document_structure`, `test_control_strings_in_the_caption_do_not_disturb_decoding` |
| 2 | Codes written by a different codec (other `K`, `V`, frame rate, or simply another RVQ with the same shape) | Refused at load time, not discovered as noise after a week of training | `test_codes_of_another_shape_dtype_or_range_are_refused`, `test_checkpoint_round_trip_and_codec_guard` |
| 3 | The model writes a malformed or impossible plan in self-planned mode | Retried, then a clear error — never a crash inside the decoder or a piece of the wrong length | `test_an_invalid_plan_is_retried_and_after_eight_attempts_refused`, `test_malformed_or_invalid_plans_are_rejected` |
| 4 | A document longer than the backbone can position (30,300 tokens against 32,768; any `max_seconds` change) | Refused before training starts and before generation starts | `test_main_refuses_documents_longer_than_the_backbone_positions`, `test_documents_longer_than_the_backbone_can_position_are_refused` |
| 5 | Padding in a batch of documents of different lengths | The loss is identical to the unpadded loss | `test_padding_does_not_change_the_loss` |

## 1. Background and design

### 1.1 What the plan asks for

**Plan §3.3.1** describes the backbone: grouped-query attention, SwiGLU, rotary position embedding, QKV bias, pre-norm RMSNorm. All five are properties of the stock Qwen2.5 architecture, so **nothing in §3.3.1 is implemented by hand**. You can see each of them in the loaded config or module:

| Plan §3.3.1 item | Where it shows in `transformers` (**Verified** on `Qwen/Qwen2.5-0.5B`, contract §3.2) |
|---|---|
| Grouped-query attention | `config.num_attention_heads = 14`, `config.num_key_value_heads = 2` |
| SwiGLU | `config.hidden_act = "silu"` with a gated MLP of width `config.intermediate_size = 4864` |
| Rotary position embedding | no position-embedding table in the state dict; `config.rope_theta` |
| QKV bias | `q_proj`, `k_proj`, `v_proj` have a `bias` parameter |
| Pre-norm RMSNorm | `config.rms_norm_eps = 1e-6`; a norm before attention and before the MLP |

**Plan §3.3.2** defines the document and the coarse-to-fine order; contract §2.2 has the equations and §8 the exact layout. **Plan §3.6** defines two training phases and the decoding rule.

### 1.2 Data flow

```
manifest row + codes/<tag>/<clip_id>.npy  int16 [K, 25*duration]
        |  ClipDataset.__getitem__  ->  build_document
        v
input_ids Long[S], labels Long[S]            S <= about 30,300
        |  TokenBudgetSampler + collate (right-pad)
        v
batch: input_ids, attention_mask, labels     each Long[B, S]
        |  MusicLM.forward  (Qwen2Model -> hidden states Float[B, S, d])
        v
restricted_loss: one cross-entropy per id range   ->  {"loss", "ce_text", "ce_k0" ... "ce_k{K-1}", "tokens"}

prompt + Plan  --generate_codes-->  schedule (allowed id range of every position)  -->  codes Long[K, 25*duration]
```

### 1.3 The loss — why not `labels=`

A naive causal-LM loss computes logits over the whole vocabulary at every position:

`30,000 positions × 191,670 tokens × 4 bytes = 23 GB` of logits for one five-minute document, before the softmax and its gradient double it.

**Decision — restricted cross-entropy.** Decoding is schedule-constrained (contract D9), so at every audio position the target is known to lie in one codebook's range. The loss therefore scores the hidden state against **that range's rows only**:

`30,000 × 10,000 × 4 bytes = 1.2 GB`.

Positions whose target is a text, structure or special token (a few hundred per document) are scored against rows `[0, audio_offset)`. `test_restricted_loss_equals_the_full_vocabulary_loss_with_the_same_mask` proves the equivalence: the restricted loss equals the full-vocabulary cross-entropy after masking the same columns to `-inf`.

What this implies: the model is never trained to *avoid* an audio id at a text position, or the wrong codebook at an audio position. That is exactly what the decoder guarantees by construction — and it means **this model cannot be decoded with the plan's literal rule** ("audio range until `<EOA>` is predicted", plan §3.6) without a schedule. If you need schedule-free decoding, change the `ranges` list in `restricted_loss` to one range covering all audio ids plus `<EOA>` (40,001 columns, about 4.8 GB per document) and accept the memory cost.

The same function gives per-level logs for free: `ce_k0 … ce_k3` and `ce_text`. At initialisation each level's cross-entropy is `ln V` (9.21 for V = 10,000) and the text loss is about `ln 151,670 = 11.93` — `test_initial_cross_entropy_of_every_level_is_ln_V` checks this, and the cross-chapter integration test in chapter 04 sees the same numbers in a real training log.

### 1.4 New embedding rows

Qwen2.5's embedding matrix has 151,936 rows but only 151,665 are used (contract §3.2). Our five special tokens land on existing-but-untrained rows, and the audio tokens straddle old and new rows. **Decision:** shrink the matrix to 151,665 rows, then grow it to `vocab.size`, so that every row from `<EOD>` onwards is created by the same call and starts from the same distribution. `transformers` initialises new rows from a normal distribution with the mean of the old rows and a tiny covariance (**Verified** by `test_new_rows_are_initialised_uniformly`), so at step 0 all audio rows are nearly identical — which is why every level starts at exactly `ln V`.

Input and output embeddings are tied (`config.tie_word_embeddings = True`), so the output matrix grows with the input matrix and `lm_head.weight` is absent from saved checkpoints (contract §7.4). `load_lm` loads with `safetensors.torch.load_model`, which is strict but knows which missing keys are tied aliases.

### 1.5 Decoding — why not `model.generate()`

**Decision — a 60-line explicit loop.** The whole layout after the prompt is known in advance (contract §8.5), so decoding is: prefill the prompt, then for each scheduled position either *feed* the forced token or *sample* from `F.linear(h, weight[lo:hi])` — 10,000 dot products instead of 191,670 — and feed the result. `model.generate()` would compute full-vocabulary logits at each of 30,000 steps, apply its own length and stopping defaults, and need a logits processor to re-derive the position on every call. `test_greedy_cached_decoding_equals_the_argmax_of_one_full_forward` shows the cached loop gives the same result as one uncached forward pass.

**Plan §2.2 writes `x̂ = argmax p`.** Greedy decoding of audio tokens collapses into loops and held notes. **Decision:** sample by default (`infer.temperature: 1.0`, `infer.top_k: 250`, both in `configs/base.yaml`); `temperature=0` gives the plan's argmax. Treat the two numbers as a starting point to tune by ear.

**Self-planned mode** (`plan=None`): the model samples plan text from ordinary text ids only, until it writes `[start_of_seg]`; the text is parsed by the strict `Plan.from_text`; an invalid plan is resampled, up to 8 times, then `ValueError`. A valid model-written plan is re-encoded canonically before decoding, so the prompt the model continues from is byte-identical to a training prompt (`test_the_generation_prompt_is_a_prefix_of_the_training_document`).

### 1.6 Decisions in this chapter

| | Decision | Reason | How to change |
|---|---|---|---|
| 3.1 | Restricted cross-entropy (§1.3) | 23 GB → 1.2 GB per document | edit `ranges` in `restricted_loss` |
| 3.2 | Shrink-then-grow embedding resize (§1.4) | uniform initialisation of all new rows | `resize_for_audio` |
| 3.3 | Explicit decoding loop (§1.5) | 19× fewer dot products per step; no fight with `generate()` defaults | — |
| 3.4 | Sampling, not argmax, by default | greedy audio decoding degenerates | `infer.temperature: 0` |
| 3.5 | `Vocab` also carries `frame_rate`, `codec_tag`, the structure id lists, and writes `vocab.json` next to the checkpoints | a checkpoint must refuse a config with another codec | — |
| 3.6 | Batches are built by a token budget (`lm.max_tokens`), documents sorted by length | documents range from 3,000 to 30,000 tokens; fixed batch sizes waste most of the batch on padding | `lm.max_tokens` |
| 3.7 | Fine-tuning uses only clips with a caption; pre-training uses every clip | contract §7.2 allows `caption: null` for pre-training | — |

## 2. File map

| File | Responsibility |
|---|---|
| `dcttgen/plan.py` | `plan_sections`, `max_clip_seconds` (contract §8.3, unchanged), `Plan` with strict text round trip |
| `dcttgen/lm/vocab.py` | `Vocab`: tokenizer, the five special ids, audio id arithmetic, safe text encoding, checkpoint guard |
| `dcttgen/lm/sequence.py` | `flatten_c2f`, `unflatten_c2f`, `build_document`, `doc_length`, `parse_document` |
| `dcttgen/lm/model.py` | `resize_for_audio`, `restricted_loss`, `MusicLM`, `load_lm` |
| `dcttgen/lm/data.py` | `ClipDataset`, `collate`, `TokenBudgetSampler` |
| `dcttgen/lm/generate.py` | `build_schedule`, sampling, `generate_codes` |
| `dcttgen/lm/train.py` | entry point for both phases; `--probe` |
| `tests/lm_testkit.py` | shared test helpers: tiny Qwen2, config namespace, synthetic dataset |
| `tests/test_plan.py`, `tests/test_lm_*.py` | 45 tests |

## 3. Tasks

Every task follows the same five steps. Run a test file either way: `pytest -q tests/test_x.py` or `python tests/test_x.py`.

### Task 1: sections and the plan text

**Files:** Create `dcttgen/plan.py` · Test `tests/test_plan.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `max_clip_seconds(position) -> int`, `plan_sections(duration, position) -> list[tuple[str, int]]`, `Plan(bpm, duration, sections, moods, instruments)` with `to_text()`, `from_text(text)`, `from_manifest(row)`; `validate_sections(sections) -> int`; constants `INSTRUMENTS`, `POSITIONS`, `MIN_CLIP_S`, `SECTION_ORDER`. Chapter 01 imports `plan_sections`, `max_clip_seconds`, `Plan` and `INSTRUMENTS`.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_plan.py</code> — 104 lines (click to expand)</summary>

```python
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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_plan.py` → `ModuleNotFoundError: No module named 'dcttgen.plan'`
- [ ] **Step 3: implement.** The first two functions are the contract's code, unchanged. `Plan.__post_init__` enforces every rule of contract §7.2 that concerns plan fields, so *an instance that exists is valid*. `from_text` accepts only the canonical rendering: it parses, rebuilds, and compares with the input, which rejects leading zeros, extra spaces and reordered fields.

**`dcttgen/plan.py`** — 105 lines

```python
"""Sections and the plan text. The first two functions are contract 8.3, copied unchanged."""
from __future__ import annotations

import re
from dataclasses import dataclass

EDGE_MAX_S, MAIN_MAX_S, MIN_CLIP_S = 30, 240, 30          # Plan §3.3.2; MIN_CLIP_S is Decision D4
POSITIONS = ("whole", "first", "middle", "last")


def max_clip_seconds(position: str = "whole") -> int:
    return MAIN_MAX_S + EDGE_MAX_S * (position in ("whole", "first")) + EDGE_MAX_S * (position in ("whole", "last"))


def plan_sections(duration: int, position: str = "whole") -> list[tuple[str, int]]:
    if position not in POSITIONS:
        raise ValueError(f"unknown position {position!r}")
    if not MIN_CLIP_S <= duration <= max_clip_seconds(position):
        raise ValueError(f"duration {duration}s is outside [{MIN_CLIP_S}, {max_clip_seconds(position)}] for position={position}")
    edge = min(EDGE_MAX_S, duration // 5)
    intro = edge if position in ("whole", "first") else 0
    outro = edge if position in ("whole", "last") else 0
    parts = [("intro", intro), ("main", duration - intro - outro), ("outro", outro)]
    return [(name, sec) for name, sec in parts if sec > 0]


# ---- contract 7.2 field rules, used by Plan below and by sequence.build_document -------------------------
SECTION_ORDER = ("intro", "main", "outro")
SECTION_MAX_S = {"intro": EDGE_MAX_S, "main": MAIN_MAX_S, "outro": EDGE_MAX_S}
INSTRUMENTS = ("moon-shaped lute", "two-string fiddle", "zither", "monochord", "bamboo flute", "gong ban")
BPM_MIN, BPM_MAX, MAX_MOODS = 30, 240, 3
_MOOD = re.compile(r"[a-z]+(?:[ -][a-z]+)*")      # one lowercase term; never contains , ; : [ ] so the text stays parseable
_TEXT = re.compile(
    r"bpm: ([0-9]{1,3}); duration: ([0-9]{1,3}); sections: (\[[a-z]+\] [0-9]{1,3}(?:, \[[a-z]+\] [0-9]{1,3})*); "
    r"moods: ([^;:\[\]]+); instruments: ([^;:\[\]]+)"
)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_sections(sections) -> int:
    """Contract 7.2 `sections` rule: labels in intro < main < outro order, each at most once, 1 <= seconds <= its cap.
    Returns the total number of seconds."""
    if not isinstance(sections, (list, tuple)) or not sections or not all(isinstance(s, (list, tuple)) and len(s) == 2 for s in sections):
        raise ValueError(f"sections must be a non-empty list of (label, seconds) pairs, got {sections!r}")
    last = -1
    for label, sec in sections:
        if label not in SECTION_ORDER:
            raise ValueError(f"unknown section label {label!r}")
        if SECTION_ORDER.index(label) <= last:
            raise ValueError(f"sections must follow {SECTION_ORDER}, each at most once: {[s[0] for s in sections]}")
        last = SECTION_ORDER.index(label)
        if not _is_int(sec) or not 1 <= sec <= SECTION_MAX_S[label]:
            raise ValueError(f"[{label}] must be an integer in 1..{SECTION_MAX_S[label]} seconds, got {sec!r}")
    return sum(sec for _, sec in sections)


@dataclass
class Plan:
    """The `<PLAN>` metadata. An instance that exists satisfies contract 7.2 (checked in __post_init__)."""
    bpm: int
    duration: int
    sections: list[tuple[str, int]]
    moods: list[str]
    instruments: list[str]

    def __post_init__(self):
        total = validate_sections(self.sections)
        self.sections = [(a, b) for a, b in self.sections]
        if not _is_int(self.bpm) or not BPM_MIN <= self.bpm <= BPM_MAX:
            raise ValueError(f"bpm must be an integer in {BPM_MIN}..{BPM_MAX}, got {self.bpm!r}")
        if self.duration != total or total < MIN_CLIP_S:
            raise ValueError(f"duration {self.duration!r} must equal the sum of the sections ({total}) and be >= {MIN_CLIP_S}")
        self.moods, self.instruments = list(self.moods), list(self.instruments)
        if not 1 <= len(self.moods) <= MAX_MOODS or len(set(self.moods)) != len(self.moods) \
                or not all(isinstance(m, str) and _MOOD.fullmatch(m) for m in self.moods):
            raise ValueError(f"moods must be 1..{MAX_MOODS} distinct lowercase terms, got {self.moods!r}")
        if not self.instruments or len(set(self.instruments)) != len(self.instruments) \
                or not all(i in INSTRUMENTS for i in self.instruments):
            raise ValueError(f"instruments must be a non-empty set drawn from {INSTRUMENTS}, got {self.instruments!r}")

    def to_text(self) -> str:                      # contract 8.3, without the "<PLAN> " prefix
        secs = ", ".join(f"[{name}] {sec}" for name, sec in self.sections)
        return (f"bpm: {self.bpm}; duration: {self.duration}; sections: {secs}; "
                f"moods: {', '.join(self.moods)}; instruments: {', '.join(self.instruments)}")

    @staticmethod
    def from_text(text: str) -> "Plan":            # strict: only the canonical rendering of a valid plan is accepted
        m = _TEXT.fullmatch(text) if isinstance(text, str) else None
        if m is None:
            raise ValueError(f"malformed plan text: {text!r}")
        sections = [(a, int(b)) for a, b in re.findall(r"\[([a-z]+)\] ([0-9]+)", m.group(3))]
        plan = Plan(int(m.group(1)), int(m.group(2)), sections, m.group(4).split(", "), m.group(5).split(", "))
        if plan.to_text() != text:                 # leading zeros and other spellings of the same numbers
            raise ValueError(f"plan text is not in canonical form: {text!r}")
        return plan

    @staticmethod
    def from_manifest(row: dict) -> "Plan":
        try:
            return Plan(row["bpm"], row["duration"], [tuple(s) for s in row["sections"]], list(row["moods"]), list(row["instruments"]))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"manifest row {row.get('clip_id', '?')!r}: {e!r}") from None
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_plan.py` → `test_plan: 5 passed`
- [ ] **Step 5: commit** — `git add dcttgen/plan.py tests/test_plan.py && git commit -m "feat: plan sections and strict plan text"`

### Task 2: vocabulary

**Files:** Create `dcttgen/lm/__init__.py` (empty), `dcttgen/lm/vocab.py`, `tests/lm_testkit.py` · Test `tests/test_lm_vocab.py`

**Interfaces:**
- Consumes: `cfg.lm.backbone`, `cfg.codec.num_codebooks`, `cfg.codec.codebook_size`, `cfg.codec.tag`, `cfg.audio.frame_rate`.
- Produces: `Vocab.build(cfg)`; fields `tokenizer, eod, soa, eoa, inst, plan, pad, audio_offset, K, V, frame_rate, codec_tag, seg_start, seg_end, label`; property `size`; `enc(text) -> list[int]`; `save(run_dir)`; `check_checkpoint(path)`.

**Untrusted text.** `tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)` makes the tokenizer treat `<EOA>`, `<|endoftext|>` and every other control string as plain characters. **Verified** with `Qwen2TokenizerFast` in `transformers` 4.57.1 by `test_untrusted_text_never_becomes_control_tokens`. `enc` additionally refuses any id at or above the 151,643 regular tokens, so a future change in the library cannot silently reopen the hole.

**One call or two.** `[<INST>] + enc(" " + caption)` gives the same ids as encoding `"<INST> " + caption` in one call — **Verified** by `test_inst_plus_encoded_caption_equals_one_call`. The guide uses the two-part form because it is the only one that is safe for untrusted captions.

- [ ] **Step 1: write the test helpers and the failing test**

<details>
<summary><code>tests/lm_testkit.py</code> — 105 lines (click to expand)</summary>

```python
"""Shared helpers for the LM tests: a config namespace, the real Qwen2.5 tokenizer (from the Hugging Face cache) and a tiny random Qwen2."""
import atexit
import json
import pathlib
import shutil
import sys
import tempfile
from functools import lru_cache
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))     # lets `python tests/test_x.py` run without installing the package

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from dcttgen.lm.sequence import build_document
from dcttgen.lm.vocab import Vocab
from dcttgen.plan import Plan, plan_sections

CAPTION = "A joyful and uplifting Don ca tai tu piece with fast tempo, performed by zither, two-string fiddle, and moon-shaped lute"


@lru_cache(maxsize=None)
def backbone_dir() -> str:
    """A random Qwen2 (hidden 32, 2 layers, the real 151,936 embedding rows) saved next to the real Qwen2.5 tokenizer files.
    `from_pretrained(backbone_dir())` then behaves like `from_pretrained("Qwen/Qwen2.5-0.5B")`, without any weights."""
    d = tempfile.mkdtemp(prefix="tiny_qwen_")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=4096, rope_theta=1e6, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    return d


def make_cfg(K=2, V=16, frame_rate=25, data_root="data", **lm):
    return NS(paths=NS(data_root=data_root, runs="runs"), audio=NS(sample_rate=32000, frame_rate=frame_rate),
              codec=NS(tag=f"k{K}v{V}", num_codebooks=K, codebook_size=V),
              lm=NS(**{**dict(backbone=backbone_dir(), loss_on_plan=True, checkpoint=None, attn="sdpa", grad_checkpointing=False,
                              max_tokens=4096, max_seconds=None, num_workers=0, temperature=1.0, top_k=None, top_p=None), **lm}))


@lru_cache(maxsize=None)
def get_vocab(K=2, V=16, frame_rate=25) -> Vocab:
    return Vocab.build(make_cfg(K, V, frame_rate))


def random_rows_model(K=2, V=16, frame_rate=1, seed=1):
    """load_lm, then random new rows. At initialisation every row of a codebook is the same, which would make most tests vacuous."""
    from dcttgen.lm.model import load_lm
    v = get_vocab(K, V, frame_rate)
    model = load_lm(make_cfg(K, V, frame_rate), v)
    torch.manual_seed(seed)
    with torch.no_grad():
        model.head_weight[v.eod:] += 0.5 * torch.randn_like(model.head_weight[v.eod:])
    return model


def make_plan(duration, position="whole", **kw):
    return Plan(**{**dict(bpm=80, duration=duration, sections=plan_sections(duration, position), moods=["uplifting", "joyful"],
                          instruments=["zither", "two-string fiddle", "moon-shaped lute"]), **kw})


def make_doc(v, duration=150, position="whole", fine=True, loss_on_plan=True, caption=CAPTION, seed=0):
    """-> (codes Long[K, T], plan, (input_ids, labels)) for random codes."""
    p = make_plan(duration, position)
    codes = torch.randint(0, v.V, (v.K, duration * v.frame_rate), generator=torch.Generator().manual_seed(seed))
    return codes, p, build_document(v, codes, p.sections, caption if fine else None, p if fine else None, loss_on_plan=loss_on_plan)


def raises(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return True
    return False


CLIPS = [("whole", 150), ("first", 270), ("middle", 120), ("last", 90), ("whole", 30), ("first", 60)]      # (position, seconds)


def write_dataset(root, caption_none=(), n=6, K=2, V=16, fr=1, clips=CLIPS):
    """A synthetic data/ tree in the contract 7.2 / 7.3 formats: manifest/{train,val}.jsonl and codes/k{K}v{V}/<clip_id>.npy (int16 [K, fr * duration])."""
    root = pathlib.Path(root)
    (root / "manifest").mkdir(parents=True)
    (root / "codes" / f"k{K}v{V}").mkdir(parents=True)
    rng, rows = np.random.default_rng(0), []
    for i in range(n):
        position, duration = clips[i % len(clips)]
        rows.append({"clip_id": f"c{i}", "recording_id": f"r{i // 2}", "position": position, "audio": f"audio/c{i}.flac", "duration": duration, "bpm": 80,
                     "moods": ["calm"], "instruments": ["zither"], "sections": [list(s) for s in plan_sections(duration, position)],
                     "caption": None if i in caption_none else "A calm piece for zither", "split": "train"})
        np.save(root / "codes" / f"k{K}v{V}" / f"c{i}.npy", rng.integers(0, V, (K, duration * fr)).astype(np.int16))
    for name, part in (("train", rows), ("val", rows[:2])):                  # val reuses two clips: the loaders only need a readable file
        (root / "manifest" / f"{name}.jsonl").write_text("".join(json.dumps({**r, "split": name}) + "\n" for r in part), encoding="utf-8")
    return rows


def run_all(ns):
    tests = [f for n, f in sorted(ns.items()) if n.startswith("test_") and callable(f)]
    for f in tests:
        f()
    print(f"{pathlib.Path(ns['__file__']).stem}: {len(tests)} passed")
```

</details>

<details>
<summary><code>tests/test_lm_vocab.py</code> — 85 lines (click to expand)</summary>

```python
import dataclasses
import tempfile
from unittest import mock

from lm_testkit import get_vocab, make_cfg, run_all

from transformers import AutoTokenizer

from dcttgen.lm import vocab as vocab_mod
from dcttgen.lm.vocab import Vocab

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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_lm_vocab.py` → `ModuleNotFoundError` (no `dcttgen.lm` module exists yet)
- [ ] **Step 3: implement**

**`dcttgen/lm/vocab.py`** — 76 lines

```python
"""Vocabulary layout, contract 8.1: Qwen2.5 tokens, five added special tokens, then K * V audio ids."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

from dcttgen.plan import SECTION_ORDER

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
```

- [ ] **Step 4: run it, expect a pass** — the shared test kit imports `sequence.py`, so this test first runs after Task 3's Step 3: `python tests/test_lm_vocab.py` → `test_lm_vocab: 5 passed`
- [ ] **Step 5: commit** — `git add dcttgen/lm tests/lm_testkit.py tests/test_lm_vocab.py && git commit -m "feat: vocabulary layout with safe text encoding"`

### Task 3: documents

**Files:** Create `dcttgen/lm/sequence.py` · Test `tests/test_lm_sequence.py`

**Interfaces:**
- Consumes: `Vocab`, `Plan`, `validate_sections`.
- Produces: `flatten_c2f(codes, audio_offset, V)`, `unflatten_c2f(ids, audio_offset, K, V)` (contract §8.2); `build_document(vocab, codes, sections, caption, plan, *, loss_on_plan=True) -> (input_ids, labels)` (contract §9); `doc_length(vocab, sections, caption, plan) -> int`; `parse_document(vocab, ids) -> dict`; `instruct_ids`, `metadata_ids`; `CAPTION_MAX_TOKENS = PLAN_MAX_TOKENS = 128`.

`parse_document` is the exact inverse of `build_document`. It is not only a test helper: `generate_codes` runs every generated sequence through it, so a decoder bug surfaces as a parse error instead of as wrong audio.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_lm_sequence.py</code> — 119 lines (click to expand)</summary>

```python
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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_lm_sequence.py` → `ModuleNotFoundError: No module named 'dcttgen.lm.sequence'`
- [ ] **Step 3: implement**

**`dcttgen/lm/sequence.py`** — 111 lines

```python
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_lm_sequence.py` → `test_lm_sequence: 10 passed`
- [ ] **Step 5: commit** — `git add dcttgen/lm/sequence.py tests/test_lm_sequence.py && git commit -m "feat: Music Chain-of-Thought documents"`

### Task 4: the model and its loss

**Files:** Create `dcttgen/lm/model.py` · Test `tests/test_lm_model.py`

**Interfaces:**
- Consumes: `Vocab`; `cfg.lm.backbone`, `cfg.lm.checkpoint`, `cfg.lm.attn`, `cfg.lm.grad_checkpointing`.
- Produces: `load_lm(cfg, vocab, checkpoint=None) -> MusicLM` (contract §9); `MusicLM.forward(batch) -> {"loss", "tokens", "ce_text", "ce_k0", …}`; properties `backbone` (the `Qwen2Model`) and `head_weight` (`Float[size, d]`); `restricted_loss(h, tgt, weight, audio_offset, K, V)`.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_lm_model.py</code> — 134 lines (click to expand)</summary>

```python
import dataclasses
import math
import pathlib
import tempfile

import torch
import torch.nn.functional as F
from lm_testkit import CAPTION, backbone_dir, get_vocab, make_cfg, make_doc, raises, random_rows_model, run_all
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from dcttgen.lm.data import collate
from dcttgen.lm.generate import build_schedule
from dcttgen.lm.model import load_lm, resize_for_audio
from dcttgen.lm.sequence import instruct_ids, metadata_ids

# K = 2 codebooks of 16 codes at 1 frame per second: a 30 s document has ~250 tokens, so the reference below can afford [B, S, 151,702] logits
CFG, V = make_cfg(2, 16, 1), get_vocab(2, 16, 1)


def make_batch(specs):
    """specs: (duration, loss_on_plan) per document -> (padded batch, the range each position's target is scored against)."""
    docs, ranges = [], []
    for duration, on in specs:
        _, plan, (ids, labels) = make_doc(V, duration, loss_on_plan=on, seed=duration)
        docs.append({"input_ids": ids, "labels": labels})
        r = [(0, V.audio_offset)] * (len(instruct_ids(V, CAPTION)) + len(metadata_ids(V, plan.to_text())))
        for lo, hi, n in build_schedule(V, plan.sections):                 # decoding schedule: a free run -> its codebook; everything else -> text rows
            r += [(lo, hi) if hi - lo > 1 else (0, V.audio_offset)] * n
        assert len(r) == len(ids)
        ranges.append(r)
    return collate(docs, V.pad), ranges


def masked_full_loss(model, batch, ranges):
    """Reference: logits of all 151,702 rows from the Hugging Face module, every column outside the target's range set to -inf."""
    logits = model.lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()[:, :-1]   # Float[B, S-1, size]; row j-1 predicts token j
    allowed = torch.ones_like(logits, dtype=torch.bool)                                                              # unlabelled rows stay untouched (an all -inf row would give NaN gradients)
    for b, rng in enumerate(ranges):
        for j in range(1, len(rng)):
            if batch["labels"][b, j] != -100:
                allowed[b, j - 1] = False
                allowed[b, j - 1, rng[j][0]:rng[j][1]] = True
    return F.cross_entropy(logits.masked_fill(~allowed, float("-inf")).flatten(0, 1), batch["labels"][:, 1:].flatten(), ignore_index=-100)


def test_new_rows_are_initialised_uniformly():
    lm = AutoModelForCausalLM.from_pretrained(backbone_dir(), dtype=torch.float32)
    w = lm.get_input_embeddings().weight
    old = w.detach()[:V.eod].clone()
    w.data[V.eod:] = 5.0                                                    # rows 151,665 .. 151,935 exist in Qwen2.5 but are untrained; poison them
    resize_for_audio(lm, V)
    w = lm.get_input_embeddings().weight.detach()
    assert w.shape == (V.size, 32) == (151702, 32) and lm.config.vocab_size == V.size
    assert lm.get_output_embeddings().weight is lm.get_input_embeddings().weight                           # tied weights stay tied
    assert torch.equal(w[:V.eod], old)                                      # pre-trained rows untouched
    new = w[V.eod:]                                                         # 5 special tokens (their ids already had rows) + 32 audio rows (they did not)
    assert (new - old.mean(0)).abs().max() < 1e-3 * old.std()               # both groups: the mean of the old rows
    assert (new[:5] - new[5:].mean(0)).abs().max() < 1e-3 * old.std()


def test_initial_cross_entropy_of_every_level_is_ln_V():
    model = load_lm(CFG, V)
    batch, _ = make_batch([(30, True)])
    out = model(batch)
    assert all(abs(out[f"ce_k{k}"].item() - math.log(16)) < 1e-3 for k in range(2)), out
    assert out["loss"].requires_grad and out["tokens"].item() == (batch["labels"][:, 1:] != -100).sum().item()


def test_restricted_loss_equals_the_full_vocabulary_loss_with_the_same_mask():
    model = random_rows_model()
    batch, ranges = make_batch([(30, True), (40, False)])                  # two lengths (padding) and both loss_on_plan modes
    out, ref = model(batch), masked_full_loss(model, batch, ranges)
    assert torch.allclose(out["loss"], ref, atol=1e-5), (out["loss"].item(), ref.item())
    g1 = torch.autograd.grad(out["loss"], list(model.parameters()), allow_unused=True)
    g2 = torch.autograd.grad(masked_full_loss(model, batch, ranges), list(model.parameters()), allow_unused=True)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g1, g2) if a is not None or b is not None)   # same gradients, not just the same value
    naive = F.cross_entropy(model.lm(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits[:, :-1].float().flatten(0, 1),
                            batch["labels"][:, 1:].flatten(), ignore_index=-100)
    assert abs(naive.item() - out["loss"].item()) > 1e-2                    # the model's built-in loss is a different objective


def test_padding_does_not_change_the_loss():
    model = random_rows_model()
    batch, _ = make_batch([(30, True), (60, True)])
    one, two = make_batch([(30, True)])[0], make_batch([(60, True)])[0]
    pad = batch["input_ids"].shape[1] - one["input_ids"].shape[1]          # padding added to the shorter document
    assert pad > 0 and (batch["input_ids"][0, -pad:] == V.pad).all() and (batch["labels"][0, -pad:] == -100).all() and (batch["attention_mask"][0, -pad:] == 0).all()
    ab, a, b = model(batch), model(one), model(two)
    assert ab["tokens"] == a["tokens"] + b["tokens"]                       # padding positions are not counted
    assert torch.allclose(ab["loss"] * ab["tokens"], a["loss"] * a["tokens"] + b["loss"] * b["tokens"], rtol=1e-4)


def test_checkpoint_round_trip_and_codec_guard():
    model = random_rows_model()
    with tempfile.TemporaryDirectory() as d:
        run = pathlib.Path(d) / "lm_pretrain" / "x"
        V.save(run)
        step = run / "step_0000010"
        step.mkdir()
        save_file({k: t for k, t in model.state_dict().items() if k != "lm.lm_head.weight"}, str(step / "model.safetensors"))   # shared tensor stored once, as the engine does
        again = load_lm(CFG, V, str(step))
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values()))
        assert torch.equal(load_lm(make_cfg(2, 16, 1, checkpoint=str(step)), V).head_weight, model.head_weight)     # cfg.lm.checkpoint is the default
        assert raises(load_lm, CFG, dataclasses.replace(V, codec_tag="another_rvq"), str(step))                     # tokens of another RVQ


def test_gradient_checkpointing_gives_the_same_gradients():
    a = random_rows_model()
    b = load_lm(make_cfg(2, 16, 1, grad_checkpointing=True), V)
    b.load_state_dict(a.state_dict())
    batch, _ = make_batch([(30, True)])
    for m in (a, b):
        m.train()
    ga = torch.autograd.grad(a(batch)["loss"], list(a.parameters()))
    gb = torch.autograd.grad(b(batch)["loss"], list(b.parameters()))
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(ga, gb))


def test_a_tiny_model_can_overfit_one_batch():
    model = load_lm(CFG, V)
    batch, _ = make_batch([(30, True), (40, True)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    first = model(batch)["loss"].item()
    for _ in range(80):
        opt.zero_grad()
        loss = model(batch)["loss"]
        loss.backward()
        opt.step()
    assert first > math.log(16) and loss.item() < 0.5 * first, (first, loss.item())


if __name__ == "__main__":
    run_all(globals())
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_lm_model.py` → `ModuleNotFoundError: No module named 'dcttgen.lm.model'`
- [ ] **Step 3: implement.** `MusicLM.forward` calls the `Qwen2Model` (not the `…ForCausalLM` wrapper), takes the hidden states at positions `0 … S-2` to predict tokens `1 … S-1`, drops the positions whose label is `-100`, and hands the rest to `restricted_loss`.

**`dcttgen/lm/model.py`** — 79 lines

```python
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_lm_model.py` → `test_lm_model: 7 passed`
- [ ] **Step 5: commit** — `git add dcttgen/lm/model.py tests/test_lm_model.py && git commit -m "feat: Qwen2.5 with audio rows and a restricted loss"`

### Task 5: dataset, collator, token-budget batches

**Files:** Create `dcttgen/lm/data.py` · Test `tests/test_lm_data.py`

**Interfaces:**
- Consumes: `manifest/<split>.jsonl` (contract §7.2), `codes/<tag>/<clip_id>.npy` (contract §7.3), `build_document`, `doc_length`; `cfg.lm.loss_on_plan`, `cfg.lm.max_seconds`.
- Produces: `ClipDataset(cfg, vocab, split, phase)` with `.lengths`; `collate(batch, pad_id) -> {"input_ids", "attention_mask", "labels"}`; `TokenBudgetSampler(lengths, max_tokens, shuffle, seed=0)` with `set_epoch(epoch)`; `PHASES = ("pretrain", "finetune")`.

The dataset refuses to start if any clip lacks a code file, and refuses any code file whose shape, dtype or value range disagrees with the config — the two ways a codec mismatch shows up.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_lm_data.py</code> — 104 lines (click to expand)</summary>

```python
import pathlib
import random
import tempfile

import numpy as np
import torch
from lm_testkit import get_vocab, make_cfg, random_rows_model, raises, run_all, write_dataset

from dcttgen.lm.data import ClipDataset, TokenBudgetSampler, collate
from dcttgen.lm.sequence import parse_document
from dcttgen.lm.train import make_loader

K, V, FR = 2, 16, 1                                  # one frame per second keeps the documents small; the code reads it from cfg


def cfg_for(root, **lm):
    return make_cfg(K, V, FR, data_root=str(root), **lm)


def test_documents_match_the_manifest_and_the_saved_codes_in_both_phases():
    v = get_vocab(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        rows = write_dataset(d)
        for phase in ("pretrain", "finetune"):
            ds = ClipDataset(cfg_for(d), v, "train", phase)
            assert len(ds) == len(rows)
            for i, row in enumerate(rows):
                item = ds[i]
                out = parse_document(v, item["input_ids"])
                saved = torch.from_numpy(np.load(f"{d}/codes/k{K}v{V}/{row['clip_id']}.npy").astype(np.int64))
                assert torch.equal(out["codes"], saved) and out["sections"] == [tuple(s) for s in row["sections"]]
                assert (out["caption"] is not None) == (phase == "finetune") and len(item["input_ids"]) == ds.lengths[i]
                assert item["input_ids"].dtype == item["labels"].dtype == torch.long


def test_finetuning_skips_clips_without_a_caption_pretraining_keeps_them():
    v = get_vocab(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        write_dataset(d, caption_none=(1, 4))
        assert len(ClipDataset(cfg_for(d), v, "train", "pretrain")) == 6
        assert [r["clip_id"] for r in ClipDataset(cfg_for(d), v, "train", "finetune").rows] == ["c0", "c2", "c3", "c5"]
        assert [r["clip_id"] for r in ClipDataset(cfg_for(d, max_seconds=100), v, "train", "pretrain").rows] == ["c3", "c4", "c5"]   # max_seconds drops 150, 270, 120 s
        assert raises(ClipDataset, cfg_for(d), v, "train", "pretrainx") and raises(ClipDataset, cfg_for(d, max_seconds=5), v, "train", "pretrain")


def test_codes_of_another_shape_dtype_or_range_are_refused():           # a stale or foreign codes/<tag>/ directory must not train silently
    v = get_vocab(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        write_dataset(d)
        ds, path = ClipDataset(cfg_for(d), v, "train", "pretrain"), pathlib.Path(d, "codes", f"k{K}v{V}", "c0.npy")
        good = np.load(path)
        for name, bad in (("one codebook", good[:1]), ("one frame short", good[:, :-1]), ("float", good.astype(np.float32)),
                          ("code >= V", good + V), ("negative code", good - V)):
            np.save(path, bad)
            assert raises(ds.__getitem__, 0), name
        np.save(path, good)
        assert ds[0]["input_ids"].numel() == ds.lengths[0]
        path.unlink()
        try:
            ClipDataset(cfg_for(d), v, "train", "pretrain")
        except FileNotFoundError as e:
            assert "c0" in str(e) and "tokenize" in str(e)
        else:
            raise AssertionError("missing codes were accepted")


def test_collate_pads_input_labels_and_mask():
    a = {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([-100, 6, 7])}
    b = {"input_ids": torch.tensor([8, 9]), "labels": torch.tensor([8, 9])}
    out = collate([a, b], pad_id=151643)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 9, 151643]] and out["labels"].tolist() == [[-100, 6, 7], [8, 9, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_token_budget_sampler():
    rnd = random.Random(0)
    lengths = [rnd.randint(24000, 30300) if rnd.random() < 0.7 else rnd.randint(3000, 24000) for _ in range(400)]   # most clips are near the 300 s cap
    s = TokenBudgetSampler(lengths, 32768, shuffle=True, seed=0)
    epoch0 = list(s)
    assert sorted(i for b in epoch0 for i in b) == list(range(400)) and len(epoch0) == len(s)                  # every document exactly once
    assert all(len(b) * max(lengths[i] for i in b) <= 32768 or len(b) == 1 for b in epoch0)                    # the budget, padding included
    padded, real = sum(len(b) * max(lengths[i] for i in b) for b in epoch0), sum(lengths)
    assert padded / real < 1.1, padded / real                                                                  # sorting keeps the padding below 10 %
    assert list(s) == epoch0                                                                                   # same (seed, epoch): same order
    s.set_epoch(1)
    epoch1 = list(s)
    assert epoch1 != epoch0 and sorted(map(tuple, epoch1)) == sorted(map(tuple, epoch0)) and len(epoch1) == len(epoch0)   # new order, same batches
    assert [i for b in TokenBudgetSampler(lengths, 32768, shuffle=False) for i in b] == sorted(range(400), key=lambda i: (lengths[i], i))
    assert TokenBudgetSampler([40000], 32768, shuffle=False).batches == [[0]]                                   # a document over the budget goes alone


def test_loader_batches_are_accepted_by_the_model():
    v, model = get_vocab(K, V, FR), random_rows_model(K, V, FR)
    with tempfile.TemporaryDirectory() as d:
        write_dataset(d)
        loader = make_loader(cfg_for(d, max_tokens=700), v, "train", "finetune", shuffle=True)
        batches = list(loader)
        assert sum(b["input_ids"].shape[0] for b in batches) == 6 and len(batches) == len(loader)
        assert all(b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape for b in batches)
        assert all(torch.isfinite(model(b)["loss"]) for b in batches)


if __name__ == "__main__":
    run_all(globals())
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_lm_data.py` → `ModuleNotFoundError: No module named 'dcttgen.lm.data'`
- [ ] **Step 3: implement**

**`dcttgen/lm/data.py`** — 94 lines

```python
"""Manifest rows + codes/<tag>/<clip_id>.npy -> documents -> padded batches of about `lm.max_tokens` tokens."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dcttgen.lm.sequence import build_document, doc_length
from dcttgen.lm.vocab import Vocab
from dcttgen.plan import Plan

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
            raise FileNotFoundError(f"{len(missing)} clips have no codes in {self.codes_dir} (first: {missing[:3]}); run dcttgen.codec.tokenize first")
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_lm_data.py` → `test_lm_data: 6 passed`
- [ ] **Step 5: commit** — `git add dcttgen/lm/data.py tests/test_lm_data.py && git commit -m "feat: LM dataset with token-budget batches"`

### Task 6: schedule-constrained decoding

**Files:** Create `dcttgen/lm/generate.py` · Test `tests/test_lm_generate.py`

**Interfaces:**
- Consumes: `MusicLM.backbone`, `MusicLM.head_weight`, `Vocab`, `Plan`, `instruct_ids`, `metadata_ids`, `parse_document`.
- Produces: `generate_codes(model, vocab, caption, plan=None, *, temperature=1.0, top_k=None, top_p=None, seed=None) -> (Plan, Long[K, frame_rate × duration])` (contract §9); `build_schedule(vocab, sections) -> list[Step]`.

- [ ] **Step 1: write the failing test**

<details>
<summary><code>tests/test_lm_generate.py</code> — 129 lines (click to expand)</summary>

```python
from functools import lru_cache
from unittest import mock

import torch
import torch.nn.functional as F
from lm_testkit import CAPTION, get_vocab, make_doc, make_plan, raises, random_rows_model, run_all

from dcttgen.lm import generate as G
from dcttgen.lm.generate import build_schedule, generate_codes, range_logits, sample
from dcttgen.lm.model import load_lm
from dcttgen.lm.sequence import build_document, instruct_ids, metadata_ids
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
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_lm_generate.py` → `ModuleNotFoundError: No module named 'dcttgen.lm.generate'`
- [ ] **Step 3: implement**

**`dcttgen/lm/generate.py`** — 117 lines

```python
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
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_lm_generate.py` → `test_lm_generate: 9 passed`
- [ ] **Step 5: commit** — `git add dcttgen/lm/generate.py tests/test_lm_generate.py && git commit -m "feat: schedule-constrained decoding"`

### Task 7: the training entry point

**Files:** Create `dcttgen/lm/train.py` · Test `tests/test_lm_train.py`

**Interfaces:**
- Consumes: `load_config` and `fit` (chapter 04), everything above; `cfg.lm.max_tokens`, `cfg.lm.num_workers`, `cfg.paths.runs`.
- Produces: `python -m dcttgen.lm.train --config C --phase pretrain|finetune [--override a.b=value …] [--name N] [--probe]`. Run directory: `runs/lm_<phase>/<name>/` containing `tokenizer/`, `vocab.json`, and the engine's `step_*/`, `log.jsonl`, `config.yaml`.

- [ ] **Step 1: write the failing test** (it stands in for chapter 04's `load_config` and `fit`; the real pair is exercised by `tests/test_integration.py` in chapter 04)

<details>
<summary><code>tests/test_lm_train.py</code> — 90 lines (click to expand)</summary>

```python
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
from unittest import mock

import torch
from lm_testkit import get_vocab, make_cfg, random_rows_model, run_all, write_dataset
from safetensors.torch import load_file, save_file

from dcttgen.lm import train as T


def fake_chapter_04(root, runs, calls):
    """Stand-ins for dcttgen.config.load_config and dcttgen.engine.fit, which chapter 04 owns."""
    def load_config(path, overrides):
        cfg = make_cfg(2, 16, 1, data_root=str(root))
        cfg.paths.runs = str(runs)
        for o in overrides:                                                    # "a.b=value"
            key, value = o.split("=", 1)
            *parents, leaf = key.split(".")
            obj = cfg
            for p in parents:
                obj = getattr(obj, p)
            try:
                value = json.loads(value)
            except ValueError:
                pass
            setattr(obj, leaf, value)
        return cfg

    def fit(model, train, val, cfg, run_dir):
        calls.append(dict(train=len(train.dataset), val=len(val.dataset), start=model.head_weight.detach().clone(), losses=[]))
        opt, _ = torch.optim.AdamW(model.parameters(), lr=1e-2), model.train()
        for epoch in range(3):
            train.batch_sampler.set_epoch(epoch)
            for batch in train:
                opt.zero_grad()
                loss = model(batch)["loss"]
                loss.backward()
                opt.step()
                calls[-1]["losses"].append(loss.item())
        step = pathlib.Path(run_dir) / "step_0000003"
        step.mkdir(parents=True)
        save_file({k: t.contiguous() for k, t in model.state_dict().items() if k != "lm.lm_head.weight"}, str(step / "model.safetensors"))

    return {"dcttgen.config": types.SimpleNamespace(load_config=load_config), "dcttgen.engine": types.SimpleNamespace(fit=fit)}


def test_main_runs_both_phases_and_the_weights_carry_over():
    with tempfile.TemporaryDirectory() as d:
        root, runs, calls = pathlib.Path(d) / "data", pathlib.Path(d) / "runs", []
        write_dataset(root, caption_none=(1,))
        with mock.patch.dict(sys.modules, fake_chapter_04(root, runs, calls)):
            T.main(["--config", "x.yaml", "--phase", "pretrain", "--name", "t"])
            pre = runs / "lm_pretrain" / "t"
            assert (pre / "vocab.json").is_file() and (pre / "tokenizer" / "tokenizer.json").is_file() and (pre / "step_0000003" / "model.safetensors").is_file()
            T.main(["--config", "x.yaml", "--phase", "finetune", "--name", "t", "--override", f"lm.checkpoint={pre / 'step_0000003'}"])
        a, b = calls
        assert (a["train"], a["val"], b["train"], b["val"]) == (6, 2, 5, 1)                  # fine-tuning leaves out the clip without a caption
        assert a["losses"][-1] < a["losses"][0]                                              # the loop trains the model it was given
        assert torch.equal(b["start"], load_file(str(pre / "step_0000003" / "model.safetensors"))["lm.model.embed_tokens.weight"])   # fine-tuning starts from it
        assert (runs / "lm_finetune" / "t" / "vocab.json").is_file()


def test_main_refuses_documents_longer_than_the_backbone_positions():
    with tempfile.TemporaryDirectory() as d:
        root, runs = pathlib.Path(d) / "data", pathlib.Path(d) / "runs"
        write_dataset(root, n=1, fr=25, clips=[("whole", 300)])                              # 300 s at 25 Hz, K = 2: 15,000+ tokens; the tiny backbone has 4,096 positions
        with mock.patch.dict(sys.modules, fake_chapter_04(root, runs, [])):
            try:
                T.main(["--config", "x.yaml", "--phase", "pretrain", "--override", "audio.frame_rate=25"])
            except SystemExit as e:
                assert "positions only 4096" in str(e) and "lm.max_seconds" in str(e)
            else:
                raise AssertionError("a document longer than the backbone's positions was accepted")


def test_probe_reports_speed_on_a_synthetic_document():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        T.probe(make_cfg(2, 16, 1), get_vocab(2, 16, 1), random_rows_model(2, 16, 1), seconds=30, steps=1)
    assert "parameters" in out.getvalue() and "tokens/s" in out.getvalue() and "peak GPU memory" in out.getvalue(), out.getvalue()


if __name__ == "__main__":
    run_all(globals())
```

</details>

- [ ] **Step 2: run it, expect failure** — `python tests/test_lm_train.py` → `ImportError: cannot import name 'train' from 'dcttgen.lm'`
- [ ] **Step 3: implement**

**`dcttgen/lm/train.py`** — 82 lines

```python
"""python -m dcttgen.lm.train --config C --phase pretrain|finetune [--override a.b=value ...] [--name N] [--probe]

Builds the vocabulary, the model and the two loaders, then hands them to the engine's fit() (chapter 04). Nothing here trains by itself."""
from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dcttgen.lm.data import PHASES, ClipDataset, TokenBudgetSampler, collate
from dcttgen.lm.model import load_lm
from dcttgen.lm.sequence import build_document
from dcttgen.lm.vocab import Vocab
from dcttgen.plan import INSTRUMENTS, Plan, plan_sections


def make_loader(cfg, vocab: Vocab, split: str, phase: str, shuffle: bool) -> DataLoader:
    ds = ClipDataset(cfg, vocab, split, phase)
    sampler = TokenBudgetSampler(ds.lengths, cfg.lm.max_tokens, shuffle)      # the engine calls loader.batch_sampler.set_epoch()
    return DataLoader(ds, batch_sampler=sampler, collate_fn=partial(collate, pad_id=vocab.pad), num_workers=cfg.lm.num_workers)


def probe(cfg, vocab: Vocab, model, seconds: int = 300, steps: int = 3) -> None:
    """Speed and peak memory of one training step on a synthetic document (random codes, the longest legal plan and caption)."""
    dev = next(model.parameters()).device
    sections = plan_sections(seconds)
    plan = Plan(120, seconds, sections, ["calm"], list(INSTRUMENTS))
    ids, labels = build_document(vocab, torch.randint(0, vocab.V, (vocab.K, seconds * vocab.frame_rate)), sections, "word " * 200, plan)
    batch = {k: v.to(dev) for k, v in collate([{"input_ids": ids, "labels": labels}], vocab.pad).items()}
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)       # lr 0: the weights stay as they are, the optimiser state is allocated as in training
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = model(batch)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if dev.type == "cuda" else "n/a (CPU)"
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters | {ids.numel()} tokens | {dt:.2f} s/step | "
          f"{ids.numel() / dt:,.0f} tokens/s | peak GPU memory {peak}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value", help="repeatable; e.g. lm.checkpoint=runs/lm_pretrain/<name>/step_0010000")
    ap.add_argument("--name", help="run directory name; default <codec.tag>-<backbone>")
    ap.add_argument("--probe", action="store_true", help="time one training step on a synthetic 300 s document and exit")
    args = ap.parse_args(argv)
    from dcttgen.config import load_config      # chapter 04
    from dcttgen.engine import fit              # chapter 04
    cfg = load_config(args.config, args.override)
    vocab = Vocab.build(cfg)
    model = load_lm(cfg, vocab)
    if args.probe:
        probe(cfg, vocab, model.to("cuda" if torch.cuda.is_available() else "cpu"))
        return
    train, val = make_loader(cfg, vocab, "train", args.phase, True), make_loader(cfg, vocab, "val", args.phase, False)
    longest = max(train.dataset.lengths + val.dataset.lengths)
    if longest > model.lm.config.max_position_embeddings:
        raise SystemExit(f"the longest document has {longest} tokens but {cfg.lm.backbone} positions only {model.lm.config.max_position_embeddings}; "
                         f"set lm.max_seconds or use a backbone with a longer context")
    run_dir = Path(cfg.paths.runs) / f"lm_{args.phase}" / (args.name or f"{cfg.codec.tag}-{Path(cfg.lm.backbone).name.lower()}")
    if os.environ.get("RANK", "0") == "0":      # `accelerate launch` sets RANK; only one process writes the vocabulary files
        vocab.save(run_dir)
    fit(model, train, val, cfg, str(run_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run it, expect a pass** — `python tests/test_lm_train.py` → `test_lm_train: 3 passed`
- [ ] **Step 5: commit** — `git add dcttgen/lm/train.py tests/test_lm_train.py && git commit -m "feat: LM training entry point for both phases"`

## 4. Running it

All commands run in the language-model environment (chapter 04). `C` is a comma-separated list of config overlays.

```bash
# 0. how fast and how large is one training step on this GPU?  (synthetic 300 s document; nothing is trained)
python -m dcttgen.lm.train --config configs/plan_k4.yaml --phase pretrain --probe

# 1. phase 1 - pre-training on audio tokens only (plan 3.6)
accelerate launch -m dcttgen.lm.train --config configs/plan_k4.yaml --phase pretrain

# 2. phase 2 - fine-tuning with caption and plan, starting from the pre-training checkpoint
accelerate launch -m dcttgen.lm.train --config configs/plan_k4.yaml --phase finetune \
    --override lm.checkpoint=runs/lm_pretrain/k4v10000-qwen2.5-0.5b/step_0020000
```

**What carries over between the phases:** the model weights only (`lm.checkpoint`). The fine-tuning run has its own directory, its own optimiser state and its own learning-rate schedule.

**How to tell that training is healthy** (read `log.jsonl`):

| Signal | Healthy | Unhealthy |
|---|---|---|
| `ce_k0 … ce_k3` at step 1 | each `ln V` = 9.21 (V = 10,000) or 9.70 (V = 16,384) | anything else: the resize or the loss ranges are wrong |
| `ce_k0` over time | falls first and furthest: the coarse level carries melody and rhythm | flat for thousands of steps: learning rate too low, or the codes are noise (wrong codec) |
| `ce_k1 … ce_k3` | fall after `ce_k0`, ending higher than it | all four identical for long: the model is not using the coarser levels as context |
| `ce_text` (fine-tuning) | drops quickly towards 0–1: plan text is highly predictable | stays near 11.9: the plan span is masked (`lm.loss_on_plan: false`) or captions are missing |
| `val_loss` | tracks the training loss, then flattens | rises while training loss falls: overfitting — expected early with 20 hours of data |

**Memory and speed.** No measurement was possible on this machine (CPU only). The `--probe` command prints parameters, tokens, seconds per step, tokens per second and peak GPU memory for one worst-case document; run it once per backbone and GPU before choosing `lm.max_tokens`. **Unverified** rough expectations, to be replaced by the probe's numbers: the 0.5B backbone with `lm.grad_checkpointing: true` should fit one 30,000-token document on a 24 GB GPU; the 1.5B backbone needs an 80 GB GPU for the same document.

**Generation speed.** A 300 s piece is 30,000 sampled tokens, one forward pass each, with the key-value cache.
`# ponytail:` the cache grows by concatenation, which copies it at every step; over 30,000 steps that is quadratic work. If the probe-measured tokens per second at inference fall well below the training-time rate, switch `_feed` to a pre-allocated cache (`transformers.StaticCache`).

**Scale-down path.**

| Setting | Pilot (`configs/pilot.yaml`) | Why |
|---|---|---|
| `lm.max_seconds` | 120 | documents of at most 12,000 audio tokens |
| `lm.max_tokens` | 12288 | one long document or several short ones per batch |
| `lm.backbone` | `Qwen/Qwen2.5-0.5B` | 494 M parameters before the audio rows |
| codec | `configs/bootstrap_k1.yaml` | K = 1: 25 tokens per second instead of 100 |

With the bootstrap codec a 120 s clip is 3,000 audio tokens, which trains on a single 24 GB GPU and even on a free notebook GPU with `lm.grad_checkpointing: true`.

## 5. What will go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: token layout … differs from contract 8.1` | another tokenizer, or the five tokens were added in another order | use a Qwen2.5 tokenizer; never reorder `SPECIALS` |
| `N clips have no codes in data/codes/<tag>` | the dataset was not tokenised with this `codec.tag` | run `python -m dcttgen.codec.tokenize` (chapter 02) |
| `codes are int16(4, 3751), expected …(4, 3750)` | the codec returned one frame too many, or the clip is not a whole number of seconds | fix upstream: contract §7.1 and §7.3 are exact |
| `checkpoint … was trained with {…}, but the config now says {…}` | `codec.tag`, K, V or the frame rate changed since training | restore the config, or retrain: tokens of two RVQs are not interchangeable |
| `the longest document has N tokens but … positions only 32768` | `Qwen2.5-0.5B` has 32,768 positions; a long caption plus a 300 s clip fits, anything larger does not | set `lm.max_seconds`, or use the 1.5B backbone (131,072 positions) |
| CUDA out of memory at the first step | `lm.max_tokens` too high for this GPU | run `--probe`, lower `lm.max_tokens`, enable `lm.grad_checkpointing` |
| Loss is `nan` after some steps | half-precision overflow in a long sequence | keep `train.mixed_precision: bf16` (not `fp16`); lower `train.lr` for fine-tuning (contract D10) |
| Generated pieces loop a short pattern | sampling too greedy | raise `infer.temperature` towards 1.0; raise `infer.top_k` |
| Generated pieces are noise | sampling too flat, or decoding with a codec other than the one that tokenised the data | lower `infer.top_k`; check `data/codes/<tag>/meta.json` against the codec (chapter 02) |
| `the model did not write a valid plan in 8 attempts` | the model was trained with `lm.loss_on_plan: false`, or too little | pass all four plan fields; or train with `loss_on_plan: true` |

## 6. Open questions for the research team

1. **Backbone size.** The plan does not name one. Default: `Qwen2.5-0.5B` for bring-up, `Qwen2.5-1.5B` for the full run (contract D11). Decide after the pilot shows how fast `ce_k0` falls.
2. **Loss on the plan text.** Default `lm.loss_on_plan: true`, so the model can write its own plan (contract D8). If the paper should match the equation literally — conditions `c = {c_inst, c_meta}` are given, never predicted — set it to `false` and always supply the four plan fields.
3. **Schedule-constrained decoding** is stricter than the plan's rule (contract D9) and the loss in §1.3 relies on it. The paper should describe the decoder as it is built.
4. **Sampling.** The plan's equation is an argmax. Recommended: report the sampling settings actually used (`temperature`, `top_k`).
5. **Learning rate for fine-tuning.** Contract D10 reads the plan as "3e-4 decaying to 3e-5" for every stage. A pre-trained backbone is usually fine-tuned more gently; if fine-tuning diverges, try `train.lr=3e-5`, the other reading of the plan.
6. **Whole-section coarse-to-fine.** In a 240 s section the codebook-1 token of a frame sits 6,000 positions after its codebook-0 token. If `ce_k1 … ce_k3` refuse to fall, the first ablation to run is frame-interleaved order (chapter 04, §ablations): it changes only `flatten_c2f`, `unflatten_c2f` and `build_schedule`.
