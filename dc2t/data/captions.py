"""Captions: a deterministic template (free; the pilot uses it) and the plan's GPT-4o step with validation."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dc2t.data.common import read_jsonl

MAX_WORDS = 60
# Every English word that counts as a mention of one of the six instruments. "gong" alone is read as gong ban.
ALIASES = {
    "moon-shaped lute": ("moon-shaped lute", "moon lute", "lute"),
    "two-string fiddle": ("two-string fiddle", "two-stringed fiddle", "fiddle"),
    "zither": ("zither",),
    "monochord": ("monochord",),
    "bamboo flute": ("bamboo flute", "flute"),
    "gong ban": ("gong ban", "gong"),
}
# Instruments the dataset does not annotate: a caption naming one is a hallucination. Extend as GPT invents new ones.
NOT_ANNOTATED = ("piano", "guitar", "violin", "cello", "harp", "drum", "drums", "percussion", "sitar", "erhu", "pipa",
                 "guzheng", "koto", "saxophone", "trumpet", "organ", "synthesizer", "accordion", "banjo", "ukulele",
                 "oboe", "clarinet", "harmonica", "orchestra", "xylophone", "mandolin", "gamelan")
# The training data is instrumental (see "Vocals"): a caption must not suggest singing.
VOCAL = ("vocal", "vocals", "vocalist", "singer", "singing", "sung", "voice", "voices", "lyrics", "choir", "chanting", "humming")
_OWNER = {a: inst for inst, names in ALIASES.items() for a in names}
_MENTION = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted([*_OWNER, *NOT_ANNOTATED, *VOCAL], key=len, reverse=True))) + r")\b", re.I)


def tempo_word(bpm: float, slow_below: float = 70, fast_from: float = 110) -> str:
    return "slow" if bpm < slow_below else "fast" if bpm >= fast_from else "moderate"


def join_and(items) -> str:
    items = list(items)
    return " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", and " + items[-1]


def template_caption(moods: list[str], instruments: list[str], bpm: float, slow_below: float = 70, fast_from: float = 110) -> str:
    """The plan's own sentence shape, filled from the manifest fields. No period, as in the plan's example."""
    article = "An" if moods[0][0] in "aeiou" else "A"
    return (f"{article} {join_and(moods)} Don ca tai tu piece with {tempo_word(bpm, slow_below, fast_from)} tempo, "
            f"performed by {join_and(instruments)}")


def validate_caption(caption, instruments: list[str], moods: list[str] = ()) -> list[str]:
    """Problems with a caption (empty list = acceptable). Instruments are checked both ways: every annotated
    instrument is named, and no other instrument (or anything vocal) is. Untrusted text: GPT output is data."""
    if not isinstance(caption, str) or not caption.strip():
        return ["the caption is empty"]
    bad = []
    if re.search(r"[\r\n\t]", caption):
        bad.append("must be one line")
    if not caption.isascii():
        bad.append("must be plain ASCII (no accents, no typographic quotes or dashes)")
    if "<" in caption or ">" in caption:
        bad.append("must not contain < or >")                                  # never let text look like a control token
    if len(caption.split()) > MAX_WORDS:
        bad.append(f"must be at most {MAX_WORDS} words")
    if "don ca tai tu" not in caption.lower():
        bad.append("must say 'Don ca tai tu'")
    named = set()
    for m in _MENTION.finditer(caption):                                       # longest alias first: 'two-string fiddle' is one hit
        w = m.group().lower()
        if w in _OWNER:
            named.add(_OWNER[w])
            if _OWNER[w] not in instruments:
                bad.append(f"names {_OWNER[w]}, which is not annotated for this clip")
        elif w in VOCAL:
            bad.append(f"mentions vocals ('{w}'); the recordings are instrumental")
        else:
            bad.append(f"names '{w}', which is not one of the annotated instruments")
    bad += [f"does not name the annotated instrument {i}" for i in instruments if i not in named]
    if moods and not any(re.search(rf"\b{re.escape(m)}\b", caption, re.I) for m in moods):
        bad.append("uses none of the annotated mood words")
    return list(dict.fromkeys(bad))


SYSTEM = ("You write one-sentence English captions for instrumental recordings of Don ca tai tu, a traditional "
          "Southern Vietnamese music form. Use only the facts you are given. Name every listed instrument exactly as "
          "spelled, and no other instrument. Use at least one of the listed moods as a word. Do not mention singing, "
          "voices or lyrics. Do not invent titles, places, people or dates. Write the genre as \"Don ca tai tu\". "
          "Plain ASCII, one line, at most 40 words. Example for tempo slow, moods calm and meditative, instruments X and Y: "
          "\"A calm and meditative Don ca tai tu piece at a slow tempo, played by X and Y.\"")
FORMAT = {"format": {"type": "json_schema", "name": "caption", "strict": True,
                     "schema": {"type": "object", "properties": {"caption": {"type": "string"}},
                                "required": ["caption"], "additionalProperties": False}}}


def user_message(row: dict, bands, problems=()) -> str:
    facts = {"bpm": row["bpm"], "tempo": tempo_word(row["bpm"], *bands), "moods": row["moods"], "instruments": row["instruments"]}
    msg = json.dumps(facts)
    return msg + (f"\nYour previous caption was rejected: {'; '.join(problems)}. Write a new one." if problems else "")


def inputs_hash(row: dict, model: str, bands) -> str:
    """Identifies everything the caption depends on. A cached caption is reused only if this still matches,
    so re-annotating a clip or editing the prompt can never leave a stale caption behind."""
    facts = [row["bpm"], tempo_word(row["bpm"], *bands), row["moods"], row["instruments"], model, SYSTEM]
    return hashlib.sha1(json.dumps(facts).encode()).hexdigest()[:12]


def caption_one(row: dict, client, model: str, attempts: int, bands) -> dict:
    """Ask up to `attempts` times, feeding the validator's complaints back. source is 'gpt' or 'failed'."""
    problems, used = [], {"input_tokens": 0, "output_tokens": 0}
    rec = {"clip_id": row["clip_id"], "inputs": inputs_hash(row, model, bands), "model": model}
    for attempt in range(1, attempts + 1):
        resp = client.responses.create(model=model, instructions=SYSTEM, input=user_message(row, bands, problems),
                                       text=FORMAT, temperature=0.7, max_output_tokens=150)
        used["input_tokens"] += resp.usage.input_tokens
        used["output_tokens"] += resp.usage.output_tokens
        try:
            caption = json.loads(resp.output_text)["caption"].strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            problems = ["the reply was not the requested JSON object"]
            continue
        problems = validate_caption(caption, row["instruments"], row["moods"])
        if not problems:
            return {**rec, "caption": caption, "source": "gpt", "attempts": attempt, **used}
    return {**rec, "caption": None, "source": "failed", "attempts": attempts, "problems": problems, **used}


def generate(rows, client, cache_path, *, model: str, attempts: int = 3, workers: int = 8, bands=(70, 110)) -> int:
    """rows: dicts with clip_id, bpm, moods, instruments. Appends one record per call to cache_path (JSONL; the last
    record of a clip wins), skipping clips whose cached caption is valid for the current inputs. Returns #clips asked."""
    cache_path = Path(cache_path)
    cache = {r["clip_id"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    todo = [r for r in rows if not (cache.get(r["clip_id"], {}).get("source") == "gpt"
                                    and cache[r["clip_id"]]["inputs"] == inputs_hash(r, model, bands))]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex, open(cache_path, "a", encoding="utf-8", newline="\n") as f:
        for fut in as_completed([ex.submit(caption_one, r, client, model, attempts, bands) for r in todo]):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()                                                          # a crash loses at most the calls in flight
    return len(todo)


def cached_captions(rows, cache_path, model: str, bands=(70, 110)) -> dict[str, str]:
    """clip_id -> caption for the cached GPT captions that are still valid for each row's CURRENT fields."""
    if not Path(cache_path).exists():
        return {}
    cache = {r["clip_id"]: r for r in read_jsonl(cache_path)}                # the last record of a clip wins
    return {r["clip_id"]: cache[r["clip_id"]]["caption"] for r in rows
            if cache.get(r["clip_id"], {}).get("source") == "gpt" and cache[r["clip_id"]]["inputs"] == inputs_hash(r, model, bands)}
