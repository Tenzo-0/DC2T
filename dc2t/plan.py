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
