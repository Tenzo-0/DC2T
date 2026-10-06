import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import numpy as np

from dcttgen.data.features import bpm_and_confidence, clip_patches, fold_bpm, pick_moods, vocal_fraction
from dcttgen.data.vocab import MOOD_VOTES, MOODS

# The 56 classes of mtg_jamendo_moodtheme-discogs-effnet-1, copied from the "classes" list of its JSON file
# (https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json)
LABELS = ["action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas", "commercial",
          "cool", "corporate", "dark", "deep", "documentary", "drama", "dramatic", "dream", "emotional", "energetic",
          "epic", "fast", "film", "fun", "funny", "game", "groovy", "happy", "heavy", "holiday", "hopeful", "inspiring",
          "love", "meditative", "melancholic", "melodic", "motivational", "movie", "nature", "party", "positive",
          "powerful", "relaxing", "retro", "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space", "sport",
          "summer", "trailer", "travel", "upbeat", "uplifting"]


def test_vocabulary_is_closed_and_votes_only_for_real_labels():
    assert len(LABELS) == 56 and len(set(LABELS)) == 56
    assert all(l in LABELS for votes in MOOD_VOTES.values() for l in votes)      # no word votes for a label that is not there
    assert all(w.isalpha() and w.islower() for w in MOODS)                      # one lowercase word each (contract 7.2)
    assert "uplifting" in MOODS and "joyful" in MOODS                           # the plan's own example moods


def p_with(**labels):
    p = np.full(56, 0.01)
    for name, v in labels.items():
        p[LABELS.index(name)] = v
    return p


def test_pick_moods():
    assert pick_moods(p_with(happy=0.4, uplifting=0.3, calm=0.01), LABELS) == (["joyful", "uplifting"], 0.4)
    # the runner-up is dropped when it is below half the best score ...
    assert pick_moods(p_with(sad=0.4, calm=0.1), LABELS)[0] == ["sad"]
    # ... or below the absolute floor; at most three words
    assert pick_moods(p_with(sad=0.04, calm=0.03), LABELS)[0] == ["sad"]
    four = p_with(sad=0.4, calm=0.39, hopeful=0.38, dream=0.37, romantic=0.36)
    assert pick_moods(four, LABELS)[0] == ["sad", "calm", "hopeful"]
    # a flat output still gives one word: the manifest needs at least one mood per clip
    words, top = pick_moods(np.zeros(56), LABELS)
    assert len(words) == 1 and words[0] in MOODS and top == 0.0
    # a label vote is by name, so a different label order must give the same answer
    order = list(reversed(range(56)))
    assert pick_moods(p_with(happy=0.4)[order], [LABELS[i] for i in order])[0] == ["joyful"]


def test_fold_bpm():
    assert [fold_bpm(x) for x in (0, -5, 120, 20, 300, 39.9, 200, 250)] == [0.0, 0.0, 120, 40, 150, 79.8, 200, 125]


def test_vocal_fraction_and_patch_slicing():
    assert vocal_fraction(np.array([0.9, 0.1, 0.6, 0.2])) == 0.5
    assert vocal_fraction(np.array([])) == 1.0                                  # no evidence is treated as vocal
    a = np.arange(100)                                                          # 100 patches over a 100 s recording
    assert list(clip_patches(a, 100.0, 20, 30)) == list(range(20, 50))
    assert list(clip_patches(a, 100.0, 70, 30)) == list(range(70, 100))
    assert len(clip_patches(np.arange(3), 100.0, 0, 30)) >= 1                   # never empty


def plucks(times, seconds, sr):
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    n, t = int(0.4 * sr), np.arange(int(0.4 * sr)) / sr
    for k, t0 in enumerate(times):
        i = int(t0 * sr)
        if i + n <= len(y):
            y[i:i + n] += (np.exp(-8 * t) * np.sin(2 * np.pi * 220 * (1 + 0.25 * (k % 3)) * t)).astype(np.float32)
    return y


def test_bpm_confidence_separates_steady_from_free_rhythm():
    try:
        import librosa  # noqa: F401
    except ImportError:
        return print("skipped: librosa not installed")
    sr = 32000
    steady, conf_s = bpm_and_confidence(plucks(np.arange(0.5, 60, 0.6), 60, sr), sr)             # 100 bpm
    assert abs(steady - 100) < 4 and conf_s > 0.6, (steady, conf_s)
    rng = np.random.default_rng(0)
    _, conf_f = bpm_and_confidence(plucks(np.sort(rng.uniform(0.5, 59, 100)), 60, sr), sr)      # no pulse
    assert conf_f < 0.3 < conf_s, (conf_f, conf_s)
    assert bpm_and_confidence(np.zeros(sr * 30, dtype=np.float32), sr) == (0.0, 0.0)             # silence: nothing to track


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
