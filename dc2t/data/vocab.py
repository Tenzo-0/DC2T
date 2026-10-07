"""Closed vocabularies of the manifest (contract 7.2)."""
from dc2t.plan import INSTRUMENTS  # noqa: F401 - the six canonical names, in the contract's order; defined once, in plan.py

# Mood word -> labels of Essentia's mtg_jamendo_moodtheme model whose probabilities vote for it.
# Every label below is one of the model's 56 classes (checked by tests/test_data_features.py).
MOOD_VOTES = {
    "calm": ("calm", "soft"),
    "relaxing": ("relaxing",),
    "meditative": ("meditative",),
    "melancholic": ("melancholic",),
    "sad": ("sad",),
    "romantic": ("romantic", "love"),
    "emotional": ("emotional",),
    "dreamy": ("dream",),
    "hopeful": ("hopeful",),
    "uplifting": ("uplifting", "inspiring", "positive", "motivational"),
    "joyful": ("happy", "fun", "upbeat"),
    "energetic": ("energetic",),
    "dramatic": ("dramatic", "drama"),
}
MOODS = tuple(MOOD_VOTES)   # the closed mood vocabulary: one lowercase word each
