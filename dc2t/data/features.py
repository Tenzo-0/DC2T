"""Per-clip musical features. The tempo comes from librosa; the vocal and mood decisions are plain numpy
functions applied to the outputs of Essentia's models (embed.py), so they run and are tested without TensorFlow."""
import numpy as np

from dc2t.data.vocab import MOOD_VOTES


def fold_bpm(bpm: float, lo: float = 40.0, hi: float = 200.0) -> float:
    """Tempo estimators often answer half or double the felt tempo: halve or double until the value is in [lo, hi]."""
    if bpm <= 0:
        return 0.0
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def bpm_and_confidence(y: np.ndarray, sr: int, hop: int = 512) -> tuple[float, float]:
    """y: Float[N] mono. Returns (bpm, confidence in [0, 1]); (0.0, 0.0) if there is nothing to track.
    Confidence is the normalised autocorrelation of the onset envelope at one or two beat periods: close to 1 for
    a steady pulse, close to 0 for free rhythm. (beat_track's own beats are no evidence: its dynamic programme
    returns evenly spaced beats even for music that has none.)"""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, _ = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0 or len(env) < 16 or not np.any(env):
        return 0.0, 0.0
    ac = librosa.autocorrelate(env - env.mean(), max_size=len(env) // 2)
    if ac[0] <= 0:
        return 0.0, 0.0
    ac = ac / ac[0]
    lag = 60.0 * sr / (hop * bpm)                                       # frames per beat
    peaks = [ac[int(round(m * lag)) - 1: int(round(m * lag)) + 2].max() for m in (1, 2) if int(round(m * lag)) + 2 < len(ac)]
    return fold_bpm(bpm), float(np.clip(max(peaks, default=0.0), 0.0, 1.0))


def pick_moods(p: np.ndarray, labels: list[str], k: int = 3, min_p: float = 0.05, rel: float = 0.5):
    """p: Float[56], mtg_jamendo_moodtheme's sigmoid output averaged over a clip's patches. labels: the model's 56
    class names in output order (read them from its JSON file; never type them). Returns (words, top score):
    1..k words of the closed vocabulary. The best word is always kept (the manifest needs at least one); the others
    need score >= min_p and >= rel * best, which stops a weak runner-up from being written as a mood."""
    idx = {name: i for i, name in enumerate(labels)}
    score = {w: max(float(p[idx[name]]) for name in votes) for w, votes in MOOD_VOTES.items()}
    ranked = sorted(score, key=score.get, reverse=True)
    top = score[ranked[0]]
    return [ranked[0]] + [w for w in ranked[1:k] if score[w] >= min_p and score[w] >= rel * top], top


def vocal_fraction(p_voice: np.ndarray, thr: float = 0.5) -> float:
    """p_voice: Float[n], P(voice) per ~1 s patch. Fraction of patches above thr. No patches = no evidence: 1.0."""
    return float(np.mean(p_voice > thr)) if len(p_voice) else 1.0


def clip_patches(a: np.ndarray, total_s: float, start: int, duration: int) -> np.ndarray:
    """a: [n, ...], one row per patch over a recording of total_s seconds. The rows inside [start, start + duration)."""
    n = len(a)
    i, j = int(start * n / total_s), int(np.ceil((start + duration) * n / total_s))
    return a[i:max(j, i + 1)]
