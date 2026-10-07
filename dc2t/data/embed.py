"""Essentia models over a standardised recording -> work/embed/<recording_id>.npz with
    p_voice Float[n]     P(voice) of each patch, from voice_instrumental-discogs-effnet-1
    mood    Float[n, 56] mtg_jamendo_moodtheme-discogs-effnet-1 activations of each patch
Linux only (essentia-tensorflow publishes no Windows wheel). Not executed in this guide: the node names, class
order and sample rate below are read from the .json metadata published next to each model file."""
import json
from pathlib import Path

import numpy as np

EFFNET = "discogs-effnet-bs64-1"
MOOD = "mtg_jamendo_moodtheme-discogs-effnet-1"
VOICE = "voice_instrumental-discogs-effnet-1"


def mood_labels(models_dir) -> list[str]:
    """The 56 class names in output order: read them from the model's own metadata, never type them."""
    return json.loads((Path(models_dir) / f"{MOOD}.json").read_text(encoding="utf-8"))["classes"]


def embed(flac, out, models_dir) -> None:
    from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    m = Path(models_dir)
    audio = MonoLoader(filename=str(flac), sampleRate=16000, resampleQuality=4)()                       # the models' sample rate
    emb = TensorflowPredictEffnetDiscogs(graphFilename=str(m / f"{EFFNET}.pb"), output="PartitionedCall:1")(audio)   # Float[n, 1280]
    mood = TensorflowPredict2D(graphFilename=str(m / f"{MOOD}.pb"), input="model/Placeholder", output="model/Sigmoid")(emb)     # Float[n, 56]
    voice = TensorflowPredict2D(graphFilename=str(m / f"{VOICE}.pb"), input="model/Placeholder", output="model/Softmax")(emb)   # Float[n, 2]
    classes = json.loads((m / f"{VOICE}.json").read_text(encoding="utf-8"))["classes"]                  # ["instrumental", "voice"]
    tmp = Path(out).with_name(Path(out).name + ".tmp.npz")
    np.savez(tmp, p_voice=np.asarray(voice)[:, classes.index("voice")], mood=np.asarray(mood))
    tmp.replace(out)
