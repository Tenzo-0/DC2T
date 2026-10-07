import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # lets `python tests/test_x.py` run without installing the package
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dc2t.data.common import read_jsonl
from dc2t.data.quality import centre_excerpt, combine_scores, parse_fadtk_csv, run, select_discard

try:
    import soundfile as sf
except ImportError:
    sf = None


def test_parse_fadtk_csv():
    text = "/p/pool/r001.flac,12.5\n/p/pool/r002.flac,nan_x\n/p/pool/r003.flac,0.25\n\n"
    assert parse_fadtk_csv(text) == {"r001": 12.5, "r003": 0.25}          # headerless; unparsable scores are skipped
    assert parse_fadtk_csv("C:\\data\\pool\\r9.flac,3.0") == {"r9": 3.0}


def scores(ids, worst, scale):
    """100 plausible per-file scores; the first `worst` files are 8 times worse than the rest."""
    base = np.random.default_rng(1).lognormal(1.0, 0.2, len(ids))
    return {i: float(scale * b * (8 if k < worst else 1)) for k, (i, b) in enumerate(zip(ids, base))}


def test_fad_discards_the_high_tail_and_no_embedding_dominates():
    ids = [f"r{i:03d}" for i in range(100)]
    per_model = {"vggish": scores(ids, 5, 1.0), "clap-laion-music": scores(ids, 5, 1000.0), "encodec-emb": scores(ids, 5, 0.01)}
    z, failed = combine_scores(per_model, ids)
    assert failed == [] and len(z) == 100
    assert select_discard(z, drop_fraction=0.05) == set(ids[:5])          # R5: the 5 worst files, not the 5 best
    assert select_discard(z, z_max=5.0) == set(ids[:5])                     # bad files sit at z of about 11-14, good ones below 3
    assert min(z[i] for i in ids[:5]) > max(z[i] for i in ids[5:])        # the bad files are separated, not just ranked first
    # scale invariance: x1000 on one model changes nothing, whereas a raw mean would be decided by that model alone
    z2, _ = combine_scores({**per_model, "encodec-emb": {i: v * 1000 for i, v in per_model["encodec-emb"].items()}}, ids)
    assert max(abs(z[i] - z2[i]) for i in ids) < 1e-9
    raw = {i: np.mean([m[i] for m in per_model.values()]) for i in ids}
    assert max(abs(raw[i] - per_model["clap-laion-music"][i] / 3) for i in ids) < 0.1 * max(raw.values())


def test_missing_or_invalid_scores_are_failures_not_passes():
    ids = ["a", "b", "c", "d"]
    per_model = {"m1": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "m2": {"a": 1.0, "b": 2.0, "c": 3.0}}   # d missing from m2
    z, failed = combine_scores(per_model, ids)
    assert failed == ["d"] and set(z) == {"a", "b", "c"}
    per_model["m2"]["d"] = float("nan")
    assert combine_scores(per_model, ids)[1] == ["d"]
    per_model["m2"]["d"] = -0.3                                           # a negative FAD is numerical garbage
    assert combine_scores(per_model, ids)[1] == ["d"]


def test_centre_excerpt():
    y = np.arange(100)
    assert list(centre_excerpt(y, 4, 10)) == list(range(30, 70))
    assert list(centre_excerpt(y, 20, 10)) == list(range(100))           # shorter than the excerpt: whole file


def make_cfg(root, **data):
    d = dict(vocal_prob=0.5, vocal_max_recording=0.2, fad_excerpt_s=3, fad_models=["vggish", "encodec-emb"],
             fad_reference="pool", fad_drop_fraction=0.15, fad_z_max=None)
    return SimpleNamespace(paths=SimpleNamespace(data_root=str(root)), data=SimpleNamespace(**{**d, **data}))


def test_stage_applies_both_gates_and_resumes():
    if sf is None:
        return print("skipped: soundfile not installed")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "work" / "rec").mkdir(parents=True)
        (root / "work" / "embed").mkdir(parents=True)
        for k in range(8):                                                # r0..r7, 6 s each at 8 kHz
            sf.write(root / "work" / "rec" / f"r{k}.flac", np.zeros(48000, dtype=np.int16) + k, 8000, format="FLAC", subtype="PCM_16")
            voice = np.full(100, 0.9 if k == 7 else 0.05)                 # r7 is a vocal recording
            np.savez(root / "work" / "embed" / f"r{k}.npz", p_voice=voice)
        calls = []

        def fake_fad(model, baseline, eval_dir, csv):                     # stands in for the fadtk command line
            calls.append(model)
            files = sorted(Path(eval_dir).glob("*.flac"))
            assert {f.stem for f in files} == {f"r{k}" for k in range(7)}  # the vocal recording is not scored
            assert all(sf.info(f).duration == 3.0 for f in files)          # excerpts of fad_excerpt_s seconds
            csv.write_text("".join(f"{f},{10.0 if f.stem == 'r3' else 1.0 + int(f.stem[1:]) / 100}\n" for f in files))

        rows = {r["recording_id"]: r for r in run(make_cfg(root), fad=fake_fad)}
        assert rows["r7"]["status"] == "vocal" and rows["r3"]["status"] == "fad_high"
        assert [r["status"] for k, r in sorted(rows.items()) if k not in ("r3", "r7")] == ["ok"] * 6
        assert sorted(r["recording_id"] for r in read_jsonl(root / "work" / "selected.jsonl")) == ["r0", "r1", "r2", "r4", "r5", "r6"]
        assert calls == ["vggish", "encodec-emb"]
        run(make_cfg(root), fad=fake_fad)
        assert calls == ["vggish", "encodec-emb"]                         # same pool: the CSVs are reused, fadtk is not called


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
