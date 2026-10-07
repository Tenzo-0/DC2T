"""python -m dc2t.infer --config C --prompt "..." --out out.wav [--duration S --bpm N --moods a,b --instruments "x,y"] [--seed N]
python -m dc2t.infer --config C --prompt "..." --out piece.npy --stage codes     (language-model environment)
python -m dc2t.infer --config C --codes piece.npy --out out.wav --stage audio    (codec environment)

Text -> music in two steps: prompt_to_codes (chapter 03's generate_codes) and codes_to_wav (chapter 02's Codec.decode).
The steps can run in different Python environments, because MuCodec and the language model need different library versions."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dc2t.plan import Plan, plan_sections

_LOADED: dict = {}


def _once(key, make):
    if key not in _LOADED:
        _LOADED[key] = make()
    return _LOADED[key]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lm_side(cfg):
    """(vocab, model), loaded once per process. Needs the language-model environment."""
    def make():
        from dc2t.lm.model import load_lm
        from dc2t.lm.vocab import Vocab
        if not cfg.lm.checkpoint:
            raise ValueError("lm.checkpoint is not set: there is no trained language model to generate with")
        vocab = Vocab.build(cfg)
        return vocab, load_lm(cfg, vocab).to(_device(), dtype=getattr(torch, cfg.infer.dtype)).eval()
    return _once(("lm", id(cfg)), make)


def load_codec(cfg):
    """The codec with its decoder, loaded once per process. Needs the codec environment."""
    def make():
        from dc2t.codec.codec import Codec
        return Codec.load(cfg, _device(), encoder=False)
    return _once(("codec", id(cfg)), make)


def clean_prompt(prompt, max_chars: int) -> str:
    """The prompt is untrusted text: one line, bounded length. (Control-token strings in it are harmless: Vocab.enc
    tokenises them as ordinary text.)"""
    if not isinstance(prompt, str):
        raise TypeError("the prompt must be a string")
    prompt = " ".join(prompt.split())                # collapses new lines, tabs and runs of spaces
    if not prompt:
        raise ValueError("the prompt is empty")
    if len(prompt) > max_chars:
        raise ValueError(f"the prompt has {len(prompt)} characters; the limit is {max_chars}")
    return prompt


def make_plan(duration, bpm, moods, instruments) -> Plan | None:
    """All four fields -> a Plan (which validates them); none -> None (the model writes the plan); anything else is an error."""
    given = [x is not None for x in (duration, bpm, moods, instruments)]
    if not any(given):
        return None
    if not all(given):
        raise ValueError("give all of duration, bpm, moods and instruments, or none of them (the model then writes the plan)")
    return Plan(bpm, duration, plan_sections(duration), list(moods), list(instruments))


def prompt_to_codes(prompt: str, cfg, *, duration: int | None = None, bpm: int | None = None, moods: list[str] | None = None,
                    instruments: list[str] | None = None, seed: int | None = None) -> tuple[Plan, torch.Tensor]:
    """-> (the plan that was used, codes Long[K, 25 * plan.duration])."""
    caption = clean_prompt(prompt, cfg.infer.max_prompt_chars)
    plan = make_plan(duration, bpm, moods, instruments)          # everything is validated before any model is loaded
    from dc2t.lm import generate as lm_generate               # imported here so that the codec environment never imports it
    vocab, model = load_lm_side(cfg)
    i = cfg.infer
    return lm_generate.generate_codes(model, vocab, caption, plan, temperature=i.temperature, top_k=i.top_k, top_p=i.top_p, seed=seed)


def codes_to_wav(codes: torch.Tensor, cfg, *, seed: int | None = None) -> tuple[torch.Tensor, int]:
    """codes Long[K, T] -> (wav Float[T * sample_rate // 25] in [-1, 1], sample_rate)."""
    codec, i = load_codec(cfg), cfg.infer
    return codec.decode(codes[None], steps=i.steps, cfg_scale=i.cfg_scale, seed=seed)[0], codec.sample_rate


def text_to_music(prompt: str, cfg, *, duration: int | None = None, bpm: int | None = None, moods: list[str] | None = None,
                  instruments: list[str] | None = None, seed: int | None = None) -> tuple[torch.Tensor, int]:
    """Both steps in one process (contract 9): -> (wav Float[N], sample_rate). Needs one environment that can import both sides."""
    _, codes = prompt_to_codes(prompt, cfg, duration=duration, bpm=bpm, moods=moods, instruments=instruments, seed=seed)
    return codes_to_wav(codes, cfg, seed=seed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("all", "codes", "audio"), default="all")
    ap.add_argument("--prompt")
    ap.add_argument("--codes", help="a .npy file written by --stage codes")
    ap.add_argument("--duration", type=int)
    ap.add_argument("--bpm", type=int)
    ap.add_argument("--moods", help="comma-separated")
    ap.add_argument("--instruments", help="comma-separated")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--override", action="append", default=[], metavar="a.b=value")
    args = ap.parse_args(argv)
    from dc2t.config import load_config
    cfg = load_config(args.config, args.override)
    items = lambda s: None if s is None else [x.strip() for x in s.split(",") if x.strip()]
    if args.stage == "audio":
        if not args.codes:
            ap.error("--stage audio needs --codes")
        codes = torch.from_numpy(np.load(args.codes).astype(np.int64))
    else:
        if not args.prompt:
            ap.error("--prompt is required")
        plan, codes = prompt_to_codes(args.prompt, cfg, duration=args.duration, bpm=args.bpm, moods=items(args.moods),
                                      instruments=items(args.instruments), seed=args.seed)
        print("plan:", plan.to_text())
        if args.stage == "codes":
            np.save(args.out, codes.cpu().numpy().astype(np.int16))
            return
    import soundfile as sf
    wav, sr = codes_to_wav(codes, cfg, seed=args.seed)
    sf.write(args.out, wav.numpy(), sr, subtype="PCM_16")
    print(f"wrote {args.out}: {len(wav) / sr:.0f} s at {sr} Hz")


if __name__ == "__main__":
    main()
