"""Configuration (contract section 9): configs/base.yaml <- overlay file(s) <- "a.b=value" overrides.

Strict on purpose: a key that is not already in base.yaml is an error, so a typo cannot silently do nothing.
"""
import copy
import difflib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # repository root: the directory that holds configs/


class Config(dict):
    """A dict whose items are also attributes: cfg.codec.tag is cfg["codec"]["tag"]."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}


def _wrap(x):
    return Config({k: _wrap(v) for k, v in x.items()}) if isinstance(x, dict) else x


def _coerce(key, old, new):
    """Convert new to the type of old or raise TypeError. PyYAML reads 1e-4 (no dot) as a string; float() fixes that."""
    if old is None or (new is None and isinstance(old, str)):
        return new  # a null default accepts anything; a string key may be reset to null
    if isinstance(old, bool):
        ok = isinstance(new, bool)
    elif isinstance(old, (int, float)):
        try:
            num = None if isinstance(new, bool) else float(new)
        except (TypeError, ValueError):
            num = None
        if num is not None and (isinstance(old, float) or num.is_integer()):
            return num if isinstance(old, float) else int(num)
        ok = False
    else:
        ok = isinstance(new, type(old))
    if not ok:
        raise TypeError(f"{key}: expected {type(old).__name__}, got {new!r}")
    return new


def _merge(base: dict, over: dict, prefix: str = "") -> None:
    for key, val in over.items():
        if key not in base:
            near = difflib.get_close_matches(key, list(base), n=1)
            hint = f" (did you mean {(prefix + near[0])!r}?)" if near else ""
            raise KeyError(f"unknown config key {(prefix + key)!r}{hint}; add it to configs/base.yaml first")
        if isinstance(base[key], dict) and isinstance(val, dict):
            _merge(base[key], val, f"{prefix}{key}.")
        else:
            base[key] = _coerce(f"{prefix}{key}", base[key], val)


def _override(cfg: dict, item: str) -> None:
    dotted, sep, raw = item.partition("=")
    if not sep or not dotted:
        raise ValueError(f"override {item!r} must look like section.key=value")
    parents, leaf = dotted.split(".")[:-1], dotted.split(".")[-1]
    node = cfg
    for p in parents:
        node = node.get(p) if isinstance(node, dict) else None
    old = node.get(leaf) if isinstance(node, dict) else None
    val = yaml.safe_load(raw)
    if isinstance(old, str) and val is not None and not isinstance(val, str):
        val = raw  # codec.tag=123 stays the string "123"
    for p in [leaf, *reversed(parents)]:
        val = {p: val}
    _merge(cfg, val)


def load_config(path: str, overrides: list[str] = ()) -> Config:
    """Deep-merge path over configs/base.yaml, then apply "a.b=value" overrides.

    path may name several overlays separated by commas ("configs/bootstrap_k1.yaml,configs/pilot.yaml"); later ones win.
    Relative entries under paths: become absolute (against the repository root), so a later os.chdir cannot break them.
    """
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    for p in str(path).split(","):
        over = yaml.safe_load(Path(p.strip()).read_text(encoding="utf-8")) or {}
        if not isinstance(over, dict):
            raise ValueError(f"{p}: the top level must be a mapping")
        _merge(cfg, over)
    for item in overrides:
        _override(cfg, item)
    for k, v in cfg["paths"].items():
        if not os.path.isabs(v):
            cfg["paths"][k] = str(ROOT / v)
    return _wrap(cfg)
