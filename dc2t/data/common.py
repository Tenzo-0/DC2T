"""Helpers shared by the data stages: atomic writes, JSONL, markers for safe re-runs, a failure-tolerant runner."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def tmp_name(path) -> Path:
    """Sibling temp path. Write there, then os.replace(): a crash never leaves a half-written final file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    tmp = tmp_name(path)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def fingerprint(path) -> list[int]:
    """[size, mtime_ns] of an input file: cheap, and it changes whenever the file is rewritten."""
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def load_marker(path, key):
    """The payload saved by save_marker if it was saved for the same `key` (inputs + parameters), else None.
    A stage skips an item only when its marker matches, so changing an upstream file or a setting redoes it."""
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m["payload"] if m.get("key") == json.loads(json.dumps(key)) else None


def save_marker(path, key, payload) -> None:
    """Write the marker LAST, after every output of the item exists: a crash then leaves no marker."""
    tmp = tmp_name(path)
    tmp.write_text(json.dumps({"key": key, "payload": payload}), encoding="utf-8")
    os.replace(tmp, path)


class _Safe:
    """Call fn(item); turn an exception into a record. A class, not a closure, so it can be pickled."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, item):
        try:
            self.fn(item)
            return None
        except Exception as e:   # noqa: BLE001 - one bad file must not stop a 2,000-hour run
            return {"item": str(item), "error": f"{type(e).__name__}: {e}"}


def run_stage(items, fn, *, workers: int = 1, failed_path=None) -> tuple[int, int]:
    """Run fn(item) over all items; failures are recorded in failed_path (JSONL) and the rest still run.
    Returns (done, failed). With workers > 1, fn must be a module-level function."""
    items, safe = list(items), _Safe(fn)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(safe, items))
    else:
        results = [safe(i) for i in items]
    failed = [r for r in results if r]
    if failed_path is not None:
        write_jsonl(failed_path, failed)
    return len(items) - len(failed), len(failed)
