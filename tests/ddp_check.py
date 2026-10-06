"""Two-process check of fit() on CPU (gloo), not part of the pytest suite:  python tests/ddp_check.py"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_engine import TinyLM, cfg_for, loader  # noqa: E402

from dcttgen.engine import fit, latest_checkpoint  # noqa: E402


class Stop(Exception):
    pass


def run(model, cfg, run_dir, stop_at=None):
    hook = model.on_step_end

    def on_step_end(step):
        hook(step)
        if step == stop_at:
            raise Stop

    model.on_step_end = on_step_end
    try:
        fit(model, loader(10), None, cfg, run_dir)
    except Stop:
        pass


def worker(rank, world, store, shared):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT="29533", ACCELERATE_USE_CPU="true", USE_LIBUV="0")
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=world)
    straight, crashed = Path(shared) / "straight", Path(shared) / "crashed"
    cfg = cfg_for("train.max_steps=6", "train.grad_accum=2", "train.save_every=2", "train.lr=0.03")
    a = TinyLM(dropout=0.2)
    run(a, cfg, str(straight))
    gathered = [torch.zeros_like(a.emb.weight) for _ in range(world)]
    dist.all_gather(gathered, a.emb.weight.detach().clone())
    assert torch.equal(gathered[0], gathered[1]), "ranks drifted apart: gradients were not synchronised"
    run(TinyLM(dropout=0.2), cfg, str(crashed), stop_at=3)  # every rank stops after step 3; the checkpoint is at step 2
    b = TinyLM(dropout=0.2, seed=99)
    run(b, cfg, str(crashed))
    assert torch.equal(a.emb.weight, b.emb.weight), "the resumed 2-process run differs from the uninterrupted one"
    if rank == 0:
        ckpt = Path(latest_checkpoint(straight))
        files = sorted(p.name for p in ckpt.iterdir())
        print("files:", files)
        print("safetensors keys:", sorted(load_file(str(ckpt / "model.safetensors"))))
        steps = [json.loads(line)["step"] for line in (straight / "log.jsonl").read_text().splitlines()]
        assert "random_states_0.pkl" in files and "random_states_1.pkl" in files  # one RNG stream per process
        assert steps == [1, 2, 3, 4, 5, 6]  # logged once, by the main process only
        print("DDP CHECK OK")


if __name__ == "__main__":
    shared = Path(tempfile.gettempdir()) / "dcttgen_ddp_check"
    shutil.rmtree(shared, ignore_errors=True)
    shared.mkdir()
    mp.spawn(worker, args=(2, (shared / "store").as_uri(), str(shared)), nprocs=2)
