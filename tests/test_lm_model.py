import dataclasses
import math
import pathlib
import tempfile

import torch
import torch.nn.functional as F
from lm_testkit import CAPTION, backbone_dir, get_vocab, make_cfg, make_doc, raises, random_rows_model, run_all
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from dc2t.lm.data import collate
from dc2t.lm.generate import build_schedule
from dc2t.lm.model import load_lm, resize_for_audio
from dc2t.lm.sequence import instruct_ids, metadata_ids

# K = 2 codebooks of 16 codes at 1 frame per second: a 30 s document has ~250 tokens, so the reference below can afford [B, S, 151,702] logits
CFG, V = make_cfg(2, 16, 1), get_vocab(2, 16, 1)


def make_batch(specs):
    """specs: (duration, loss_on_plan) per document -> (padded batch, the range each position's target is scored against)."""
    docs, ranges = [], []
    for duration, on in specs:
        _, plan, (ids, labels) = make_doc(V, duration, loss_on_plan=on, seed=duration)
        docs.append({"input_ids": ids, "labels": labels})
        r = [(0, V.audio_offset)] * (len(instruct_ids(V, CAPTION)) + len(metadata_ids(V, plan.to_text())))
        for lo, hi, n in build_schedule(V, plan.sections):                 # decoding schedule: a free run -> its codebook; everything else -> text rows
            r += [(lo, hi) if hi - lo > 1 else (0, V.audio_offset)] * n
        assert len(r) == len(ids)
        ranges.append(r)
    return collate(docs, V.pad), ranges


def masked_full_loss(model, batch, ranges):
    """Reference: logits of all 151,702 rows from the Hugging Face module, every column outside the target's range set to -inf."""
    logits = model.lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()[:, :-1]   # Float[B, S-1, size]; row j-1 predicts token j
    allowed = torch.ones_like(logits, dtype=torch.bool)                                                              # unlabelled rows stay untouched (an all -inf row would give NaN gradients)
    for b, rng in enumerate(ranges):
        for j in range(1, len(rng)):
            if batch["labels"][b, j] != -100:
                allowed[b, j - 1] = False
                allowed[b, j - 1, rng[j][0]:rng[j][1]] = True
    return F.cross_entropy(logits.masked_fill(~allowed, float("-inf")).flatten(0, 1), batch["labels"][:, 1:].flatten(), ignore_index=-100)


def test_new_rows_are_initialised_uniformly():
    lm = AutoModelForCausalLM.from_pretrained(backbone_dir(), dtype=torch.float32)
    w = lm.get_input_embeddings().weight
    old = w.detach()[:V.eod].clone()
    w.data[V.eod:] = 5.0                                                    # rows 151,665 .. 151,935 exist in Qwen2.5 but are untrained; poison them
    resize_for_audio(lm, V)
    w = lm.get_input_embeddings().weight.detach()
    assert w.shape == (V.size, 32) == (151702, 32) and lm.config.vocab_size == V.size
    assert lm.get_output_embeddings().weight is lm.get_input_embeddings().weight                           # tied weights stay tied
    assert torch.equal(w[:V.eod], old)                                      # pre-trained rows untouched
    new = w[V.eod:]                                                         # 5 special tokens (their ids already had rows) + 32 audio rows (they did not)
    assert (new - old.mean(0)).abs().max() < 1e-3 * old.std()               # both groups: the mean of the old rows
    assert (new[:5] - new[5:].mean(0)).abs().max() < 1e-3 * old.std()


def test_initial_cross_entropy_of_every_level_is_ln_V():
    model = load_lm(CFG, V)
    batch, _ = make_batch([(30, True)])
    out = model(batch)
    assert all(abs(out[f"ce_k{k}"].item() - math.log(16)) < 1e-3 for k in range(2)), out
    assert out["loss"].requires_grad and out["tokens"].item() == (batch["labels"][:, 1:] != -100).sum().item()


def test_restricted_loss_equals_the_full_vocabulary_loss_with_the_same_mask():
    model = random_rows_model()
    batch, ranges = make_batch([(30, True), (40, False)])                  # two lengths (padding) and both loss_on_plan modes
    out, ref = model(batch), masked_full_loss(model, batch, ranges)
    assert torch.allclose(out["loss"], ref, atol=1e-5), (out["loss"].item(), ref.item())
    g1 = torch.autograd.grad(out["loss"], list(model.parameters()), allow_unused=True)
    g2 = torch.autograd.grad(masked_full_loss(model, batch, ranges), list(model.parameters()), allow_unused=True)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g1, g2) if a is not None or b is not None)   # same gradients, not just the same value
    naive = F.cross_entropy(model.lm(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits[:, :-1].float().flatten(0, 1),
                            batch["labels"][:, 1:].flatten(), ignore_index=-100)
    assert abs(naive.item() - out["loss"].item()) > 1e-2                    # the model's built-in loss is a different objective


def test_padding_does_not_change_the_loss():
    model = random_rows_model()
    batch, _ = make_batch([(30, True), (60, True)])
    one, two = make_batch([(30, True)])[0], make_batch([(60, True)])[0]
    pad = batch["input_ids"].shape[1] - one["input_ids"].shape[1]          # padding added to the shorter document
    assert pad > 0 and (batch["input_ids"][0, -pad:] == V.pad).all() and (batch["labels"][0, -pad:] == -100).all() and (batch["attention_mask"][0, -pad:] == 0).all()
    ab, a, b = model(batch), model(one), model(two)
    assert ab["tokens"] == a["tokens"] + b["tokens"]                       # padding positions are not counted
    assert torch.allclose(ab["loss"] * ab["tokens"], a["loss"] * a["tokens"] + b["loss"] * b["tokens"], rtol=1e-4)


def test_checkpoint_round_trip_and_codec_guard():
    model = random_rows_model()
    with tempfile.TemporaryDirectory() as d:
        run = pathlib.Path(d) / "lm_pretrain" / "x"
        V.save(run)
        step = run / "step_0000010"
        step.mkdir()
        save_file({k: t for k, t in model.state_dict().items() if k != "lm.lm_head.weight"}, str(step / "model.safetensors"))   # shared tensor stored once, as the engine does
        again = load_lm(CFG, V, str(step))
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values()))
        assert torch.equal(load_lm(make_cfg(2, 16, 1, checkpoint=str(step)), V).head_weight, model.head_weight)     # cfg.lm.checkpoint is the default
        assert raises(load_lm, CFG, dataclasses.replace(V, codec_tag="another_rvq"), str(step))                     # tokens of another RVQ


def test_gradient_checkpointing_gives_the_same_gradients():
    a = random_rows_model()
    b = load_lm(make_cfg(2, 16, 1, grad_checkpointing=True), V)
    b.load_state_dict(a.state_dict())
    batch, _ = make_batch([(30, True)])
    for m in (a, b):
        m.train()
    ga = torch.autograd.grad(a(batch)["loss"], list(a.parameters()))
    gb = torch.autograd.grad(b(batch)["loss"], list(b.parameters()))
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(ga, gb))


def test_a_tiny_model_can_overfit_one_batch():
    model = load_lm(CFG, V)
    batch, _ = make_batch([(30, True), (40, True)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    first = model(batch)["loss"].item()
    for _ in range(80):
        opt.zero_grad()
        loss = model(batch)["loss"]
        loss.backward()
        opt.step()
    assert first > math.log(16) and loss.item() < 0.5 * first, (first, loss.item())


if __name__ == "__main__":
    run_all(globals())
