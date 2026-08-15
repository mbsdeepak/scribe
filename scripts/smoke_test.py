"""Fast correctness checks — no data download, runs in seconds on CPU.

1. shapes: forward produces [B,T,vocab] logits and a scalar loss.
2. sanity of the loss at init: cross-entropy ~= ln(vocab_size) for a fresh model.
3. overfit-one-batch: trained repeatedly on a SINGLE random batch, the loss must
   collapse toward zero. If the model, the causal mask, or the loss wiring were
   wrong, it could not memorize one batch — this is the single most useful test.
4. weight-tying + key names match what ember expects.
5. generate() runs and returns the right length.
"""
from __future__ import annotations

import math

import torch

from config import GPTConfig
from src.model import GPT


def main() -> None:
    torch.manual_seed(0)
    gc = GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(gc)
    B, T = 4, gc.block_size

    x = torch.randint(0, gc.vocab_size, (B, T))
    y = torch.randint(0, gc.vocab_size, (B, T))

    # (1) shapes
    logits, loss = model(x, y)
    assert logits.shape == (B, T, gc.vocab_size), logits.shape
    assert loss.dim() == 0
    print(f"[1] shapes OK  logits={tuple(logits.shape)}")

    # (2) init loss ~ ln(V)
    expected = math.log(gc.vocab_size)
    print(f"[2] init loss {loss.item():.3f}  (expected ~{expected:.3f} = ln {gc.vocab_size})")
    assert abs(loss.item() - expected) < 1.0

    # (4) weight tying + ember-compatible key names
    keys = set(model.state_dict().keys())
    for expected_key in ("wte.weight", "wpe.weight", "ln_f.weight",
                          "h.0.attn.c_attn.weight", "h.0.attn.c_proj.weight",
                          "h.0.mlp.c_fc.weight", "h.0.mlp.c_proj.weight",
                          "h.0.ln_1.weight", "h.0.ln_2.weight"):
        assert expected_key in keys, f"missing ember-compatible key: {expected_key}"
    assert "lm_head.weight" not in keys, "LM head must be tied to wte, not a separate param"
    print(f"[4] ember-compatible keys OK  ({len(keys)} tensors, head tied to wte)")

    # (3) overfit one batch
    opt = model.configure_optimizers(0.0, 3e-3, (0.9, 0.95))
    first = loss.item()
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    last = loss.item()
    print(f"[3] overfit one batch: {first:.3f} -> {last:.4f}")
    assert last < 0.1, f"model failed to memorize one batch (loss {last:.3f})"

    # (5) generate
    out = model.generate(x[:, :1], max_new_tokens=10, top_k=5)
    assert out.shape == (B, 11), out.shape
    print(f"[5] generate OK  {tuple(out.shape)}")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
