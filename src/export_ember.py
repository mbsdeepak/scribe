"""Export a trained checkpoint into the format ember serves.

scribe's `GPT` was built with the same module names and tensor layout as ember's
inference-time `GPT2`, so "conversion" is really just: save the weights as
safetensors and write a small config.json describing the shape. ember loads both
directly (see ember's local-weights loader).

    python -m src.export_ember --ckpt checkpoints/ckpt.pt --out ember_export
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import save_file

from config import GPTConfig
from src.model import GPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--out", default="ember_export")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gc = GPTConfig(**ck["config"])
    model = GPT(gc)
    model.load_state_dict(ck["model"])

    os.makedirs(args.out, exist_ok=True)
    # Contiguous CPU float32 tensors for safetensors.
    state = {k: v.contiguous().float().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(args.out, "model.safetensors"))

    # ember's ModelConfig fields — enough for it to build the matching GPT2 module.
    ember_cfg = {
        "name": "scribe",
        "n_layer": gc.n_layer,
        "n_head": gc.n_head,
        "n_kv_head": gc.n_head,
        "n_embd": gc.n_embd,
        "vocab_size": gc.vocab_size,
        "max_position": gc.block_size,
        "dtype": "float32",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(ember_cfg, f, indent=2)

    n = sum(v.numel() for v in state.values())
    print(f"exported {len(state)} tensors ({n/1e6:.2f}M params) -> {args.out}/")
    print("  model.safetensors + config.json  (ember-loadable)")


if __name__ == "__main__":
    main()
