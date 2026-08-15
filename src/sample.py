"""Generate text from a trained checkpoint (a quick local sampler).

    python -m src.sample --prompt "Once upon a time" --tokens 200

The real serving path is ember; this is here so you can eyeball the model
immediately after training without spinning up the server.
"""
from __future__ import annotations

import argparse

import torch

from config import GPTConfig, TrainConfig
from src.model import GPT
from src.tokenizer import EOT, Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tc = TrainConfig()
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=tc.device, weights_only=False)
    gc = GPTConfig(**ck["config"])
    model = GPT(gc).to(tc.device).eval()
    model.load_state_dict(ck["model"])

    tok = Tokenizer()
    ids = tok.encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=tc.device)
    out = model.generate(x, args.tokens, temperature=args.temperature, top_k=args.top_k)[0].tolist()

    # Stop at the first end-of-text after the prompt, if any.
    gen = out[len(ids):]
    if EOT in gen:
        gen = gen[:gen.index(EOT)]
    print(args.prompt + tok.decode(gen))


if __name__ == "__main__":
    main()
