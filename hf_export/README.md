---
license: mit
tags:
  - gpt
  - language-model
  - text-generation
  - tinystories
  - pytorch
  - from-scratch
  - slm
library_name: pytorch
pipeline_tag: text-generation
datasets:
  - roneneldan/TinyStories
---

# scribe

A **~30M-parameter small language model built from scratch in PyTorch** and trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). It's a decoder-only GPT —
hand-written multi-head causal attention, pre-LayerNorm blocks, GELU MLP, tied LM head — with no
`transformers`/`nanoGPT` modelling code. The architecture is deliberately **GPT-2-compatible**, so
these weights load directly into [`ember`](https://github.com/mbsdeepak/ember), a from-scratch
LLM inference server.

📦 **Code, training, and full write-up (incl. the maths):** https://github.com/mbsdeepak/scribe

## What it does

Give it a prompt and it continues it as a simple, coherent children's story:

> **"Once upon a time"** → *there was a little girl named Lily. She loved to read magazines. One
> day, she found a magazine in her mom's room... "Mommy, look at that magazine!" Lily said... Her
> mom looked at the magazine and smiled. "That's very nice, Lily. But be careful with it, it might
> break."*
>
> **"One day, a boy named Deepak"** → *went to the park every day. He saw a large tree and wanted to
> climb it. He started to climb it, but it was too heavy for him. He tried to jump, but he fell.*

## Model

| | |
| :-- | :-- |
| Architecture | decoder-only transformer (GPT-2 family) |
| Parameters | ~30.0M (19.3M of it the tied 50257-token embedding) |
| Layers / heads / width | 6 / 6 / 384 |
| Context length | 256 tokens |
| Tokenizer | GPT-2 BPE (`tiktoken`) |
| Training | TinyStories, AdamW + warmup/cosine, ~3.4h on Apple-Silicon MPS |
| Validation loss | **1.74** (cross-entropy) |

## Files

- `model.safetensors` — the weights (fp32)
- `config.json` — architecture config to rebuild the model

## Usage

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json, torch

# get the code: git clone https://github.com/mbsdeepak/scribe
from config import GPTConfig
from src.model import GPT
from src.tokenizer import Tokenizer

cfg_json = json.load(open(hf_hub_download("mbsdeepak/scribe", "config.json")))
gc = GPTConfig(n_layer=cfg_json["n_layer"], n_head=cfg_json["n_head"],
               n_embd=cfg_json["n_embd"], block_size=cfg_json["max_position"])
model = GPT(gc).eval()
model.load_state_dict(load_file(hf_hub_download("mbsdeepak/scribe", "model.safetensors")))

tok = Tokenizer()
ids = torch.tensor([tok.encode("Once upon a time")])
out = model.generate(ids, max_new_tokens=120, temperature=0.8, top_k=200)[0].tolist()
print(tok.decode(out))
```

Or serve it with [ember](https://github.com/mbsdeepak/ember) for a streaming, OpenAI-compatible API.

## Limitations

A 30M model on TinyStories writes simple, coherent children's-story English — not general-purpose
text. It's a learning/portfolio project: the goal is understanding an LLM end-to-end (architecture,
training, serving), not competing with production models. Context is 256 tokens; architecture is
vanilla GPT-2 (no RoPE/RMSNorm/SwiGLU).

## License

MIT
