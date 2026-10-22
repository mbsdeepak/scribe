---
name: scribe-slm
description: personal project scribe — a from-scratch small language model (GPT) trained on TinyStories
metadata: 
  node_type: memory
  type: project
  originSessionId: bb2e7363-3812-40fc-bd3c-f1d31a7d1307
---

Personal project at `~/projects/scribe` — a **small language model built from
scratch** in PyTorch, the language-side companion to [[personal-github-identity]]'s
diffusion project. ~30M-param decoder-only GPT (6 layers / 6 heads / 384 dim,
context 256, GPT-2 BPE via tiktoken), trained on **TinyStories**.

Deliberately **GPT-2-architecture-compatible** (module names/layout match
[[ember-inference-server]]'s GPT2) so the trained weights serve in ember directly
— that's the payoff: trained the model AND built the engine that runs it.

**Trained (2026-08-16):** to step 3700, **val loss 1.74** (down from 10.8), LR
fully decayed via warmup+cosine; ~3.4h total on MPS. Writes coherent little
stories with dialogue ("Once upon a time, there was a little girl named Lily...";
handles a "Deepak" protagonist). `checkpoints/best.pt` (344MB, gitignored).

**Key gotcha (16GB Mac):** `batch_size=32` OOM'd — the `[B,T,vocab]` training
logits (`[32,256,50257]`≈1.6GB + grad) blew past RAM and MPS wedged in a Metal
wait while swapping. Fix: `batch_size=8, grad_accum=16` (same effective batch
128). Also torch 2.6 `weights_only=True` rejects numpy RNG in checkpoints → load
with `weights_only=False`. Uses arm64 venv ([[macos-arm64-python]]).

**Trainer features:** epoch-based, pausable (Ctrl-C→save+exit / SIGINT for
background), auto-resume from `last.pt`, `--until-epoch` / `--epochs` /
`--max-steps` (max-steps sets stop point AND cosine horizon so LR decays by
then). Full-state checkpoints (model/optim/step/RNG) every 500 steps + on interrupt.

**Status (2026-08-16):** **PUBLISHED** — github.com/mbsdeepak/scribe (public,
signed/Verified) + weights on huggingface.co/mbsdeepak/scribe (model.safetensors
+ config.json + model card). Push uses the gmail SSH key ([[personal-github-identity]]).
Next: (1) **serve in ember** — needs a small ember addition (non-preset ModelConfig
from config.json + a local/HF-safetensors loader; scribe's keys already match
ember's GPT2 exactly, no transpose), (2) optional more training / v2 modern arch
(RoPE/RMSNorm/SwiGLU + matching ember backend).
