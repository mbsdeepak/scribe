# scribe

A **small language model, trained from scratch in PyTorch** on TinyStories — a
~30M-parameter GPT that learns to write coherent little stories. It's the
language-side companion to my [from-scratch diffusion model](https://github.com/mbsdeepak/text-diffusion-fashion-mnist):
implement the transformer, the training loop, and the sampler myself rather than
calling a high-level library.

The payoff: scribe is deliberately **GPT-2-architecture-compatible**, so the
weights it trains serve directly in [**ember**](https://github.com/mbsdeepak/ember),
my from-scratch inference server — *I trained the model and built the engine that
runs it, and they compose.*

> **Status:** learning / portfolio project. The goal is to understand every part
> of an LLM pipeline end-to-end — the architecture, the next-token objective, the
> optimizer schedule, and pausable/resumable training — small enough to train on
> an Apple-Silicon MacBook (MPS), no cloud GPU. Results section fills in when the
> current training run completes.

---

## What it demonstrates

| Area | In this repo |
|------|--------------|
| **PyTorch from scratch** | Hand-written decoder-only transformer — multi-head causal attention, pre-LN blocks, GELU MLP, tied LM head — no `transformers`/`nanoGPT` dependency ([`src/model.py`](src/model.py)) |
| **Language-model training** | Next-token cross-entropy, AdamW, linear-warmup + cosine LR, gradient accumulation, global-norm clipping ([`src/train.py`](src/train.py)) |
| **Real training ergonomics** | Epoch-based training with **pause (Ctrl-C), automatic resume, `--until-epoch` / `--epochs` targets**; full-state checkpoints (model, optimizer, epoch, global step, RNG) |
| **Serving interop** | Same weight layout as [ember](https://github.com/mbsdeepak/ember)'s `GPT2`; [`src/export_ember.py`](src/export_ember.py) emits `safetensors` + `config.json` the engine loads directly |
| **The maths** | Every equation derived and tied to code in [`docs/math.md`](docs/math.md) |

---

## How it works

> 📄 **Full derivations, every equation:** [`docs/math.md`](docs/math.md)

### 1. The objective (next-token prediction)
A language model factorizes a sequence left-to-right, $p_\theta(x)=\prod_t p_\theta(x_t\mid x_{<t})$,
and is trained by maximum likelihood — i.e. minimizing the **cross-entropy**
between the predicted next-token distribution and the true next token:

```math
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\!\left(x_{t}\mid x_{<t}\right)
```

The target is just the input shifted by one; "teacher forcing" feeds the true
prefix at every position.

### 2. The model (a decoder-only transformer)
Token + learned position embeddings feed $L$ pre-LayerNorm blocks, each a
residual **causal self-attention** followed by a residual **GELU MLP**:

```math
Q,K,V = XW_{Q},XW_{K},XW_{V}\qquad
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\tfrac{QK^{\top}}{\sqrt{d_k}}+M\right)V
```

The causal mask $M$ (upper-triangular $-\infty$) is the one constraint that makes
this a valid autoregressive factorization: position $t$ never sees the future.
The output head shares the token-embedding matrix (**weight tying**).

### 3. Optimization
AdamW ($\beta=0.9,0.95$, decoupled weight decay on 2-D weights only) with **linear
warmup then cosine decay**, keyed off the global step so a paused run resumes
exactly on its LR curve. Gradient accumulation gives a larger effective batch
than memory would otherwise allow.

### 4. Generation
Autoregressive sampling with temperature / top-k / top-p. The production serving
path is [ember](https://github.com/mbsdeepak/ember) (paged KV cache + continuous
batching); [`src/sample.py`](src/sample.py) is a quick local sampler for eyeballing
the model right after training.

---

## Model

| | |
| :-- | :-- |
| Architecture | decoder-only transformer (GPT-2 family) |
| Parameters | **≈ 30.0M** (19.3M of it the tied 50257-token embedding) |
| Layers / heads / width | 6 / 6 / 384 |
| Context length | 256 tokens |
| Tokenizer | GPT-2 BPE (`tiktoken`, vocab 50257) |
| Data | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~2.1M synthetic short stories) |

---

## Results

> _Training in progress._ Target validation loss ≈ **1.2–1.5** (where TinyStories
> models start writing coherent stories); this section will be updated with the
> final loss curve and sample generations once the 3-epoch run finishes.

Reproduce once trained:

```bash
python -m src.sample --prompt "Once upon a time, a little robot" --tokens 200
```

---

## Project layout

```
config.py              # GPTConfig (model) + TrainConfig (epoch-based schedule)
src/
  tokenizer.py         # GPT-2 BPE via tiktoken (ember-compatible)
  model.py             # from-scratch GPT — training-time dense causal attention
  data.py              # TinyStories -> uint16 memmap; shuffled epoch DataLoader
  train.py             # epoch-based trainer: pause / resume / --until-epoch / --epochs
  sample.py            # local autoregressive sampler
  export_ember.py      # export weights (safetensors + config.json) for ember
scripts/
  smoke_test.py        # fast correctness checks (overfit one batch, no download)
docs/
  math.md              # full derivations
```

---

## Quickstart

> **Apple Silicon note:** use a **native arm64** Python (e.g. Homebrew's
> `/opt/homebrew/bin/python3.11`). A Rosetta/x86_64 Python can't install recent
> PyTorch and has no MPS. Check: `python3 -c "import platform;print(platform.machine())"`
> → should say `arm64`.

```bash
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fast correctness check — no download, a few seconds:
python -m scripts.smoke_test

# 2. Train (downloads the ~2GB TinyStories train set once, then a few hours on MPS):
python -m src.train --prepare --until-epoch 3

# 3. Generate from the trained checkpoint:
python -m src.sample --prompt "Once upon a time" --tokens 200

# 4. Export for the ember inference server:
python -m src.export_ember --ckpt checkpoints/best.pt --out ember_export
```

### Training controls

Training is epoch-based, pausable, and resumable:

```bash
python -m src.train --until-epoch 3     # train until epoch 3 (total)
python -m src.train --epochs 2          # train 2 MORE epochs from where it stopped
# Ctrl-C                                # finishes the step, saves last.pt, exits
python -m src.train                     # just run again to resume
python -m src.train --fresh             # ignore the checkpoint and start over
```

Checkpoints are written to `checkpoints/last.pt` (rolling, resumable) and
`checkpoints/best.pt` (lowest val loss) — every epoch, every 500 steps, and on
interrupt — so a pause loses at most a few hundred steps.

---

## Design choices & what I learned

- **GPT-2-compatible on purpose.** Learned position embeddings + LayerNorm + GELU
  + tied head, with module names matching ember's `GPT2`, so the trained weights
  serve with no conversion. It's the fastest way to close the train→serve loop; a
  modern Llama-style variant (RoPE/RMSNorm/SwiGLU) is the planned v2.
- **TinyStories is the right dataset for a small model.** It's synthetic text
  constrained to a toddler's vocabulary, so a 30M model can actually learn
  coherent grammar and simple narrative — the language analog of using
  Fashion-MNIST to learn diffusion.
- **The embedding table dominates.** 19.3M of ~30M params is the 50257-token
  embedding; weight tying (sharing it with the output head) is a big deal at this
  scale, both for parameter count and regularization.
- **Global-step LR schedule = clean resume.** Making the LR a pure function of the
  global step (not "iterations this run") is what lets a paused run pick up
  exactly on its cosine curve.
- **Epochs need a real DataLoader.** nanoGPT samples random windows (no real
  "epoch"). To make `--until-epoch`/`--epochs` meaningful I use non-overlapping
  shuffled windows so one pass over the corpus is a well-defined epoch.

## Honest limitations

- 30M params on TinyStories: it writes simple, coherent children's-story English,
  not general-purpose text. The point is understanding the method end-to-end.
- Context is 256 tokens — plenty for TinyStories, small by modern standards.
- Vanilla GPT-2 architecture, not the current (RoPE/RMSNorm/SwiGLU/GQA) recipe.

## Possible extensions

- **v2 architecture:** RoPE + RMSNorm + SwiGLU + grouped-query attention, and a
  matching backend in [ember](https://github.com/mbsdeepak/ember) so it serves
  both families.
- Scale up (more layers/width, longer context) on a real corpus.
- Publish trained weights to the Hugging Face Hub (as with the diffusion model).

---

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2019)
- Eldan & Li, *TinyStories* (2023)
- Karpathy, *nanoGPT* — reference minimal GPT training loop

## License

MIT
