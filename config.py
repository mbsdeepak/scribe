"""Central configuration — model shape and training hyperparameters in one place.

The default model is ~30M parameters: small enough to train from scratch on a
Mac (MPS) in a few hours, large enough that on TinyStories it writes coherent
little stories. The architecture is deliberately GPT-2-shaped (learned position
embeddings, LayerNorm, GELU, tied LM head) so the trained weights load straight
into the `ember` inference server with no conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class GPTConfig:
    # GPT-2-family shape. n_kv_head == n_head (no grouped-query attention) so the
    # weights are byte-identical in layout to ember's GPT2 module.
    vocab_size: int = 50257       # GPT-2 BPE (tiktoken "gpt2")
    block_size: int = 256         # context length; TinyStories are short
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True             # GPT-2 uses bias in Linear + LayerNorm

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    # data
    dataset_repo: str = "roneneldan/TinyStories"
    data_dir: str = "data"

    # schedule is epoch-based; one epoch = one shuffled pass over the corpus.
    epochs: int = 3               # default target horizon (overridable per run)

    # optimization (AdamW + cosine schedule with linear warmup, keyed off global step)
    # batch_size kept small so the [B, T, vocab] training logits fit in 16GB MPS;
    # grad_accum makes up the effective batch (8 * 16 = 128).
    batch_size: int = 8
    grad_accum: int = 16          # effective batch = batch_size * grad_accum = 128
    warmup_steps: int = 200
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # eval / checkpoint cadence (in optimizer steps)
    eval_interval: int = 500      # eval on val + write a mid-epoch checkpoint
    eval_iters: int = 100
    log_interval: int = 20

    # io / runtime
    out_dir: str = "checkpoints"
    seed: int = 1337
    device: str = field(default_factory=pick_device)
    dtype: str = "float32"        # float32 is the safe choice on MPS


def get_configs() -> tuple[GPTConfig, TrainConfig]:
    return GPTConfig(), TrainConfig()
