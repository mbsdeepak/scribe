"""TinyStories data pipeline.

`prepare()` downloads a TinyStories text split from the Hub, tokenizes it with
the GPT-2 BPE, and writes a flat `uint16` token stream to `data/<split>.bin`
(uint16 is enough: the vocab is 50257 < 65536). Stories are separated by the EOT
token so the model learns story boundaries.

`get_batch()` memory-maps that file and draws random contiguous windows — the
standard, dead-simple language-model training sampler. Targets are inputs shifted
by one, which is exactly the next-token objective.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from src.tokenizer import EOT, Tokenizer

# Candidate filenames on the Hub (dataset repo), newest naming first.
_SPLIT_FILES = {
    "train": ["TinyStories-train.txt", "TinyStoriesV2-GPT4-train.txt"],
    "val":   ["TinyStories-valid.txt", "TinyStoriesV2-GPT4-valid.txt"],
}

_STORY_SEP = "<|endoftext|>"  # how stories are delimited in the raw text


def _download(repo: str, split: str) -> str:
    from huggingface_hub import hf_hub_download

    last_err = None
    for fname in _SPLIT_FILES[split]:
        try:
            return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
        except Exception as e:  # try the next candidate name
            last_err = e
    raise RuntimeError(f"could not download {split} split from {repo}: {last_err}")


def prepare(repo: str, data_dir: str, split: str, limit: Optional[int] = None) -> str:
    """Tokenize a split to data/<split>.bin. `limit` caps the number of stories
    (handy for a fast smoke run). Returns the output path."""
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{split}.bin")
    src_path = _download(repo, split)

    tok = Tokenizer()
    n_stories, n_tokens = 0, 0
    buf: list[str] = []

    def flush_story(f, lines: list[str]) -> int:
        text = "".join(lines).strip()
        if not text:
            return 0
        ids = tok.encode(text) + [EOT]
        np.asarray(ids, dtype=np.uint16).tofile(f)
        return len(ids)

    with open(out_path, "wb") as fout, open(src_path, "r", encoding="utf-8") as fin:
        pbar = tqdm(fin, desc=f"tokenizing {split}", unit=" lines")
        for line in pbar:
            if line.strip() == _STORY_SEP:
                n_tokens += flush_story(fout, buf)
                buf = []
                n_stories += 1
                if limit and n_stories >= limit:
                    break
            else:
                buf.append(line)
        if buf and not (limit and n_stories >= limit):  # trailing story with no separator
            n_tokens += flush_story(fout, buf)
            n_stories += 1

    print(f"  {split}: {n_stories} stories, {n_tokens:,} tokens -> {out_path}")
    return out_path


class TokenWindows(torch.utils.data.Dataset):
    """Non-overlapping (input, target) windows over the token stream.

    One pass over this dataset == one epoch. Target is the input shifted by one,
    which is the next-token objective. Using non-overlapping windows makes an
    "epoch" a clean, well-defined single pass over the whole corpus.
    """

    def __init__(self, data_dir: str, split: str, block_size: int):
        self.path = os.path.join(data_dir, f"{split}.bin")
        self.block_size = block_size
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n = (len(self.data) - 1) // block_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        b = self.block_size
        start = i * b
        x = torch.from_numpy(self.data[start:start + b].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1:start + 1 + b].astype(np.int64))
        return x, y


def make_loader(data_dir: str, split: str, block_size: int, batch_size: int,
                shuffle: bool) -> torch.utils.data.DataLoader:
    ds = TokenWindows(data_dir, split, block_size)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=True,
        num_workers=0, pin_memory=False,  # 0 workers is safest on macOS/MPS
    )
