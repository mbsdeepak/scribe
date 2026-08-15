"""GPT-2 BPE tokenizer via tiktoken.

Using the exact GPT-2 vocabulary (50257 tokens, EOT = 50256) is a deliberate
choice: it's what `ember` expects, so a model trained here serves there with no
tokenizer conversion.
"""
from __future__ import annotations

from typing import List

import tiktoken

EOT = 50256  # <|endoftext|> — also used as the story separator in the data


class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self._enc = tiktoken.get_encoding(name)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text, allowed_special=set())

    def encode_with_eot(self, text: str) -> List[int]:
        return self._enc.encode(text, allowed_special=set()) + [EOT]

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab
