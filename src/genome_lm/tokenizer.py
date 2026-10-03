from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DNATokenizer:
    """Tokenizer for DNA nucleotides with special MLM symbols."""

    vocab: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.vocab is None:
            object.__setattr__(
                self,
                "vocab",
                {
                    "A": 0,
                    "C": 1,
                    "G": 2,
                    "T": 3,
                    "[MASK]": 4,
                    "[PAD]": 5,
                    "N": 6,
                },
            )

    @property
    def inv_vocab(self) -> dict[int, str]:
        return {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def mask_id(self) -> int:
        return self.vocab["[MASK]"]

    @property
    def pad_id(self) -> int:
        return self.vocab["[PAD]"]

    @property
    def n_id(self) -> int:
        return self.vocab["N"]

    def encode(self, seq: str) -> List[int]:
        return [self.vocab.get(ch.upper(), self.n_id) for ch in seq]

    def decode(self, token_ids: List[int]) -> str:
        inv = self.inv_vocab
        return "".join(inv.get(idx, "N") for idx in token_ids)
