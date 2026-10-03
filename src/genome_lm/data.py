from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import Dataset

from .tokenizer import DNATokenizer


@dataclass
class DNAWindow:
    seq_id: str
    start: int
    end: int
    strand: str
    sequence: str
    repeat_fraction: float


def _repeat_fraction(window: str) -> float:
    if not window:
        return 0.0
    repeat_count = sum(1 for ch in window if ch.islower())
    return repeat_count / len(window)


def iter_fasta_windows(
    fasta_path: Path,
    window_size: int,
    stride: int,
    max_windows: int | None = None,
    min_acgt_fraction: float = 0.9,
) -> Iterable[DNAWindow]:
    """Yield fixed-length windows with genomic coordinates from FASTA."""
    produced = 0
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        raw = str(record.seq)
        upper = raw.upper()
        for start in range(0, max(0, len(upper) - window_size + 1), stride):
            end = start + window_size
            window_upper = upper[start:end]
            if len(window_upper) != window_size:
                continue

            acgt = sum(1 for ch in window_upper if ch in {"A", "C", "G", "T"})
            if (acgt / window_size) < min_acgt_fraction:
                continue

            window_raw = raw[start:end]
            yield DNAWindow(
                seq_id=record.id,
                start=start,
                end=end,
                strand="+",
                sequence=window_upper,
                repeat_fraction=_repeat_fraction(window_raw),
            )

            produced += 1
            if max_windows is not None and produced >= max_windows:
                return


class DNAMaskedLMDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[DNAWindow],
        tokenizer: DNATokenizer,
        mask_probability: float = 0.15,
        seed: int = 13,
    ) -> None:
        self.windows = list(windows)
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.windows)

    def _mask_tokens(self, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels = np.full_like(token_ids, fill_value=-100)
        valid_positions = np.where(
            (token_ids != self.tokenizer.n_id) & (token_ids != self.tokenizer.pad_id)
        )[0]

        if len(valid_positions) == 0:
            return token_ids, labels

        mask_flags = self.rng.random(len(valid_positions)) < self.mask_probability
        masked_positions = valid_positions[mask_flags]
        labels[masked_positions] = token_ids[masked_positions]

        if len(masked_positions) == 0:
            return token_ids, labels

        # BERT-style masking: 80% [MASK], 10% random base, 10% unchanged.
        probs = self.rng.random(len(masked_positions))

        mask_mask = probs < 0.8
        random_mask = (probs >= 0.8) & (probs < 0.9)

        token_ids = token_ids.copy()
        token_ids[masked_positions[mask_mask]] = self.tokenizer.mask_id

        if np.any(random_mask):
            random_base_ids = self.rng.integers(low=0, high=4, size=np.sum(random_mask))
            token_ids[masked_positions[random_mask]] = random_base_ids

        return token_ids, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | float]:
        window = self.windows[idx]
        token_ids = np.array(self.tokenizer.encode(window.sequence), dtype=np.int64)
        input_ids, labels = self._mask_tokens(token_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "seq_id": window.seq_id,
            "start": window.start,
            "end": window.end,
            "repeat_fraction": window.repeat_fraction,
        }


def build_windows(
    fasta_path: str,
    window_size: int,
    stride: int,
    max_windows: int | None,
    min_acgt_fraction: float,
) -> List[DNAWindow]:
    return list(
        iter_fasta_windows(
            fasta_path=Path(fasta_path),
            window_size=window_size,
            stride=stride,
            max_windows=max_windows,
            min_acgt_fraction=min_acgt_fraction,
        )
    )
