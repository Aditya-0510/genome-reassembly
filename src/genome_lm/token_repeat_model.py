from __future__ import annotations

import torch
from torch import nn


class TokenRepeatCNN(nn.Module):
    """Per-base repeat labeling model over DNA token IDs."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        channels: int = 128,
        num_layers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj_in = nn.Conv1d(embedding_dim, channels, kernel_size=1)

        blocks: list[nn.Module] = []
        dilation = 1
        for _ in range(num_layers):
            blocks.extend(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dilation = min(2 * dilation, 32)

        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]
        x = self.proj_in(x)
        x = self.encoder(x)
        logits = self.head(x).squeeze(1)  # [B, L]
        return logits
