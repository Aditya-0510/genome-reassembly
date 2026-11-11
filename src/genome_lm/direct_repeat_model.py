from __future__ import annotations

import torch
from torch import nn


class DirectRepeatCNN(nn.Module):
    """Direct window-level repeat classifier over DNA token IDs."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Conv1d(embedding_dim, channels, kernel_size=1)

        blocks: list[nn.Module] = []
        dilation = 1
        for _ in range(num_layers):
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dilation = min(dilation * 2, 16)

        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]
        x = self.proj(x)
        x = self.encoder(x)
        logits = self.head(x).squeeze(-1)  # [B]
        return logits
