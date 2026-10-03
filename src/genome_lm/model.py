from __future__ import annotations

import torch
from torch import nn


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class GenomicCNNMLM(nn.Module):
    """Dilated CNN model for DNA masked language modeling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        channels: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj_in = nn.Conv1d(embedding_dim, channels, kernel_size=1)

        dilations = [2 ** (i % 4) for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [DilatedConvBlock(channels=channels, dilation=d, dropout=dropout) for d in dilations]
        )

        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]
        x = self.proj_in(x)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)  # [B, L, C]
        x = self.norm(x)
        logits = self.head(x)

        if return_embeddings:
            return logits, x
        return logits
