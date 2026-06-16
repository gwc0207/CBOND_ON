from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorLSTMModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bool(bidirectional),
        )
        out_dim = int(hidden_size) * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class FactorCNN1DModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = int(n_features)
        out_channels = int(channels)
        padding = max(0, int(kernel_size) // 2)
        for _ in range(max(1, int(num_layers))):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=int(kernel_size), padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_channels = out_channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(out_channels * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(out_channels * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is (batch, time, features); Conv1d expects (batch, features, time).
        h = x.transpose(1, 2)
        h = self.net(h)
        avg_pool = h.mean(dim=2)
        max_pool = F.adaptive_max_pool1d(h, output_size=1).squeeze(-1)
        return self.head(torch.cat([avg_pool, max_pool], dim=1)).squeeze(-1)
