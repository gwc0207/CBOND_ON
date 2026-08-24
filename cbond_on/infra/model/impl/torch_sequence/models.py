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


class _CausalConv1d(nn.Module):
    """A length-preserving convolution whose output at t sees inputs <= t only."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = max(0, (int(kernel_size) - 1) * int(dilation))
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class _ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each temporal position.

    Normalizing across the time axis would leak future sequence positions into
    earlier hidden states, which is not valid for a daily causal TCN.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _FactorTCNBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = _CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm1 = _ChannelLayerNorm(int(out_channels))
        self.norm2 = _ChannelLayerNorm(int(out_channels))
        self.dropout = nn.Dropout(float(dropout))
        self.residual = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        h = self.dropout(self.norm2(self.conv2(h)))
        return F.gelu(h + self.residual(x))


class FactorTCNModel(nn.Module):
    """Daily-factor TCN with strictly left-padded temporal convolutions.

    The scorer takes the final sequence state only.  Each hidden state is
    prefix-causal, so the final state uses the factor path through the target
    score day but never a later day.
    """

    def __init__(
        self,
        *,
        n_features: int,
        channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(int(n_features))
        blocks: list[nn.Module] = []
        in_channels = int(n_features)
        for layer_idx in range(max(1, int(num_layers))):
            blocks.append(
                _FactorTCNBlock(
                    in_channels=in_channels,
                    out_channels=int(channels),
                    kernel_size=max(2, int(kernel_size)),
                    dilation=max(1, int(dilation_base)) ** layer_idx,
                    dropout=float(dropout),
                )
            )
            in_channels = int(channels)
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(int(channels)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(channels), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x).transpose(1, 2)
        h = self.net(h)
        return self.head(h[:, :, -1]).squeeze(-1)
