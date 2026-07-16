from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSummaryPool(nn.Module):
    def __init__(self, recent_steps: int = 20) -> None:
        super().__init__()
        self.recent_steps = max(1, int(recent_steps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recent = x[:, :, -min(self.recent_steps, x.shape[-1]) :]
        return torch.cat(
            [
                x.mean(dim=2),
                F.adaptive_max_pool1d(x, output_size=1).squeeze(-1),
                x[:, :, -1],
                recent.mean(dim=2),
            ],
            dim=1,
        )


class IntradayCNN1DModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        channels: int = 48,
        depth: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
        recent_steps: int = 20,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = int(n_features)
        padding = int(kernel_size) // 2
        for _ in range(max(1, int(depth))):
            blocks.extend(
                [
                    nn.Conv1d(current, int(channels), int(kernel_size), padding=padding, bias=False),
                    nn.BatchNorm1d(int(channels)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current = int(channels)
        self.encoder = nn.Sequential(*blocks)
        self.pool = TemporalSummaryPool(recent_steps=recent_steps)
        self.head = nn.Sequential(
            nn.LayerNorm(current * 4),
            nn.Dropout(float(dropout)),
            nn.Linear(current * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.transpose(1, 2))
        return self.head(self.pool(encoded)).squeeze(-1)


class _InceptionBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        branch_channels: int,
        kernels: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        bottleneck_channels = max(branch_channels, in_channels // 2)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    branch_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        out_channels = branch_channels * 4
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(float(dropout))
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        merged = torch.cat(
            [*[branch(bottleneck) for branch in self.branches], self.pool_branch(x)],
            dim=1,
        )
        return self.dropout(F.silu(self.norm(merged) + self.residual(x)))


class IntradayInceptionModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        branch_channels: int = 16,
        depth: int = 3,
        kernels: tuple[int, int, int] = (3, 7, 15),
        dropout: float = 0.1,
        recent_steps: int = 20,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = int(n_features)
        for _ in range(max(1, int(depth))):
            block = _InceptionBlock(
                in_channels=current,
                branch_channels=int(branch_channels),
                kernels=tuple(int(value) for value in kernels),
                dropout=float(dropout),
            )
            blocks.append(block)
            current = int(branch_channels) * 4
        self.encoder = nn.Sequential(*blocks)
        self.pool = TemporalSummaryPool(recent_steps=recent_steps)
        self.head = nn.Sequential(
            nn.LayerNorm(current * 4),
            nn.Dropout(float(dropout)),
            nn.Linear(current * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.transpose(1, 2))
        return self.head(self.pool(encoded)).squeeze(-1)


class _TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.silu(self.net(x) + self.residual(x)))


class IntradayTCNModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        channels: int = 48,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        kernel_size: int = 3,
        dropout: float = 0.1,
        recent_steps: int = 20,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = int(n_features)
        for dilation in dilations:
            blocks.append(
                _TemporalResidualBlock(
                    in_channels=current,
                    out_channels=int(channels),
                    dilation=int(dilation),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
            )
            current = int(channels)
        self.encoder = nn.Sequential(*blocks)
        self.pool = TemporalSummaryPool(recent_steps=recent_steps)
        self.head = nn.Sequential(
            nn.LayerNorm(current * 4),
            nn.Dropout(float(dropout)),
            nn.Linear(current * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.transpose(1, 2))
        return self.head(self.pool(encoded)).squeeze(-1)


class IntradayGRUModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        recent_steps: int = 20,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(int(n_features))
        self.gru = nn.GRU(
            input_size=int(n_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.recent_steps = max(1, int(recent_steps))
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_size) * 3),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) * 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(self.input_norm(x))
        recent = encoded[:, -min(self.recent_steps, encoded.shape[1]) :, :]
        pooled = torch.cat([encoded.mean(dim=1), encoded[:, -1, :], recent.mean(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)


class IntradayTransformerModel(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        sequence_length: int = 120,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        recent_steps: int = 20,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(int(n_features), int(d_model))
        self.position = nn.Parameter(torch.zeros(1, int(sequence_length), int(d_model)))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.recent_steps = max(1, int(recent_steps))
        self.head = nn.Sequential(
            nn.LayerNorm(int(d_model) * 3),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(x) + self.position[:, : x.shape[1], :]
        encoded = self.encoder(encoded)
        recent = encoded[:, -min(self.recent_steps, encoded.shape[1]) :, :]
        pooled = torch.cat([encoded.mean(dim=1), encoded[:, -1, :], recent.mean(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)


def build_intraday_sequence_model(
    architecture: str,
    *,
    n_features: int,
    sequence_length: int,
    model_params: dict,
) -> nn.Module:
    kind = str(architecture).strip().lower()
    params = dict(model_params)
    if kind == "cnn1d":
        return IntradayCNN1DModel(n_features=n_features, **params)
    if kind == "inception":
        kernels = params.pop("kernels", (3, 7, 15))
        return IntradayInceptionModel(
            n_features=n_features,
            kernels=tuple(int(value) for value in kernels),
            **params,
        )
    if kind == "tcn":
        dilations = params.pop("dilations", (1, 2, 4, 8, 16))
        return IntradayTCNModel(
            n_features=n_features,
            dilations=tuple(int(value) for value in dilations),
            **params,
        )
    if kind == "gru":
        return IntradayGRUModel(n_features=n_features, **params)
    if kind == "transformer":
        return IntradayTransformerModel(
            n_features=n_features,
            sequence_length=sequence_length,
            **params,
        )
    raise ValueError(f"unsupported intraday sequence architecture: {architecture}")
