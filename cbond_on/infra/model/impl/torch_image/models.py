from __future__ import annotations

import torch
import torch.nn as nn


class ConvImageEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        channels = [
            int(base_channels),
            int(base_channels) * 2,
            int(base_channels) * 4,
            int(base_channels) * 6,
        ]
        blocks: list[nn.Module] = []
        current = int(in_channels)
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                ]
            )
            current = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.output_channels = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x))


class KlineImageCNN(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = ConvImageEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
        )
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.encoder.output_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


class KlineTimeFrequencyFusionCNN(nn.Module):
    def __init__(
        self,
        *,
        candle_channels: int = 3,
        time_frequency_channels: int = 3,
        base_channels: int = 12,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.candle_encoder = ConvImageEncoder(
            in_channels=candle_channels,
            base_channels=base_channels,
        )
        self.time_frequency_encoder = ConvImageEncoder(
            in_channels=time_frequency_channels,
            base_channels=base_channels,
        )
        merged_channels = (
            self.candle_encoder.output_channels + self.time_frequency_encoder.output_channels
        )
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(merged_channels, 1),
        )

    def forward(self, candles: torch.Tensor, time_frequency: torch.Tensor) -> torch.Tensor:
        candle_features = self.candle_encoder(candles)
        time_frequency_features = self.time_frequency_encoder(time_frequency)
        return self.head(torch.cat([candle_features, time_frequency_features], dim=1)).squeeze(-1)
