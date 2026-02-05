from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """
    Strict causal conv on time dimension.
    Input: (N, C, T, L)
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.k_t, self.k_l = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        pad_t = self.k_t - 1
        pad_l = self.k_l // 2
        x = F.pad(x, (pad_l, pad_l, pad_t, 0))
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        n, c, t, l = x.shape
        w = x.mean(dim=(2, 3))
        w = self.fc1(w)
        w = F.silu(w)
        w = self.fc2(w)
        w = torch.sigmoid(w)
        w = w.view(n, c, 1, 1)
        return x * w


class LevelRBFEmbed(nn.Module):
    def __init__(self, L: int, C_out: int, num_bases: int = 16, sigma: float = 0.5):
        super().__init__()
        self.L = L
        self.num_bases = num_bases
        self.sigma = sigma

        levels = torch.linspace(-1.0, 1.0, steps=L).view(1, 1, 1, L)
        centers = torch.linspace(-1.0, 1.0, steps=num_bases).view(1, num_bases, 1, 1)
        diff = levels - centers
        emb = torch.exp(-0.5 * (diff / sigma) ** 2)
        self.register_buffer("rbf_emb", emb)

        self.proj = nn.Conv2d(C_out + num_bases, C_out, kernel_size=1)

    def forward(self, x):
        n, c, t, l = x.shape
        if l != self.L:
            raise ValueError(f"LevelRBFEmbed expects L={self.L}, got {l}")
        emb = self.rbf_emb.expand(n, -1, t, -1)
        x_cat = torch.cat([x, emb], dim=1)
        return self.proj(x_cat)


class LevelTimeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        k_time: int = 3,
        num_time_convs: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.gn1 = nn.GroupNorm(num_groups, mid_channels)
        self.gn2 = nn.GroupNorm(num_groups, mid_channels)

        self.time_convs = nn.ModuleList()
        self.time_gns = nn.ModuleList()
        for _ in range(num_time_convs):
            self.time_convs.append(CausalConv2d(mid_channels, mid_channels, kernel_size=(k_time, 1)))
            self.time_gns.append(nn.GroupNorm(num_groups, mid_channels))

        if in_channels != mid_channels:
            self.proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        residual = self.proj(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = F.silu(out)

        for conv, gn in zip(self.time_convs, self.time_gns):
            out = conv(out)
            out = gn(out)
            out = F.silu(out)

        return out + residual


class SpatioTemporalResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        C_mid: int,
        k_time_3: int = 3,
        k_time_5: int = 5,
        num_groups: int = 8,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.bottleneck = nn.Conv2d(in_channels, C_mid, kernel_size=1)
        self.bn_gn = nn.GroupNorm(num_groups, C_mid)

        self.branch1 = nn.Conv2d(C_mid, C_mid, kernel_size=1)
        self.branch2 = CausalConv2d(C_mid, C_mid, kernel_size=(k_time_3, 3))
        self.branch3 = CausalConv2d(C_mid, C_mid, kernel_size=(k_time_5, 5))

        self.post_concat_gn = nn.GroupNorm(num_groups, C_mid * 3)
        self.conv_1x2 = nn.Conv2d(C_mid * 3, out_channels, kernel_size=(1, 2), padding=0)
        self.post_conv_gn = nn.GroupNorm(num_groups, out_channels)

        self.se = SEBlock(out_channels, reduction=se_reduction)
        self.shortcut_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        _, _, _, L = x.shape
        if L % 2 == 1:
            x_padded = F.pad(x, (0, 1, 0, 0))
        else:
            x_padded = x
        pooled = F.max_pool2d(x_padded, kernel_size=(1, 2), stride=(1, 2))
        shortcut = self.shortcut_proj(pooled)

        out = self.bottleneck(pooled)
        out = self.bn_gn(out)
        out = F.silu(out)

        b1 = self.branch1(out)
        b2 = self.branch2(out)
        b3 = self.branch3(out)

        merged = torch.cat([b1, b2, b3], dim=1)
        merged = self.post_concat_gn(merged)
        merged = F.silu(merged)

        merged_padded = F.pad(merged, (0, 1, 0, 0))
        merged = self.conv_1x2(merged_padded)
        merged = self.post_conv_gn(merged)
        merged = F.silu(merged)

        merged = self.se(merged)
        return merged + shortcut


class TemporalHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        fc_in_dim = lstm_hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in_dim, 1)

    def forward(self, x):
        x = x.mean(dim=3)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        y_pred = self.fc(last_out)
        return y_pred.squeeze(-1)


class LOBSpatioTemporalModel(nn.Module):
    def __init__(
        self,
        C_in: int = 2,
        depth_levels: int = 10,
        C_first: int = 64,
        LT1_mid_channels: int = 128,
        LT1_k_time: int = 3,
        LT2_mid_channels: int = 256,
        LT2_k_time: int = 3,
        rbf_num_bases: int = 16,
        rbf_sigma: float = 0.5,
        ST1_C_mid: int = 64,
        ST2_C_mid: int = 128,
        ST3_C_mid: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
    ):
        super().__init__()
        self.depth_levels = depth_levels

        self.first_conv = nn.Conv2d(C_in, C_first, kernel_size=1)
        self.first_gn = nn.GroupNorm(8, C_first)

        self.ltb1 = LevelTimeBlock(
            in_channels=C_first,
            mid_channels=LT1_mid_channels,
            k_time=LT1_k_time,
            num_time_convs=2,
            num_groups=8,
        )

        self.rbf_embed = LevelRBFEmbed(
            L=depth_levels,
            C_out=LT1_mid_channels,
            num_bases=rbf_num_bases,
            sigma=rbf_sigma,
        )

        self.ltb2 = LevelTimeBlock(
            in_channels=LT1_mid_channels,
            mid_channels=LT2_mid_channels,
            k_time=LT2_k_time,
            num_time_convs=2,
            num_groups=8,
        )

        self.st1 = SpatioTemporalResBlock(
            in_channels=LT2_mid_channels,
            out_channels=256,
            C_mid=ST1_C_mid,
            k_time_3=3,
            k_time_5=5,
            num_groups=8,
            se_reduction=4,
        )
        self.st2 = SpatioTemporalResBlock(
            in_channels=256,
            out_channels=512,
            C_mid=ST2_C_mid,
            k_time_3=3,
            k_time_5=5,
            num_groups=8,
            se_reduction=4,
        )
        self.st3 = SpatioTemporalResBlock(
            in_channels=512,
            out_channels=1024,
            C_mid=ST3_C_mid,
            k_time_3=3,
            k_time_5=5,
            num_groups=8,
            se_reduction=4,
        )

        self.temporal_head = TemporalHead(
            in_channels=1024,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            bidirectional=False,
        )

    def forward(self, x):
        _, _, _, L = x.shape
        if L != self.depth_levels:
            raise ValueError(f"expected L={self.depth_levels}, got {L}")

        x = torch.log1p(torch.clamp(x, min=0.0))

        x = self.first_conv(x)
        x = self.first_gn(x)
        x = F.silu(x)

        x = self.ltb1(x)
        x = self.rbf_embed(x)
        x = self.ltb2(x)

        x = self.st1(x)
        x = self.st2(x)
        x = self.st3(x)

        return self.temporal_head(x)
