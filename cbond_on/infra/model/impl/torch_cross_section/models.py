from __future__ import annotations

import math

import torch
import torch.nn as nn


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    count = weights.sum(dim=0, keepdim=True).clamp_min(1.0)
    return (values * weights).sum(dim=0, keepdim=True) / count


def _masked_max(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
    out = masked.max(dim=0, keepdim=True).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


class CrossSectionMLP(nn.Module):
    """Shared per-bond MLP baseline for a daily cross-sectional loss.

    This model itself has no cross-bond context; its role is to distinguish the
    incremental value of the daily ListNet objective from explicit set models.
    It is nevertheless permutation equivariant because each bond uses shared
    parameters only.
    """

    def __init__(self, *, n_features: int, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(n_features)),
            nn.Linear(int(n_features), int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), max(8, int(hidden_size) // 2)),
            nn.GELU(),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"cross-sectional inputs must be (n_bonds, n_features), got {tuple(x.shape)}")
        out = self.net(x).squeeze(-1)
        if valid_mask is not None:
            out = out.masked_fill(~valid_mask, 0.0)
        return out


class CrossSectionLinearModel(nn.Module):
    """Low-capacity permutation-equivariant linear daily scorer.

    Scores are trained with the same daily ListNet objective as the set models;
    only the inductive bias is linear.  It is deliberately free of security
    identifiers, temporal state, and cross-sectional pooling.
    """

    def __init__(self, *, n_features: int, bias: bool = False) -> None:
        super().__init__()
        self.head = nn.Linear(int(n_features), 1, bias=bool(bias))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"cross-sectional inputs must be (n_bonds, n_features), got {tuple(x.shape)}")
        out = self.head(x).squeeze(-1)
        if valid_mask is not None:
            out = out.masked_fill(~valid_mask, 0.0)
        return out


class DeepSetsModel(nn.Module):
    """Permutation-equivariant scorer with mean/max same-day set context."""

    def __init__(self, *, n_features: int, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        width = int(hidden_size)
        self.input_norm = nn.LayerNorm(int(n_features))
        self.phi = nn.Sequential(
            nn.Linear(int(n_features), width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"cross-sectional inputs must be (n_bonds, n_features), got {tuple(x.shape)}")
        mask = (
            torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            if valid_mask is None
            else valid_mask.to(dtype=torch.bool, device=x.device)
        )
        h = self.phi(self.input_norm(x))
        mean = _masked_mean(h, mask).expand_as(h)
        max_value = _masked_max(h, mask).expand_as(h)
        out = self.rho(torch.cat([h, mean, max_value], dim=-1)).squeeze(-1)
        return out.masked_fill(~mask, 0.0)


class SetTransformerModel(nn.Module):
    """Small set-attention scorer without security IDs or positional encoding."""

    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        width = int(hidden_size)
        heads = int(num_heads)
        if width % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_norm = nn.LayerNorm(int(n_features))
        self.embed = nn.Sequential(nn.Linear(int(n_features), width), nn.GELU())
        self.attn = nn.MultiheadAttention(width, heads, dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width * 2, width),
        )
        self.norm2 = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"cross-sectional inputs must be (n_bonds, n_features), got {tuple(x.shape)}")
        mask = (
            torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            if valid_mask is None
            else valid_mask.to(dtype=torch.bool, device=x.device)
        )
        h = self.embed(self.input_norm(x)).unsqueeze(0)
        # key_padding_mask prevents invalid padded set members from becoming
        # attention keys.  Their eventual score is hard-zeroed below.
        key_padding_mask = (~mask).unsqueeze(0)
        attended, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ffn(h)).squeeze(0)
        context = _masked_mean(h, mask).expand_as(h)
        out = self.head(torch.cat([h, context], dim=-1)).squeeze(-1)
        return out.masked_fill(~mask, 0.0)


def build_cross_section_model(
    architecture: str,
    *,
    n_features: int,
    model_params: dict | None = None,
) -> nn.Module:
    params = dict(model_params or {})
    name = str(architecture).strip().lower()
    common = {
        "n_features": int(n_features),
        "hidden_size": int(params.get("hidden_size", 64)),
        "dropout": float(params.get("dropout", 0.1)),
    }
    if name in {"mlp", "cross_section_mlp"}:
        return CrossSectionMLP(**common)
    if name in {"linear", "cross_section_linear"}:
        return CrossSectionLinearModel(
            n_features=int(n_features),
            bias=bool(params.get("bias", False)),
        )
    if name in {"deepsets", "deep_sets"}:
        return DeepSetsModel(**common)
    if name in {"settransformer", "set_transformer"}:
        return SetTransformerModel(
            **common,
            num_heads=int(params.get("num_heads", 4)),
        )
    raise ValueError(f"unsupported cross-sectional architecture: {architecture}")
