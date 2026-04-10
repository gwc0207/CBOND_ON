from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._bond_stock_utils import build_bond_stock_latest_frame, to_dt_code_series
from cbond_on.domain.factors.defs._intraday_utils import _compute_backend_runtime, _torch_device_available

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@FactorRegistry.register("premium_momentum_proxy_v1")
class PremiumMomentumProxyV1Factor(Factor):
    name = "premium_momentum_proxy_v1"
    requires_stock_panel = True
    requires_bond_stock_map = True

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        frame = build_bond_stock_latest_frame(
            ctx,
            bond_cols=["last", "prev_bar_close"],
            stock_cols=["last", "prev_bar_close"],
        )
        if frame.empty:
            out = pd.Series(dtype="float64")
            out.name = self.output_name(self.name)
            return out

        backend, device = _compute_backend_runtime(ctx.params)
        if backend == "torch_cuda" and _torch_device_available(device) and torch is not None:
            try:
                bond_last_t = torch.as_tensor(
                    pd.to_numeric(frame["last"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
                    dtype=torch.float64,
                    device=device,
                )
                bond_pre_close_t = torch.as_tensor(
                    pd.to_numeric(frame["prev_bar_close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
                    dtype=torch.float64,
                    device=device,
                )
                stock_last_t = torch.as_tensor(
                    pd.to_numeric(frame["stock_last"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
                    dtype=torch.float64,
                    device=device,
                )
                stock_prev_bar_close_t = torch.as_tensor(
                    pd.to_numeric(frame["stock_prev_bar_close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
                    dtype=torch.float64,
                    device=device,
                )
                bond_strength_t = (bond_last_t - bond_pre_close_t) / (bond_pre_close_t + 1e-8)
                stock_strength_t = (stock_last_t - stock_prev_bar_close_t) / (stock_prev_bar_close_t + 1e-8)
                values = pd.Series(
                    (bond_strength_t - stock_strength_t).detach().cpu().numpy(),
                    index=frame.index,
                    dtype="float64",
                )
                return to_dt_code_series(frame, values, name=self.output_name(self.name))
            except Exception:
                pass

        bond_last = pd.to_numeric(frame["last"], errors="coerce")
        bond_pre_close = pd.to_numeric(frame["prev_bar_close"], errors="coerce")
        stock_last = pd.to_numeric(frame["stock_last"], errors="coerce")
        stock_prev_bar_close = pd.to_numeric(frame["stock_prev_bar_close"], errors="coerce")

        bond_strength = (bond_last - bond_pre_close) / (bond_pre_close + 1e-8)
        stock_strength = (stock_last - stock_prev_bar_close) / (stock_prev_bar_close + 1e-8)
        values = bond_strength - stock_strength
        return to_dt_code_series(frame, values, name=self.output_name(self.name))

