from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, ensure_trade_time


@FactorRegistry.register("volen")
class VolenFactor(Factor):
    name = "volen"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        levels = int(ctx.params.get("levels", 5))
        fast = int(ctx.params.get("fast", 60))
        slow = int(ctx.params.get("slow", 10))
        if fast <= 0 or slow <= 0:
            raise ValueError("fast/slow must be > 0")

        ask_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]
        bid_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        missing = [c for c in ask_cols + bid_cols if c not in panel.columns]
        if missing:
            raise KeyError(f"volen missing columns: {missing}")

        cols = ask_cols + bid_cols
        n_cols = len(cols)

        # Fully vectorized row-wise total depth volume.
        # Keep old strict semantics: any NaN in required depth levels => row total NaN.
        total = (
            panel.loc[:, cols]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=n_cols)
            .astype("float64")
        )

        # Fully vectorized tail-window aggregation (no per-group Python loop).
        group_levels = ["dt", "code"]
        grouped = total.groupby(level=group_levels, sort=False)
        group_size = grouped.size()
        forward_pos = grouped.cumcount()
        group_size_row = group_size.reindex(total.index)
        reverse_pos = group_size_row.to_numpy(dtype=np.int64) - forward_pos.to_numpy(dtype=np.int64) - 1
        reverse_pos = pd.Series(reverse_pos, index=total.index, dtype="int64")

        fast_mask = reverse_pos < fast
        slow_mask = reverse_pos < slow

        fast_sum = total.where(fast_mask).groupby(level=group_levels, sort=False).sum(min_count=fast)
        slow_sum = total.where(slow_mask).groupby(level=group_levels, sort=False).sum(min_count=slow)

        fast_last = fast_sum / float(fast)
        slow_last = slow_sum / float(slow)

        ratio = fast_last / slow_last
        valid = fast_last.notna() & slow_last.notna() & slow_last.abs().gt(EPS)
        ratio = ratio.where(valid, 0.0)
        ratio = ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
        ratio.name = self.output_name(self.name)
        return ratio
