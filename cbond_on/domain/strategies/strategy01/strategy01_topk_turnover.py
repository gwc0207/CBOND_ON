from __future__ import annotations

import pandas as pd

from cbond_on.domain.strategies.base import Strategy, StrategyContext
from cbond_on.domain.strategies.registry import StrategyRegistry


@StrategyRegistry.register("strategy01_topk_turnover")
class Strategy01TopKTurnover(Strategy):
    strategy_id = "strategy01_topk_turnover"

    def select(self, universe: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        if universe is None or universe.empty:
            return pd.DataFrame(columns=["code", "score", "weight", "rank"])
        if "code" not in universe.columns or "score" not in universe.columns:
            raise KeyError("strategy input must include code and score")

        cfg = dict(ctx.config or {})
        top_k = max(1, int(cfg.get("top_k", 20)))
        max_weight = float(cfg.get("max_weight", 0.05))
        turnover_ratio = float(cfg.get("turnover_ratio", 1.0))
        turnover_ratio = min(1.0, max(0.0, turnover_ratio))

        ranked = universe[["code", "score"]].copy()
        ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
        ranked = ranked.dropna(subset=["score"]).sort_values("score", ascending=False)
        ranked = ranked.drop_duplicates(subset=["code"], keep="first")
        if ranked.empty:
            return pd.DataFrame(columns=["code", "score", "weight", "rank"])

        prev = ctx.prev_positions if ctx.prev_positions is not None else pd.DataFrame()
        prev_codes: list[str] = []
        if not prev.empty and "code" in prev.columns and turnover_ratio < 1.0:
            prev_codes = [str(c) for c in prev["code"].astype(str).tolist()]

        keep_codes: list[str] = []
        if prev_codes and turnover_ratio < 1.0:
            keep_count = int(round(top_k * (1.0 - turnover_ratio)))
            keep_count = min(top_k, max(0, keep_count))
            if keep_count > 0:
                keep_ranked = ranked[ranked["code"].isin(prev_codes)]
                keep_codes = keep_ranked.head(keep_count)["code"].astype(str).tolist()

        remain = top_k - len(keep_codes)
        new_ranked = ranked[~ranked["code"].isin(keep_codes)]
        select_codes = keep_codes + new_ranked.head(remain)["code"].astype(str).tolist()
        picks = ranked[ranked["code"].isin(select_codes)].copy()
        picks = picks.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        if picks.empty:
            return pd.DataFrame(columns=["code", "score", "weight", "rank"])

        picks["rank"] = picks.index + 1
        base_weight = 1.0 / len(picks)
        picks["weight"] = min(base_weight, max_weight)
        return picks[["code", "score", "weight", "rank"]]


