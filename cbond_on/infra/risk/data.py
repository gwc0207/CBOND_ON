from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from cbond_on.core.trading_days import next_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.domain.risk import RiskModelError
from cbond_on.infra.benchmark.service import (
    BenchmarkPoolConfig,
    build_strict_buy_holdings_from_selection,
    compute_strict_cycle_detail_for_holdings,
)
from cbond_on.infra.data.io import read_table_range
from cbond_on.infra.risk.exposure import normalize_cbond_codes
from cbond_on.infra.universe.pool_filter import UpstreamPoolConfig, resolve_pool_codes_for_trade_day


@dataclass(frozen=True)
class RiskDayInput:
    trade_day: date
    exposure_day: date
    sell_day: date
    panel: pd.DataFrame
    diagnostics: dict[str, Any]


def _pool_config(inputs: Mapping[str, Any]) -> UpstreamPoolConfig:
    return UpstreamPoolConfig(
        pool_table=str(inputs["pool_table"]),
        positive_field=str(inputs.get("pool_positive_field", "factor_value")),
        positive_fallback_field=str(inputs.get("pool_positive_fallback_field", "weight")),
        positive_threshold=float(inputs.get("pool_positive_threshold", 0.0)),
        pool_lag_trading_days=max(0, int(inputs.get("pool_lag_trading_days", 1))),
        pool_asset=str(inputs.get("pool_asset", "cbond")),
    )


def _strict_pool_config(inputs: Mapping[str, Any]) -> BenchmarkPoolConfig:
    return BenchmarkPoolConfig(
        method="strict_official_prev_close_split",
        pool_table=str(inputs["pool_table"]),
        buy_twap_col=str(inputs["buy_twap_col"]),
        sell_twap_col=str(inputs["sell_twap_col"]),
        use_window_data=False,
        window_data_root="",
        min_price=0.0,
        max_price=float("inf"),
        positive_field=str(inputs.get("pool_positive_field", "factor_value")),
        positive_fallback_field=str(inputs.get("pool_positive_fallback_field", "weight")),
        positive_threshold=float(inputs.get("pool_positive_threshold", 0.0)),
        pool_lag_trading_days=max(0, int(inputs.get("pool_lag_trading_days", 1))),
        pool_asset=str(inputs.get("pool_asset", "cbond")),
        use_pool_weight=False,
    )


class LocalDataHubRiskSource:
    """Read-only DataHub-local source for the offline CB-Risk core.

    The source deliberately selects the static exposure table at T-1.  It is
    labelled PIT-unverified because old local daily files lack the immutable
    availability metadata required to claim a certified live replay.
    """

    def __init__(self, raw_data_root: str | Path, cfg: Mapping[str, Any]):
        self.raw_data_root = Path(raw_data_root)
        self.cfg = dict(cfg)
        self.inputs = dict(self.cfg.get("inputs") or {})
        self.asof = dict(self.cfg.get("asof") or {})
        if bool(self.asof.get("require_pit_certified", False)):
            raise RiskModelError(
                "LocalDataHubRiskSource has no available_at/revision manifest; "
                "it cannot serve a PIT-certified risk run"
            )
        if str(self.inputs.get("source")) != "datahub_local_only":
            raise RiskModelError("LocalDataHubRiskSource requires datahub_local_only")
        self.pool_cfg = _pool_config(self.inputs)
        self.strict_cfg = _strict_pool_config(self.inputs)

    def build_day(self, trade_day: date) -> RiskDayInput | None:
        lag = max(1, int(self.asof.get("static_exposure_lag_trading_days", 1)))
        previous = prev_trading_days_from_raw(
            self.raw_data_root,
            trade_day,
            lag,
            kind="snapshot",
            asset="cbond",
        )
        following = next_trading_days_from_raw(
            self.raw_data_root,
            trade_day,
            1,
            kind="snapshot",
            asset="cbond",
        )
        if len(previous) < lag or not following:
            return None
        exposure_day = previous[-1]
        sell_day = following[0]
        pool_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=self.raw_data_root,
            trade_day=trade_day,
            pool_cfg=self.pool_cfg,
            enabled=True,
        )
        diagnostics: dict[str, Any] = {
            "trade_date": trade_day.isoformat(),
            "exposure_date": exposure_day.isoformat(),
            "sell_date": sell_day.isoformat(),
            "pit_status": str(self.asof.get("pit_status", "unverified_research_only")),
            **pool_info,
        }
        if pool_codes is None:
            diagnostics.update({"status": "missing_o0005_pool", "panel_count": 0})
            return RiskDayInput(trade_day, exposure_day, sell_day, pd.DataFrame(), diagnostics)

        base = read_table_range(
            self.raw_data_root,
            str(self.inputs["base_table"]),
            exposure_day,
            exposure_day,
        )
        if base.empty:
            diagnostics.update({"status": "missing_tminus1_base", "panel_count": 0})
            return RiskDayInput(trade_day, exposure_day, sell_day, pd.DataFrame(), diagnostics)
        base = base.copy()
        base["code"] = normalize_cbond_codes(base)
        base = base.drop_duplicates("code", keep="last")
        base = base[base["code"].isin(pool_codes)].copy()
        diagnostics["pool_base_count"] = int(len(base))

        selection = pd.DataFrame({"code": sorted(pool_codes), "weight": 1.0})
        try:
            buy_holdings = build_strict_buy_holdings_from_selection(
                raw_data_root=self.raw_data_root,
                buy_day=trade_day,
                selection=selection,
                buy_bps=0.0,
                pool_cfg=self.strict_cfg,
                weight_col="weight",
                normalize=True,
            )
            detail = compute_strict_cycle_detail_for_holdings(
                raw_data_root=self.raw_data_root,
                buy_day=trade_day,
                sell_day=sell_day,
                buy_holdings=buy_holdings,
                sell_bps=0.0,
                pool_cfg=self.strict_cfg,
            )
        except Exception as exc:
            diagnostics.update({"status": "execution_return_error", "panel_count": 0, "message": str(exc)})
            return RiskDayInput(trade_day, exposure_day, sell_day, pd.DataFrame(), diagnostics)
        if detail.empty or "return_gross" not in detail.columns:
            diagnostics.update({"status": "missing_execution_return", "panel_count": 0})
            return RiskDayInput(trade_day, exposure_day, sell_day, pd.DataFrame(), diagnostics)
        returns = detail.loc[:, ["code", "return_gross"]].copy()
        returns["code"] = normalize_cbond_codes(returns)
        returns["gross_return"] = pd.to_numeric(returns["return_gross"], errors="coerce")
        returns = returns.dropna(subset=["gross_return"]).drop_duplicates("code", keep="last")
        panel = base.merge(returns.loc[:, ["code", "gross_return"]], on="code", how="inner")
        diagnostics.update(
            {
                "execution_return_count": int(len(returns)),
                "panel_count": int(len(panel)),
                "return_coverage_of_pool_base": float(len(panel) / len(base)) if len(base) else 0.0,
                "status": "ok" if not panel.empty else "empty_after_return_join",
                "base_update_time_min": str(base.get("update_time", pd.Series(dtype=object)).min()),
                "base_update_time_max": str(base.get("update_time", pd.Series(dtype=object)).max()),
            }
        )
        panel["trade_date"] = pd.Timestamp(trade_day)
        panel["exposure_date"] = pd.Timestamp(exposure_day)
        panel["sell_date"] = pd.Timestamp(sell_day)
        return RiskDayInput(trade_day, exposure_day, sell_day, panel, diagnostics)
