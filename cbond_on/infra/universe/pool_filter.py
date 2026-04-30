from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from cbond_on.core.config import load_config_file
from cbond_on.core.trading_days import prev_trading_days_from_raw
from cbond_on.infra.data.io import read_table_range


@dataclass(frozen=True)
class UpstreamPoolConfig:
    pool_table: str
    positive_field: str
    positive_fallback_field: str
    positive_threshold: float
    pool_lag_trading_days: int
    pool_asset: str


def load_upstream_pool_config(cfg: dict | None = None) -> UpstreamPoolConfig:
    raw = dict(cfg or load_config_file("benchmark"))
    return UpstreamPoolConfig(
        pool_table=str(raw.get("pool_table", "quant_factor_dev.researcher_xuvb.o_0005")),
        positive_field=str(raw.get("positive_field", "factor_value")),
        positive_fallback_field=str(raw.get("positive_fallback_field", "weight")),
        positive_threshold=float(raw.get("positive_threshold", 0.0)),
        pool_lag_trading_days=max(0, int(raw.get("pool_lag_trading_days", 1))),
        pool_asset=str(raw.get("pool_asset", "cbond")),
    )


def _normalize_code_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def _normalize_pool_code(df: pd.DataFrame) -> pd.Series:
    if "code" in df.columns:
        return _normalize_code_series(df["code"])
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        left = _normalize_code_series(df["instrument_code"])
        right = _normalize_code_series(df["exchange_code"])
        return left + "." + right
    if "instrument_code" in df.columns:
        return _normalize_code_series(df["instrument_code"])
    raise RuntimeError("pool file missing code columns (need code or instrument_code/exchange_code)")


def _pool_table_root(raw_data_root: str | Path, pool_table: str) -> Path:
    return Path(raw_data_root) / str(pool_table).replace(".", "__")


def _pool_day_path(raw_data_root: str | Path, pool_table: str, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return _pool_table_root(raw_data_root, pool_table) / month / filename


@lru_cache(maxsize=32)
def _pool_available_days(raw_data_root: str, pool_table: str) -> tuple[date, ...]:
    base = _pool_table_root(raw_data_root, pool_table)
    if not base.exists():
        return tuple()
    days: set[date] = set()
    for path in base.glob("*/*.parquet"):
        stem = path.stem.strip()
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except Exception:
            continue
        days.add(day)
    return tuple(sorted(days))


def _nearest_available_pool_day(target_day: date, available_days: tuple[date, ...]) -> date | None:
    if not available_days:
        return None
    prev_or_equal = [d for d in available_days if d <= target_day]
    if prev_or_equal:
        return prev_or_equal[-1]
    return available_days[0]


def _load_pool_codes_for_pool_day(
    *,
    raw_data_root: str | Path,
    pool_day: date,
    pool_cfg: UpstreamPoolConfig,
) -> tuple[set[str] | None, str]:
    path = _pool_day_path(raw_data_root, pool_cfg.pool_table, pool_day)
    if not path.exists():
        return None, "missing_pool_day_file"
    pool_df = read_table_range(raw_data_root, pool_cfg.pool_table, pool_day, pool_day)
    if pool_df.empty:
        return None, "empty_pool_day_file"
    pool_df = pool_df.copy()
    pool_df["code"] = _normalize_pool_code(pool_df)
    if pool_cfg.positive_field in pool_df.columns:
        mask = (
            pd.to_numeric(pool_df[pool_cfg.positive_field], errors="coerce")
            > pool_cfg.positive_threshold
        )
    elif pool_cfg.positive_fallback_field in pool_df.columns:
        mask = (
            pd.to_numeric(pool_df[pool_cfg.positive_fallback_field], errors="coerce")
            > pool_cfg.positive_threshold
        )
    else:
        raise RuntimeError(
            "pool file missing positive-filter columns: "
            f"{pool_cfg.positive_field} / {pool_cfg.positive_fallback_field}"
        )
    codes = pool_df.loc[mask, "code"].dropna()
    codes = _normalize_code_series(codes).drop_duplicates()
    if codes.empty:
        return None, "empty_after_positive_filter"
    return set(codes.tolist()), "ok"


def resolve_pool_codes_for_trade_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    pool_cfg: UpstreamPoolConfig | None = None,
    enabled: bool = True,
) -> tuple[set[str] | None, dict[str, Any]]:
    cfg = pool_cfg or load_upstream_pool_config()
    info: dict[str, Any] = {
        "trade_day": trade_day,
        "pool_table": cfg.pool_table,
        "pool_lag_trading_days": int(cfg.pool_lag_trading_days),
        "pool_enabled": False,
        "pool_codes_count": 0,
        "pool_day_expected": None,
        "pool_day_used": None,
        "fallback_no_filter": False,
        "fallback_reason": "",
        "nearest_pool_day": None,
    }
    if not enabled:
        info["fallback_no_filter"] = True
        info["fallback_reason"] = "pool_filter_disabled"
        return None, info

    lag = max(0, int(cfg.pool_lag_trading_days))
    expected_pool_day: date | None
    if lag <= 0:
        expected_pool_day = trade_day
    else:
        prev_days = prev_trading_days_from_raw(
            raw_data_root,
            trade_day,
            lag,
            kind="snapshot",
            asset=cfg.pool_asset,
        )
        expected_pool_day = prev_days[-1] if len(prev_days) >= lag else None
    info["pool_day_expected"] = expected_pool_day

    available_days = _pool_available_days(str(raw_data_root), cfg.pool_table)
    nearest_day = _nearest_available_pool_day(expected_pool_day or trade_day, available_days)
    info["nearest_pool_day"] = nearest_day

    if expected_pool_day is None:
        info["fallback_no_filter"] = True
        info["fallback_reason"] = "missing_prev_trading_day"
        return None, info

    pool_codes, status = _load_pool_codes_for_pool_day(
        raw_data_root=raw_data_root,
        pool_day=expected_pool_day,
        pool_cfg=cfg,
    )
    if pool_codes is None:
        info["fallback_no_filter"] = True
        info["fallback_reason"] = status
        return None, info

    info["pool_enabled"] = True
    info["pool_codes_count"] = int(len(pool_codes))
    info["pool_day_used"] = expected_pool_day
    return pool_codes, info


def apply_pool_filter_to_universe(
    universe_df: pd.DataFrame,
    *,
    pool_codes: set[str] | None,
) -> pd.DataFrame:
    if pool_codes is None:
        return universe_df
    if "code" not in universe_df.columns:
        raise KeyError("universe df missing code column for pool filtering")
    work = universe_df.copy()
    work["code"] = _normalize_code_series(work["code"])
    return work[work["code"].isin(pool_codes)]
