from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from cbond_on.core.config import load_config_file
from cbond_on.core.trading_days import list_available_trading_days_from_raw
from cbond_on.core.universe import normalize_price_bound
from cbond_on.infra.backtest.execution import bps_to_rate
from cbond_on.infra.data.io import read_table_range
from cbond_on.infra.io.market_twap import read_price_daily, read_twap_daily
from cbond_on.infra.universe.pool_filter import (
    UpstreamPoolConfig,
    resolve_pool_codes_for_trade_day,
)


CLOSE_PRICE_CANDIDATES = ("close", "close_price", "close_px", "closePrice")
PREV_CLOSE_PRICE_CANDIDATES = ("prev_close_price", "prev_close", "pre_close")
_WINDOW_TWAP_DAY_CACHE: dict[tuple[str, date, tuple[str, ...]], pd.DataFrame] = {}


@dataclass(frozen=True)
class BenchmarkPoolConfig:
    method: str
    pool_table: str
    buy_twap_col: str
    sell_twap_col: str
    use_window_data: bool
    window_data_root: str
    min_price: float
    max_price: float
    positive_field: str
    positive_fallback_field: str
    positive_threshold: float
    pool_lag_trading_days: int
    pool_asset: str
    use_pool_weight: bool


@dataclass(frozen=True)
class BenchmarkReturnBreakdown:
    full_cycle_ret_net: float
    buy_leg_ret_net: float
    sell_leg_ret_net: float
    count: int
    sell_count: int = 0
    fallback_sell_codes: int = 0
    fallback_sell_weight: float = 0.0


def load_benchmark_pool_config(cfg: dict | None = None) -> BenchmarkPoolConfig:
    raw = dict(cfg or load_config_file("benchmark"))
    return BenchmarkPoolConfig(
        method=str(raw.get("method", "strict_official_prev_close_split")),
        pool_table=str(raw.get("pool_table", "quant_factor_dev.researcher_xuvb.o_0005")),
        buy_twap_col=str(raw.get("buy_twap_col", "twap_1442_1457")),
        sell_twap_col=str(raw.get("sell_twap_col", "twap_0930_0939")),
        use_window_data=bool(raw.get("use_window_data", True)),
        window_data_root=str(
            raw.get(
                "window_data_root",
                r"\\nfs\10.1.30.100\data\yinhe-data\kline\cbond\window-data",
            )
        ),
        min_price=normalize_price_bound(raw.get("min_price", 0.0)),
        max_price=normalize_price_bound(raw.get("max_price")),
        positive_field=str(raw.get("positive_field", "factor_value")),
        positive_fallback_field=str(raw.get("positive_fallback_field", "weight")),
        positive_threshold=float(raw.get("positive_threshold", 0.0)),
        pool_lag_trading_days=max(0, int(raw.get("pool_lag_trading_days", 1))),
        pool_asset=str(raw.get("pool_asset", "cbond")),
        use_pool_weight=bool(raw.get("use_pool_weight", False)),
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
        right = _normalize_code_series(df["exchange_code"]).str.upper()
        return left + "." + right
    if "instrument_code" in df.columns:
        return _normalize_code_series(df["instrument_code"])
    raise RuntimeError("benchmark pool missing code columns")


def _ensure_code_column(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    work = df.copy()
    if "code" in work.columns:
        work["code"] = _normalize_code_series(work["code"])
        return work
    if "instrument_code" in work.columns and "exchange_code" in work.columns:
        left = _normalize_code_series(work["instrument_code"])
        right = _normalize_code_series(work["exchange_code"]).str.upper()
        work["code"] = left + "." + right
        return work
    raise RuntimeError(f"{label} missing code columns")


def _normalize_weight_series(weight: pd.Series) -> pd.Series:
    w = pd.to_numeric(weight, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return w / total


def _first_price_series(df: pd.DataFrame, candidates: Iterable[str], *, label: str) -> pd.Series:
    if df.empty:
        raise RuntimeError(f"benchmark daily_price missing for {label}")
    work = _ensure_code_column(df, label=f"{label} daily_price")
    for col in candidates:
        if col in work.columns:
            out = work[["code", col]].copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=["code"])
            return out.dropna(subset=[col]).groupby("code", sort=False)[col].mean()
    raise RuntimeError(f"benchmark daily_price missing {label} columns: {list(candidates)}")


def _twap_col_to_window_file(col: str) -> str:
    text = str(col).strip()
    if text.startswith("twap_"):
        text = text[5:]
    return text.replace("_", "-")


def _window_file_to_col(window_file: str) -> str:
    return "twap_" + str(window_file).strip().replace("-", "_")


def _parse_window_file_bounds(window_file: str) -> tuple[int, int]:
    text = str(window_file).strip()
    if "-" not in text:
        raise ValueError(f"invalid twap window: {window_file}")
    start_s, end_s = text.split("-", 1)
    if len(start_s) != 4 or len(end_s) != 4 or not start_s.isdigit() or not end_s.isdigit():
        raise ValueError(f"invalid twap window: {window_file}")
    return int(start_s), int(end_s)


def _snapshot_root_from_cfg(cfg: BenchmarkPoolConfig) -> Path:
    root = Path(cfg.window_data_root)
    name = root.name.lower()
    if name in {"window-data", "window_data"}:
        return root.parent / "from_snapshot"
    if name == "from_snapshot":
        return root
    return root / "from_snapshot"


def _read_window_twap_daily(
    day: date,
    cfg: BenchmarkPoolConfig,
    *,
    twap_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    cols = [str(c).strip() for c in (twap_cols or [cfg.buy_twap_col, cfg.sell_twap_col]) if str(c).strip()]
    windows = []
    for col in cols:
        window = _twap_col_to_window_file(col)
        if window and window not in windows:
            windows.append(window)
    if not windows:
        return pd.DataFrame()

    root = _snapshot_root_from_cfg(cfg)
    key = (str(root), day, tuple(windows))
    cached = _WINDOW_TWAP_DAY_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    snapshot_file = root / f"{day:%Y-%m}" / f"{day:%Y-%m-%d}.parquet"
    if not snapshot_file.exists():
        out = pd.DataFrame()
        _WINDOW_TWAP_DAY_CACHE[key] = out
        return out.copy()

    try:
        one = pd.read_parquet(
            snapshot_file,
            columns=["instrument_code", "exchange_code", "trade_time", "twap"],
        )
    except Exception:
        try:
            one = pd.read_parquet(snapshot_file)
        except Exception:
            out = pd.DataFrame()
            _WINDOW_TWAP_DAY_CACHE[key] = out
            return out.copy()
    if one.empty or "trade_time" not in one.columns or "twap" not in one.columns:
        out = pd.DataFrame()
        _WINDOW_TWAP_DAY_CACHE[key] = out
        return out.copy()

    one = _ensure_code_column(one, label=f"from_snapshot {day:%Y-%m-%d}")
    minute_hhmm = pd.to_numeric(
        one["trade_time"].astype(str).str.slice(0, 5).str.replace(":", "", regex=False),
        errors="coerce",
    )
    work = pd.DataFrame(
        {
            "code": _normalize_code_series(one["code"]),
            "minute_hhmm": minute_hhmm,
            "twap": pd.to_numeric(one["twap"], errors="coerce"),
        }
    )
    work = work[work["minute_hhmm"].notna() & work["twap"].notna()].copy()
    if work.empty:
        out = pd.DataFrame()
        _WINDOW_TWAP_DAY_CACHE[key] = out
        return out.copy()

    merged: pd.DataFrame | None = None
    for window in windows:
        try:
            start_hhmm, end_hhmm = _parse_window_file_bounds(window)
        except ValueError:
            continue
        mask = (work["minute_hhmm"] > start_hhmm) & (work["minute_hhmm"] <= end_hhmm)
        tmp = work.loc[mask, ["code", "twap"]].copy()
        if tmp.empty:
            continue
        col = _window_file_to_col(window)
        tmp = tmp.groupby("code", as_index=False)["twap"].mean().rename(columns={"twap": col})
        merged = tmp if merged is None else merged.merge(tmp, on="code", how="outer")

    out = merged if merged is not None else pd.DataFrame()
    _WINDOW_TWAP_DAY_CACHE[key] = out
    return out.copy()


def _read_benchmark_twap_daily(
    raw_data_root: str | Path,
    day: date,
    cfg: BenchmarkPoolConfig,
    *,
    twap_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    if cfg.use_window_data:
        return _read_window_twap_daily(day, cfg, twap_cols=twap_cols)
    df = read_twap_daily(str(raw_data_root), day)
    if df.empty:
        return df
    if twap_cols is None:
        return df
    keep = ["code", *[c for c in twap_cols if c in df.columns]]
    return _ensure_code_column(df, label="benchmark daily_twap")[[c for c in keep if c in df.columns]]


def _load_target_pool_for_buy_day(
    *,
    raw_data_root: str | Path,
    buy_day: date,
    cfg: BenchmarkPoolConfig,
) -> pd.DataFrame:
    upstream_cfg = UpstreamPoolConfig(
        pool_table=cfg.pool_table,
        positive_field=cfg.positive_field,
        positive_fallback_field=cfg.positive_fallback_field,
        positive_threshold=cfg.positive_threshold,
        pool_lag_trading_days=cfg.pool_lag_trading_days,
        pool_asset=cfg.pool_asset,
    )
    pool_codes, pool_info = resolve_pool_codes_for_trade_day(
        raw_data_root=raw_data_root,
        trade_day=buy_day,
        pool_cfg=upstream_cfg,
    )
    if pool_codes is None:
        raise RuntimeError(
            "benchmark target pool unavailable: "
            f"buy_day={buy_day:%Y-%m-%d} reason={pool_info.get('fallback_reason')}"
        )
    pool_day = pool_info.get("pool_day_used")
    if pool_day is None:
        raise RuntimeError(f"benchmark target pool missing used day: buy_day={buy_day:%Y-%m-%d}")

    pool_df = read_table_range(raw_data_root, cfg.pool_table, pool_day, pool_day)
    if pool_df.empty:
        raise RuntimeError(f"benchmark target pool file empty: pool_day={pool_day:%Y-%m-%d}")
    pool_df = pool_df.copy()
    pool_df["code"] = _normalize_pool_code(pool_df)
    if cfg.positive_field in pool_df.columns:
        mask = pd.to_numeric(pool_df[cfg.positive_field], errors="coerce") > cfg.positive_threshold
    elif cfg.positive_fallback_field in pool_df.columns:
        mask = pd.to_numeric(pool_df[cfg.positive_fallback_field], errors="coerce") > cfg.positive_threshold
    else:
        raise RuntimeError(
            "benchmark target pool missing positive fields: "
            f"{cfg.positive_field}/{cfg.positive_fallback_field}"
        )

    keep_cols = ["code"]
    if "weight" in pool_df.columns:
        keep_cols.append("weight")
    target = pool_df.loc[mask, keep_cols].copy()
    target["code"] = _normalize_code_series(target["code"])
    target = target[target["code"].isin(pool_codes)]
    target = target.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="first")
    if target.empty:
        raise RuntimeError(f"benchmark target pool empty after filter: buy_day={buy_day:%Y-%m-%d}")

    if cfg.use_pool_weight and "weight" in target.columns:
        normalized = _normalize_weight_series(target["weight"])
        target["target_weight"] = normalized.reindex(target.index).fillna(0.0) if not normalized.empty else 1.0 / len(target)
    else:
        target["target_weight"] = 1.0 / len(target)
    target["pool_day"] = pd.to_datetime(pool_day)
    target["buy_trade_day"] = pd.to_datetime(buy_day)
    return target[["code", "target_weight", "pool_day", "buy_trade_day"]].reset_index(drop=True)


def _available_trade_days(raw_data_root: str | Path, cfg: BenchmarkPoolConfig) -> list[date]:
    days = list_available_trading_days_from_raw(
        raw_data_root,
        kind="snapshot",
        asset=cfg.pool_asset,
    )
    if days:
        return days
    if cfg.use_window_data:
        root = _snapshot_root_from_cfg(cfg)
        out: set[date] = set()
        if root.exists():
            for path in root.glob("*/*.parquet"):
                parsed = pd.to_datetime(path.stem, errors="coerce")
                if not pd.isna(parsed):
                    out.add(parsed.date())
        return sorted(out)
    return days


def _previous_trading_day(raw_data_root: str | Path, day: date, cfg: BenchmarkPoolConfig) -> date:
    days = _available_trade_days(raw_data_root, cfg)
    prior = [d for d in days if d < day]
    if not prior:
        raise RuntimeError(f"benchmark previous trading day missing for: {day:%Y-%m-%d}")
    return prior[-1]


def load_strict_market_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> pd.DataFrame:
    cfg = pool_cfg or load_benchmark_pool_config()
    twap = _read_benchmark_twap_daily(
        raw_data_root,
        trade_day,
        cfg,
        twap_cols=[cfg.buy_twap_col, cfg.sell_twap_col],
    )
    if twap.empty:
        twap = pd.DataFrame(columns=["code", cfg.buy_twap_col, cfg.sell_twap_col])
    else:
        twap = _ensure_code_column(twap, label="strict market twap")
        keep = ["code", *[c for c in [cfg.buy_twap_col, cfg.sell_twap_col] if c in twap.columns]]
        twap = twap[keep].copy()

    price_df = read_price_daily(str(raw_data_root), trade_day)
    close = _first_price_series(price_df, CLOSE_PRICE_CANDIDATES, label="close_price").rename("buy_close_price")
    prev_close = _first_price_series(
        price_df,
        PREV_CLOSE_PRICE_CANDIDATES,
        label="strict_prev_close_price",
    ).rename("strict_prev_close_price")
    price = close.reset_index().merge(prev_close.reset_index(), on="code", how="outer")
    market = price.merge(twap, on="code", how="outer")

    buy_price = pd.to_numeric(market.get(cfg.buy_twap_col), errors="coerce")
    buy_close = pd.to_numeric(market["buy_close_price"], errors="coerce")
    buy_rate = bps_to_rate(buy_bps)
    market["buy_price"] = buy_price
    market["buy_leg_ret_gross"] = buy_close / buy_price - 1.0
    market["buy_leg_ret_net"] = market["buy_leg_ret_gross"] - buy_rate

    sell_raw = pd.to_numeric(market.get(cfg.sell_twap_col), errors="coerce")
    strict_prev_close = pd.to_numeric(market["strict_prev_close_price"], errors="coerce")
    sell_valid = sell_raw.notna() & (sell_raw > 0)
    market["strict_sell_price"] = sell_raw.where(sell_valid, strict_prev_close)
    market["sell_price"] = market["strict_sell_price"]
    market["sell_price_source"] = "sell_twap"
    market.loc[~sell_valid, "sell_price_source"] = "official_prev_close_fallback"
    market["sell_missing_fallback"] = ~sell_valid
    sell_rate = bps_to_rate(sell_bps)
    market["strict_sell_leg_gross_ret"] = market["strict_sell_price"] / strict_prev_close - 1.0
    market["strict_sell_leg_net_ret"] = market["strict_sell_price"] * (1.0 - sell_rate) / strict_prev_close - 1.0
    market["buy_cost_bps"] = float(buy_bps)
    market["sell_cost_bps"] = float(sell_bps)
    market["trade_day"] = pd.to_datetime(trade_day)
    return market.sort_values("code", kind="mergesort").reset_index(drop=True)


def _build_buy_holdings_for_day(
    *,
    raw_data_root: str | Path,
    buy_day: date,
    buy_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> pd.DataFrame:
    cfg = pool_cfg or load_benchmark_pool_config()
    target = _load_target_pool_for_buy_day(raw_data_root=raw_data_root, buy_day=buy_day, cfg=cfg)
    market = load_strict_market_day(
        raw_data_root=raw_data_root,
        trade_day=buy_day,
        buy_bps=buy_bps,
        sell_bps=0.0,
        pool_cfg=cfg,
    )
    required = ["code", "buy_price", "buy_close_price", "buy_leg_ret_gross", "buy_leg_ret_net"]
    holdings = target.merge(market[required], on="code", how="inner")
    holdings["buy_price"] = pd.to_numeric(holdings["buy_price"], errors="coerce")
    holdings["buy_close_price"] = pd.to_numeric(holdings["buy_close_price"], errors="coerce")
    holdings = holdings[
        holdings["buy_price"].notna()
        & holdings["buy_close_price"].notna()
        & (holdings["buy_price"] > 0)
        & (holdings["buy_close_price"] > 0)
    ].copy()
    if holdings.empty:
        raise RuntimeError(f"benchmark buy holdings empty: buy_day={buy_day:%Y-%m-%d}")

    weights = _normalize_weight_series(holdings["target_weight"])
    holdings["weight"] = weights.reindex(holdings.index).fillna(0.0) if not weights.empty else 1.0 / len(holdings)
    holdings["buy_weight_base"] = pd.to_numeric(holdings["weight"], errors="coerce")
    holdings["close_price"] = holdings["buy_close_price"]
    holdings["prev_close_price"] = holdings["buy_close_price"]
    holdings["buy_cost_bps"] = float(buy_bps)
    holdings["weighted_buy_leg_ret_net"] = (
        pd.to_numeric(holdings["weight"], errors="coerce")
        * pd.to_numeric(holdings["buy_leg_ret_net"], errors="coerce")
    )
    holdings["weighted_buy_leg_ret_gross"] = (
        pd.to_numeric(holdings["weight"], errors="coerce")
        * pd.to_numeric(holdings["buy_leg_ret_gross"], errors="coerce")
    )
    return holdings.sort_values("code", kind="mergesort").reset_index(drop=True)


def build_strict_buy_holdings_from_selection(
    *,
    raw_data_root: str | Path,
    buy_day: date,
    selection: pd.DataFrame,
    buy_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
    weight_col: str = "weight",
    normalize: bool = True,
) -> pd.DataFrame:
    if selection is None or selection.empty:
        return pd.DataFrame()
    cfg = pool_cfg or load_benchmark_pool_config()
    selected = _ensure_code_column(selection, label="strict selection").copy()
    selected = selected.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="first")
    market = load_strict_market_day(
        raw_data_root=raw_data_root,
        trade_day=buy_day,
        buy_bps=buy_bps,
        sell_bps=0.0,
        pool_cfg=cfg,
    )
    required = ["code", "buy_price", "buy_close_price", "buy_leg_ret_gross", "buy_leg_ret_net"]
    holdings = selected.merge(market[required], on="code", how="inner")
    holdings["buy_price"] = pd.to_numeric(holdings["buy_price"], errors="coerce")
    holdings["buy_close_price"] = pd.to_numeric(holdings["buy_close_price"], errors="coerce")
    holdings = holdings[
        holdings["buy_price"].notna()
        & holdings["buy_close_price"].notna()
        & (holdings["buy_price"] > 0)
        & (holdings["buy_close_price"] > 0)
    ].copy()
    if holdings.empty:
        return holdings

    if weight_col in holdings.columns:
        raw_weight = pd.to_numeric(holdings[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if normalize:
            normalized = _normalize_weight_series(raw_weight)
            holdings["weight"] = normalized.reindex(holdings.index).fillna(0.0) if not normalized.empty else 1.0 / len(holdings)
        else:
            holdings["weight"] = raw_weight
    else:
        holdings["weight"] = 1.0 / len(holdings)
    holdings["buy_trade_day"] = pd.to_datetime(buy_day)
    holdings["buy_weight_base"] = pd.to_numeric(holdings["weight"], errors="coerce")
    holdings["close_price"] = holdings["buy_close_price"]
    holdings["prev_close_price"] = holdings["buy_close_price"]
    holdings["buy_cost_bps"] = float(buy_bps)
    holdings["weighted_buy_leg_ret_net"] = (
        pd.to_numeric(holdings["weight"], errors="coerce")
        * pd.to_numeric(holdings["buy_leg_ret_net"], errors="coerce")
    )
    return holdings.sort_values("code", kind="mergesort").reset_index(drop=True)


def _to_strict_sell_holdings(buy_holdings: pd.DataFrame) -> pd.DataFrame:
    if buy_holdings is None or buy_holdings.empty:
        return pd.DataFrame()
    work = _ensure_code_column(buy_holdings, label="strict sell holdings").copy()
    if "weight" not in work.columns:
        work["weight"] = 1.0 / len(work)
    if "prev_close_price" not in work.columns:
        if "buy_close_price" in work.columns:
            work["prev_close_price"] = work["buy_close_price"]
        elif "close_price" in work.columns:
            work["prev_close_price"] = work["close_price"]
    return work.reset_index(drop=True)


def compute_strict_sell_detail_for_holdings(
    *,
    raw_data_root: str | Path,
    sell_day: date,
    prev_holdings: pd.DataFrame,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> pd.DataFrame:
    if prev_holdings is None or prev_holdings.empty:
        return pd.DataFrame()
    cfg = pool_cfg or load_benchmark_pool_config()
    held = _ensure_code_column(prev_holdings, label="strict previous holdings").copy()
    if "weight" not in held.columns:
        held["weight"] = 1.0 / len(held)

    twap = _read_benchmark_twap_daily(
        raw_data_root,
        sell_day,
        cfg,
        twap_cols=[cfg.sell_twap_col],
    )
    if twap.empty or cfg.sell_twap_col not in twap.columns:
        sell = pd.DataFrame(columns=["code", cfg.sell_twap_col])
    else:
        sell = _ensure_code_column(twap, label="strict sell twap")[["code", cfg.sell_twap_col]].copy()

    price_df = read_price_daily(str(raw_data_root), sell_day)
    strict_prev = _first_price_series(
        price_df,
        PREV_CLOSE_PRICE_CANDIDATES,
        label="strict_prev_close_price",
    ).rename("strict_prev_close_price")

    detail = held.merge(strict_prev.reset_index(), on="code", how="inner").merge(sell, on="code", how="left")
    detail["strict_prev_close_price"] = pd.to_numeric(detail["strict_prev_close_price"], errors="coerce")
    detail = detail[detail["strict_prev_close_price"].notna() & (detail["strict_prev_close_price"] > 0)].copy()
    if detail.empty:
        return detail

    sell_raw = pd.to_numeric(detail.get(cfg.sell_twap_col), errors="coerce")
    sell_valid = sell_raw.notna() & (sell_raw > 0)
    detail["strict_sell_price"] = sell_raw.where(sell_valid, detail["strict_prev_close_price"])
    detail["sell_price"] = detail["strict_sell_price"]
    detail["sell_price_source"] = "sell_twap"
    detail.loc[~sell_valid, "sell_price_source"] = "official_prev_close_fallback"
    detail["sell_missing_fallback"] = ~sell_valid
    detail["sell_trade_day"] = pd.to_datetime(sell_day)
    detail["sell_cost_bps"] = float(sell_bps)

    sell_rate = bps_to_rate(sell_bps)
    detail["strict_sell_leg_gross_ret"] = detail["strict_sell_price"] / detail["strict_prev_close_price"] - 1.0
    detail["strict_sell_leg_net_ret"] = (
        detail["strict_sell_price"] * (1.0 - sell_rate) / detail["strict_prev_close_price"] - 1.0
    )
    detail["weighted_sell_leg_ret_net"] = (
        pd.to_numeric(detail["weight"], errors="coerce")
        * pd.to_numeric(detail["strict_sell_leg_net_ret"], errors="coerce")
    )
    detail["weighted_sell_leg_ret_gross"] = (
        pd.to_numeric(detail["weight"], errors="coerce")
        * pd.to_numeric(detail["strict_sell_leg_gross_ret"], errors="coerce")
    )
    detail["strict_sell_fee_weighted"] = pd.to_numeric(detail["weight"], errors="coerce") * sell_rate
    return detail.sort_values("code", kind="mergesort").reset_index(drop=True)


def _numeric_detail_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def compute_strict_cycle_detail_for_holdings(
    *,
    raw_data_root: str | Path,
    buy_day: date,
    sell_day: date,
    buy_holdings: pd.DataFrame,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> pd.DataFrame:
    """Return strict buy-day plus sell-day cycle detail for an existing buy basket."""
    if buy_holdings is None or buy_holdings.empty:
        return pd.DataFrame()
    cfg = pool_cfg or load_benchmark_pool_config()
    buy_part = _ensure_code_column(buy_holdings, label="strict cycle buy holdings").copy()
    if "weight" not in buy_part.columns:
        buy_part["weight"] = 1.0 / len(buy_part)

    sell_detail = compute_strict_sell_detail_for_holdings(
        raw_data_root=raw_data_root,
        sell_day=sell_day,
        prev_holdings=_to_strict_sell_holdings(buy_part),
        sell_bps=sell_bps,
        pool_cfg=cfg,
    )

    buy_cols = [
        "code",
        "pool_day",
        "buy_trade_day",
        "weight",
        "buy_price",
        "buy_close_price",
        "buy_leg_ret_gross",
        "buy_leg_ret_net",
        "weighted_buy_leg_ret_net",
        "weighted_buy_leg_ret_gross",
        "buy_cost_bps",
    ]
    sell_cols = [
        "code",
        "sell_trade_day",
        "strict_prev_close_price",
        cfg.sell_twap_col,
        "strict_sell_price",
        "sell_price",
        "sell_price_source",
        "sell_missing_fallback",
        "strict_sell_leg_gross_ret",
        "strict_sell_leg_net_ret",
        "weighted_sell_leg_ret_net",
        "weighted_sell_leg_ret_gross",
        "strict_sell_fee_weighted",
        "sell_cost_bps",
    ]
    if "code" in sell_detail.columns:
        sell_part = sell_detail[[c for c in sell_cols if c in sell_detail.columns]]
    else:
        sell_part = pd.DataFrame({"code": pd.Series(dtype=str)})
    detail = buy_part[[c for c in buy_cols if c in buy_part.columns]].merge(sell_part, on="code", how="left")

    weight = _numeric_detail_column(detail, "weight").fillna(0.0)
    buy_net = _numeric_detail_column(detail, "buy_leg_ret_net")
    sell_net = _numeric_detail_column(detail, "strict_sell_leg_net_ret")
    buy_gross = _numeric_detail_column(detail, "buy_leg_ret_gross")
    sell_gross = _numeric_detail_column(detail, "strict_sell_leg_gross_ret")

    detail["trade_date"] = pd.to_datetime(buy_day)
    detail["trade_day"] = pd.to_datetime(buy_day)
    detail["next_day"] = pd.to_datetime(sell_day)
    detail["sell_day"] = pd.to_datetime(sell_day)
    detail["buy_twap_col"] = cfg.buy_twap_col
    detail["sell_twap_col"] = cfg.sell_twap_col
    detail["return_net"] = buy_net + sell_net
    detail["return_gross"] = buy_gross + sell_gross
    detail["weighted_buy_leg_ret_net"] = weight * buy_net
    detail["weighted_sell_leg_ret_net"] = weight * sell_net
    detail["weighted_return"] = weight * detail["return_net"]
    detail["full_cycle_ret_net"] = detail["return_net"]
    return detail.sort_values("code", kind="mergesort").reset_index(drop=True)


def compute_benchmark_cycle_detail_for_day(
    *,
    raw_data_root: str | Path,
    buy_day: date,
    sell_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> pd.DataFrame:
    cfg = pool_cfg or load_benchmark_pool_config()
    buy_holdings = _build_buy_holdings_for_day(
        raw_data_root=raw_data_root,
        buy_day=buy_day,
        buy_bps=buy_bps,
        pool_cfg=cfg,
    )
    return compute_strict_cycle_detail_for_holdings(
        raw_data_root=raw_data_root,
        buy_day=buy_day,
        sell_day=sell_day,
        buy_holdings=buy_holdings,
        sell_bps=sell_bps,
        pool_cfg=cfg,
    )


def compute_benchmark_detail_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> pd.DataFrame:
    del next_day
    cfg = pool_cfg or load_benchmark_pool_config()
    prev_day = _previous_trading_day(raw_data_root, trade_day, cfg)
    prev_holdings = _to_strict_sell_holdings(
        _build_buy_holdings_for_day(
            raw_data_root=raw_data_root,
            buy_day=prev_day,
            buy_bps=buy_bps,
            pool_cfg=cfg,
        )
    )
    sell_detail = compute_strict_sell_detail_for_holdings(
        raw_data_root=raw_data_root,
        sell_day=trade_day,
        prev_holdings=prev_holdings,
        sell_bps=sell_bps,
        pool_cfg=cfg,
    )
    buy_holdings = _build_buy_holdings_for_day(
        raw_data_root=raw_data_root,
        buy_day=trade_day,
        buy_bps=buy_bps,
        pool_cfg=cfg,
    )

    buy_cols = [
        "code",
        "pool_day",
        "buy_trade_day",
        "weight",
        "buy_price",
        "buy_close_price",
        "buy_leg_ret_gross",
        "buy_leg_ret_net",
        "weighted_buy_leg_ret_net",
    ]
    buy_detail = buy_holdings[[c for c in buy_cols if c in buy_holdings.columns]].rename(
        columns={"weight": "buy_weight"}
    )
    sell_cols = [
        "code",
        "buy_trade_day",
        "sell_trade_day",
        "weight",
        "strict_prev_close_price",
        cfg.sell_twap_col,
        "strict_sell_price",
        "sell_price",
        "sell_price_source",
        "sell_missing_fallback",
        "strict_sell_leg_gross_ret",
        "strict_sell_leg_net_ret",
        "weighted_sell_leg_ret_net",
    ]
    sell_part = sell_detail[[c for c in sell_cols if c in sell_detail.columns]].rename(
        columns={"weight": "sell_weight"}
    )
    detail = buy_detail.merge(sell_part, on="code", how="outer", suffixes=("", "_sell"))
    detail["trade_date"] = pd.to_datetime(trade_day)
    detail["trade_day"] = pd.to_datetime(trade_day)
    detail["next_day"] = pd.to_datetime(trade_day)
    detail["weight"] = pd.to_numeric(detail.get("buy_weight"), errors="coerce").fillna(
        pd.to_numeric(detail.get("sell_weight"), errors="coerce")
    )
    detail["weighted_buy_leg_ret_net"] = pd.to_numeric(
        detail.get("weighted_buy_leg_ret_net"),
        errors="coerce",
    ).fillna(0.0)
    detail["weighted_sell_leg_ret_net"] = pd.to_numeric(
        detail.get("weighted_sell_leg_ret_net"),
        errors="coerce",
    ).fillna(0.0)
    detail["weighted_return"] = detail["weighted_buy_leg_ret_net"] + detail["weighted_sell_leg_ret_net"]
    detail["return_net"] = detail["weighted_return"]
    detail["full_cycle_ret_net"] = detail["weighted_return"]
    detail["buy_cost_bps"] = float(buy_bps)
    detail["sell_cost_bps"] = float(sell_bps)
    return detail.sort_values("code", kind="mergesort").reset_index(drop=True)


def compute_benchmark_return_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> float:
    del next_day
    breakdown = compute_benchmark_breakdown_for_day(
        raw_data_root=raw_data_root,
        trade_day=trade_day,
        next_day=trade_day,
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        pool_cfg=pool_cfg,
    )
    return float(breakdown.full_cycle_ret_net)


def _build_next_day_map(
    *,
    raw_data_root: str | Path,
    trade_days: Iterable[date],
) -> dict[date, date]:
    wanted = sorted(set(trade_days))
    if not wanted:
        return {}
    cfg = load_benchmark_pool_config()
    all_days = _available_trade_days(raw_data_root, cfg)
    pos_map = {d: i for i, d in enumerate(all_days)}
    out: dict[date, date] = {}
    for day in wanted:
        idx = pos_map.get(day)
        if idx is None or idx + 1 >= len(all_days):
            raise RuntimeError(f"benchmark next trading day missing for: {day:%Y-%m-%d}")
        out[day] = all_days[idx + 1]
    return out


def compute_benchmark_breakdowns_for_days(
    *,
    raw_data_root: str | Path,
    trade_days: Iterable[date],
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
    skip_failed_days: bool = False,
) -> pd.DataFrame:
    cfg = pool_cfg or load_benchmark_pool_config()
    days = sorted(set(trade_days))
    if not days:
        return pd.DataFrame()

    try:
        prev_day = _previous_trading_day(raw_data_root, days[0], cfg)
        prev_holdings = _to_strict_sell_holdings(
            _build_buy_holdings_for_day(
                raw_data_root=raw_data_root,
                buy_day=prev_day,
                buy_bps=buy_bps,
                pool_cfg=cfg,
            )
        )
    except Exception:
        if not skip_failed_days:
            raise
        return pd.DataFrame()

    rows: list[dict] = []
    nav = 1.0
    for day in days:
        try:
            sell_detail = compute_strict_sell_detail_for_holdings(
                raw_data_root=raw_data_root,
                sell_day=day,
                prev_holdings=prev_holdings,
                sell_bps=sell_bps,
                pool_cfg=cfg,
            )
            if sell_detail.empty:
                raise RuntimeError(f"benchmark strict sell detail empty: sell_day={day:%Y-%m-%d}")
            buy_holdings = _build_buy_holdings_for_day(
                raw_data_root=raw_data_root,
                buy_day=day,
                buy_bps=buy_bps,
                pool_cfg=cfg,
            )
        except Exception:
            if not skip_failed_days:
                raise
            continue

        buy_net_return = float(pd.to_numeric(buy_holdings["weighted_buy_leg_ret_net"], errors="coerce").sum())
        buy_gross_return = float(pd.to_numeric(buy_holdings["weighted_buy_leg_ret_gross"], errors="coerce").sum())
        sell_net_return = float(pd.to_numeric(sell_detail["weighted_sell_leg_ret_net"], errors="coerce").sum())
        sell_gross_return = float(pd.to_numeric(sell_detail["weighted_sell_leg_ret_gross"], errors="coerce").sum())
        benchmark_return = sell_net_return + buy_net_return
        nav *= 1.0 + benchmark_return

        buy_weight = pd.to_numeric(buy_holdings["weight"], errors="coerce").fillna(0.0)
        sell_weight = pd.to_numeric(sell_detail["weight"], errors="coerce").fillna(0.0)
        fallback_mask = sell_detail["sell_missing_fallback"].astype(bool)
        rows.append(
            {
                "trade_date": day,
                "trade_day": day,
                "sell_trade_day": day,
                "buy_trade_day": day,
                "prev_buy_trade_day": pd.to_datetime(prev_holdings["buy_trade_day"].iloc[0]).date()
                if "buy_trade_day" in prev_holdings.columns and not prev_holdings.empty
                else pd.NaT,
                "next_day": day,
                "pool_day": pd.to_datetime(buy_holdings["pool_day"].iloc[0]).date()
                if "pool_day" in buy_holdings.columns and not buy_holdings.empty
                else pd.NaT,
                "benchmark_return": float(benchmark_return),
                "strict_daily_return": float(benchmark_return),
                "full_cycle_ret_net": float(benchmark_return),
                "buy_leg_ret_net": float(buy_net_return),
                "sell_leg_ret_net": float(sell_net_return),
                "strict_buy_leg_net_return": float(buy_net_return),
                "strict_buy_leg_gross_return": float(buy_gross_return),
                "strict_sell_leg_net_return": float(sell_net_return),
                "strict_sell_leg_gross_return": float(sell_gross_return),
                "buy_fee_weighted": float(buy_weight.sum() * bps_to_rate(buy_bps)),
                "strict_sell_fee_weighted": float(sell_weight.sum() * bps_to_rate(sell_bps)),
                "count": int(buy_holdings["code"].nunique()),
                "buy_count": int(buy_holdings["code"].nunique()),
                "sell_count": int(sell_detail["code"].nunique()),
                "strict_sell_count": int(sell_detail["code"].nunique()),
                "fallback_sell_codes": int(fallback_mask.sum()),
                "fallback_sell_weight": float(sell_weight.loc[fallback_mask].sum()),
                "buy_cost_bps_mean": float(buy_bps),
                "sell_cost_bps_mean": float(sell_bps),
                "nav": float(nav),
                "benchmark_method": "strict_official_prev_close_split",
                "buy_twap_col": cfg.buy_twap_col,
                "sell_twap_col": cfg.sell_twap_col,
            }
        )
        prev_holdings = _to_strict_sell_holdings(buy_holdings)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def compute_benchmark_breakdown_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> BenchmarkReturnBreakdown:
    del next_day
    frame = compute_benchmark_breakdowns_for_days(
        raw_data_root=raw_data_root,
        trade_days=[trade_day],
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        pool_cfg=pool_cfg,
        skip_failed_days=False,
    )
    if frame.empty:
        raise RuntimeError(f"benchmark failed for day: {trade_day:%Y-%m-%d}")
    row = frame.iloc[0]
    return BenchmarkReturnBreakdown(
        full_cycle_ret_net=float(row["benchmark_return"]),
        buy_leg_ret_net=float(row["strict_buy_leg_net_return"]),
        sell_leg_ret_net=float(row["strict_sell_leg_net_return"]),
        count=int(row["buy_count"]),
        sell_count=int(row["strict_sell_count"]),
        fallback_sell_codes=int(row.get("fallback_sell_codes", 0)),
        fallback_sell_weight=float(row.get("fallback_sell_weight", 0.0)),
    )


def compute_benchmark_returns_for_days(
    *,
    raw_data_root: str | Path,
    trade_days: Iterable[date],
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
    skip_failed_days: bool = False,
) -> pd.Series:
    frame = compute_benchmark_breakdowns_for_days(
        raw_data_root=raw_data_root,
        trade_days=trade_days,
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        pool_cfg=pool_cfg,
        skip_failed_days=skip_failed_days,
    )
    if frame.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        pd.to_numeric(frame["benchmark_return"], errors="coerce").values,
        index=pd.to_datetime(frame["trade_date"], errors="coerce").dt.date,
        dtype=float,
    ).dropna().sort_index()
