from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import (
    ensure_trade_time,
    _iter_dt_code_groups,
    slice_window,
)


EPS = 1e-8


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"panel missing column: {col}")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _valid_price(series: pd.Series) -> pd.Series:
    return series.where(series > 0)


def _last_valid(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def _ret(series: pd.Series) -> float:
    px = _valid_price(pd.to_numeric(series, errors="coerce")).dropna()
    if len(px) < 2:
        return np.nan
    start = float(px.iloc[0])
    end = float(px.iloc[-1])
    if start <= 0:
        return np.nan
    return end / start - 1.0


def _range(series: pd.Series) -> float:
    px = _valid_price(pd.to_numeric(series, errors="coerce")).dropna()
    if px.empty:
        return np.nan
    end = float(px.iloc[-1])
    if end <= 0:
        return np.nan
    return (float(px.max()) - float(px.min())) / end


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def _safe_sum(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan
    return float(s.sum())


def _safe_std(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=0))


def _signed_log(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.sign(value) * np.log1p(abs(value)))


def _signed_log_series(series: pd.Series) -> pd.Series:
    out = np.sign(series) * np.log1p(series.abs())
    return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _level_depth(df: pd.DataFrame, side: str, level: int) -> pd.Series:
    parts = []
    side = str(side).strip().lower()
    prefix = "bid_volume" if side == "bid" else "ask_volume"
    for i in range(1, max(1, int(level)) + 1):
        parts.append(_num(df, f"{prefix}{i}"))
    return sum(parts)


def _depth_imbalance(df: pd.DataFrame, level: int) -> pd.Series:
    bid = _level_depth(df, "bid", level)
    ask = _level_depth(df, "ask", level)
    return (bid - ask) / (bid + ask + EPS)


def _spread(df: pd.DataFrame) -> pd.Series:
    ask = _valid_price(_num(df, "ask_price1"))
    bid = _valid_price(_num(df, "bid_price1"))
    mid = (ask + bid) / 2.0
    return (ask - bid) / (mid + EPS)


def _mid(df: pd.DataFrame) -> pd.Series:
    ask = _valid_price(_num(df, "ask_price1"))
    bid = _valid_price(_num(df, "bid_price1"))
    return (ask + bid) / 2.0


def _microprice_gap(df: pd.DataFrame) -> pd.Series:
    bid_p = _valid_price(_num(df, "bid_price1"))
    ask_p = _valid_price(_num(df, "ask_price1"))
    bid_v = _num(df, "bid_volume1").where(_num(df, "bid_volume1") > 0)
    ask_v = _num(df, "ask_volume1").where(_num(df, "ask_volume1") > 0)
    mid = (bid_p + ask_p) / 2.0
    micro = (bid_p * ask_v + ask_p * bid_v) / (bid_v + ask_v + EPS)
    return (micro - mid) / (mid + EPS)


def _vwap(df: pd.DataFrame) -> pd.Series:
    amount = _num(df, "amount")
    volume = _num(df, "volume")
    return amount / (volume + EPS)


def _last_vwap_gap(df: pd.DataFrame) -> pd.Series:
    last = _valid_price(_num(df, "last"))
    vwap = _valid_price(_vwap(df))
    return (last - vwap) / (vwap + EPS)


def _split_recent_early(df: pd.DataFrame, recent_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("trade_time")
    recent = slice_window(df, int(recent_minutes))
    if recent.empty:
        return recent, df.iloc[:0]
    cutoff = recent["trade_time"].iloc[0]
    early = df[df["trade_time"] < cutoff]
    return recent, early


_METRIC_COLUMNS = [
    "ret_w",
    "range_w",
    "liquidity",
    "trade_size",
    "recent_share",
    "amount_recent_share",
    "volume_accel",
    "amount_accel",
    "trade_accel",
    "vol_per_trade_recent",
    "vol_per_trade_all",
    "ret_recent",
    "ret_early",
    "depth_recent",
    "depth_early",
    "spread_recent",
    "spread_early",
    "micro_gap_last",
    "vwap_gap_last",
    "mid_ret",
    "last_std_ratio",
    "last_pos",
]


def _nan_metrics() -> dict[str, float]:
    return {col: np.nan for col in _METRIC_COLUMNS}


def _calc_metrics(group: pd.DataFrame, *, window_minutes: int, recent_minutes: int, level: int) -> dict[str, float]:
    group = group.sort_values("trade_time")
    window = slice_window(group, window_minutes)
    if window.empty:
        return _nan_metrics()
    recent, early = _split_recent_early(window, recent_minutes)
    if recent.empty:
        return _nan_metrics()

    last = _valid_price(_num(window, "last"))
    ret_w = _ret(last)
    vol_w = _safe_sum(_num(window, "volume"))
    amt_w = _safe_sum(_num(window, "amount"))
    trades_w = _safe_sum(_num(window, "num_trades"))
    vol_recent = _safe_sum(_num(recent, "volume"))
    amt_recent = _safe_sum(_num(recent, "amount"))
    trades_recent = _safe_sum(_num(recent, "num_trades"))
    vol_early = _safe_sum(_num(early, "volume")) if not early.empty else np.nan
    amt_early = _safe_sum(_num(early, "amount")) if not early.empty else np.nan

    depth_recent = _safe_mean(_depth_imbalance(recent, level))
    spread_recent = _safe_mean(_spread(recent))
    px = _last_valid(last)
    lo = float(last.min(skipna=True))
    hi = float(last.max(skipna=True))
    last_std = _safe_std(last)

    out = _nan_metrics()
    out.update(
        {
            "ret_w": ret_w,
            "range_w": _range(last),
            "liquidity": np.log1p(max(amt_w, 0.0)) if not pd.isna(amt_w) else np.nan,
            "trade_size": amt_w / (trades_w + EPS),
            "recent_share": vol_recent / (vol_w + EPS),
            "amount_recent_share": amt_recent / (amt_w + EPS),
            "volume_accel": vol_recent / (vol_early + EPS) if not pd.isna(vol_early) else np.nan,
            "amount_accel": amt_recent / (amt_early + EPS) if not pd.isna(amt_early) else np.nan,
            "trade_accel": trades_recent / (trades_w - trades_recent + EPS),
            "vol_per_trade_recent": amt_recent / (trades_recent + EPS),
            "vol_per_trade_all": amt_w / (trades_w + EPS),
            "ret_recent": _ret(_num(recent, "last")),
            "ret_early": _ret(_num(early, "last")) if not early.empty else np.nan,
            "depth_recent": depth_recent,
            "depth_early": _safe_mean(_depth_imbalance(early, level)) if not early.empty else np.nan,
            "spread_recent": spread_recent,
            "spread_early": _safe_mean(_spread(early)) if not early.empty else np.nan,
            "micro_gap_last": _last_valid(_microprice_gap(window)),
            "vwap_gap_last": _last_valid(_last_vwap_gap(window)),
            "mid_ret": _ret(_mid(window)),
            "last_std_ratio": last_std / (px + EPS) if not pd.isna(last_std) and not pd.isna(px) else np.nan,
            "last_pos": (px - lo) / (hi - lo + EPS) if not pd.isna(px) else np.nan,
        }
    )
    return out


def _build_metrics(panel: pd.DataFrame, *, window_minutes: int, recent_minutes: int, level: int) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, str, dict[str, float]]] = []
    for dt, code, group in _iter_dt_code_groups(panel):
        try:
            values = _calc_metrics(
                group,
                window_minutes=window_minutes,
                recent_minutes=recent_minutes,
                level=level,
            )
        except Exception:
            values = _nan_metrics()
        rows.append((dt, str(code), values))
    if not rows:
        return pd.DataFrame(columns=_METRIC_COLUMNS, dtype="float64")
    idx = pd.MultiIndex.from_tuples([(dt, code) for dt, code, _ in rows], names=["dt", "code"])
    out = pd.DataFrame([values for _, _, values in rows], index=idx, columns=_METRIC_COLUMNS, dtype="float64")
    return out.replace([np.inf, -np.inf], np.nan)


@FactorRegistry.register("ai_factory_wave80_intraday_v1")
class AiFactoryWave80IntradayV1(Factor):
    """Research-only 14:30 intraday composite factors generated for AI factory wave80."""

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        mode = str(ctx.params.get("mode", "")).strip()
        if not mode:
            raise ValueError("ai_factory_wave80_intraday_v1 requires params.mode")

        level = int(ctx.params.get("level", 1))
        required = {
            "last",
            "volume",
            "amount",
            "num_trades",
            "ask_price1",
            "ask_volume1",
            "bid_price1",
            "bid_volume1",
        }
        for i in range(1, level + 1):
            required.add(f"ask_volume{i}")
            required.add(f"bid_volume{i}")
        missing = sorted(c for c in required if c not in panel.columns)
        if missing:
            raise KeyError(f"panel missing columns for wave80 factor: {missing}")

        window_minutes = int(ctx.params.get("window_minutes", 30))
        recent_minutes = int(ctx.params.get("recent_minutes", max(5, window_minutes // 3)))
        power = float(ctx.params.get("power", 1.0))
        scale = float(ctx.params.get("scale", 1.0))
        sign = float(ctx.params.get("sign", 1.0))

        cache_key = (
            "ai_factory_wave80_intraday_v1",
            int(window_minutes),
            int(recent_minutes),
            int(level),
        )
        with ctx.cache_lock:
            metrics = ctx.cache.get(cache_key)
            if metrics is None:
                metrics = _build_metrics(
                    panel,
                    window_minutes=window_minutes,
                    recent_minutes=recent_minutes,
                    level=level,
                )
                ctx.cache[cache_key] = metrics

        if mode == "ret_depth_liq":
            value = metrics["ret_w"] * metrics["depth_recent"] * metrics["liquidity"]
        elif mode == "ret_spread_inverse":
            value = metrics["ret_w"] / (metrics["spread_recent"] + EPS)
        elif mode == "vwap_gap_depth":
            value = metrics["vwap_gap_last"] * metrics["depth_recent"]
        elif mode == "micro_gap_liq":
            value = metrics["micro_gap_last"] * metrics["liquidity"]
        elif mode == "range_amount_pressure":
            value = metrics["range_w"] * metrics["amount_recent_share"] * np.sign(metrics["ret_recent"])
        elif mode == "depth_shift_return":
            value = (metrics["depth_recent"] - metrics["depth_early"]) * metrics["ret_recent"]
        elif mode == "spread_shift_reversal":
            value = (metrics["spread_recent"] - metrics["spread_early"]) * -metrics["ret_recent"]
        elif mode == "tail_volume_reversal":
            value = metrics["recent_share"] * (metrics["ret_recent"] - metrics["ret_early"])
        elif mode == "trade_size_momentum":
            value = (
                metrics["vol_per_trade_recent"] / (metrics["vol_per_trade_all"] + EPS) - 1.0
            ) * metrics["ret_recent"]
        elif mode == "mid_vwap_contrast":
            value = (metrics["mid_ret"] - metrics["ret_w"]) * metrics["vwap_gap_last"]
        elif mode == "depth_spread_absorption":
            value = metrics["depth_recent"] / (metrics["spread_recent"] + EPS) * np.sign(metrics["ret_recent"])
        elif mode == "liquidity_volatility_balance":
            value = metrics["liquidity"] / (metrics["last_std_ratio"] + EPS) * np.sign(metrics["ret_w"])
        elif mode == "amount_accel_depth":
            value = metrics["amount_accel"] * metrics["depth_recent"]
        elif mode == "volume_accel_spread":
            value = metrics["volume_accel"] / (metrics["spread_recent"] + EPS)
        elif mode == "trade_accel_micro":
            value = metrics["trade_accel"] * metrics["micro_gap_last"]
        elif mode == "price_position_depth":
            value = (metrics["last_pos"] - 0.5) * metrics["depth_recent"]
        else:
            raise ValueError(f"unsupported wave80 mode: {mode}")

        value = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if power != 1.0:
            value = np.sign(value) * np.power(value.abs(), power)
        value = _signed_log_series(value * scale) * sign
        return value.astype("float64")
