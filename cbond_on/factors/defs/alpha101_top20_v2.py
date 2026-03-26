from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time

EPS = 1e-8


def _prepare_panel(ctx: FactorComputeContext, required: list[str]) -> pd.DataFrame:
    panel = ensure_trade_time(ctx.panel)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"alpha101 missing columns: {missing}")
    frame = panel.reset_index()[["dt", "code", "seq", *required]].copy()
    frame = frame.sort_values(["dt", "code", "seq"], kind="mergesort")
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _group_scalar(frame: pd.DataFrame, func) -> pd.Series:
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for (dt, code), g in frame.groupby(["dt", "code"], sort=False):
        g = g.sort_values("seq", kind="mergesort")
        try:
            val = float(func(g))
        except Exception:
            val = np.nan
        rows.append((dt, str(code), val))
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.MultiIndex.from_tuples([(dt, code) for dt, code, _ in rows], names=["dt", "code"])
    out = pd.Series([v for _, _, v in rows], index=idx, dtype="float64")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _cs_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.groupby(level="dt").rank(pct=True, method="average")


def _delta_last(series: pd.Series, periods: int) -> float:
    periods = max(1, int(periods))
    if len(series) <= periods:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-1 - periods])


def _delay_last(series: pd.Series, periods: int) -> float:
    periods = max(1, int(periods))
    if series.empty:
        return 0.0
    if len(series) <= periods:
        return float(series.iloc[0])
    return float(series.iloc[-1 - periods])


def _ts_rank_last(series: pd.Series, window: int) -> float:
    window = max(1, int(window))
    tail = series.tail(window).dropna()
    if tail.empty:
        return 0.0
    ranked = tail.rank(pct=True, method="average")
    return float(ranked.iloc[-1])


def _corr_last(x: pd.Series, y: pd.Series, window: int) -> float:
    window = max(2, int(window))
    tail = pd.concat([x, y], axis=1).dropna().tail(window)
    if len(tail) < 2:
        return 0.0
    a = tail.iloc[:, 0].astype("float64")
    b = tail.iloc[:, 1].astype("float64")
    if float(a.std(ddof=0)) <= EPS or float(b.std(ddof=0)) <= EPS:
        return 0.0
    corr = a.corr(b)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _cov_last(x: pd.Series, y: pd.Series, window: int) -> float:
    window = max(2, int(window))
    tail = pd.concat([x, y], axis=1).dropna().tail(window)
    if len(tail) < 2:
        return 0.0
    cov = tail.iloc[:, 0].astype("float64").cov(tail.iloc[:, 1].astype("float64"))
    if pd.isna(cov):
        return 0.0
    return float(cov)


def _open_like(g: pd.DataFrame) -> pd.Series:
    open_px: pd.Series | None = None
    if "open" in g.columns:
        open_px = g["open"].astype("float64")
    mid: pd.Series | None = None
    if "ask_price1" in g.columns and "bid_price1" in g.columns:
        ask = g["ask_price1"].astype("float64")
        bid = g["bid_price1"].astype("float64")
        mid = (ask + bid) / 2.0
    if mid is not None and open_px is not None:
        return mid.where(mid.notna(), open_px)
    if mid is not None:
        return mid
    if open_px is not None:
        return open_px
    raise KeyError("open-like requires open or [ask_price1, bid_price1]")


class _AlphaBase(Factor):
    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        raise NotImplementedError

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        out = self._compute_series(ctx)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out.name = self.output_name(self.name)
        return out


@FactorRegistry.register("alpha001_signed_power_v1")
class Alpha001SignedPowerV1Factor(_AlphaBase):
    name = "alpha001_signed_power_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        stddev_window = int(ctx.params.get("stddev_window", 20))
        ts_max_window = int(ctx.params.get("ts_max_window", 5))
        frame = _prepare_panel(ctx, ["last", "pre_close"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            std_ret = returns.rolling(max(2, stddev_window), min_periods=2).std().fillna(0.0)
            base = np.where(returns < 0.0, std_ret, last_px)
            sp = np.sign(base) * np.power(np.abs(base), 2.0)
            ts_max_sp = pd.Series(sp).rolling(max(1, ts_max_window), min_periods=1).max()
            return float(ts_max_sp.iloc[-1])

        raw = _group_scalar(frame, _calc)
        return _cs_rank(raw) - 0.5


@FactorRegistry.register("alpha002_corr_volume_return_v1")
class Alpha002CorrVolumeReturnV1Factor(_AlphaBase):
    name = "alpha002_corr_volume_return_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 6))
        frame = _prepare_panel(ctx, ["volume", "last", "open", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64").clip(lower=0.0)
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            log_volume = np.log(volume + EPS)
            delta_log_vol = log_volume.diff(2)
            ret = (last_px - open_) / (open_ + EPS)
            x = delta_log_vol.rank(pct=True, method="average")
            y = ret.rank(pct=True, method="average")
            return -_corr_last(x, y, corr_window)

        return _group_scalar(frame, _calc)


@FactorRegistry.register("alpha003_corr_open_volume_v1")
class Alpha003CorrOpenVolumeV1Factor(_AlphaBase):
    name = "alpha003_corr_open_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(ctx, ["open", "ask_price1", "bid_price1", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            open_rank = _open_like(g).rank(pct=True, method="average")
            vol_rank = g["volume"].astype("float64").rank(pct=True, method="average")
            return -_corr_last(open_rank, vol_rank, corr_window)

        return _group_scalar(frame, _calc)


@FactorRegistry.register("alpha004_ts_rank_low_v1")
class Alpha004TsRankLowV1Factor(_AlphaBase):
    name = "alpha004_ts_rank_low_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_rank_window = int(ctx.params.get("ts_rank_window", 9))
        frame = _prepare_panel(ctx, ["low"])

        def _calc(g: pd.DataFrame) -> float:
            low_rank = g["low"].astype("float64").rank(pct=True, method="average")
            return -_ts_rank_last(low_rank, ts_rank_window)

        return _group_scalar(frame, _calc)


@FactorRegistry.register("alpha005_vwap_gap_v1")
class Alpha005VwapGapV1Factor(_AlphaBase):
    name = "alpha005_vwap_gap_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        vwap_window = int(ctx.params.get("vwap_window", 10))
        frame = _prepare_panel(ctx, ["amount", "volume", "open", "ask_price1", "bid_price1", "last"])

        def _gap1(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            open_ = _open_like(g)
            vwap = amount / (volume + EPS)
            avg_vwap = vwap.rolling(max(1, vwap_window), min_periods=1).mean()
            return float(open_.iloc[-1] - avg_vwap.iloc[-1])

        def _gap2(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            return float(last_px.iloc[-1] - vwap.iloc[-1])

        gap1 = _group_scalar(frame, _gap1)
        gap2 = _group_scalar(frame, _gap2)
        return _cs_rank(gap1) * (-_cs_rank(gap2).abs())


@FactorRegistry.register("alpha006_corr_open_volume_neg_v1")
class Alpha006CorrOpenVolumeNegV1Factor(_AlphaBase):
    name = "alpha006_corr_open_volume_neg_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(ctx, ["open", "ask_price1", "bid_price1", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            volume = g["volume"].astype("float64")
            return -_corr_last(open_, volume, corr_window)

        return _group_scalar(frame, _calc)


@FactorRegistry.register("alpha007_volume_breakout_v1")
class Alpha007VolumeBreakoutV1Factor(_AlphaBase):
    name = "alpha007_volume_breakout_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 20))
        delta_window = int(ctx.params.get("delta_window", 7))
        ts_rank_window = int(ctx.params.get("ts_rank_window", 60))
        frame = _prepare_panel(ctx, ["amount", "volume", "last"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            delta_last = last_px.diff(max(1, delta_window))
            if float(adv.iloc[-1]) >= float(volume.iloc[-1]):
                return -1.0
            sign_delta = float(np.sign(delta_last.iloc[-1])) if pd.notna(delta_last.iloc[-1]) else 0.0
            ts_rank_abs = _ts_rank_last(delta_last.abs(), ts_rank_window)
            return float((-1.0 * ts_rank_abs) * sign_delta)

        return _group_scalar(frame, _calc)


@FactorRegistry.register("alpha008_open_return_momentum_v1")
class Alpha008OpenReturnMomentumV1Factor(_AlphaBase):
    name = "alpha008_open_return_momentum_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        sum_window = int(ctx.params.get("sum_window", 5))
        delay_window = int(ctx.params.get("delay_window", 10))
        frame = _prepare_panel(ctx, ["open", "ask_price1", "bid_price1", "last"])

        def _calc(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            last_px = g["last"].astype("float64")
            ret = (last_px - open_) / (open_ + EPS)
            sum_open = open_.rolling(max(1, sum_window), min_periods=1).sum()
            sum_ret = ret.rolling(max(1, sum_window), min_periods=1).sum()
            prod = sum_open * sum_ret
            delayed = prod.shift(max(1, delay_window))
            delay_val = float(delayed.iloc[-1]) if pd.notna(delayed.iloc[-1]) else float(prod.iloc[0])
            return float(prod.iloc[-1] - delay_val)

        raw = _group_scalar(frame, _calc)
        return -_cs_rank(raw)


@FactorRegistry.register("alpha009_close_change_filter_v1")
class Alpha009CloseChangeFilterV1Factor(_AlphaBase):
    name = "alpha009_close_change_filter_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_window = int(ctx.params.get("ts_window", 5))
        frame = _prepare_panel(ctx, ["last"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta_last = last_px.diff(1)
            ts_min = delta_last.rolling(max(1, ts_window), min_periods=1).min().iloc[-1]
            ts_max = delta_last.rolling(max(1, ts_window), min_periods=1).max().iloc[-1]
            d = float(delta_last.iloc[-1]) if pd.notna(delta_last.iloc[-1]) else 0.0
            if float(ts_min) > 0.0:
                return d
            if float(ts_max) < 0.0:
                return d
            return -d

        return _group_scalar(frame, _calc)


@FactorRegistry.register("alpha010_close_change_rank_v1")
class Alpha010CloseChangeRankV1Factor(_AlphaBase):
    name = "alpha010_close_change_rank_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_window = int(ctx.params.get("ts_window", 4))
        frame = _prepare_panel(ctx, ["last"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta_last = last_px.diff(1)
            ts_min = delta_last.rolling(max(1, ts_window), min_periods=1).min().iloc[-1]
            ts_max = delta_last.rolling(max(1, ts_window), min_periods=1).max().iloc[-1]
            d = float(delta_last.iloc[-1]) if pd.notna(delta_last.iloc[-1]) else 0.0
            if float(ts_min) > 0.0:
                return d
            if float(ts_max) < 0.0:
                return d
            return -d

        raw = _group_scalar(frame, _calc)
        return _cs_rank(raw)


@FactorRegistry.register("alpha011_vwap_close_volume_v1")
class Alpha011VwapCloseVolumeV1Factor(_AlphaBase):
    name = "alpha011_vwap_close_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_window = int(ctx.params.get("ts_window", 3))
        frame = _prepare_panel(ctx, ["amount", "volume", "last"])

        def _max_diff(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            diff = vwap - last_px
            return float(diff.rolling(max(1, ts_window), min_periods=1).max().iloc[-1])

        def _min_diff(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            vwap = amount / (volume + EPS)
            diff = vwap - last_px
            return float(diff.rolling(max(1, ts_window), min_periods=1).min().iloc[-1])

        def _delta_vol(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            return _delta_last(volume, ts_window)

        max_diff = _group_scalar(frame, _max_diff)
        min_diff = _group_scalar(frame, _min_diff)
        delta_vol = _group_scalar(frame, _delta_vol)
        return (_cs_rank(max_diff) + _cs_rank(min_diff)) * _cs_rank(delta_vol)


@FactorRegistry.register("alpha012_volume_close_reversal_v1")
class Alpha012VolumeCloseReversalV1Factor(_AlphaBase):
    name = "alpha012_volume_close_reversal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(ctx, ["volume", "last"])

        def _calc(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            d_vol = _delta_last(volume, 1)
            d_last = _delta_last(last_px, 1)
            return float(np.sign(d_vol) * (-d_last))

        return _group_scalar(frame, _calc)


@FactorRegistry.register("alpha013_cov_close_volume_v1")
class Alpha013CovCloseVolumeV1Factor(_AlphaBase):
    name = "alpha013_cov_close_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        cov_window = int(ctx.params.get("cov_window", 5))
        frame = _prepare_panel(ctx, ["last", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            close_rank = g["last"].astype("float64").rank(pct=True, method="average")
            vol_rank = g["volume"].astype("float64").rank(pct=True, method="average")
            return _cov_last(close_rank, vol_rank, cov_window)

        raw = _group_scalar(frame, _calc)
        return -_cs_rank(raw)


@FactorRegistry.register("alpha014_return_open_volume_v1")
class Alpha014ReturnOpenVolumeV1Factor(_AlphaBase):
    name = "alpha014_return_open_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delta_window = int(ctx.params.get("delta_window", 3))
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(
            ctx,
            ["last", "pre_close", "open", "ask_price1", "bid_price1", "volume"],
        )

        def _delta_ret(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            return _delta_last(returns, delta_window)

        def _corr_ov(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            volume = g["volume"].astype("float64")
            return _corr_last(open_, volume, corr_window)

        delta_ret = _group_scalar(frame, _delta_ret)
        corr_ov = _group_scalar(frame, _corr_ov)
        return (-_cs_rank(delta_ret)) * corr_ov


@FactorRegistry.register("alpha015_high_volume_corr_v1")
class Alpha015HighVolumeCorrV1Factor(_AlphaBase):
    name = "alpha015_high_volume_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 3))
        sum_window = int(ctx.params.get("sum_window", 3))
        frame = _prepare_panel(ctx, ["high", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            high_rank = g["high"].astype("float64").rank(pct=True, method="average")
            vol_rank = g["volume"].astype("float64").rank(pct=True, method="average")
            corr_series = high_rank.rolling(max(2, corr_window), min_periods=2).corr(vol_rank)
            corr_series = corr_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            ranked_corr = corr_series.rank(pct=True, method="average")
            return float(ranked_corr.tail(max(1, sum_window)).sum())

        raw = _group_scalar(frame, _calc)
        return -raw


@FactorRegistry.register("alpha016_cov_high_volume_v1")
class Alpha016CovHighVolumeV1Factor(_AlphaBase):
    name = "alpha016_cov_high_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        cov_window = int(ctx.params.get("cov_window", 5))
        frame = _prepare_panel(ctx, ["high", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            high_rank = g["high"].astype("float64").rank(pct=True, method="average")
            vol_rank = g["volume"].astype("float64").rank(pct=True, method="average")
            return _cov_last(high_rank, vol_rank, cov_window)

        raw = _group_scalar(frame, _calc)
        return -_cs_rank(raw)


@FactorRegistry.register("alpha017_close_rank_volume_v1")
class Alpha017CloseRankVolumeV1Factor(_AlphaBase):
    name = "alpha017_close_rank_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 20))
        ts_rank_close_window = int(ctx.params.get("ts_rank_close_window", 10))
        ts_rank_vol_window = int(ctx.params.get("ts_rank_vol_window", 5))
        frame = _prepare_panel(ctx, ["last", "amount", "volume"])

        def _term1(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            return _ts_rank_last(last_px, ts_rank_close_window)

        def _term2(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta2 = last_px.diff(1).diff(1)
            return float(delta2.iloc[-1]) if pd.notna(delta2.iloc[-1]) else 0.0

        def _term3(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            ratio = volume / (adv + EPS)
            return _ts_rank_last(ratio, ts_rank_vol_window)

        t1 = _group_scalar(frame, _term1)
        t2 = _group_scalar(frame, _term2)
        t3 = _group_scalar(frame, _term3)
        return (-_cs_rank(t1)) * _cs_rank(t2) * _cs_rank(t3)


@FactorRegistry.register("alpha018_close_open_vol_v1")
class Alpha018CloseOpenVolV1Factor(_AlphaBase):
    name = "alpha018_close_open_vol_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        stddev_window = int(ctx.params.get("stddev_window", 5))
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(ctx, ["last", "open", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            diff = last_px - open_
            std_diff = diff.abs().rolling(max(2, stddev_window), min_periods=2).std().fillna(0.0)
            corr_co = _corr_last(last_px, open_, corr_window)
            return float(std_diff.iloc[-1] + diff.iloc[-1] + corr_co)

        raw = _group_scalar(frame, _calc)
        return -_cs_rank(raw)


@FactorRegistry.register("alpha019_close_momentum_sign_v1")
class Alpha019CloseMomentumSignV1Factor(_AlphaBase):
    name = "alpha019_close_momentum_sign_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delta_window = int(ctx.params.get("delta_window", 7))
        sum_window = int(ctx.params.get("sum_window", 250))
        frame = _prepare_panel(ctx, ["last", "pre_close"])

        def _sign_term(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta_last = last_px.diff(max(1, delta_window))
            last_change = float(last_px.iloc[-1] - _delay_last(last_px, delta_window))
            delta_value = float(delta_last.iloc[-1]) if pd.notna(delta_last.iloc[-1]) else 0.0
            return float(np.sign(last_change + delta_value))

        def _sum_ret(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            return float(returns.rolling(max(1, sum_window), min_periods=1).sum().iloc[-1])

        sign_term = _group_scalar(frame, _sign_term)
        sum_ret = _group_scalar(frame, _sum_ret)
        return (-sign_term) * (1.0 + _cs_rank(1.0 + sum_ret))


@FactorRegistry.register("alpha020_open_delay_range_v1")
class Alpha020OpenDelayRangeV1Factor(_AlphaBase):
    name = "alpha020_open_delay_range_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delay_window = int(ctx.params.get("delay_window", 1))
        frame = _prepare_panel(ctx, ["open", "ask_price1", "bid_price1", "high", "low", "last"])

        def _d1(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            high = g["high"].astype("float64")
            return float(open_.iloc[-1] - _delay_last(high, delay_window))

        def _d2(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            last_px = g["last"].astype("float64")
            return float(open_.iloc[-1] - _delay_last(last_px, delay_window))

        def _d3(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            low = g["low"].astype("float64")
            return float(open_.iloc[-1] - _delay_last(low, delay_window))

        d1 = _group_scalar(frame, _d1)
        d2 = _group_scalar(frame, _d2)
        d3 = _group_scalar(frame, _d3)
        return (-_cs_rank(d1)) * _cs_rank(d2) * _cs_rank(d3)
