from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from cbond_on.infra.data.snapshot_loader import pick_twap_column


def compute_mark_nav(targets: pd.DataFrame, panel) -> pd.Series:
    data = panel.data
    nav = []
    cash = 1.0
    positions = {}
    for dt in targets.index:
        if dt not in data.index.get_level_values(0):
            continue
        snap = data.xs(dt, level=0)
        if "signal_price" in snap.columns:
            mark_col = "signal_price"
        else:
            mark_col = pick_twap_column(snap)
            if mark_col is None:
                mark_col = "signal_price"
        mark = snap[mark_col]
        nav_value = cash + sum(positions.get(c, 0) * mark.get(c, 0) for c in positions)

        weights = targets.loc[dt].copy()
        weights = weights[weights > 0]
        target_shares = {}
        if not weights.empty:
            for code, w in weights.items():
                px = mark.get(code, pd.NA)
                if pd.isna(px) or px <= 0:
                    continue
                target_notional = nav_value * float(w)
                target_shares[code] = target_notional / px
        positions = {k: v for k, v in target_shares.items() if v != 0}
        cash = nav_value - sum(positions.get(c, 0) * mark.get(c, 0) for c in positions)

        nav.append((dt, nav_value))
    return pd.Series(dict(nav)).sort_index()


def compute_weighted_last(
    targets: pd.DataFrame, panel, *, price_col: str = "signal_price"
) -> pd.Series:
    weighted_last = []
    for dt in targets.index:
        if dt not in panel.data.index.get_level_values(0):
            continue
        snap = panel.data.xs(dt, level=0)
        if price_col not in snap.columns:
            weighted_last.append((dt, pd.NA))
            continue
        weights = targets.loc[dt].copy()
        weights = weights[weights > 0]
        if weights.empty:
            weighted_last.append((dt, pd.NA))
            continue
        w = weights / weights.sum()
        prices = snap[price_col]
        w = w[w.index.isin(prices.index)]
        if w.empty:
            weighted_last.append((dt, pd.NA))
            continue
        weighted_last.append((dt, (w * prices.loc[w.index]).sum()))
    return pd.Series(dict(weighted_last)).sort_index().dropna()


def compress_lunch(index: Iterable[pd.Timestamp]) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    minutes = idx.hour.astype(int) * 60 + idx.minute.astype(int)
    lunch_start = 11 * 60 + 30
    lunch_end = 13 * 60
    adj = pd.Series(minutes)
    adj = adj.where(adj < lunch_end, adj - (lunch_end - lunch_start))
    trading_day_minutes = 120 + 105  # 9:30-11:30 + 13:00-14:45

    # 只按有数据的交易日做连续映射，剔除周末/节假日空档
    trade_days = pd.Index(sorted(idx.normalize().unique()))
    day_map = {d: i for i, d in enumerate(trade_days)}
    day_offsets = idx.normalize().map(day_map).to_numpy(dtype=int) * trading_day_minutes
    return day_offsets + (adj - (9 * 60 + 30))


def prepend_start(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    first_day = series.index.min().normalize()
    start_dt = first_day + pd.Timedelta(hours=9, minutes=30)
    if start_dt in series.index:
        series.loc[start_dt] = 1.0
        return series.sort_index()
    return pd.concat([pd.Series({start_dt: 1.0}), series]).sort_index()


def plot_nav(
    nav_series: pd.Series,
    mark_series: pd.Series,
    weighted_last: pd.Series,
    *,
    ax=None,
    benchmark_series: Optional[pd.Series] = None,
    benchmark_color: str = "red",
    title: str = "NAV 对比（执行前估值 vs 执行后净值）",
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    nav_series = prepend_start(nav_series.copy())
    mark_series = prepend_start(mark_series.copy())
    if benchmark_series is not None:
        benchmark_series = prepend_start(benchmark_series.copy())

    nav_x = compress_lunch(nav_series.index)
    mark_x = compress_lunch(mark_series.index)
    last_x = compress_lunch(weighted_last.index)

    ax.plot(nav_x, nav_series.values, label="执行后净值")
    ax.plot(mark_x, mark_series.values, label="执行前估值", linestyle="--")
    if benchmark_series is not None:
        bench_x = compress_lunch(benchmark_series.index)
        ax.plot(
            bench_x,
            benchmark_series.values,
            label="基准净值",
            linestyle=":",
            color=benchmark_color,
        )

    ax2 = ax.twinx()
    ax2.plot(
        last_x,
        weighted_last.values,
        label="组合加权 last",
        linestyle="-.",
        color="tab:green",
    )

    ax.set_title(title)
    tick_index = nav_series.index
    tick_x = compress_lunch(tick_index)
    tick_labels = [t.strftime("%H:%M") for t in tick_index]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_xlabel("time (lunch removed)")
    ax.set_ylabel("nav")
    ax2.set_ylabel("last")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)

    return fig, ax

