from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import time
from cbond_on.core.plotting import compress_lunch

if TYPE_CHECKING:
    from cbond_on.factor_batch.runner import FactorBacktestResult


def _summary_stats(result: Any) -> Dict[str, float]:
    rets = result.returns.dropna()
    if rets.empty:
        return {
            "sharpe": 0.0,
            "maxdd": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "ic_mean": 0.0,
            "ic_ir": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_ir": 0.0,
        }
    mean = float(rets.mean())
    std = float(rets.std(ddof=0))
    sharpe = float((mean / std) * (252.0 ** 0.5)) if std else 0.0
    nav = result.nav.dropna()
    if nav.empty:
        maxdd = 0.0
    else:
        running_max = nav.cummax()
        maxdd = float((nav / running_max - 1.0).min())
    win_rate = float((rets > 0).mean())
    ic = result.ic.dropna()
    rank_ic = result.rank_ic.dropna()
    ic_mean = float(ic.mean()) if not ic.empty else 0.0
    ic_ir = float(ic.mean() / ic.std()) if ic.std() else 0.0
    rank_ic_mean = float(rank_ic.mean()) if not rank_ic.empty else 0.0
    rank_ic_ir = float(rank_ic.mean() / rank_ic.std()) if rank_ic.std() else 0.0
    return {
        "sharpe": sharpe,
        "maxdd": maxdd,
        "win_rate": win_rate,
        "turnover": 0.0,
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_ir": rank_ic_ir,
    }


def save_single_factor_report(
    result: Any,
    out_dir: Path,
    *,
    factor_name: str,
    factor_col: str,
    trading_days: set | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = getattr(result, "diagnostics", pd.DataFrame())
    if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
        diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    bin_ok = getattr(result, "bin_ok", None)
    bin_fail = getattr(result, "bin_fail", None)
    if bin_ok is not None or bin_fail is not None:
        pd.DataFrame(
            [{"bin_ok": bin_ok or 0, "bin_fail": bin_fail or 0}]
        ).to_csv(out_dir / "bin_summary.csv", index=False)
    summary = _summary_stats(result)
    pd.DataFrame([summary]).to_csv(out_dir / "factor_metrics.csv", index=False)

    ic_df = result.daily_stats.copy()
    if not ic_df.empty:
        ic_df = ic_df.sort_values("trade_time")
        if trading_days is not None:
            ic_df = ic_df[ic_df["trade_time"].dt.date.isin(trading_days)]
        ic_df = _filter_trading_times(ic_df, "trade_time")
        ic_df["ic"] = pd.to_numeric(ic_df["ic"], errors="coerce")
        ic_df["rank_ic"] = pd.to_numeric(ic_df["rank_ic"], errors="coerce")
        ic_df["ic_cum"] = ic_df["ic"].fillna(0.0).cumsum()
        ic_df["rank_ic_cum"] = ic_df["rank_ic"].fillna(0.0).cumsum()
    ic_df.to_csv(out_dir / "ic_series.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    fig.suptitle(f"{factor_name} ({factor_col})", fontsize=12)

    def _plot_compressed(ax, x_index, y, label=None, color=None):
        x = compress_lunch(pd.DatetimeIndex(pd.to_datetime(x_index)))
        if color is None:
            ax.plot(x, y, label=label)
        else:
            ax.plot(x, y, label=label, color=color)
        _set_time_ticks(ax, x_index, x)

    def _set_time_ticks(ax, x_index, x_vals, max_ticks: int = 8):
        if len(x_vals) == 0:
            return
        step = max(1, len(x_vals) // max_ticks)
        idx = list(range(0, len(x_vals), step))
        ax.set_xticks([x_vals[i] for i in idx])
        labels = [pd.to_datetime(x_index[i]).strftime("%Y-%m-%d %H:%M") for i in idx]
        ax.set_xticklabels(labels, rotation=30, fontsize=8)

    ax = axes[0, 0]
    if not ic_df.empty:
        _plot_compressed(ax, ic_df["trade_time"], ic_df["ic"], label="Mean Daily IC")
        _plot_compressed(ax, ic_df["trade_time"], ic_df["ic_cum"], label="Accumulative IC")
        _plot_compressed(ax, ic_df["trade_time"], ic_df["rank_ic"], label="Mean Daily Rank IC")
        _plot_compressed(ax, ic_df["trade_time"], ic_df["rank_ic_cum"], label="Accumulative Rank IC")
    ax.set_title("Mean IC & Accumulative IC by Date")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    metric_labels = ["IC", "IR", "Rank_IC", "Rank_IR"]
    values = [
        summary["ic_mean"],
        summary["ic_ir"],
        summary["rank_ic_mean"],
        summary["rank_ic_ir"],
    ]
    ax.bar(metric_labels, values, color=["#3C8DBC", "#1F77B4", "#9467BD", "#FF7F0E"])
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("IC & IR (mean)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 2]
    bin_time_df = _normalize_bin_time(result.bin_returns)
    if not bin_time_df.empty and trading_days is not None:
        bin_time_df = bin_time_df[bin_time_df["trade_time"].dt.date.isin(trading_days)]
    if not bin_time_df.empty:
        bin_time_df = _filter_trading_times(bin_time_df, "trade_time")
    bin_time_df.to_csv(out_dir / "bin_time_returns.csv", index=False)
    if bin_time_df.empty:
        raise ValueError(f"bin_time_df is empty for {factor_name} ({factor_col})")
    if not bin_time_df.empty:
        pivot = bin_time_df.pivot_table(
            index="trade_time", columns="bin", values="return_mean", aggfunc="mean"
        ).sort_index()
        nav = (1.0 + pivot.fillna(0.0)).cumprod()
        colors = plt.cm.viridis(np.linspace(0, 1, len(nav.columns)))
        for color, col in zip(colors, nav.columns):
            _plot_compressed(ax, nav.index, nav[col], label=f"bin {col}", color=color)
        ax.legend(fontsize=7, ncol=2)
    ax.set_title("Bin cumulative NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    bin_stats = _calc_bin_stats(bin_time_df)
    if bin_stats.empty:
        raise ValueError(f"bin_stats is empty for {factor_name} ({factor_col})")
    bin_stats.to_csv(out_dir / "factor_bins.csv", index=False)
    if not bin_stats.empty:
        ax.bar(bin_stats["bin"].astype(int), bin_stats["total_return"], color="#1F77B4")
    ax.set_title("Bin total return")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Total Return")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    metric_names = ["sharpe", "maxdd", "win_rate", "turnover"]
    metric_vals = [summary[m] for m in metric_names]
    ax.bar(metric_names, metric_vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    for idx, val in enumerate(metric_vals):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Performance metrics")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 2]
    if not result.returns.empty:
        factor_nav = (1.0 + result.returns.fillna(0.0)).cumprod()
        if trading_days is not None:
            factor_nav = factor_nav[factor_nav.index.to_series().dt.date.isin(trading_days)]
        factor_nav = factor_nav[factor_nav.index.to_series().pipe(lambda s: _mask_trading_times(s))]
        _plot_compressed(ax, factor_nav.index, factor_nav.values, label="Factor")
        ax.legend(fontsize=8)
    ax.set_title("Factor NAV")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "factor_report.png", dpi=150)
    plt.close(fig)


def _normalize_bin_time(bin_returns: Any) -> pd.DataFrame:
    if bin_returns is None:
        return pd.DataFrame()
    if isinstance(bin_returns, pd.DataFrame):
        df = bin_returns.copy()
        if "trade_time" in df.columns and "bin" in df.columns and "return_mean" in df.columns:
            return df
        if df.index.nlevels == 2:
            df = df.stack().reset_index()
            df.columns = ["trade_time", "bin", "return_mean"]
            return df
        # dt index + bin columns
        if df.index.nlevels == 1 and df.shape[1] > 0:
            stacked = df.stack().reset_index()
            stacked.columns = ["trade_time", "bin", "return_mean"]
            return stacked
        return pd.DataFrame()
    if isinstance(bin_returns, pd.Series) and bin_returns.index.nlevels == 2:
        df = bin_returns.reset_index()
        df.columns = ["trade_time", "bin", "return_mean"]
        return df
    return pd.DataFrame()


def _calc_bin_stats(bin_time_df: pd.DataFrame) -> pd.DataFrame:
    if bin_time_df.empty:
        return pd.DataFrame(columns=["bin", "nav_end", "total_return"])
    pivot = bin_time_df.pivot_table(
        index="trade_time", columns="bin", values="return_mean", aggfunc="mean"
    ).sort_index()
    nav = (1.0 + pivot.fillna(0.0)).cumprod()
    nav_end = nav.tail(1).T.reset_index()
    nav_end.columns = ["bin", "nav_end"]
    nav_end["total_return"] = nav_end["nav_end"] - 1.0
    return nav_end


def _mask_trading_times(index: pd.Series) -> pd.Series:
    t = index.dt.time
    morning = (t >= time(9, 30)) & (t < time(11, 30))
    afternoon = (t >= time(13, 0)) & (t <= time(14, 45))
    return morning | afternoon


def _filter_trading_times(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    series = pd.to_datetime(df[col], errors="coerce")
    mask = _mask_trading_times(series)
    return df[mask]
