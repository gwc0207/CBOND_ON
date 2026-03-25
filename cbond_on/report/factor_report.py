from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbond_on.core.plotting import compress_lunch


def _summary_stats(result: Any) -> dict[str, float]:
    rets = pd.to_numeric(result.returns, errors="coerce").dropna()
    if rets.empty:
        return {
            "sharpe": 0.0,
            "maxdd": 0.0,
            "win_rate": 0.0,
            "ic_mean": 0.0,
            "ic_ir": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_ir": 0.0,
            "ret_total": 0.0,
        }

    nav = (1.0 + rets).cumprod()
    running_max = nav.cummax()
    maxdd = float((nav / running_max - 1.0).min()) if not nav.empty else 0.0

    ic = pd.to_numeric(result.ic, errors="coerce").dropna()
    rank_ic = pd.to_numeric(result.rank_ic, errors="coerce").dropna()
    ic_std = float(ic.std(ddof=0)) if not ic.empty else 0.0
    rank_ic_std = float(rank_ic.std(ddof=0)) if not rank_ic.empty else 0.0

    mean = float(rets.mean())
    std = float(rets.std(ddof=0))
    sharpe = float((mean / std) * np.sqrt(252.0)) if std > 0 else 0.0
    return {
        "sharpe": sharpe,
        "maxdd": maxdd,
        "win_rate": float((rets > 0).mean()),
        "ic_mean": float(ic.mean()) if not ic.empty else 0.0,
        "ic_ir": float(ic.mean() / ic_std) if ic_std > 0 else 0.0,
        "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else 0.0,
        "rank_ic_ir": float(rank_ic.mean() / rank_ic_std) if rank_ic_std > 0 else 0.0,
        "ret_total": float(nav.iloc[-1] - 1.0) if not nav.empty else 0.0,
    }


def _mask_trading_times(index: pd.Series) -> pd.Series:
    ts = pd.to_datetime(index, errors="coerce")
    t = ts.dt.time
    morning = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("11:30").time())
    afternoon = (t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("14:57").time())
    return morning | afternoon


def _filter_trading_times(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    return df[_mask_trading_times(df[col])]


def _normalize_time_column(
    df: pd.DataFrame,
    *,
    target: str = "trade_time",
    candidates: tuple[str, ...] = ("trade_time", "dt", "trade_date", "index", "level_0"),
) -> pd.DataFrame:
    if df.empty:
        return df
    if target in df.columns:
        return df
    for name in candidates:
        if name in df.columns:
            return df.rename(columns={name: target})
    # fallback: first non-value column
    for col in df.columns:
        if col != "nav":
            return df.rename(columns={col: target})
    return df


def _normalize_bin_time(bin_returns: Any) -> pd.DataFrame:
    if bin_returns is None:
        return pd.DataFrame(columns=["trade_time", "bin", "return_mean"])
    if isinstance(bin_returns, pd.DataFrame):
        df = bin_returns.copy()
        if {"trade_time", "bin", "return_mean"}.issubset(df.columns):
            out = df[["trade_time", "bin", "return_mean"]].copy()
            out["trade_time"] = pd.to_datetime(out["trade_time"], errors="coerce")
            return out.dropna(subset=["trade_time"])
        if df.empty:
            return pd.DataFrame(columns=["trade_time", "bin", "return_mean"])
        stacked = df.stack().reset_index()
        stacked.columns = ["trade_time", "bin", "return_mean"]
        stacked["trade_time"] = pd.to_datetime(stacked["trade_time"], errors="coerce")
        return stacked.dropna(subset=["trade_time"])
    if isinstance(bin_returns, pd.Series) and bin_returns.index.nlevels == 2:
        out = bin_returns.reset_index()
        out.columns = ["trade_time", "bin", "return_mean"]
        out["trade_time"] = pd.to_datetime(out["trade_time"], errors="coerce")
        return out.dropna(subset=["trade_time"])
    return pd.DataFrame(columns=["trade_time", "bin", "return_mean"])


def _calc_bin_stats(bin_time_df: pd.DataFrame) -> pd.DataFrame:
    if bin_time_df.empty:
        return pd.DataFrame(columns=["bin", "nav_end", "total_return"])
    pivot = bin_time_df.pivot_table(
        index="trade_time",
        columns="bin",
        values="return_mean",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return pd.DataFrame(columns=["bin", "nav_end", "total_return"])
    nav = (1.0 + pivot.fillna(0.0)).cumprod()
    nav_end = nav.tail(1).T.reset_index()
    nav_end.columns = ["bin", "nav_end"]
    nav_end["total_return"] = pd.to_numeric(nav_end["nav_end"], errors="coerce") - 1.0
    return nav_end


def _plot_compressed(ax: Any, x_index: Any, y: Any, *, label: str | None = None, color: Any = None) -> None:
    ts = pd.DatetimeIndex(pd.to_datetime(x_index, errors="coerce"))
    valid = ~ts.isna()
    if not valid.any():
        return
    ts = ts[valid]
    y_series = pd.Series(y).reset_index(drop=True)
    y_series = y_series[valid]
    x = compress_lunch(ts)
    if color is None:
        ax.plot(x, y_series.to_numpy(), label=label)
    else:
        ax.plot(x, y_series.to_numpy(), label=label, color=color)


def save_single_factor_report(
    result: Any,
    out_dir: Path,
    *,
    factor_name: str,
    factor_col: str,
    trading_days: set | None = None,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = getattr(result, "diagnostics", pd.DataFrame())
    if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
        diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)

    bin_ok = int(getattr(result, "bin_ok", 0) or 0)
    bin_fail = int(getattr(result, "bin_fail", 0) or 0)
    pd.DataFrame([{"bin_ok": bin_ok, "bin_fail": bin_fail}]).to_csv(
        out_dir / "bin_summary.csv",
        index=False,
    )

    summary = _summary_stats(result)
    pd.DataFrame([summary]).to_csv(out_dir / "factor_metrics.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ic_df = result.daily_stats.copy() if isinstance(result.daily_stats, pd.DataFrame) else pd.DataFrame()
    if not ic_df.empty:
        ic_df["trade_time"] = pd.to_datetime(ic_df["trade_time"], errors="coerce")
        ic_df = ic_df.dropna(subset=["trade_time"]).sort_values("trade_time")
        if trading_days is not None:
            ic_df = ic_df[ic_df["trade_time"].dt.date.isin(trading_days)]
        ic_df = _filter_trading_times(ic_df, "trade_time")
        ic_df["ic"] = pd.to_numeric(ic_df.get("ic"), errors="coerce")
        ic_df["rank_ic"] = pd.to_numeric(ic_df.get("rank_ic"), errors="coerce")
        ic_df["ic_cum"] = ic_df["ic"].fillna(0.0).cumsum()
        ic_df["rank_ic_cum"] = ic_df["rank_ic"].fillna(0.0).cumsum()
    ic_df.to_csv(out_dir / "ic_series.csv", index=False)

    bin_time_df = _normalize_bin_time(getattr(result, "bin_returns", None))
    if not bin_time_df.empty and trading_days is not None:
        bin_time_df = bin_time_df[bin_time_df["trade_time"].dt.date.isin(trading_days)]
    if not bin_time_df.empty:
        bin_time_df = _filter_trading_times(bin_time_df, "trade_time")
    bin_time_df.to_csv(out_dir / "bin_time_returns.csv", index=False)

    bin_stats = _calc_bin_stats(bin_time_df)
    bin_stats.to_csv(out_dir / "factor_bins.csv", index=False)

    nav_series = pd.to_numeric(getattr(result, "nav", pd.Series(dtype=float)), errors="coerce").dropna()
    if not nav_series.empty:
        nav_out = nav_series.rename("nav").reset_index()
        nav_out = _normalize_time_column(nav_out, target="trade_time")
        nav_out["trade_time"] = pd.to_datetime(nav_out["trade_time"], errors="coerce")
        nav_out = nav_out.dropna(subset=["trade_time"])
        if trading_days is not None:
            nav_out = nav_out[nav_out["trade_time"].dt.date.isin(trading_days)]
        nav_out = _filter_trading_times(nav_out, "trade_time")
        nav_out.to_csv(out_dir / "nav_series.csv", index=False)
    else:
        pd.DataFrame(columns=["trade_time", "nav"]).to_csv(out_dir / "nav_series.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip factor report image ({factor_name}): {exc}")
        return summary

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    fig.suptitle(f"{factor_name} ({factor_col})", fontsize=12)

    ax = axes[0, 0]
    if not ic_df.empty:
        _plot_compressed(ax, ic_df["trade_time"], ic_df["ic"], label="IC")
        _plot_compressed(ax, ic_df["trade_time"], ic_df["rank_ic"], label="RankIC")
        _plot_compressed(ax, ic_df["trade_time"], ic_df["ic_cum"], label="IC Cumsum")
        _plot_compressed(ax, ic_df["trade_time"], ic_df["rank_ic_cum"], label="RankIC Cumsum")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No IC data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("IC Series")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ic_metric_names = ["ic_mean", "ic_ir", "rank_ic_mean", "rank_ic_ir"]
    ic_metric_vals = [summary[k] for k in ic_metric_names]
    ax.bar(ic_metric_names, ic_metric_vals, color=["#3C8DBC", "#1F77B4", "#9467BD", "#FF7F0E"])
    for idx, val in enumerate(ic_metric_vals):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("IC Metrics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)

    ax = axes[0, 2]
    if not bin_time_df.empty:
        pivot = bin_time_df.pivot_table(
            index="trade_time",
            columns="bin",
            values="return_mean",
            aggfunc="mean",
        ).sort_index()
        if not pivot.empty:
            nav = (1.0 + pivot.fillna(0.0)).cumprod()
            colors = plt.cm.viridis(np.linspace(0, 1, len(nav.columns)))
            for color, col in zip(colors, nav.columns):
                _plot_compressed(ax, nav.index, nav[col], label=f"bin {col}", color=color)
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.text(0.5, 0.5, "No bin NAV", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No bin data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Bin Cumulative NAV")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if not bin_stats.empty:
        x = pd.to_numeric(bin_stats["bin"], errors="coerce")
        y = pd.to_numeric(bin_stats["total_return"], errors="coerce")
        ax.bar(x, y, color="#1F77B4")
    else:
        ax.text(0.5, 0.5, "No bin stats", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Bin Total Return")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    perf_metric_names = ["ret_total", "sharpe", "maxdd", "win_rate"]
    perf_metric_vals = [summary[k] for k in perf_metric_names]
    ax.bar(perf_metric_names, perf_metric_vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    for idx, val in enumerate(perf_metric_vals):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Performance")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)

    ax = axes[1, 2]
    if not nav_series.empty:
        nav_ts = pd.Series(nav_series.values, index=pd.to_datetime(nav_series.index, errors="coerce")).dropna()
        if not nav_ts.empty:
            _plot_compressed(ax, nav_ts.index, nav_ts.values, label="Factor NAV")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No NAV data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No NAV data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Factor NAV")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "factor_report.png", dpi=150)
    plt.close(fig)
    return summary
