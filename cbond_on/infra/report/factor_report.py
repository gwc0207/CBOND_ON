from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbond_on.core.plotting import compress_lunch


def _summary_stats(result: Any) -> dict[str, float]:
    rets = pd.to_numeric(result.returns, errors="coerce").dropna()
    benchmark_rets = pd.to_numeric(
        getattr(result, "benchmark_returns", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    trade_rets = pd.to_numeric(
        getattr(result, "trade_returns", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not trade_rets.empty:
        win_rate = float((trade_rets > 0).mean())
    elif not rets.empty:
        win_rate = float((rets > 0).mean())
    else:
        win_rate = 0.0
    if rets.empty:
        return {
            "sharpe": 0.0,
            "benchmark_sharpe": 0.0,
            "alpha_sharpe": 0.0,
            "maxdd": 0.0,
            "win_rate": win_rate,
            "ic_mean": 0.0,
            "ic_ir": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_ir": 0.0,
            "ret_total": 0.0,
            "benchmark_ret_total": 0.0,
            "alpha_ret_total": 0.0,
        }

    nav = (1.0 + rets).cumprod()
    running_max = nav.cummax()
    maxdd = float((nav / running_max - 1.0).min()) if not nav.empty else 0.0
    benchmark_nav = (1.0 + benchmark_rets).cumprod() if not benchmark_rets.empty else pd.Series(dtype=float)

    ic = pd.to_numeric(result.ic, errors="coerce").dropna()
    rank_ic = pd.to_numeric(result.rank_ic, errors="coerce").dropna()
    ic_std = float(ic.std(ddof=0)) if not ic.empty else 0.0
    rank_ic_std = float(rank_ic.std(ddof=0)) if not rank_ic.empty else 0.0

    mean = float(rets.mean())
    std = float(rets.std(ddof=0))
    sharpe = float((mean / std) * np.sqrt(252.0)) if std > 0 else 0.0
    benchmark_mean = float(benchmark_rets.mean()) if not benchmark_rets.empty else 0.0
    benchmark_std = float(benchmark_rets.std(ddof=0)) if not benchmark_rets.empty else 0.0
    benchmark_sharpe = (
        float((benchmark_mean / benchmark_std) * np.sqrt(252.0))
        if benchmark_std > 0
        else 0.0
    )
    common_idx = rets.index.intersection(benchmark_rets.index)
    alpha_rets = rets.loc[common_idx] - benchmark_rets.loc[common_idx] if len(common_idx) else pd.Series(dtype=float)
    alpha_mean = float(alpha_rets.mean()) if not alpha_rets.empty else 0.0
    alpha_std = float(alpha_rets.std(ddof=0)) if not alpha_rets.empty else 0.0
    alpha_sharpe = float((alpha_mean / alpha_std) * np.sqrt(252.0)) if alpha_std > 0 else 0.0
    alpha_nav = (1.0 + alpha_rets).cumprod() if not alpha_rets.empty else pd.Series(dtype=float)
    return {
        "sharpe": sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "alpha_sharpe": alpha_sharpe,
        "maxdd": maxdd,
        "win_rate": win_rate,
        "ic_mean": float(ic.mean()) if not ic.empty else 0.0,
        "ic_ir": float(ic.mean() / ic_std) if ic_std > 0 else 0.0,
        "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else 0.0,
        "rank_ic_ir": float(rank_ic.mean() / rank_ic_std) if rank_ic_std > 0 else 0.0,
        "ret_total": float(nav.iloc[-1] - 1.0) if not nav.empty else 0.0,
        "benchmark_ret_total": float(benchmark_nav.iloc[-1] - 1.0) if not benchmark_nav.empty else 0.0,
        "alpha_ret_total": float(alpha_nav.iloc[-1] - 1.0) if not alpha_nav.empty else 0.0,
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


def _mean_bin_monotonicity(bin_time_df: pd.DataFrame) -> float:
    if bin_time_df.empty:
        return float("nan")
    day_df = bin_time_df.copy()
    day_df["trade_date"] = pd.to_datetime(day_df["trade_time"], errors="coerce").dt.normalize()
    day_df = day_df.dropna(subset=["trade_date"])
    if day_df.empty:
        return float("nan")
    rhos: list[float] = []
    for _, g in day_df.groupby("trade_date", sort=True):
        x = pd.to_numeric(g.get("bin"), errors="coerce")
        y = pd.to_numeric(g.get("return_mean"), errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        if x.nunique() < 2:
            continue
        rho = x.corr(y, method="spearman")
        if pd.notna(rho):
            rhos.append(float(rho))
    if not rhos:
        return float("nan")
    return float(pd.Series(rhos, dtype=float).mean())


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


def _rolling_mean(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window, min_periods=min_periods).mean()


def _rolling_tstat(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    roll = s.rolling(window=window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std(ddof=0)
    cnt = roll.count()
    t_val = (mean / std) * np.sqrt(cnt)
    return t_val.replace([np.inf, -np.inf], np.nan)


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
    benchmark_nav_series = pd.to_numeric(
        getattr(result, "benchmark_nav", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not nav_series.empty:
        nav_out = nav_series.rename("nav").reset_index()
        nav_out = _normalize_time_column(nav_out, target="trade_time")
        nav_out["trade_time"] = pd.to_datetime(nav_out["trade_time"], errors="coerce")
        nav_out = nav_out.dropna(subset=["trade_time"])
        if not benchmark_nav_series.empty:
            bench_out = benchmark_nav_series.rename("benchmark_nav").reset_index()
            bench_out = _normalize_time_column(bench_out, target="trade_time")
            bench_out["trade_time"] = pd.to_datetime(bench_out["trade_time"], errors="coerce")
            bench_out = bench_out.dropna(subset=["trade_time"])
            nav_out = nav_out.merge(bench_out, on="trade_time", how="left")
        if trading_days is not None:
            nav_out = nav_out[nav_out["trade_time"].dt.date.isin(trading_days)]
        nav_out = _filter_trading_times(nav_out, "trade_time")
        nav_out.to_csv(out_dir / "nav_series.csv", index=False)
    else:
        pd.DataFrame(columns=["trade_time", "nav", "benchmark_nav"]).to_csv(
            out_dir / "nav_series.csv",
            index=False,
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
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
    benchmark_line_plotted = False
    if not benchmark_nav_series.empty:
        bench_plot = pd.Series(
            benchmark_nav_series.values,
            index=pd.to_datetime(benchmark_nav_series.index, errors="coerce"),
        ).dropna()
        if not bench_plot.empty:
            bench_plot.index = bench_plot.index.normalize()
            bench_plot = bench_plot.groupby(bench_plot.index).last().sort_index()
            if not bench_plot.empty:
                ax.plot(
                    bench_plot.index,
                    bench_plot.values,
                    label="benchmark",
                    color="red",
                    linestyle="--",
                    linewidth=1.3,
                )
                benchmark_line_plotted = True
    if not bin_time_df.empty:
        day_df = bin_time_df.copy()
        day_df["trade_date"] = pd.to_datetime(day_df["trade_time"], errors="coerce").dt.normalize()
        day_df = day_df.dropna(subset=["trade_date"])
        pivot = day_df.pivot_table(
            index="trade_date",
            columns="bin",
            values="return_mean",
            aggfunc="mean",
        ).sort_index()
        if not pivot.empty:
            nav = (1.0 + pivot.fillna(0.0)).cumprod()
            colors = plt.cm.viridis(np.linspace(0, 1, len(nav.columns)))
            for color, col in zip(colors, nav.columns):
                x = pd.to_datetime(nav.index, errors="coerce")
                y = pd.to_numeric(nav[col], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.any():
                    ax.plot(x[valid], y[valid], label=f"bin {col}", color=color)
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(axis="x", labelrotation=30)
            ax.legend(fontsize=7, ncol=2)
        else:
            if not benchmark_line_plotted:
                ax.text(0.5, 0.5, "No bin NAV", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.legend(fontsize=7, ncol=2)
    else:
        if not benchmark_line_plotted:
            ax.text(0.5, 0.5, "No bin data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.legend(fontsize=7, ncol=2)
    ax.set_title("Bin Cumulative NAV")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    factor_values = pd.to_numeric(
        getattr(result, "factor_values", pd.Series(dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if not factor_values.empty:
        q01 = float(factor_values.quantile(0.01))
        q99 = float(factor_values.quantile(0.99))
        clipped = factor_values.clip(lower=q01, upper=q99) if q99 > q01 else factor_values
        bins = int(min(80, max(20, np.sqrt(max(1, len(clipped))))))
        ax.hist(clipped, bins=bins, color="#1F77B4", alpha=0.85, edgecolor="white", linewidth=0.3)
        mean_val = float(clipped.mean())
        med_val = float(clipped.median())
        ax.axvline(mean_val, color="#E45756", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.4g}")
        ax.axvline(med_val, color="#54A24B", linestyle="-.", linewidth=1.2, label=f"median={med_val:.4g}")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No factor values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Factor Value Distribution")
    ax.set_xlabel(factor_col)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    mono = _mean_bin_monotonicity(bin_time_df)

    perf_metrics = [
        ("strategy_ret", float(summary.get("ret_total", float("nan")))),
        ("strategy_sharpe", float(summary.get("sharpe", float("nan")))),
        ("maxdd", float(summary.get("maxdd", float("nan")))),
        ("alpha_ret", float(summary.get("alpha_ret_total", float("nan")))),
        ("mono_rankcorr", mono),
    ]
    perf_metric_names = [name for name, _ in perf_metrics]
    perf_metric_vals = [val for _, val in perf_metrics]
    ax.bar(
        perf_metric_names,
        perf_metric_vals,
        color=["#4C78A8", "#F58518", "#E45756", "#54A24B", "#9C755F"],
    )
    for idx, val in enumerate(perf_metric_vals):
        if pd.notna(val):
            ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Factor + Strategy Metrics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)

    ax = axes[1, 2]
    factor_daily = pd.to_numeric(getattr(result, "returns", pd.Series(dtype=float)), errors="coerce").dropna()
    benchmark_daily = pd.to_numeric(
        getattr(result, "benchmark_returns", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    rank_ic_daily = pd.to_numeric(getattr(result, "rank_ic", pd.Series(dtype=float)), errors="coerce").dropna()
    if not factor_daily.empty:
        factor_daily = factor_daily.sort_index()
        rank_ic_daily = rank_ic_daily.sort_index()
        common_idx = factor_daily.index.intersection(benchmark_daily.index)
        alpha_daily = (
            factor_daily.loc[common_idx] - benchmark_daily.loc[common_idx]
            if len(common_idx)
            else pd.Series(dtype=float)
        ).sort_index()

        obs = int(max(len(factor_daily), len(rank_ic_daily), len(alpha_daily)))
        # Adaptive rolling window by observation count (trading-day based).
        roll_win = max(20, min(120, obs // 4 if obs > 0 else 20))
        min_roll = max(10, roll_win // 2)

        rank_ic_roll = _rolling_mean(rank_ic_daily, window=roll_win, min_periods=min_roll)
        alpha_t_roll = _rolling_tstat(alpha_daily, window=roll_win, min_periods=min_roll)
        alpha_hit_roll = _rolling_mean((alpha_daily > 0).astype(float), window=roll_win, min_periods=min_roll)

        daily_export = pd.DataFrame(index=factor_daily.index.union(alpha_daily.index).union(rank_ic_daily.index).sort_values())
        daily_export.index.name = "trade_time"
        daily_export["factor_daily_return"] = factor_daily.reindex(daily_export.index)
        daily_export["alpha_daily_return"] = alpha_daily.reindex(daily_export.index)
        daily_export["rank_ic_daily"] = rank_ic_daily.reindex(daily_export.index)
        daily_export["rank_ic_roll_mean"] = rank_ic_roll.reindex(daily_export.index)
        daily_export["alpha_roll_tstat"] = alpha_t_roll.reindex(daily_export.index)
        daily_export["alpha_roll_hitrate"] = alpha_hit_roll.reindex(daily_export.index)
        daily_export.reset_index().to_csv(out_dir / "factor_effectiveness_stability.csv", index=False)

        plotted = False
        if not rank_ic_roll.dropna().empty:
            x = pd.to_datetime(rank_ic_roll.index, errors="coerce")
            y = pd.to_numeric(rank_ic_roll, errors="coerce")
            valid = x.notna() & y.notna()
            if valid.any():
                ax.plot(x[valid], y[valid], label=f"RankIC Mean({roll_win}d)", color="#4C78A8")
                plotted = True
        ax.axhline(0.0, color="#4C78A8", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.set_ylabel("RankIC")

        ax2 = ax.twinx()
        if not alpha_t_roll.dropna().empty:
            x = pd.to_datetime(alpha_t_roll.index, errors="coerce")
            y = pd.to_numeric(alpha_t_roll, errors="coerce")
            valid = x.notna() & y.notna()
            if valid.any():
                ax2.plot(x[valid], y[valid], label=f"Alpha t-stat({roll_win}d)", color="#54A24B")
                plotted = True
        ax2.axhline(2.0, color="#54A24B", linewidth=1.0, linestyle="--", alpha=0.7)
        ax2.axhline(-2.0, color="#E45756", linewidth=1.0, linestyle="--", alpha=0.7)
        ax2.set_ylabel("Alpha t-stat")
        if plotted:
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(axis="x", labelrotation=30)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, "No effectiveness stability data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No return data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Effectiveness Stability")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "factor_report.png", dpi=150)
    plt.close(fig)
    return summary
