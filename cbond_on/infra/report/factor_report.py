from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbond_on.core.plotting import compress_lunch


def _merge_best_bin_metrics(result: Any, summary: dict[str, float]) -> dict[str, float]:
    best_metrics = getattr(result, "best_bin_metrics", {}) or {}
    if isinstance(best_metrics, dict):
        for key, value in best_metrics.items():
            try:
                summary[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return summary


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
        return _merge_best_bin_metrics(result, {
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
        })
    rets_for_alpha = rets.copy()
    rets_idx = pd.to_datetime(rets_for_alpha.index, errors="coerce")
    if rets_idx.notna().any():
        rets_for_alpha = pd.Series(rets_for_alpha.values, index=rets_idx).dropna()
        rets_for_alpha.index = rets_for_alpha.index.normalize()
        rets_for_alpha = rets_for_alpha.groupby(rets_for_alpha.index).mean().sort_index()
    benchmark_for_alpha = benchmark_rets.copy()
    bench_idx = pd.to_datetime(benchmark_for_alpha.index, errors="coerce")
    if bench_idx.notna().any():
        benchmark_for_alpha = pd.Series(benchmark_for_alpha.values, index=bench_idx).dropna()
        benchmark_for_alpha.index = benchmark_for_alpha.index.normalize()
        benchmark_for_alpha = benchmark_for_alpha.groupby(benchmark_for_alpha.index).mean().sort_index()

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
    common_idx = rets_for_alpha.index.intersection(benchmark_for_alpha.index)
    alpha_rets = (
        rets_for_alpha.loc[common_idx] - benchmark_for_alpha.loc[common_idx]
        if len(common_idx)
        else pd.Series(dtype=float)
    )
    alpha_mean = float(alpha_rets.mean()) if not alpha_rets.empty else 0.0
    alpha_std = float(alpha_rets.std(ddof=0)) if not alpha_rets.empty else 0.0
    alpha_sharpe = float((alpha_mean / alpha_std) * np.sqrt(252.0)) if alpha_std > 0 else 0.0
    alpha_nav = (1.0 + alpha_rets).cumprod() if not alpha_rets.empty else pd.Series(dtype=float)
    return _merge_best_bin_metrics(result, {
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
    })


def _mask_trading_times(index: pd.Series) -> pd.Series:
    ts = pd.to_datetime(index, errors="coerce")
    t = ts.dt.time
    morning = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("11:30").time())
    afternoon = (t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("14:57").time())
    return morning | afternoon


def _has_intraday_clock(index: pd.Series) -> bool:
    ts = pd.to_datetime(index, errors="coerce")
    valid = ts.dropna()
    if valid.empty:
        return False
    midnight = pd.Timestamp("00:00").time()
    return bool((valid.dt.time != midnight).any())


def _filter_trading_times(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    if not _has_intraday_clock(df[col]):
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


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _newey_west_tstat(series: pd.Series, *, lag: int | None = None) -> tuple[float, float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    n = int(s.shape[0])
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mean = float(s.mean())
    e = s.to_numpy() - mean
    if lag is None:
        lag = int(max(1, min(n - 1, round(4.0 * (n / 100.0) ** (2.0 / 9.0)))))
    else:
        lag = int(max(1, min(n - 1, lag)))
    gamma0 = float(np.dot(e, e) / n)
    long_run_var = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(e[k:], e[:-k]) / n)
        weight = 1.0 - (k / (lag + 1.0))
        long_run_var += 2.0 * weight * gamma_k
    var_mean = long_run_var / n
    if not np.isfinite(var_mean) or var_mean <= 0:
        return mean, float("nan"), float("nan")
    t_val = mean / math.sqrt(var_mean)
    p_val = 2.0 * (1.0 - _normal_cdf(abs(float(t_val))))
    return mean, float(t_val), float(p_val)


def _series_sign_flip_count(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    signs = np.sign(s.to_numpy())
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def save_single_factor_report(
    result: Any,
    out_dir: Path,
    *,
    factor_name: str,
    factor_col: str,
    trading_days: set | None = None,
    alpha_significance_window: int | None = None,
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
    any_nav_line_plotted = False
    strategy_plot: pd.Series | None = None
    benchmark_plot: pd.Series | None = None
    if not nav_series.empty:
        strat_plot = pd.Series(
            nav_series.values,
            index=pd.to_datetime(nav_series.index, errors="coerce"),
        ).dropna()
        if not strat_plot.empty:
            strat_plot.index = strat_plot.index.normalize()
            strat_plot = strat_plot.groupby(strat_plot.index).last().sort_index()
            if not strat_plot.empty:
                strategy_plot = strat_plot
    if not benchmark_nav_series.empty:
        bench_plot = pd.Series(
            benchmark_nav_series.values,
            index=pd.to_datetime(benchmark_nav_series.index, errors="coerce"),
        ).dropna()
        if not bench_plot.empty:
            bench_plot.index = bench_plot.index.normalize()
            bench_plot = bench_plot.groupby(bench_plot.index).last().sort_index()
            if not bench_plot.empty:
                benchmark_plot = bench_plot
    if strategy_plot is not None:
        ax.plot(
            strategy_plot.index,
            strategy_plot.values,
            label="wf_strategy(net)",
            color="black",
            linewidth=2.4,
            zorder=12,
        )
        any_nav_line_plotted = True
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
                    ax.plot(x[valid], y[valid], label=f"bin {col}", color=color, zorder=2)
            if benchmark_plot is not None:
                ax.plot(
                    benchmark_plot.index,
                    benchmark_plot.values,
                    label="benchmark",
                    color="red",
                    linestyle="--",
                    linewidth=2.0,
                    zorder=10,
                )
                any_nav_line_plotted = True
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(axis="x", labelrotation=30)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                order = sorted(
                    range(len(labels)),
                    key=lambda i: (
                        0 if labels[i] == "wf_strategy(net)" else 1 if labels[i] == "benchmark" else 2,
                        labels[i],
                    ),
                )
                ax.legend(
                    [handles[i] for i in order],
                    [labels[i] for i in order],
                    fontsize=7,
                    ncol=2,
                )
        else:
            if benchmark_plot is not None:
                ax.plot(
                    benchmark_plot.index,
                    benchmark_plot.values,
                    label="benchmark",
                    color="red",
                    linestyle="--",
                    linewidth=2.0,
                    zorder=10,
                )
                any_nav_line_plotted = True
            if not any_nav_line_plotted:
                ax.text(0.5, 0.5, "No bin NAV", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.legend(fontsize=7, ncol=2)
    else:
        if benchmark_plot is not None:
            ax.plot(
                benchmark_plot.index,
                benchmark_plot.values,
                label="benchmark",
                color="red",
                linestyle="--",
                linewidth=2.0,
                zorder=10,
            )
            any_nav_line_plotted = True
        if not any_nav_line_plotted:
            ax.text(0.5, 0.5, "No bin data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.legend(fontsize=7, ncol=2)
    ax.set_title("Bin NAV + WF Strategy (Strict Official Split Net)")
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
    perf_metrics = [
        ("wf_ret", float(summary.get("ret_total", float("nan")))),
        ("wf_sharpe", float(summary.get("sharpe", float("nan")))),
        ("wf_maxdd", float(summary.get("maxdd", float("nan")))),
        ("wf_alpha", float(summary.get("alpha_ret_total", float("nan")))),
        ("best_ret", float(summary.get("best_bin_ret_total", float("nan")))),
        ("best_sharpe", float(summary.get("best_bin_sharpe", float("nan")))),
        ("best_maxdd", float(summary.get("best_bin_maxdd", float("nan")))),
        ("best_alpha", float(summary.get("best_bin_alpha_ret_total", float("nan")))),
    ]
    perf_metric_names = [name for name, _ in perf_metrics]
    perf_metric_vals = [val for _, val in perf_metrics]
    ax.bar(
        perf_metric_names,
        perf_metric_vals,
        color=["#222222", "#4C78A8", "#E45756", "#54A24B", "#72B7B2", "#F58518", "#B279A2", "#FF9DA6"],
    )
    for idx, val in enumerate(perf_metric_vals):
        if pd.notna(val):
            ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    best_bin = summary.get("best_bin", float("nan"))
    best_label = f"{int(best_bin)}" if pd.notna(best_bin) and best_bin >= 0 else "n/a"
    ax.set_title(f"WF/Best Metrics - Strict Split (best={best_label})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=30)

    ax = axes[1, 2]
    benchmark_daily = pd.to_numeric(
        getattr(result, "benchmark_returns", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not bin_time_df.empty and not benchmark_daily.empty:
        day_df = bin_time_df.copy()
        day_df["trade_date"] = pd.to_datetime(day_df["trade_time"], errors="coerce").dt.normalize()
        day_df = day_df.dropna(subset=["trade_date"])
        pivot = day_df.pivot_table(
            index="trade_date",
            columns="bin",
            values="return_mean",
            aggfunc="mean",
        ).sort_index()
        bench_day = pd.Series(
            benchmark_daily.values,
            index=pd.to_datetime(benchmark_daily.index, errors="coerce"),
        ).dropna()
        bench_day.index = bench_day.index.normalize()
        bench_day = bench_day.groupby(bench_day.index).mean().sort_index()

        common_days = pivot.index.intersection(bench_day.index)
        pivot = pivot.loc[common_days]
        bench_day = bench_day.loc[common_days]
        alpha_by_bin = pivot.sub(bench_day, axis=0)
        alpha_by_bin = alpha_by_bin.sort_index(axis=1)

        obs = int(alpha_by_bin.shape[0])
        configured_win = int(alpha_significance_window) if alpha_significance_window is not None else 40
        configured_win = max(2, configured_win)
        if obs > 0:
            roll_win = int(min(configured_win, obs))
        else:
            roll_win = configured_win
        min_roll = max(2, roll_win // 2)

        alpha_roll_t = pd.DataFrame(index=alpha_by_bin.index)
        summary_rows: list[dict[str, float | int]] = []
        for col in alpha_by_bin.columns:
            s = pd.to_numeric(alpha_by_bin[col], errors="coerce")
            t_roll = _rolling_tstat(s, window=roll_win, min_periods=min_roll)
            alpha_roll_t[col] = t_roll
            mean_val, nw_t, nw_p = _newey_west_tstat(s)
            pos_ratio = float((pd.to_numeric(t_roll, errors="coerce") > 2.0).mean()) if t_roll.notna().any() else 0.0
            neg_ratio = float((pd.to_numeric(t_roll, errors="coerce") < -2.0).mean()) if t_roll.notna().any() else 0.0
            flip_count = _series_sign_flip_count(t_roll)
            summary_rows.append(
                {
                    "bin": int(col),
                    "sample_days": int(s.notna().sum()),
                    "alpha_mean": float(mean_val),
                    "alpha_nw_tstat": float(nw_t) if pd.notna(nw_t) else float("nan"),
                    "alpha_nw_pvalue": float(nw_p) if pd.notna(nw_p) else float("nan"),
                    "roll_t_pos_ratio": pos_ratio,
                    "roll_t_neg_ratio": neg_ratio,
                    "roll_t_sign_flip": int(flip_count),
                }
            )
        pd.DataFrame(summary_rows).sort_values("bin").to_csv(
            out_dir / "factor_bin_alpha_significance.csv",
            index=False,
        )

        heat = alpha_roll_t.T
        if not heat.empty and heat.notna().any().any():
            x = pd.to_datetime(heat.columns, errors="coerce")
            valid_cols = ~x.isna()
            heat = heat.loc[:, valid_cols]
            x = x[valid_cols]
            if heat.shape[1] > 0:
                x_num = mdates.date2num(x.to_pydatetime())
                x0 = float(x_num.min())
                x1 = float(x_num.max())
                if x0 == x1:
                    x0 -= 0.5
                    x1 += 0.5
                im = ax.imshow(
                    np.clip(heat.to_numpy(dtype=float), -3.0, 3.0),
                    aspect="auto",
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-3.0,
                    vmax=3.0,
                    extent=[x0, x1, -0.5, heat.shape[0] - 0.5],
                )
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Rolling t-stat", fontsize=8)
                y_vals = list(range(heat.shape[0]))
                y_labels = [f"bin {int(b)}" for b in heat.index]
                ax.set_yticks(y_vals)
                ax.set_yticklabels(y_labels, fontsize=7)
                locator = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                ax.tick_params(axis="x", labelrotation=30)
            else:
                ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No rolling t-stat data", ha="center", va="center", transform=ax.transAxes)
    else:
        pd.DataFrame(
            columns=[
                "bin",
                "sample_days",
                "alpha_mean",
                "alpha_nw_tstat",
                "alpha_nw_pvalue",
                "roll_t_pos_ratio",
                "roll_t_neg_ratio",
                "roll_t_sign_flip",
            ]
        ).to_csv(out_dir / "factor_bin_alpha_significance.csv", index=False)
        ax.text(0.5, 0.5, "No bin/benchmark data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(
        f"Bin Alpha Significance Heatmap (rolling t, win={roll_win if 'roll_win' in locals() else 'n/a'})"
    )
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Bin")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_dir / "factor_report.png", dpi=150)
    plt.close(fig)
    return summary
