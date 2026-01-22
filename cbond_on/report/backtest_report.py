from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _calc_turnover(positions: pd.DataFrame) -> float:
    if positions.empty:
        return 0.0
    work = positions.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"]).dt.date
    turnover_vals: list[float] = []
    prev = pd.Series(dtype=float)
    for _, group in work.groupby("trade_date"):
        weights = group.set_index("code")["weight"].astype(float)
        all_codes = prev.index.union(weights.index)
        diff = (weights.reindex(all_codes).fillna(0.0) - prev.reindex(all_codes).fillna(0.0)).abs()
        turnover_vals.append(0.5 * float(diff.sum()))
        prev = weights
    if not turnover_vals:
        return 0.0
    return float(pd.Series(turnover_vals).mean())


def _calc_performance_metrics(daily_returns: pd.DataFrame, nav_curve: pd.DataFrame) -> dict:
    if daily_returns.empty or nav_curve.empty:
        return {"sharpe": 0.0, "maxdd": 0.0, "win_rate": 0.0}
    daily = daily_returns["day_return"].astype(float)
    mean_ret = float(daily.mean())
    vol = float(daily.std(ddof=0))
    sharpe = float((mean_ret / vol) * (252.0**0.5)) if vol else 0.0
    nav = nav_curve["nav"].astype(float)
    running_max = nav.cummax()
    drawdown = (nav / running_max) - 1.0
    maxdd = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((daily > 0).mean())
    return {"sharpe": sharpe, "maxdd": maxdd, "win_rate": win_rate}


def render_backtest_report(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "daily_returns.csv"
    nav_path = out_dir / "nav_curve.csv"
    ic_path = out_dir / "ic_series.csv"
    bin_nav_path = out_dir / "bin_nav.csv"
    positions_path = out_dir / "positions.csv"

    if not daily_path.exists() or not nav_path.exists():
        raise FileNotFoundError(f"missing backtest outputs under {out_dir}")

    daily = pd.read_csv(daily_path, parse_dates=["trade_date"])
    nav = pd.read_csv(nav_path, parse_dates=["trade_date"])
    ic_df = pd.read_csv(ic_path, parse_dates=["trade_date"]) if ic_path.exists() else pd.DataFrame()
    bin_nav = pd.read_csv(bin_nav_path, parse_dates=["trade_date"]) if bin_nav_path.exists() else pd.DataFrame()
    positions = pd.read_csv(positions_path) if positions_path.exists() else pd.DataFrame()

    metrics = _calc_performance_metrics(daily, nav)
    metrics["turnover"] = _calc_turnover(positions)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))

    ax = axes[0, 0]
    if not ic_df.empty:
        ic_df = ic_df.sort_values("trade_date")
        ic_df["ic_cum"] = ic_df["ic"].cumsum()
        ic_df["rank_ic_cum"] = ic_df["rank_ic"].cumsum()
        ax.plot(ic_df["trade_date"], ic_df["ic"], label="IC")
        ax.plot(ic_df["trade_date"], ic_df["ic_cum"], label="IC cum")
        ax.plot(ic_df["trade_date"], ic_df["rank_ic"], label="Rank IC")
        ax.plot(ic_df["trade_date"], ic_df["rank_ic_cum"], label="Rank IC cum")
    ax.set_title("IC series")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if not ic_df.empty:
        ic_mean = float(ic_df["ic"].mean())
        rank_ic_mean = float(ic_df["rank_ic"].mean())
        ic_ir = float(ic_df["ic"].mean() / ic_df["ic"].std()) if ic_df["ic"].std() else 0.0
        rank_ic_ir = float(ic_df["rank_ic"].mean() / ic_df["rank_ic"].std()) if ic_df["rank_ic"].std() else 0.0
        labels = ["IC", "IR", "Rank_IC", "Rank_IR"]
        values = [ic_mean, ic_ir, rank_ic_mean, rank_ic_ir]
        ax.bar(labels, values, color=["#3C8DBC", "#1F77B4", "#9467BD", "#FF7F0E"])
        for idx, val in enumerate(values):
            ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("IC summary")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 2]
    if not bin_nav.empty:
        bin_nav = bin_nav.sort_values("trade_date")
        bins = [c for c in bin_nav.columns if c != "trade_date"]
        colors = plt.cm.viridis(np.linspace(0, 1, len(bins))) if bins else []
        for color, col in zip(colors, bins):
            ax.plot(bin_nav["trade_date"], bin_nav[col], color=color, label=f"bin {col}")
        if bins:
            ax.legend(fontsize=7, ncol=2)
    ax.set_title("Bin NAV")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if not daily.empty and "benchmark_return" in daily.columns:
        daily = daily.sort_values("trade_date")
        nav_factor = (1.0 + daily["day_return"].astype(float)).cumprod()
        nav_bench = (1.0 + daily["benchmark_return"].astype(float)).cumprod()
        ax.plot(daily["trade_date"], nav_factor, label="Factor")
        ax.plot(daily["trade_date"], nav_bench, label="Benchmark")
        ax.legend(fontsize=8)
    ax.set_title("NAV vs Benchmark")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    metric_names = ["sharpe", "maxdd", "win_rate", "turnover"]
    metric_vals = [metrics[m] for m in metric_names]
    ax.bar(metric_names, metric_vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    for idx, val in enumerate(metric_vals):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Performance metrics")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 2]
    if not daily.empty:
        ax.plot(daily["trade_date"], daily["day_return"].astype(float), color="#1F77B4")
    ax.set_title("Daily returns")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "backtest_report.png", dpi=150)
    plt.close(fig)
