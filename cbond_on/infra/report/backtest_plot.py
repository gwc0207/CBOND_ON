from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_backtest_report_image(
    *,
    out_dir: Path,
    daily_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    ic_df: pd.DataFrame | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip backtest report image: {exc}")
        return

    daily = daily_df.copy()
    nav = nav_df.copy()
    if daily.empty or nav.empty:
        return
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"], errors="coerce")
    daily = daily[daily["trade_date"].notna()].sort_values("trade_date")
    nav = nav[nav["trade_date"].notna()].sort_values("trade_date")
    if daily.empty or nav.empty:
        return

    nav_series = pd.to_numeric(nav["nav"], errors="coerce").ffill()
    drawdown = nav_series / nav_series.cummax() - 1.0
    day_return = pd.to_numeric(daily["day_return"], errors="coerce").fillna(0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_nav = axes[0, 0]
    ax_nav.plot(nav["trade_date"], nav_series, label="strategy_nav", linewidth=1.8)
    if "benchmark_return" in daily.columns:
        benchmark_nav = (1.0 + pd.to_numeric(daily["benchmark_return"], errors="coerce").fillna(0.0)).cumprod()
        ax_nav.plot(daily["trade_date"], benchmark_nav, label="benchmark_nav", linestyle="--", linewidth=1.2)
    ax_nav.set_title("NAV")
    ax_nav.grid(alpha=0.25)
    ax_nav.legend(loc="best")

    ax_dd = axes[0, 1]
    ax_dd.fill_between(nav["trade_date"], drawdown, 0.0, color="tomato", alpha=0.35)
    ax_dd.plot(nav["trade_date"], drawdown, color="tomato", linewidth=1.2)
    ax_dd.set_title("Drawdown")
    ax_dd.grid(alpha=0.25)

    ax_ret = axes[1, 0]
    ax_ret.bar(daily["trade_date"], day_return, width=1.0, color="steelblue", alpha=0.8)
    ax_ret.axhline(0.0, color="black", linewidth=0.8)
    ax_ret.set_title("Daily Return")
    ax_ret.grid(alpha=0.25)

    ax_ic = axes[1, 1]
    has_ic = ic_df is not None and not ic_df.empty
    if has_ic:
        ic_work = ic_df.copy()
        ic_work["trade_date"] = pd.to_datetime(ic_work["trade_date"], errors="coerce")
        ic_work = ic_work[ic_work["trade_date"].notna()].sort_values("trade_date")
        if not ic_work.empty:
            if "ic" in ic_work.columns:
                ax_ic.plot(ic_work["trade_date"], pd.to_numeric(ic_work["ic"], errors="coerce"), label="ic")
            if "rank_ic" in ic_work.columns:
                ax_ic.plot(
                    ic_work["trade_date"],
                    pd.to_numeric(ic_work["rank_ic"], errors="coerce"),
                    label="rank_ic",
                )
            ax_ic.axhline(0.0, color="black", linewidth=0.8)
            ax_ic.legend(loc="best")
    if not has_ic:
        ax_ic.text(0.5, 0.5, "No IC series", ha="center", va="center", transform=ax_ic.transAxes)
    ax_ic.set_title("IC / RankIC")
    ax_ic.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "backtest_report.png", dpi=150)
    plt.close(fig)
