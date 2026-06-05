from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _fmt_pct(value: object, digits: int = 2) -> str:
    val = _finite(value)
    if np.isnan(val):
        return "-"
    return f"{val * 100:.{digits}f}%"


def _fmt_num(value: object, digits: int = 2) -> str:
    val = _finite(value)
    if np.isnan(val):
        return "-"
    return f"{val:.{digits}f}"


def _annual_return(final_nav: float, days: int) -> float:
    if days <= 0 or not np.isfinite(final_nav) or final_nav <= 0:
        return float("nan")
    return float(final_nav ** (TRADING_DAYS_PER_YEAR / days) - 1.0)


def _annual_vol(returns: pd.Series) -> float:
    ret = _num(returns).dropna()
    if len(ret) < 2:
        return float("nan")
    return float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sharpe(returns: pd.Series) -> float:
    ret = _num(returns).dropna()
    if len(ret) < 2:
        return float("nan")
    std = float(ret.std(ddof=1))
    if std <= 0:
        return float("nan")
    return float(ret.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _tstat(returns: pd.Series) -> float:
    ret = _num(returns).dropna()
    if len(ret) < 2:
        return float("nan")
    std = float(ret.std(ddof=1))
    if std <= 0:
        return float("nan")
    return float(ret.mean() / std * np.sqrt(len(ret)))


def _max_drawdown_info(nav: pd.Series, dates: pd.Series) -> dict[str, object]:
    vals = _num(nav).ffill()
    work = pd.DataFrame({"trade_date": pd.to_datetime(dates, errors="coerce"), "nav": vals})
    work = work.dropna(subset=["trade_date", "nav"]).sort_values("trade_date")
    if work.empty:
        return {"max_drawdown": float("nan"), "peak_date": None, "trough_date": None, "recovery_date": None}

    nav_s = work["nav"].reset_index(drop=True)
    date_s = work["trade_date"].reset_index(drop=True)
    running_max = nav_s.cummax()
    drawdown = nav_s / running_max - 1.0
    trough_idx = int(drawdown.idxmin())
    peak_idx = int(nav_s.iloc[: trough_idx + 1].idxmax())
    peak_val = float(nav_s.iloc[peak_idx])
    recover = work.iloc[trough_idx:].reset_index(drop=True)
    recover = recover[recover["nav"] >= peak_val]
    recovery_date = None
    if not recover.empty:
        recovery_date = recover["trade_date"].iloc[0].date().isoformat()
    return {
        "max_drawdown": float(drawdown.iloc[trough_idx]),
        "peak_date": date_s.iloc[peak_idx].date().isoformat(),
        "trough_date": date_s.iloc[trough_idx].date().isoformat(),
        "recovery_date": recovery_date,
    }


def _compound_return(returns: pd.Series) -> float:
    ret = _num(returns).dropna()
    if ret.empty:
        return float("nan")
    return float((1.0 + ret).prod() - 1.0)


def _return_table(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = daily.set_index("trade_date")[["day_return", "benchmark_return"]].copy()
    monthly = indexed.resample("ME").apply(_compound_return)
    monthly["excess"] = monthly["day_return"] - monthly["benchmark_return"]
    yearly = indexed.resample("YE").apply(_compound_return)
    yearly["excess"] = yearly["day_return"] - yearly["benchmark_return"]
    return monthly, yearly


def _turnover_table(positions: pd.DataFrame | None) -> pd.DataFrame:
    if positions is None or positions.empty or not {"trade_date", "code", "weight"}.issubset(positions.columns):
        return pd.DataFrame(columns=["trade_date", "turnover"])
    pos = positions.copy()
    pos["trade_date"] = pd.to_datetime(pos["trade_date"], errors="coerce")
    pos["weight"] = _num(pos["weight"])
    pos = pos.dropna(subset=["trade_date", "code", "weight"])
    if pos.empty:
        return pd.DataFrame(columns=["trade_date", "turnover"])
    wide = pos.pivot_table(index="trade_date", columns="code", values="weight", aggfunc="sum").fillna(0.0)
    wide = wide.sort_index()
    turnover = wide.diff().abs().sum(axis=1) / 2.0
    if not turnover.empty:
        turnover.iloc[0] = wide.iloc[0].abs().sum() / 2.0
    return pd.DataFrame({"trade_date": turnover.index, "turnover": turnover.values})


def _contribution_table(positions: pd.DataFrame | None) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame(columns=["code", "weighted_return", "hold_days"])
    required = {"code", "weight", "return"}
    if not required.issubset(positions.columns):
        return pd.DataFrame(columns=["code", "weighted_return", "hold_days"])
    pos = positions.copy()
    pos["weight"] = _num(pos["weight"])
    pos["return"] = _num(pos["return"])
    pos = pos.dropna(subset=["code", "weight", "return"])
    if pos.empty:
        return pd.DataFrame(columns=["code", "weighted_return", "hold_days"])
    pos["weighted_return"] = pos["weight"] * pos["return"]
    out = (
        pos.groupby("code", as_index=False)
        .agg(weighted_return=("weighted_return", "sum"), hold_days=("code", "size"))
        .sort_values("weighted_return", ascending=False)
    )
    return out


def _ic_summary(ic_df: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if ic_df is None or ic_df.empty:
        return pd.DataFrame(columns=["metric", "mean", "std", "ir", "t_stat", "positive_ratio", "median"])
    for col in ["ic", "rank_ic"]:
        if col not in ic_df.columns:
            continue
        vals = _num(ic_df[col]).dropna()
        if vals.empty:
            continue
        std = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(
            {
                "metric": col,
                "mean": float(vals.mean()),
                "std": std,
                "ir": float(vals.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else float("nan"),
                "t_stat": float(vals.mean() / std * np.sqrt(len(vals))) if std > 0 else float("nan"),
                "positive_ratio": float((vals > 0).mean()),
                "median": float(vals.median()),
            }
        )
    return pd.DataFrame(rows)


def _summary_tables(
    *,
    daily: pd.DataFrame,
    nav: pd.DataFrame,
    positions: pd.DataFrame | None,
    diagnostics: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    configured_start: object | None,
    configured_end: object | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_ret = _num(daily["day_return"]).fillna(0.0)
    bench_ret = _num(daily.get("benchmark_return", pd.Series(index=daily.index, dtype=float))).fillna(0.0)
    excess_ret = day_ret - bench_ret

    nav_series = _num(nav["nav"]).ffill()
    if "benchmark_nav" in nav.columns:
        bench_nav = _num(nav["benchmark_nav"]).ffill()
    else:
        bench_nav = (1.0 + bench_ret).cumprod()
    excess_nav = (1.0 + excess_ret).cumprod()

    days = int(len(daily))
    strategy_dd = _max_drawdown_info(nav_series, nav["trade_date"])
    bench_dd = _max_drawdown_info(bench_nav, nav["trade_date"])
    excess_dd = _max_drawdown_info(excess_nav, daily["trade_date"])

    turnover = _turnover_table(positions)
    avg_turnover = float(turnover["turnover"].mean()) if not turnover.empty else float("nan")
    median_turnover = float(turnover["turnover"].median()) if not turnover.empty else float("nan")
    diag_rows = int(len(diagnostics)) if diagnostics is not None else 0
    skip_days = 0
    missing_score_days = 0
    allowlist_table = ""
    allowlist_lag_trading_days = ""
    allowlist_applied_days = 0
    allowlist_fallback_days = 0
    avg_allowlist_count = float("nan")
    avg_pre_allowlist_count = float("nan")
    avg_post_allowlist_count = float("nan")
    security_banlist_file = ""
    security_banlist_codes_count = 0
    security_banlist_removed_total = 0
    security_banlist_removed_days = 0
    min_price = float("nan")
    max_price = float("nan")
    if diagnostics is not None and not diagnostics.empty and "status" in diagnostics.columns:
        skip_days = int((diagnostics["status"].astype(str) != "ok").sum())
        if "reason" in diagnostics.columns:
            missing_score_days = int((diagnostics["reason"].fillna("").astype(str) == "missing_score").sum())
        if "allowlist_table" in diagnostics.columns:
            values = diagnostics["allowlist_table"].dropna().astype(str)
            values = values[values != ""]
            allowlist_table = values.iloc[0] if not values.empty else ""
        if "allowlist_lag_trading_days" in diagnostics.columns:
            values = diagnostics["allowlist_lag_trading_days"].dropna()
            allowlist_lag_trading_days = str(values.iloc[0]) if not values.empty else ""
        if "allowlist_applied" in diagnostics.columns:
            allowlist_applied_days = int(diagnostics["allowlist_applied"].fillna(False).astype(bool).sum())
        if "allowlist_fallback_no_filter" in diagnostics.columns:
            allowlist_fallback_days = int(diagnostics["allowlist_fallback_no_filter"].fillna(False).astype(bool).sum())
        if "allowlist_codes_count" in diagnostics.columns:
            counts = _num(diagnostics["allowlist_codes_count"])
            counts = counts[counts > 0]
            avg_allowlist_count = float(counts.mean()) if not counts.empty else float("nan")
        if "pre_allowlist_count" in diagnostics.columns:
            counts = _num(diagnostics["pre_allowlist_count"])
            counts = counts[counts > 0]
            avg_pre_allowlist_count = float(counts.mean()) if not counts.empty else float("nan")
        if "post_allowlist_count" in diagnostics.columns:
            counts = _num(diagnostics["post_allowlist_count"])
            counts = counts[counts > 0]
            avg_post_allowlist_count = float(counts.mean()) if not counts.empty else float("nan")
        if "security_banlist_file" in diagnostics.columns:
            values = diagnostics["security_banlist_file"].dropna().astype(str)
            values = values[values != ""]
            security_banlist_file = values.iloc[0] if not values.empty else ""
        if "security_banlist_codes_count" in diagnostics.columns:
            counts = _num(diagnostics["security_banlist_codes_count"]).dropna()
            security_banlist_codes_count = int(counts.max()) if not counts.empty else 0
        if "security_banlist_removed_count" in diagnostics.columns:
            removed = _num(diagnostics["security_banlist_removed_count"]).fillna(0)
            security_banlist_removed_total = int(removed.sum())
            security_banlist_removed_days = int((removed > 0).sum())
        if "min_price" in diagnostics.columns:
            values = _num(diagnostics["min_price"]).dropna()
            values = values[values > 0]
            min_price = float(values.iloc[0]) if not values.empty else float("nan")
        if "max_price" in diagnostics.columns:
            values = _num(diagnostics["max_price"]).dropna()
            values = values[values > 0]
            max_price = float(values.iloc[0]) if not values.empty else float("nan")

    ic_stats = _ic_summary(ic_df)
    ic_mean = float("nan")
    rank_ic_mean = float("nan")
    if not ic_stats.empty:
        ic_row = ic_stats[ic_stats["metric"] == "ic"]
        rank_row = ic_stats[ic_stats["metric"] == "rank_ic"]
        if not ic_row.empty:
            ic_mean = float(ic_row["mean"].iloc[0])
        if not rank_row.empty:
            rank_ic_mean = float(rank_row["mean"].iloc[0])

    summary_rows = [
        {
            "metric": "strategy",
            "final_nav": float(nav_series.iloc[-1]),
            "total_return": float(nav_series.iloc[-1] - 1.0),
            "annual_return": _annual_return(float(nav_series.iloc[-1]), days),
            "annual_vol": _annual_vol(day_ret),
            "sharpe": _sharpe(day_ret),
            "max_drawdown": strategy_dd["max_drawdown"],
            "calmar": _annual_return(float(nav_series.iloc[-1]), days) / abs(float(strategy_dd["max_drawdown"]))
            if float(strategy_dd["max_drawdown"]) < 0
            else float("nan"),
            "win_rate": float((day_ret > 0).mean()),
            "avg_daily_return": float(day_ret.mean()),
            "t_stat": _tstat(day_ret),
            "peak_date": strategy_dd["peak_date"],
            "trough_date": strategy_dd["trough_date"],
            "recovery_date": strategy_dd["recovery_date"],
        },
        {
            "metric": "benchmark",
            "final_nav": float(bench_nav.iloc[-1]),
            "total_return": float(bench_nav.iloc[-1] - 1.0),
            "annual_return": _annual_return(float(bench_nav.iloc[-1]), days),
            "annual_vol": _annual_vol(bench_ret),
            "sharpe": _sharpe(bench_ret),
            "max_drawdown": bench_dd["max_drawdown"],
            "calmar": _annual_return(float(bench_nav.iloc[-1]), days) / abs(float(bench_dd["max_drawdown"]))
            if float(bench_dd["max_drawdown"]) < 0
            else float("nan"),
            "win_rate": float((bench_ret > 0).mean()),
            "avg_daily_return": float(bench_ret.mean()),
            "t_stat": _tstat(bench_ret),
            "peak_date": bench_dd["peak_date"],
            "trough_date": bench_dd["trough_date"],
            "recovery_date": bench_dd["recovery_date"],
        },
        {
            "metric": "excess",
            "final_nav": float(excess_nav.iloc[-1]),
            "total_return": float(excess_nav.iloc[-1] - 1.0),
            "annual_return": _annual_return(float(excess_nav.iloc[-1]), days),
            "annual_vol": _annual_vol(excess_ret),
            "sharpe": _sharpe(excess_ret),
            "max_drawdown": excess_dd["max_drawdown"],
            "calmar": _annual_return(float(excess_nav.iloc[-1]), days) / abs(float(excess_dd["max_drawdown"]))
            if float(excess_dd["max_drawdown"]) < 0
            else float("nan"),
            "win_rate": float((excess_ret > 0).mean()),
            "avg_daily_return": float(excess_ret.mean()),
            "t_stat": _tstat(excess_ret),
            "peak_date": excess_dd["peak_date"],
            "trough_date": excess_dd["trough_date"],
            "recovery_date": excess_dd["recovery_date"],
        },
    ]
    summary = pd.DataFrame(summary_rows)

    meta = pd.DataFrame(
        [
            {
                "configured_start": str(configured_start) if configured_start is not None else "",
                "configured_end": str(configured_end) if configured_end is not None else "",
                "actual_start": daily["trade_date"].min().date().isoformat(),
                "actual_end": daily["trade_date"].max().date().isoformat(),
                "trading_days": days,
                "diagnostic_rows": diag_rows,
                "skip_days": skip_days,
                "missing_score_days": missing_score_days,
                "avg_count": float(_num(daily["count"]).mean()) if "count" in daily.columns else float("nan"),
                "allowlist_table": allowlist_table,
                "allowlist_lag_trading_days": allowlist_lag_trading_days,
                "allowlist_applied_days": allowlist_applied_days,
                "allowlist_fallback_days": allowlist_fallback_days,
                "avg_allowlist_count": avg_allowlist_count,
                "avg_pre_allowlist_count": avg_pre_allowlist_count,
                "avg_post_allowlist_count": avg_post_allowlist_count,
                "security_banlist_file": security_banlist_file,
                "security_banlist_codes_count": security_banlist_codes_count,
                "security_banlist_removed_total": security_banlist_removed_total,
                "security_banlist_removed_days": security_banlist_removed_days,
                "min_price": min_price,
                "max_price": max_price,
                "avg_turnover": avg_turnover,
                "median_turnover": median_turnover,
                "ic_mean": ic_mean,
                "rank_ic_mean": rank_ic_mean,
            }
        ]
    )
    return summary, meta


def _write_report_data(
    *,
    out_dir: Path,
    daily: pd.DataFrame,
    nav: pd.DataFrame,
    positions: pd.DataFrame | None,
    diagnostics: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    configured_start: object | None,
    configured_end: object | None,
) -> dict[str, pd.DataFrame]:
    summary, meta = _summary_tables(
        daily=daily,
        nav=nav,
        positions=positions,
        diagnostics=diagnostics,
        ic_df=ic_df,
        configured_start=configured_start,
        configured_end=configured_end,
    )
    monthly, yearly = _return_table(daily)
    turnover = _turnover_table(positions)
    contribution = _contribution_table(positions)
    ic_stats = _ic_summary(ic_df)

    day_cols = ["trade_date", "day_return", "benchmark_return", "buy_leg_ret_net", "sell_leg_ret_net", "count"]
    day_cols = [c for c in day_cols if c in daily.columns]
    top_days = pd.concat(
        [
            daily.nlargest(10, "day_return")[day_cols].assign(bucket="best"),
            daily.nsmallest(10, "day_return")[day_cols].assign(bucket="worst"),
        ],
        ignore_index=True,
    )

    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    meta.to_csv(out_dir / "run_coverage_summary.csv", index=False)
    monthly.to_csv(out_dir / "monthly_returns.csv")
    yearly.to_csv(out_dir / "yearly_returns.csv")
    turnover.to_csv(out_dir / "turnover.csv", index=False)
    contribution.to_csv(out_dir / "contribution_by_code.csv", index=False)
    ic_stats.to_csv(out_dir / "ic_summary.csv", index=False)
    top_days.to_csv(out_dir / "top_days.csv", index=False)
    payload = {
        "coverage": meta.to_dict(orient="records")[0] if not meta.empty else {},
        "summary": summary.to_dict(orient="records"),
        "ic_summary": ic_stats.to_dict(orient="records"),
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "meta": meta,
        "monthly": monthly,
        "yearly": yearly,
        "turnover": turnover,
        "contribution": contribution,
        "ic_stats": ic_stats,
        "top_days": top_days,
    }


def write_backtest_report_image(
    *,
    out_dir: Path,
    daily_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    ic_df: pd.DataFrame | None,
    positions_df: pd.DataFrame | None = None,
    diagnostics_df: pd.DataFrame | None = None,
    configured_start: object | None = None,
    configured_end: object | None = None,
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

    tables = _write_report_data(
        out_dir=out_dir,
        daily=daily,
        nav=nav,
        positions=positions_df,
        diagnostics=diagnostics_df,
        ic_df=ic_df,
        configured_start=configured_start,
        configured_end=configured_end,
    )

    day_return = _num(daily["day_return"]).fillna(0.0)
    bench_return = _num(daily.get("benchmark_return", pd.Series(index=daily.index, dtype=float))).fillna(0.0)
    excess_return = day_return - bench_return
    nav_series = _num(nav["nav"]).ffill()
    benchmark_nav = _num(nav["benchmark_nav"]).ffill() if "benchmark_nav" in nav.columns else (1.0 + bench_return).cumprod()
    excess_nav = (1.0 + excess_return).cumprod()
    drawdown = nav_series / nav_series.cummax() - 1.0
    benchmark_drawdown = benchmark_nav / benchmark_nav.cummax() - 1.0
    excess_drawdown = excess_nav / excess_nav.cummax() - 1.0

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.35, 1.25])

    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis("off")
    summary = tables["summary"].set_index("metric")
    meta = tables["meta"].iloc[0].to_dict() if not tables["meta"].empty else {}
    display_rows = [
        ("Final NAV", "final_nav", _fmt_num),
        ("Total Return", "total_return", _fmt_pct),
        ("Annual Return", "annual_return", _fmt_pct),
        ("Annual Vol", "annual_vol", _fmt_pct),
        ("Sharpe / IR", "sharpe", _fmt_num),
        ("Max Drawdown", "max_drawdown", _fmt_pct),
        ("Calmar", "calmar", _fmt_num),
        ("Win Rate", "win_rate", _fmt_pct),
    ]
    table_text = []
    for label, key, formatter in display_rows:
        table_text.append(
            [
                label,
                formatter(summary.loc["strategy", key]) if "strategy" in summary.index else "-",
                formatter(summary.loc["benchmark", key]) if "benchmark" in summary.index else "-",
                formatter(summary.loc["excess", key]) if "excess" in summary.index else "-",
            ]
        )
    table = ax_summary.table(
        cellText=table_text,
        colLabels=["Metric", "Strategy", "Benchmark", "Excess"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    allowlist_note = ""
    if np.isfinite(_finite(meta.get("avg_allowlist_count"))):
        allowlist_note = (
            f" | allowlist_avg={_fmt_num(meta.get('avg_allowlist_count'), 0)}"
            f" fallback={meta.get('allowlist_fallback_days', '-')}"
        )
    banlist_note = ""
    if int(_finite(meta.get("security_banlist_codes_count")) if np.isfinite(_finite(meta.get("security_banlist_codes_count"))) else 0) > 0:
        banlist_note = (
            f" | banlist={int(meta.get('security_banlist_codes_count', 0))}"
            f" removed={int(meta.get('security_banlist_removed_total', 0))}"
        )
    price_note = ""
    has_min_price = np.isfinite(_finite(meta.get("min_price")))
    has_max_price = np.isfinite(_finite(meta.get("max_price")))
    if has_min_price and has_max_price:
        price_min = _fmt_num(meta.get("min_price"), 0) if np.isfinite(_finite(meta.get("min_price"))) else "-"
        price_max = _fmt_num(meta.get("max_price"), 0) if np.isfinite(_finite(meta.get("max_price"))) else "-"
        price_note = f" | price={price_min}-{price_max}"
    elif has_min_price:
        price_note = f" | price>={_fmt_num(meta.get('min_price'), 0)}"
    elif has_max_price:
        price_note = f" | price<={_fmt_num(meta.get('max_price'), 0)}"
    title = (
        f"Backtest Report | configured {meta.get('configured_start', '')} to {meta.get('configured_end', '')} | "
        f"actual {meta.get('actual_start', '')} to {meta.get('actual_end', '')} | "
        f"days={meta.get('trading_days', '-')} skip={meta.get('skip_days', '-')}"
        f"{allowlist_note}"
        f"{banlist_note}"
        f"{price_note}"
    )
    ax_summary.set_title(title, fontsize=13, fontweight="bold", pad=12)

    ax_nav = fig.add_subplot(gs[1, 0])
    ax_nav.plot(nav["trade_date"], nav_series, label="strategy", linewidth=1.8)
    ax_nav.plot(nav["trade_date"], benchmark_nav, label="benchmark", linestyle="--", linewidth=1.3)
    ax_nav.plot(daily["trade_date"], excess_nav, label="excess nav", linewidth=1.2, alpha=0.85)
    ax_nav.set_title("NAV / Excess NAV (Strict Official Split)")
    ax_nav.grid(alpha=0.25)
    ax_nav.legend(loc="best", fontsize=8)

    ax_dd = fig.add_subplot(gs[1, 1])
    ax_dd.fill_between(nav["trade_date"], drawdown, 0.0, color="#E45756", alpha=0.25)
    ax_dd.plot(nav["trade_date"], drawdown, color="#E45756", linewidth=1.1, label="strategy")
    ax_dd.plot(nav["trade_date"], benchmark_drawdown, color="#4C78A8", linewidth=1.0, label="benchmark")
    ax_dd.plot(daily["trade_date"], excess_drawdown, color="#F58518", linewidth=1.0, label="excess")
    ax_dd.set_title("Drawdown")
    ax_dd.grid(alpha=0.25)
    ax_dd.legend(loc="lower left", fontsize=8)

    ax_ic = fig.add_subplot(gs[1, 2])
    has_ic = ic_df is not None and not ic_df.empty
    if has_ic:
        ic_work = ic_df.copy()
        ic_work["trade_date"] = pd.to_datetime(ic_work["trade_date"], errors="coerce")
        ic_work = ic_work[ic_work["trade_date"].notna()].sort_values("trade_date")
        if not ic_work.empty:
            if "ic" in ic_work.columns:
                ax_ic.plot(ic_work["trade_date"], _num(ic_work["ic"]).rolling(20, min_periods=5).mean(), label="ic 20d")
            if "rank_ic" in ic_work.columns:
                ax_ic.plot(
                    ic_work["trade_date"],
                    _num(ic_work["rank_ic"]).rolling(20, min_periods=5).mean(),
                    label="rank_ic 20d",
                )
            ax_ic.axhline(0.0, color="black", linewidth=0.8)
            ax_ic.legend(loc="best", fontsize=8)
    if not has_ic:
        ax_ic.text(0.5, 0.5, "No IC series", ha="center", va="center", transform=ax_ic.transAxes)
    ax_ic.set_title(f"IC / RankIC | mean={_fmt_num(meta.get('ic_mean'), 3)} / {_fmt_num(meta.get('rank_ic_mean'), 3)}")
    ax_ic.grid(alpha=0.25)

    ax_turnover = fig.add_subplot(gs[2, 0])
    turnover = tables["turnover"]
    if not turnover.empty:
        ax_turnover.plot(turnover["trade_date"], turnover["turnover"], color="#54A24B", linewidth=1.0)
        ax_turnover.axhline(turnover["turnover"].mean(), color="#E45756", linestyle="--", linewidth=1.0, label="avg")
        ax_turnover.legend(loc="best", fontsize=8)
    ax_turnover.set_title(f"Turnover | avg={_fmt_pct(meta.get('avg_turnover'))}")
    ax_turnover.grid(alpha=0.25)

    ax_days = fig.add_subplot(gs[2, 1])
    ax_days.axis("off")
    best = daily.nlargest(3, "day_return")
    worst = daily.nsmallest(3, "day_return")
    day_table = []
    for _, row in best.iterrows():
        day_table.append(["Best", row["trade_date"].date().isoformat(), _fmt_pct(row["day_return"]), _fmt_pct(row.get("benchmark_return"))])
    for _, row in worst.iterrows():
        day_table.append(["Worst", row["trade_date"].date().isoformat(), _fmt_pct(row["day_return"]), _fmt_pct(row.get("benchmark_return"))])
    if day_table:
        tbl = ax_days.table(
            cellText=day_table,
            colLabels=["Type", "Date", "Return", "Benchmark"],
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.25)
    ax_days.set_title("Best / Worst Days (Strict Split)", fontsize=11)

    ax_contrib = fig.add_subplot(gs[2, 2])
    contrib = tables["contribution"]
    if not contrib.empty:
        top = contrib.head(6)
        bottom = contrib.tail(6)
        show = pd.concat([bottom, top], ignore_index=True)
        colors = ["#E45756" if x < 0 else "#54A24B" for x in show["weighted_return"]]
        ax_contrib.barh(show["code"].astype(str), show["weighted_return"], color=colors, alpha=0.8)
        ax_contrib.axvline(0.0, color="black", linewidth=0.8)
    else:
        ax_contrib.text(0.5, 0.5, "No contribution data", ha="center", va="center", transform=ax_contrib.transAxes)
    ax_contrib.set_title("Top / Bottom Code Contribution")
    ax_contrib.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(out_dir / "backtest_report.png", dpi=150)
    plt.close(fig)
