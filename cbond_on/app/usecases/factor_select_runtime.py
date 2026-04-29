from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.infra.benchmark.service import compute_benchmark_returns_for_days
from cbond_on.infra.factors.quality import expected_factor_columns_from_cfg
from cbond_on.infra.model.eval.evaluator import (
    EvaluationResult,
    evaluate_merged_scores,
    load_scores_frame,
    merge_score_with_label,
)


@dataclass
class _EvalTrial:
    trial_name: str
    trial_dir: Path
    score_output: Path
    state_dir: Path
    factors: list[str]
    summary: dict[str, Any]
    daily: pd.DataFrame


def _safe_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "empty"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _load_factor_list(path_like: str) -> list[str]:
    path = resolve_config_path(path_like)
    payload = load_json_like(path)
    raw: Any
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("factors", [])
    else:
        raw = []
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            name = str(item).strip()
            if name:
                out.append(name)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        v = str(value).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _build_trial_model_cfg(
    *,
    base_model_cfg: dict[str, Any],
    start_day: date,
    end_day: date,
    factors: list[str],
    model_name: str,
    score_output: Path,
    save_state: bool,
    state_dir: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_model_cfg)
    cfg["start"] = str(start_day)
    cfg["end"] = str(end_day)
    cfg["model_name"] = model_name
    cfg["factors"] = list(factors)
    cfg["score_output"] = str(score_output.as_posix())
    cfg["score_overwrite"] = True
    cfg["score_dedupe"] = True
    inc = dict(cfg.get("incremental", {}))
    inc["enabled"] = True
    inc["skip_existing_scores"] = False
    inc["warm_start"] = False
    inc["save_state"] = bool(save_state)
    inc["state_dir"] = str(state_dir.as_posix())
    cfg["incremental"] = inc
    return cfg


def _evaluate_score_output(
    *,
    score_output: Path,
    label_root: Path,
    factor_time: str,
    label_time: str,
    start_day: date,
    end_day: date,
    bins: int,
) -> EvaluationResult:
    scores = load_scores_frame(score_output)
    merged = merge_score_with_label(
        scores=scores,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        start=start_day,
        end=end_day,
    )
    return evaluate_merged_scores(merged, bins=bins)


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


def _mean_bin_monotonicity(bin_daily: pd.DataFrame) -> float:
    if bin_daily.empty:
        return float("nan")
    rhos: list[float] = []
    for _, g in bin_daily.groupby("trade_date", sort=True):
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


def _series_maxdd(nav_series: pd.Series) -> float:
    s = pd.to_numeric(nav_series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return float(dd.min()) if not dd.empty else float("nan")


def _build_bin_outputs(
    *,
    merged: pd.DataFrame,
    bins: int,
    benchmark_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols_daily = ["trade_date", "bin", "return_mean"]
    if merged is None or merged.empty:
        return (
            pd.DataFrame(columns=cols_daily),
            pd.DataFrame(columns=["trade_date", "benchmark_return"]),
            pd.DataFrame(columns=["trade_date"]),
        )

    work = merged.copy()
    if "trade_date" not in work.columns or "score" not in work.columns or "y" not in work.columns:
        return (
            pd.DataFrame(columns=cols_daily),
            pd.DataFrame(columns=["trade_date", "benchmark_return"]),
            pd.DataFrame(columns=["trade_date"]),
        )

    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["trade_date", "score", "y"])
    if work.empty:
        return (
            pd.DataFrame(columns=cols_daily),
            pd.DataFrame(columns=["trade_date", "benchmark_return"]),
            pd.DataFrame(columns=["trade_date"]),
        )

    q = max(2, int(bins))
    bin_rows: list[dict[str, Any]] = []
    for trade_day, group in work.groupby("trade_date", sort=True):
        g = group[["score", "y"]].dropna()
        if g.empty:
            continue
        try:
            labels = pd.qcut(
                g["score"].rank(pct=True, method="average"),
                q=q,
                labels=False,
                duplicates="drop",
            )
        except Exception:
            continue
        gg = g.assign(bin=labels).dropna(subset=["bin"])
        if gg.empty:
            continue
        per_bin = gg.groupby("bin", dropna=True)["y"].mean()
        for bin_id, ret in per_bin.items():
            bin_rows.append(
                {
                    "trade_date": trade_day,
                    "bin": int(bin_id),
                    "return_mean": float(ret),
                }
            )

    bin_daily = pd.DataFrame(bin_rows, columns=cols_daily)
    if bin_daily.empty:
        return (
            bin_daily,
            pd.DataFrame(columns=["trade_date", "benchmark_return"]),
            pd.DataFrame(columns=["trade_date"]),
        )
    bin_daily = bin_daily.sort_values(["trade_date", "bin"], kind="mergesort").reset_index(drop=True)

    bench_daily = benchmark_daily.copy() if benchmark_daily is not None else pd.DataFrame()
    if bench_daily.empty:
        raise RuntimeError("factor_select benchmark_daily is empty")
    if "trade_date" not in bench_daily.columns or "benchmark_return" not in bench_daily.columns:
        raise RuntimeError("factor_select benchmark_daily missing required columns")
    bench_daily["trade_date"] = pd.to_datetime(bench_daily["trade_date"], errors="coerce").dt.date
    bench_daily = (
        bench_daily.dropna(subset=["trade_date"])
        .drop_duplicates(subset=["trade_date"], keep="last")
        .sort_values("trade_date", kind="mergesort")
        .reset_index(drop=True)
    )

    pivot = (
        bin_daily.pivot_table(
            index="trade_date",
            columns="bin",
            values="return_mean",
            aggfunc="mean",
        )
        .sort_index()
        .fillna(0.0)
    )
    if pivot.empty:
        return (
            bin_daily,
            bench_daily,
            pd.DataFrame(columns=["trade_date"]),
        )

    nav = (1.0 + pivot).cumprod()
    nav.columns = [f"bin_{int(c)}" for c in nav.columns]

    bench_series = pd.Series(
        data=pd.to_numeric(bench_daily["benchmark_return"], errors="coerce").values,
        index=bench_daily["trade_date"],
        dtype=float,
    ).sort_index()
    bench_series = bench_series.reindex(nav.index)
    if bench_series.isna().any():
        missing_days = [d for d in nav.index if pd.isna(bench_series.loc[d])]
        sample = ",".join(str(x) for x in missing_days[:5])
        raise RuntimeError(
            "factor_select benchmark missing days for bin nav: "
            f"missing={len(missing_days)} sample={sample}"
        )
    nav.insert(0, "benchmark", (1.0 + bench_series).cumprod().values)

    nav = nav.reset_index().rename(columns={"trade_date": "trade_date"})
    return bin_daily, bench_daily, nav


def _plot_bin_nav(
    *,
    nav_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if nav_df is None or nav_df.empty:
        return
    if "trade_date" not in nav_df.columns:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    frame = nav_df.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date", kind="mergesort")
    if frame.empty:
        return

    x = frame["trade_date"]
    y_cols = [c for c in frame.columns if c != "trade_date"]
    if not y_cols:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in y_cols:
        ys = pd.to_numeric(frame[col], errors="coerce")
        if col == "benchmark":
            ax.plot(
                x,
                ys,
                color="#D62728",
                linestyle="--",
                linewidth=2.0,
                label="benchmark",
                zorder=20,
            )
        else:
            ax.plot(x, ys, linewidth=1.0, alpha=0.85, label=col)

    ax.set_title(title)
    ax.set_xlabel("trade_date")
    ax.set_ylabel("cumulative_nav")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if "benchmark" in labels:
            idx = labels.index("benchmark")
            handles = [handles[idx]] + [h for i, h in enumerate(handles) if i != idx]
            labels = [labels[idx]] + [l for i, l in enumerate(labels) if i != idx]
        ax.legend(handles, labels, ncol=2, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_trial_metrics_and_alpha(
    *,
    eval_daily: pd.DataFrame,
    bin_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    alpha_significance_window: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, pd.DataFrame]:
    metric_frame = pd.DataFrame(
        columns=[
            "trade_date",
            "strategy_return",
            "benchmark_return",
            "alpha_return",
            "strategy_nav",
            "benchmark_nav",
            "alpha_nav",
        ]
    )
    metric_summary = {
        "strategy_ret": float("nan"),
        "strategy_sharpe": float("nan"),
        "maxdd": float("nan"),
        "alpha_ret": float("nan"),
        "mono_rankcorr": float("nan"),
    }
    alpha_summary = pd.DataFrame(
        columns=[
            "bin",
            "sample_days",
            "alpha_mean",
            "alpha_nw_tstat",
            "alpha_nw_pvalue",
            "roll_t_pos_ratio",
            "roll_t_neg_ratio",
        ]
    )
    alpha_heat = pd.DataFrame()

    if bin_daily.empty:
        return metric_frame, metric_summary, alpha_summary, alpha_heat

    pivot = (
        bin_daily.pivot_table(
            index="trade_date",
            columns="bin",
            values="return_mean",
            aggfunc="mean",
        )
        .sort_index()
    )
    if pivot.empty:
        return metric_frame, metric_summary, alpha_summary, alpha_heat

    bench = (
        benchmark_daily.copy()
        if benchmark_daily is not None and not benchmark_daily.empty
        else pd.DataFrame(columns=["trade_date", "benchmark_return"])
    )
    if bench.empty:
        raise RuntimeError("factor_select benchmark_daily is empty for trial metrics")
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce").dt.date
    bench = bench.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    bench_series = pd.Series(
        pd.to_numeric(bench["benchmark_return"], errors="coerce").values,
        index=bench["trade_date"],
        dtype=float,
    ).sort_index()
    missing_bench_days = [d for d in pivot.index if d not in bench_series.index]
    if missing_bench_days:
        sample = ",".join(str(x) for x in missing_bench_days[:5])
        raise RuntimeError(
            "factor_select benchmark missing days for metrics: "
            f"missing={len(missing_bench_days)} sample={sample}"
        )

    strategy_series = pd.Series(index=pivot.index, dtype=float)
    for d in pivot.index:
        row = pd.to_numeric(pivot.loc[d], errors="coerce").dropna()
        if row.empty:
            strategy_series.loc[d] = float("nan")
        else:
            # strategy proxy: top score quantile bin (highest bin id)
            strategy_series.loc[d] = float(row.iloc[-1])

    bench_aligned = bench_series.reindex(pivot.index)
    alpha = strategy_series - bench_aligned

    metric_frame = pd.DataFrame(
        {
            "trade_date": pivot.index,
            "strategy_return": strategy_series.values,
            "benchmark_return": bench_aligned.values,
            "alpha_return": alpha.values,
        }
    ).dropna(subset=["trade_date"]).reset_index(drop=True)
    if not metric_frame.empty:
        metric_frame["strategy_nav"] = (1.0 + pd.to_numeric(metric_frame["strategy_return"], errors="coerce").fillna(0.0)).cumprod()
        metric_frame["benchmark_nav"] = (1.0 + pd.to_numeric(metric_frame["benchmark_return"], errors="coerce").fillna(0.0)).cumprod()
        metric_frame["alpha_nav"] = (1.0 + pd.to_numeric(metric_frame["alpha_return"], errors="coerce").fillna(0.0)).cumprod()

        s = pd.to_numeric(metric_frame["strategy_return"], errors="coerce").dropna()
        s_std = float(s.std(ddof=0)) if not s.empty else float("nan")
        sharpe = float((float(s.mean()) / s_std) * math.sqrt(252.0)) if s_std and s_std > 0 else float("nan")
        metric_summary["strategy_ret"] = float(metric_frame["strategy_nav"].iloc[-1] - 1.0)
        metric_summary["strategy_sharpe"] = sharpe
        metric_summary["maxdd"] = _series_maxdd(metric_frame["strategy_nav"])
        metric_summary["alpha_ret"] = float(metric_frame["alpha_nav"].iloc[-1] - 1.0)

    metric_summary["mono_rankcorr"] = _mean_bin_monotonicity(bin_daily)

    alpha_by_bin = pivot.sub(bench_series.reindex(pivot.index), axis=0)
    alpha_by_bin = alpha_by_bin.sort_index(axis=1)
    if not alpha_by_bin.empty:
        obs = int(alpha_by_bin.shape[0])
        roll_win = int(max(2, min(int(alpha_significance_window), obs)))
        min_roll = max(2, roll_win // 2)
        roll_t = pd.DataFrame(index=alpha_by_bin.index)
        rows: list[dict[str, float | int]] = []
        for col in alpha_by_bin.columns:
            s = pd.to_numeric(alpha_by_bin[col], errors="coerce")
            t_roll = _rolling_tstat(s, window=roll_win, min_periods=min_roll)
            roll_t[col] = t_roll
            mean_val, nw_t, nw_p = _newey_west_tstat(s)
            rows.append(
                {
                    "bin": int(col),
                    "sample_days": int(s.notna().sum()),
                    "alpha_mean": float(mean_val),
                    "alpha_nw_tstat": float(nw_t) if pd.notna(nw_t) else float("nan"),
                    "alpha_nw_pvalue": float(nw_p) if pd.notna(nw_p) else float("nan"),
                    "roll_t_pos_ratio": float((pd.to_numeric(t_roll, errors="coerce") > 2.0).mean()) if t_roll.notna().any() else 0.0,
                    "roll_t_neg_ratio": float((pd.to_numeric(t_roll, errors="coerce") < -2.0).mean()) if t_roll.notna().any() else 0.0,
                }
            )
        alpha_summary = pd.DataFrame(rows).sort_values("bin", kind="mergesort").reset_index(drop=True)
        alpha_heat = roll_t.T
        alpha_heat.index = [f"bin_{int(x)}" for x in alpha_heat.index]

    # backfill these fields from eval daily so trial_metrics.csv can include them together
    if eval_daily is not None and not eval_daily.empty:
        metric_summary["ic_mean"] = float(pd.to_numeric(eval_daily["ic"], errors="coerce").mean())
        metric_summary["ic_ir"] = float(pd.to_numeric(eval_daily["ic_ir"], errors="coerce").mean())
        metric_summary["rank_ic_mean"] = float(pd.to_numeric(eval_daily["rank_ic"], errors="coerce").mean())
        metric_summary["rank_ic_ir"] = float(pd.to_numeric(eval_daily["rank_ic_ir"], errors="coerce").mean())

    return metric_frame, metric_summary, alpha_summary, alpha_heat


def _plot_trial_report(
    *,
    trial_name: str,
    trial_dir: Path,
    merged: pd.DataFrame,
    eval_daily: pd.DataFrame,
    summary: dict[str, Any],
    nav_df: pd.DataFrame,
    metric_summary: dict[str, float],
    alpha_heat: pd.DataFrame,
    alpha_significance_window: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception:
        return

    daily = eval_daily.copy() if eval_daily is not None else pd.DataFrame()
    if not daily.empty:
        daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
        daily = daily.dropna(subset=["trade_date"]).sort_values("trade_date", kind="mergesort")
        daily["ic"] = pd.to_numeric(daily["ic"], errors="coerce")
        daily["rank_ic"] = pd.to_numeric(daily["rank_ic"], errors="coerce")
        daily["ic_cum"] = daily["ic"].fillna(0.0).cumsum()
        daily["rank_ic_cum"] = daily["rank_ic"].fillna(0.0).cumsum()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(f"factor_select trial: {trial_name}", fontsize=12)

    # 1) IC Series
    ax = axes[0, 0]
    if not daily.empty:
        ax.plot(daily["trade_date"], daily["ic"], label="IC")
        ax.plot(daily["trade_date"], daily["rank_ic"], label="RankIC")
        ax.plot(daily["trade_date"], daily["ic_cum"], label="IC Cumsum")
        ax.plot(daily["trade_date"], daily["rank_ic_cum"], label="RankIC Cumsum")
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No IC data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("IC Series")
    ax.grid(alpha=0.3)

    # 2) IC metrics
    ax = axes[0, 1]
    ic_names = ["ic_mean", "ic_ir", "rank_ic_mean", "rank_ic_ir"]
    ic_vals = [float(summary.get(k, float("nan"))) for k in ic_names]
    ax.bar(ic_names, ic_vals, color=["#3C8DBC", "#1F77B4", "#9467BD", "#FF7F0E"])
    for i, v in enumerate(ic_vals):
        if pd.notna(v):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("IC Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)

    # 3) Bin NAV
    ax = axes[0, 2]
    nav = nav_df.copy() if nav_df is not None else pd.DataFrame()
    if not nav.empty and "trade_date" in nav.columns:
        nav["trade_date"] = pd.to_datetime(nav["trade_date"], errors="coerce")
        nav = nav.dropna(subset=["trade_date"]).sort_values("trade_date", kind="mergesort")
        y_cols = [c for c in nav.columns if c != "trade_date"]
        for col in y_cols:
            ys = pd.to_numeric(nav[col], errors="coerce")
            if col == "benchmark":
                ax.plot(nav["trade_date"], ys, color="red", linestyle="--", linewidth=2.0, label="benchmark", zorder=20)
            else:
                ax.plot(nav["trade_date"], ys, linewidth=1.0, alpha=0.85, label=col)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="x", labelrotation=30)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if "benchmark" in labels:
                idx = labels.index("benchmark")
                handles = [handles[idx]] + [h for i, h in enumerate(handles) if i != idx]
                labels = [labels[idx]] + [l for i, l in enumerate(labels) if i != idx]
            ax.legend(handles, labels, fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "No bin NAV", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Bin Cumulative NAV")
    ax.grid(alpha=0.3)

    # 4) score distribution
    ax = axes[1, 0]
    scores = pd.to_numeric(merged["score"], errors="coerce") if merged is not None and not merged.empty else pd.Series(dtype=float)
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if not scores.empty:
        q01 = float(scores.quantile(0.01))
        q99 = float(scores.quantile(0.99))
        clipped = scores.clip(lower=q01, upper=q99) if q99 > q01 else scores
        n_bins = int(min(80, max(20, np.sqrt(max(1, len(clipped))))))
        ax.hist(clipped, bins=n_bins, color="#1F77B4", alpha=0.85, edgecolor="white", linewidth=0.3)
        mean_val = float(clipped.mean())
        med_val = float(clipped.median())
        ax.axvline(mean_val, color="#E45756", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.4g}")
        ax.axvline(med_val, color="#54A24B", linestyle="-.", linewidth=1.2, label=f"median={med_val:.4g}")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No score values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Score Distribution")
    ax.set_xlabel("score")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    # 5) trial + strategy metrics
    ax = axes[1, 1]
    perf = [
        ("strategy_ret", float(metric_summary.get("strategy_ret", float("nan")))),
        ("strategy_sharpe", float(metric_summary.get("strategy_sharpe", float("nan")))),
        ("maxdd", float(metric_summary.get("maxdd", float("nan")))),
        ("alpha_ret", float(metric_summary.get("alpha_ret", float("nan")))),
        ("mono_rankcorr", float(metric_summary.get("mono_rankcorr", float("nan")))),
    ]
    p_names = [n for n, _ in perf]
    p_vals = [v for _, v in perf]
    ax.bar(p_names, p_vals, color=["#4C78A8", "#F58518", "#E45756", "#54A24B", "#9C755F"])
    for i, v in enumerate(p_vals):
        if pd.notna(v):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Trial + Strategy Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)

    # 6) alpha significance heatmap
    ax = axes[1, 2]
    heat = alpha_heat.copy() if alpha_heat is not None else pd.DataFrame()
    if not heat.empty:
        x = pd.to_datetime(heat.columns, errors="coerce")
        valid = ~x.isna()
        heat = heat.loc[:, valid]
        x = x[valid]
        if heat.shape[1] > 0:
            x_num = mdates.date2num(x.to_pydatetime())
            x0, x1 = float(x_num.min()), float(x_num.max())
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
            ax.set_yticks(list(range(heat.shape[0])))
            ax.set_yticklabels(list(heat.index), fontsize=7)
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(axis="x", labelrotation=30)
        else:
            ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No alpha significance data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"Bin Alpha Significance Heatmap (rolling t, win={alpha_significance_window})")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Bin")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(trial_dir / "trial_report.png", dpi=150)
    plt.close(fig)


def _run_model_trial(
    *,
    trial_name: str,
    trial_dir: Path,
    factors: list[str],
    model_id: str,
    model_type: str,
    base_model_cfg: dict[str, Any],
    score_cfg: dict[str, Any],
    execution_override: dict[str, Any],
    start_day: date,
    end_day: date,
    label_root: Path,
    factor_time: str,
    label_time: str,
    bins: int,
    save_state: bool,
    alpha_significance_window: int,
    raw_data_root: str | Path,
) -> _EvalTrial:
    trial_dir.mkdir(parents=True, exist_ok=True)
    score_output = trial_dir / "scores"
    state_dir = trial_dir / "state"
    model_name = f"{base_model_cfg.get('model_name', model_id)}__factor_select__{_safe_name(trial_name)}"
    trial_cfg = _build_trial_model_cfg(
        base_model_cfg=base_model_cfg,
        start_day=start_day,
        end_day=end_day,
        factors=factors,
        model_name=model_name,
        score_output=score_output,
        save_state=save_state,
        state_dir=state_dir,
    )
    trial_cfg_path = trial_dir / "model_config.json"
    trial_cfg_path.write_text(json.dumps(trial_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    trial_score_cfg = copy.deepcopy(score_cfg)
    trial_score_cfg["model_id"] = "factor_select_trial"
    trial_score_cfg["default_model_id"] = "factor_select_trial"
    trial_score_cfg["models"] = {
        "factor_select_trial": {
            "model_type": model_type,
            "model_config": str(trial_cfg_path.as_posix()),
        }
    }
    trial_score_cfg["execution"] = execution_override

    print(f"[factor_select] trial={trial_name} start factors={len(factors)}")
    run_model_score(
        model_id="factor_select_trial",
        start=start_day,
        end=end_day,
        cfg=trial_score_cfg,
    )

    eval_res = _evaluate_score_output(
        score_output=score_output,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        start_day=start_day,
        end_day=end_day,
        bins=bins,
    )

    payload = {
        "trial_name": trial_name,
        "factor_count": len(factors),
        "factors": factors,
        "summary": eval_res.summary,
    }
    (trial_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not eval_res.daily.empty:
        eval_res.daily.to_csv(trial_dir / "evaluation_daily.csv", index=False)
    daily_ic = pd.DataFrame()
    if eval_res.daily is not None and not eval_res.daily.empty:
        daily_ic = eval_res.daily.copy()
        daily_ic["trade_date"] = pd.to_datetime(daily_ic["trade_date"], errors="coerce")
        daily_ic = daily_ic.dropna(subset=["trade_date"]).sort_values("trade_date", kind="mergesort")
        daily_ic["ic"] = pd.to_numeric(daily_ic["ic"], errors="coerce")
        daily_ic["rank_ic"] = pd.to_numeric(daily_ic["rank_ic"], errors="coerce")
        daily_ic["ic_cum"] = daily_ic["ic"].fillna(0.0).cumsum()
        daily_ic["rank_ic_cum"] = daily_ic["rank_ic"].fillna(0.0).cumsum()
        daily_ic.to_csv(trial_dir / "ic_series.csv", index=False)

    trade_days: list[date] = []
    if eval_res.merged is not None and not eval_res.merged.empty and "trade_date" in eval_res.merged.columns:
        trade_days = (
            pd.to_datetime(eval_res.merged["trade_date"], errors="coerce")
            .dt.date.dropna().drop_duplicates().sort_values().tolist()
        )
    if not trade_days and eval_res.daily is not None and not eval_res.daily.empty and "trade_date" in eval_res.daily.columns:
        trade_days = (
            pd.to_datetime(eval_res.daily["trade_date"], errors="coerce")
            .dt.date.dropna().drop_duplicates().sort_values().tolist()
        )
    buy_bps, sell_bps, _ = load_fees_buy_sell_bps()
    benchmark_series = compute_benchmark_returns_for_days(
        raw_data_root=raw_data_root,
        trade_days=trade_days,
        buy_bps=buy_bps,
        sell_bps=sell_bps,
    )
    benchmark_daily_df = pd.DataFrame(
        {
            "trade_date": list(benchmark_series.index),
            "benchmark_return": list(benchmark_series.values),
        }
    )

    bin_daily_df, benchmark_daily_df, bin_nav_df = _build_bin_outputs(
        merged=eval_res.merged,
        bins=bins,
        benchmark_daily=benchmark_daily_df,
    )
    if not bin_daily_df.empty:
        bin_daily_df.to_csv(trial_dir / "bin_daily_returns.csv", index=False)
    benchmark_daily_df.to_csv(trial_dir / "benchmark_daily_returns.csv", index=False)
    if not bin_nav_df.empty:
        bin_nav_df.to_csv(trial_dir / "bin_nav.csv", index=False)
        _plot_bin_nav(
            nav_df=bin_nav_df,
            out_path=trial_dir / "bin_nav.png",
            title=f"{trial_name} bin cumulative nav",
        )

    metric_frame, metric_summary, alpha_summary_df, alpha_heat_df = _build_trial_metrics_and_alpha(
        eval_daily=eval_res.daily,
        bin_daily=bin_daily_df,
        benchmark_daily=benchmark_daily_df,
        alpha_significance_window=alpha_significance_window,
    )
    if not metric_frame.empty:
        metric_frame.to_csv(trial_dir / "trial_daily_metrics.csv", index=False)
    pd.DataFrame([metric_summary]).to_csv(trial_dir / "trial_metrics.csv", index=False)
    alpha_summary_df.to_csv(trial_dir / "bin_alpha_significance.csv", index=False)
    if not alpha_heat_df.empty:
        alpha_heat_df.to_csv(trial_dir / "bin_alpha_significance_heatmap.csv")

    _plot_trial_report(
        trial_name=trial_name,
        trial_dir=trial_dir,
        merged=eval_res.merged,
        eval_daily=eval_res.daily,
        summary=eval_res.summary,
        nav_df=bin_nav_df,
        metric_summary=metric_summary,
        alpha_heat=alpha_heat_df,
        alpha_significance_window=alpha_significance_window,
    )

    print(
        f"[factor_select] trial={trial_name} done "
        f"rank_ic_mean={eval_res.summary.get('rank_ic_mean')} "
        f"samples={eval_res.summary.get('samples')}"
    )

    return _EvalTrial(
        trial_name=trial_name,
        trial_dir=trial_dir,
        score_output=score_output,
        state_dir=state_dir,
        factors=factors,
        summary=dict(eval_res.summary),
        daily=eval_res.daily,
    )


def _load_importance_from_state_dir(
    *,
    state_dir: Path,
    importance_type: str,
) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"lightgbm is required for factor importance export: {exc}") from exc

    if not state_dir.exists():
        return pd.DataFrame(columns=["factor", "importance_mean", "importance_std", "importance_nonzero_ratio", "checkpoints"])

    records: list[dict[str, Any]] = []
    checkpoints = sorted(state_dir.glob("*.txt"))
    for ckpt in checkpoints:
        try:
            booster = lgb.Booster(model_file=str(ckpt))
            names = list(booster.feature_name())
            importances = list(booster.feature_importance(importance_type=importance_type))
            for factor, imp in zip(names, importances):
                records.append(
                    {
                        "checkpoint": ckpt.stem,
                        "factor": str(factor),
                        "importance": float(imp),
                    }
                )
        except Exception as exc:
            print(f"[factor_select] skip checkpoint importance parse failed: {ckpt.name} ({type(exc).__name__}: {exc})")

    if not records:
        return pd.DataFrame(columns=["factor", "importance_mean", "importance_std", "importance_nonzero_ratio", "checkpoints"])

    df = pd.DataFrame(records)
    agg = (
        df.groupby("factor", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_nonzero_ratio=("importance", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            checkpoints=("checkpoint", "nunique"),
        )
        .sort_values(["importance_mean", "importance_nonzero_ratio"], ascending=[False, False], kind="mergesort")
        .reset_index(drop=True)
    )
    agg["importance_std"] = pd.to_numeric(agg["importance_std"], errors="coerce").fillna(0.0)
    return agg


def _plot_importance_topk(
    *,
    importance_df: pd.DataFrame,
    out_path: Path,
    top_k: int,
) -> None:
    if importance_df.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    top_k = max(1, int(top_k))
    top = importance_df.head(top_k).copy()
    top = top.iloc[::-1]

    fig_h = max(6, int(len(top) * 0.28) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(top["factor"], top["importance_mean"], color="#4C78A8", alpha=0.9)
    ax.set_xlabel("importance_mean")
    ax.set_title(f"Top {len(top)} Factor Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _numeric_delta_map(full_summary: dict[str, Any], topn_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted(set(full_summary.keys()) | set(topn_summary.keys()))
    for key in keys:
        fv = full_summary.get(key)
        tv = topn_summary.get(key)
        if isinstance(fv, (int, float)) and isinstance(tv, (int, float)):
            out[key] = float(tv) - float(fv)
    return out


def run(
    *,
    cfg: dict | None = None,
    config_name: str | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
) -> dict[str, Any]:
    selector_cfg = dict(cfg or load_config_file(config_name or "score/factor_select"))
    paths_cfg = load_config_file("paths")
    score_cfg = load_config_file(str(selector_cfg.get("model_score_config", "score/model_score")))
    factor_cfg = load_config_file(str(selector_cfg.get("factor_config", "factor")))

    model_id = str(
        selector_cfg.get("base_model_id")
        or score_cfg.get("model_id")
        or score_cfg.get("default_model_id", "")
    ).strip()
    if not model_id:
        raise ValueError("factor_select missing base_model_id")

    models = dict(score_cfg.get("models", {}))
    if model_id not in models:
        raise KeyError(f"base_model_id not found in model_score config: {model_id}")

    model_entry = dict(models[model_id])
    model_type = str(model_entry.get("model_type", "")).strip().lower()
    model_cfg_key = str(model_entry.get("model_config", "")).strip()
    if not model_type or not model_cfg_key:
        raise ValueError(f"invalid model entry for base model: {model_id}")
    if model_type != "lgbm":
        raise ValueError(f"factor_select importance_topn currently supports lgbm only, got: {model_type}")

    base_model_cfg_path = resolve_config_path(model_cfg_key)
    base_model_cfg = load_json_like(base_model_cfg_path)

    start_day = parse_date(start or selector_cfg.get("start") or base_model_cfg.get("start"))
    end_day = parse_date(end or selector_cfg.get("end") or base_model_cfg.get("end"))
    if start_day > end_day:
        raise ValueError("start must be <= end")

    label_root = Path(paths_cfg["label_data_root"])
    factor_time = str(base_model_cfg.get("factor_time", "14:30"))
    label_time = str(base_model_cfg.get("label_time", "14:42"))
    bins = int(selector_cfg.get("bins", base_model_cfg.get("bins", 5)))
    execution_override = dict(selector_cfg.get("execution", {}))

    baseline_factors = _load_factor_list(str(selector_cfg.get("baseline_factors_file", "score/factor_baseline_factors")))
    if not baseline_factors:
        raise ValueError("baseline factor list is empty")

    blacklist = set(_load_factor_list(str(selector_cfg.get("blacklist_file", "score/factor_blacklist"))) )

    candidate_all = expected_factor_columns_from_cfg(factor_cfg)
    baseline_set = set(baseline_factors)
    candidate_extra = [x for x in candidate_all if x not in baseline_set and x not in blacklist]
    max_candidates = int(selector_cfg.get("max_candidates", 0) or 0)
    if max_candidates > 0:
        candidate_extra = candidate_extra[:max_candidates]

    selection_cfg = dict(selector_cfg.get("selection", {}))
    pool_source = str(selection_cfg.get("pool_source", "baseline_plus_candidates")).strip().lower()
    if pool_source == "baseline_only":
        pool_factors = list(baseline_factors)
    elif pool_source == "candidates_only":
        pool_factors = [x for x in candidate_all if x not in blacklist]
    else:
        pool_factors = baseline_factors + candidate_extra
    pool_factors = _dedupe_keep_order(pool_factors)
    if not pool_factors:
        raise ValueError("factor_select pool factors is empty")

    mode = str(selection_cfg.get("mode", "importance_topn")).strip().lower()
    if mode != "importance_topn":
        raise ValueError(f"unsupported factor_select.selection.mode: {mode}")

    importance_type = str(selection_cfg.get("importance_type", "gain")).strip().lower()
    if importance_type not in {"gain", "split"}:
        raise ValueError("selection.importance_type must be one of: gain, split")

    top_n = int(selection_cfg.get("top_n", 10) or 10)
    min_importance = float(selection_cfg.get("min_importance", 0.0))
    plot_top_k = int(selection_cfg.get("plot_top_k", 30) or 30)
    alpha_significance_window = int(selector_cfg.get("alpha_significance_window", 40) or 40)

    results_root = Path(paths_cfg["results_root"])
    date_label = f"{start_day:%Y-%m-%d}_{end_day:%Y-%m-%d}"
    out_root = results_root / date_label / "Factor_Select" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    # stage 1: train with full pool
    full_dir = out_root / "stage1_full_pool"
    full_trial = _run_model_trial(
        trial_name="full_pool",
        trial_dir=full_dir,
        factors=pool_factors,
        model_id=model_id,
        model_type=model_type,
        base_model_cfg=base_model_cfg,
        score_cfg=score_cfg,
        execution_override=execution_override,
        start_day=start_day,
        end_day=end_day,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        bins=bins,
        save_state=True,
        alpha_significance_window=alpha_significance_window,
        raw_data_root=paths_cfg["raw_data_root"],
    )

    importance_df = _load_importance_from_state_dir(
        state_dir=full_trial.state_dir,
        importance_type=importance_type,
    )
    if importance_df.empty:
        raise RuntimeError(
            "failed to extract feature importance from state checkpoints; "
            "ensure rolling refit happened and incremental.save_state is enabled"
        )

    importance_df.to_csv(full_dir / "feature_importance.csv", index=False)
    _plot_importance_topk(
        importance_df=importance_df,
        out_path=full_dir / "feature_importance_topk.png",
        top_k=plot_top_k,
    )

    selected_df = importance_df[pd.to_numeric(importance_df["importance_mean"], errors="coerce") >= float(min_importance)].copy()
    if selected_df.empty:
        raise RuntimeError("no factor survives min_importance threshold")
    if top_n > 0:
        selected_df = selected_df.head(top_n)
    selected_factors = selected_df["factor"].astype(str).tolist()
    selected_factors = _dedupe_keep_order(selected_factors)
    if not selected_factors:
        raise RuntimeError("selected topN factors is empty")

    selected_payload = {
        "selection_mode": mode,
        "importance_type": importance_type,
        "min_importance": min_importance,
        "top_n": top_n,
        "selected_factor_count": len(selected_factors),
        "selected_factors": selected_factors,
    }
    (out_root / "selected_factors_topn.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # stage 2: retrain with topN factors
    topn_dir = out_root / "stage2_topn_retrain"
    topn_trial = _run_model_trial(
        trial_name="topn_retrain",
        trial_dir=topn_dir,
        factors=selected_factors,
        model_id=model_id,
        model_type=model_type,
        base_model_cfg=base_model_cfg,
        score_cfg=score_cfg,
        execution_override=execution_override,
        start_day=start_day,
        end_day=end_day,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        bins=bins,
        save_state=False,
        alpha_significance_window=alpha_significance_window,
        raw_data_root=paths_cfg["raw_data_root"],
    )

    compare_payload = {
        "model_id": model_id,
        "start": str(start_day),
        "end": str(end_day),
        "pool_factor_count": len(pool_factors),
        "selected_factor_count": len(selected_factors),
        "full_pool_summary": full_trial.summary,
        "topn_summary": topn_trial.summary,
        "delta_topn_minus_full": _numeric_delta_map(full_trial.summary, topn_trial.summary),
    }
    (out_root / "compare_summary.json").write_text(
        json.dumps(compare_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    snapshot_payload = {
        "selector_config": selector_cfg,
        "model_score_config_key": str(selector_cfg.get("model_score_config", "score/model_score")),
        "base_model_config_path": str(base_model_cfg_path.as_posix()),
        "baseline_factors_file": str(selector_cfg.get("baseline_factors_file")),
        "blacklist_file": str(selector_cfg.get("blacklist_file")),
        "baseline_factors": baseline_factors,
        "candidate_extra": candidate_extra,
        "pool_factors": pool_factors,
        "blacklist": sorted(list(blacklist)),
    }
    (out_root / "run_config_snapshot.json").write_text(
        json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_root": str(out_root.as_posix()),
        "model_id": model_id,
        "pool_factors": len(pool_factors),
        "selected_factors": len(selected_factors),
        "stage1_rank_ic_mean": full_trial.summary.get("rank_ic_mean"),
        "stage2_rank_ic_mean": topn_trial.summary.get("rank_ic_mean"),
    }
