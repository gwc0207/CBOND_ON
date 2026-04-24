from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.core.utils import progress
from cbond_on.infra.factors.pipeline import run_factor_pipeline
from cbond_on.infra.factors.quality import load_factor_specs_from_cfg, resolve_disabled_factor_names
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.report.factor_report import save_single_factor_report


@dataclass
class FactorBacktestResult:
    returns: pd.Series
    nav: pd.Series
    ic: pd.Series
    rank_ic: pd.Series
    bin_returns: pd.DataFrame
    daily_stats: pd.DataFrame
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass
class _BacktestDayPrepared:
    trade_day: date
    reason: str
    merged: pd.DataFrame | None = None
    benchmark_universe: pd.DataFrame | None = None
    joined_count: int = 0
    valid_count: int = 0


def build_signal_specs(cfg: dict) -> list[FactorSpec]:
    specs = load_factor_specs_from_cfg(cfg)
    disabled = resolve_disabled_factor_names(cfg)
    if disabled:
        print(
            "factor disabled set:",
            f"count={len(disabled)}",
        )
    return specs


def _read_label_day(label_root: Path, day: date, *, factor_time: str, label_time: str) -> pd.DataFrame:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = label_root / month / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    if "trade_time" in df.columns:
        dt_series = pd.to_datetime(df["trade_time"], errors="coerce")
    elif "dt" in df.columns:
        dt_series = pd.to_datetime(df["dt"], errors="coerce")
    else:
        return pd.DataFrame()
    df["dt"] = dt_series

    # filter by label_time and align dt to factor_time
    try:
        label_h, label_m = map(int, label_time.split(":"))
        factor_h, factor_m = map(int, factor_time.split(":"))
    except Exception:
        return pd.DataFrame()
    label_t = dt_time(label_h, label_m)
    df = df[df["dt"].dt.time == label_t]
    if df.empty:
        return df
    base_date = df["dt"].dt.normalize()
    df["dt"] = base_date + pd.Timedelta(hours=factor_h, minutes=factor_m)
    return df


def _iter_existing_label_days(label_root: Path, start: date, end: date) -> Iterable[date]:
    seen: set[date] = set()
    for path in label_root.rglob("*.parquet"):
        stem = path.stem
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except Exception:
            continue
        if start <= day <= end:
            seen.add(day)
    for day in sorted(seen):
        yield day


def _prepare_backtest_day(
    *,
    day: date,
    factor_store: FactorStore,
    label_root: Path,
    factor_col: str,
    factor_time: str,
    label_time: str,
) -> _BacktestDayPrepared:
    label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
    if label_df.empty:
        return _BacktestDayPrepared(trade_day=day, reason="missing_label")
    benchmark_universe = label_df[["dt", "code", "y"]].dropna(subset=["y"])
    if benchmark_universe.empty:
        return _BacktestDayPrepared(trade_day=day, reason="missing_label")

    factor_df = factor_store.read_day(day)
    if factor_df.empty or factor_col not in factor_df.columns:
        return _BacktestDayPrepared(
            trade_day=day,
            reason="missing_factor",
            benchmark_universe=benchmark_universe,
            joined_count=0,
            valid_count=0,
        )

    factor_df = factor_df.reset_index()
    joined = factor_df.merge(label_df, on=["dt", "code"], how="inner")
    joined = joined[[factor_col, "y", "dt", "code"]]
    joined_count = int(len(joined))
    merged = joined.dropna()
    valid_count = int(len(merged))
    if merged.empty:
        return _BacktestDayPrepared(
            trade_day=day,
            reason="merged_empty",
            benchmark_universe=benchmark_universe,
            joined_count=joined_count,
            valid_count=valid_count,
        )
    return _BacktestDayPrepared(
        trade_day=day,
        reason="",
        merged=merged,
        benchmark_universe=benchmark_universe,
        joined_count=joined_count,
        valid_count=valid_count,
    )


def _select_bins_by_mean_return_intraday(
    *,
    data: pd.DataFrame,
    factor_col: str,
    bin_count: int,
) -> list[int]:
    if data.empty:
        return []
    per_bin_returns: dict[int, list[float]] = {}
    for _, group in data.groupby("dt", sort=True):
        g = group[[factor_col, "y"]].dropna()
        if g.empty:
            continue
        try:
            bins_cat = pd.qcut(
                g[factor_col],
                bin_count,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        if bins_cat.dropna().empty:
            continue
        for bin_id in bins_cat.dropna().unique():
            mask = bins_cat == bin_id
            if mask.sum() == 0:
                continue
            per_bin_returns.setdefault(int(bin_id), []).append(float(g.loc[mask, "y"].mean()))
    if not per_bin_returns:
        return []
    mean_ret = {k: float(pd.Series(v).mean()) for k, v in per_bin_returns.items() if v}
    ranked = sorted(mean_ret.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked]


def run_intraday_factor_backtest(
    factor_store: FactorStore,
    label_root: Path,
    start: date,
    end: date,
    *,
    factor_col: str,
    factor_time: str,
    label_time: str,
    min_count: int = 30,
    ic_bins: int = 5,
    bin_count: int | None = None,
    bin_select: list[int] | None = None,
    bin_source: str = "manual",
    bin_top_k: int = 1,
    bin_lookback_days: int = 60,
    workers: int = 1,
) -> FactorBacktestResult:
    rows = []
    benchmark_rows = []
    trade_returns_rows: list[float] = []
    diagnostics: list[dict] = []
    factor_joined_total = 0
    factor_valid_total = 0
    days = list(_iter_existing_label_days(label_root, start, end))
    workers = max(1, int(workers))
    if workers <= 1:
        for day in progress(days, desc="factor_backtest", unit="day", total=len(days)):
            prepared = _prepare_backtest_day(
                day=day,
                factor_store=factor_store,
                label_root=label_root,
                factor_col=factor_col,
                factor_time=factor_time,
                label_time=label_time,
            )
            factor_joined_total += int(prepared.joined_count)
            factor_valid_total += int(prepared.valid_count)
            if prepared.benchmark_universe is not None and not prepared.benchmark_universe.empty:
                benchmark_rows.append(prepared.benchmark_universe)
            if prepared.merged is None:
                diagnostics.append({"trade_date": day, "status": "skip", "reason": prepared.reason})
                continue
            diagnostics.append({"trade_date": day, "status": "ok", "count": len(prepared.merged)})
            rows.append(prepared.merged)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_day = {
                executor.submit(
                    _prepare_backtest_day,
                    day=day,
                    factor_store=factor_store,
                    label_root=label_root,
                    factor_col=factor_col,
                    factor_time=factor_time,
                    label_time=label_time,
                ): day
                for day in days
            }
            for future in progress(
                as_completed(future_to_day),
                desc="factor_backtest",
                unit="day",
                total=len(future_to_day),
            ):
                day = future_to_day[future]
                try:
                    prepared = future.result()
                except Exception as exc:
                    raise RuntimeError(f"factor_backtest failed on {day}") from exc
                factor_joined_total += int(prepared.joined_count)
                factor_valid_total += int(prepared.valid_count)
                if prepared.benchmark_universe is not None and not prepared.benchmark_universe.empty:
                    benchmark_rows.append(prepared.benchmark_universe)
                if prepared.merged is None:
                    diagnostics.append(
                        {"trade_date": prepared.trade_day, "status": "skip", "reason": prepared.reason}
                    )
                    continue
                diagnostics.append(
                    {"trade_date": prepared.trade_day, "status": "ok", "count": len(prepared.merged)}
                )
                rows.append(prepared.merged)
    if not rows:
        empty = pd.Series(dtype=float)
        result = FactorBacktestResult(
            empty,
            empty,
            empty,
            empty,
            pd.DataFrame(),
            pd.DataFrame(),
            benchmark_returns=empty,
            benchmark_nav=empty,
        )
        result.trade_returns = empty  # type: ignore[attr-defined]
        result.factor_values = empty  # type: ignore[attr-defined]
        result.diagnostics = pd.DataFrame(diagnostics)  # type: ignore[attr-defined]
        result.factor_joined_total = int(factor_joined_total)  # type: ignore[attr-defined]
        result.factor_valid_total = int(factor_valid_total)  # type: ignore[attr-defined]
        result.factor_nan_ratio = (
            float((factor_joined_total - factor_valid_total) / factor_joined_total)
            if factor_joined_total > 0
            else float("nan")
        )  # type: ignore[attr-defined]
        result.bin_ok = 0  # type: ignore[attr-defined]
        result.bin_fail = 0  # type: ignore[attr-defined]
        return result

    data = pd.concat(rows, ignore_index=True)
    benchmark_data = (
        pd.concat(benchmark_rows, ignore_index=True)
        if benchmark_rows
        else pd.DataFrame(columns=["dt", "code", "y"])
    )
    benchmark_by_dt = (
        benchmark_data.groupby("dt", sort=True)["y"].mean()
        if not benchmark_data.empty
        else pd.Series(dtype=float)
    )

    def _calc_day(group: pd.DataFrame) -> dict:
        g = group.dropna()
        if len(g) < min_count:
            return {"ret": pd.NA, "ic": pd.NA, "rank_ic": pd.NA, "count": len(g)}
        g = g.sort_values(factor_col)
        top = g.tail(top_k)
        bottom = g.head(top_k)
        ret = top["y"].mean() - bottom["y"].mean()
        ic = g[factor_col].corr(g["y"], method="pearson")
        rank_ic = g[factor_col].corr(g["y"], method="spearman")
        return {"ret": ret, "ic": ic, "rank_ic": rank_ic, "count": len(g)}

    daily_records: list[dict] = []
    holdings: set[str] = set()
    bin_source = str(bin_source or "manual").lower()
    bin_count = int(bin_count) if bin_count is not None else int(ic_bins)
    if bin_count <= 1:
        bin_count = 2
    bin_select = list(bin_select) if bin_select is not None else None
    bin_top_k = max(1, int(bin_top_k))
    bin_lookback_days = max(1, int(bin_lookback_days))
    if bin_source == "auto":
        recent = data
        if bin_lookback_days > 0:
            recent_days = sorted(data["dt"].unique())
            recent_days = recent_days[-bin_lookback_days:] if len(recent_days) > bin_lookback_days else recent_days
            recent = data[data["dt"].isin(recent_days)]
        ranked_bins = _select_bins_by_mean_return_intraday(
            data=recent,
            factor_col=factor_col,
            bin_count=bin_count,
        )
        if ranked_bins:
            bin_select = ranked_bins[:bin_top_k]

    for dt, group in data.groupby("dt", sort=True):
        g = group[[factor_col, "y", "code"]].dropna()
        benchmark_ret = float(benchmark_by_dt.loc[dt]) if dt in benchmark_by_dt.index else pd.NA
        if len(g) < min_count:
            diagnostics.append(
                {
                    "trade_date": dt,
                    "status": "skip",
                    "reason": "min_count_not_met",
                    "count": int(len(g)),
                }
            )
            daily_records.append(
                {
                    "dt": dt,
                    "ret": pd.NA,
                    "benchmark_return": benchmark_ret,
                    "ic": pd.NA,
                    "rank_ic": pd.NA,
                    "count": len(g),
                }
            )
            continue

        ic = g[factor_col].corr(g["y"], method="pearson")
        rank_ic = g[factor_col].corr(g["y"], method="spearman")

        # bin-based selection (aligned with cbond_day style)
        try:
            bins_cat = pd.qcut(
                g[factor_col],
                bin_count,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            diagnostics.append(
                {
                    "trade_date": dt,
                    "status": "skip",
                    "reason": "binning_failed",
                }
            )
            daily_records.append(
                {
                    "dt": dt,
                    "ret": pd.NA,
                    "benchmark_return": benchmark_ret,
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "count": len(g),
                }
            )
            continue
        available_bins = sorted(bins_cat.dropna().unique().tolist())
        if not available_bins:
            diagnostics.append(
                {
                    "trade_date": dt,
                    "status": "skip",
                    "reason": "binning_failed",
                }
            )
            daily_records.append(
                {
                    "dt": dt,
                    "ret": pd.NA,
                    "benchmark_return": benchmark_ret,
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "count": len(g),
                }
            )
            continue
        n_bins = len(available_bins)

        effective_bins = bin_select
        if bin_source == "auto":
            effective_bins = bin_select
        if effective_bins is None:
            effective_bins = [max(available_bins)]
            if bin_top_k > 1:
                effective_bins = sorted(available_bins, reverse=True)[:bin_top_k]
        if max(effective_bins) >= n_bins:
            diagnostics.append(
                {
                    "trade_date": dt,
                    "status": "skip",
                    "reason": "bin_select_out_of_range",
                    "bin_used": ",".join(str(x) for x in effective_bins),
                    "bin_count_actual": int(n_bins),
                }
            )
            daily_records.append(
                {
                    "dt": dt,
                    "ret": pd.NA,
                    "benchmark_return": benchmark_ret,
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "count": len(g),
                }
            )
            continue
        picks = g[bins_cat.isin(effective_bins)]
        if len(picks) < min_count:
            diagnostics.append(
                {
                    "trade_date": dt,
                    "status": "skip",
                    "reason": "min_count_not_met",
                    "picked": int(len(picks)),
                    "bin_used": ",".join(str(x) for x in effective_bins),
                    "bin_count_actual": int(n_bins),
                }
            )
            daily_records.append(
                {
                    "dt": dt,
                    "ret": pd.NA,
                    "benchmark_return": benchmark_ret,
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "count": len(g),
                }
            )
            continue
        ret = picks["y"].mean()
        picks_y = pd.to_numeric(picks["y"], errors="coerce").dropna()
        if not picks_y.empty:
            trade_returns_rows.extend(float(v) for v in picks_y.to_numpy())

        diagnostics.append(
            {
                "trade_date": dt,
                "status": "ok",
                "reason": "",
                "bin_used": ",".join(str(x) for x in effective_bins),
                "bin_count_actual": int(n_bins),
                "picked": int(len(picks)),
                "count": int(len(g)),
            }
        )
        daily_records.append(
            {
                "dt": dt,
                "ret": ret,
                "benchmark_return": benchmark_ret,
                "ic": ic,
                "rank_ic": rank_ic,
                "count": len(g),
            }
        )

    daily = pd.DataFrame(daily_records).set_index("dt")
    returns = pd.to_numeric(daily["ret"], errors="coerce")
    benchmark_returns = pd.to_numeric(benchmark_by_dt, errors="coerce").dropna().sort_index()
    valid_mask = returns.notna()
    returns = returns.loc[valid_mask]
    ic = pd.to_numeric(daily["ic"], errors="coerce").dropna()
    rank_ic = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    benchmark_nav = (1.0 + benchmark_returns.fillna(0.0)).cumprod()

    def _bin_ret(group: pd.DataFrame) -> pd.Series:
        g = group.dropna()
        n = len(g)
        if n < 2:
            return pd.Series(dtype=float)
        bins_target = ic_bins
        if n < max(min_count, ic_bins):
            bins_target = max(2, min(ic_bins, n // 5))
        if bins_target < 2:
            return pd.Series(dtype=float)
        vals = g[factor_col]
        try:
            bins = pd.qcut(vals, bins_target, labels=False, duplicates="drop")
            if bins.nunique() < 2:
                raise ValueError("bins insufficient")
        except Exception:
            ranked = vals.rank(pct=True, method="average")
            try:
                bins = pd.qcut(ranked, bins_target, labels=False, duplicates="drop")
            except Exception:
                return pd.Series(dtype=float)
        return g.groupby(bins)["y"].mean()

    bin_rows: list[pd.Series] = []
    bin_fail = 0
    for dt, group in data.groupby("dt", sort=True):
        s = _bin_ret(group[[factor_col, "y"]])
        if s is None or s.empty or s.isna().all():
            bin_fail += 1
            continue
        s = s.astype(float)
        s.name = dt
        bin_rows.append(s)
    if bin_rows:
        bin_returns = pd.DataFrame(bin_rows).sort_index()
    else:
        bin_returns = pd.DataFrame()
    daily_stats = daily.reset_index().rename(columns={"dt": "trade_time"})
    trade_returns = pd.to_numeric(pd.Series(trade_returns_rows, dtype=float), errors="coerce").dropna()
    factor_values = pd.to_numeric(data.get(factor_col), errors="coerce").dropna()
    result = FactorBacktestResult(
        returns=returns,
        nav=nav,
        ic=ic,
        rank_ic=rank_ic,
        bin_returns=bin_returns,
        daily_stats=daily_stats,
        benchmark_returns=benchmark_returns,
        benchmark_nav=benchmark_nav,
    )
    result.trade_returns = trade_returns  # type: ignore[attr-defined]
    result.factor_values = factor_values  # type: ignore[attr-defined]
    result.diagnostics = pd.DataFrame(diagnostics)  # type: ignore[attr-defined]
    result.bin_ok = len(bin_rows)  # type: ignore[attr-defined]
    result.bin_fail = bin_fail  # type: ignore[attr-defined]
    result.factor_joined_total = int(factor_joined_total)  # type: ignore[attr-defined]
    result.factor_valid_total = int(factor_valid_total)  # type: ignore[attr-defined]
    result.factor_nan_ratio = (
        float((factor_joined_total - factor_valid_total) / factor_joined_total)
        if factor_joined_total > 0
        else float("nan")
    )  # type: ignore[attr-defined]
    return result


def _load_screening_config(cfg: dict) -> dict:
    raw = cfg.get("screening", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "ic_metric": str(raw.get("ic_metric", "rank_ic_mean")),
        "ir_metric": str(raw.get("ir_metric", "rank_ic_ir")),
        "ic_abs_min": float(raw.get("ic_abs_min", 0.0)),
        "ir_abs_min": float(raw.get("ir_abs_min", 0.0)),
        "sharpe_min": float(raw.get("sharpe_min", 0.1)),
        "copy_reports": bool(raw.get("copy_reports", True)),
    }


def _load_bad_factor_report_config(cfg: dict) -> dict:
    raw = cfg.get("bad_factor_report", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "nan_ratio_threshold": float(raw.get("nan_ratio_threshold", 0.20)),
        "bin_fail_ratio_threshold": float(raw.get("bin_fail_ratio_threshold", 0.20)),
        "skip_ratio_threshold": float(raw.get("skip_ratio_threshold", 0.30)),
        "top_n_plot": int(raw.get("top_n_plot", 30)),
        "min_days": int(raw.get("min_days", 20)),
        "required_bin_count": int(raw.get("required_bin_count", 20)),
        "mark_bad_if_any_insufficient_bin_day": bool(raw.get("mark_bad_if_any_insufficient_bin_day", True)),
    }


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _build_screening_row(
    *,
    factor_name: str,
    factor_col: str,
    summary: dict,
    screening_cfg: dict,
) -> dict:
    ic_metric = screening_cfg["ic_metric"]
    ir_metric = screening_cfg["ir_metric"]
    ic_val = _to_float(summary.get(ic_metric))
    ir_val = _to_float(summary.get(ir_metric))
    sharpe = _to_float(summary.get("sharpe"))

    pass_ic = pd.notna(ic_val) and abs(ic_val) >= float(screening_cfg["ic_abs_min"])
    pass_ir = pd.notna(ir_val) and abs(ir_val) >= float(screening_cfg["ir_abs_min"])
    pass_sharpe = pd.notna(sharpe) and sharpe >= float(screening_cfg["sharpe_min"])
    passed = bool(pass_ic and pass_ir and pass_sharpe)

    failed_rules: list[str] = []
    if not pass_ic:
        failed_rules.append("ic")
    if not pass_ir:
        failed_rules.append("ir")
    if not pass_sharpe:
        failed_rules.append("sharpe")

    row = {
        "factor_name": factor_name,
        "factor_col": factor_col,
        "ic_metric": ic_metric,
        "ir_metric": ir_metric,
        "ic_metric_value": ic_val,
        "ir_metric_value": ir_val,
        "sharpe": sharpe,
        "ic_abs_min": float(screening_cfg["ic_abs_min"]),
        "ir_abs_min": float(screening_cfg["ir_abs_min"]),
        "sharpe_min": float(screening_cfg["sharpe_min"]),
        "pass_ic": bool(pass_ic),
        "pass_ir": bool(pass_ir),
        "pass_sharpe": bool(pass_sharpe),
        "passed": passed,
        "failed_rules": ",".join(failed_rules),
    }
    for key, value in summary.items():
        if key not in row:
            row[key] = value
    return row


def _build_bad_factor_row(
    *,
    factor_name: str,
    factor_col: str,
    result: FactorBacktestResult,
    cfg: dict,
) -> dict:
    diag = getattr(result, "diagnostics", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame):
        diag = pd.DataFrame()
    total_days = int(len(diag))
    if total_days > 0 and "status" in diag.columns:
        skip_days = int((diag["status"].astype(str) != "ok").sum())
    else:
        skip_days = 0
    skip_ratio = float(skip_days / total_days) if total_days > 0 else float("nan")

    if total_days > 0 and "reason" in diag.columns:
        reason_series = diag["reason"].fillna("").astype(str)
        bin_reason_mask = reason_series.isin({"binning_failed", "bin_select_out_of_range", "min_count_not_met"})
        bin_standard_fail_days = int(bin_reason_mask.sum())
    else:
        bin_standard_fail_days = 0
    bin_standard_fail_ratio = (
        float(bin_standard_fail_days / total_days) if total_days > 0 else float("nan")
    )

    bin_ok = int(getattr(result, "bin_ok", 0) or 0)
    bin_fail = int(getattr(result, "bin_fail", 0) or 0)
    bin_total = int(bin_ok + bin_fail)
    bin_fail_ratio = float(bin_fail / bin_total) if bin_total > 0 else float("nan")

    required_bin_count = int(max(2, cfg.get("required_bin_count", 20)))
    bin_returns = getattr(result, "bin_returns", pd.DataFrame())
    if isinstance(bin_returns, pd.DataFrame) and not bin_returns.empty:
        active_bins_by_day = pd.to_numeric(bin_returns.notna().sum(axis=1), errors="coerce").fillna(0).astype(int)
        insufficient_from_ok = int((active_bins_by_day < required_bin_count).sum())
        bin_eval_days = int(len(active_bins_by_day))
    else:
        insufficient_from_ok = 0
        bin_eval_days = 0
    # Bin failures imply no valid bin panel for that day; count them as insufficient too.
    insufficient_bin_days = int(insufficient_from_ok + max(0, bin_fail))
    insufficient_bin_ratio = (
        float(insufficient_bin_days / max(1, total_days)) if total_days > 0 else float("nan")
    )

    nan_ratio = _to_float(getattr(result, "factor_nan_ratio", float("nan")))
    joined_total = int(getattr(result, "factor_joined_total", 0) or 0)
    valid_total = int(getattr(result, "factor_valid_total", 0) or 0)

    nan_bad = pd.notna(nan_ratio) and nan_ratio >= float(cfg["nan_ratio_threshold"])
    bin_bad = (
        pd.notna(bin_fail_ratio)
        and bin_total >= int(cfg["min_days"])
        and bin_fail_ratio >= float(cfg["bin_fail_ratio_threshold"])
    ) or (
        pd.notna(bin_standard_fail_ratio)
        and total_days >= int(cfg["min_days"])
        and bin_standard_fail_ratio >= float(cfg["bin_fail_ratio_threshold"])
    )
    skip_bad = (
        pd.notna(skip_ratio)
        and total_days >= int(cfg["min_days"])
        and skip_ratio >= float(cfg["skip_ratio_threshold"])
    )
    insufficient_bin_bad = (
        bool(cfg.get("mark_bad_if_any_insufficient_bin_day", True))
        and total_days >= int(cfg["min_days"])
        and insufficient_bin_days > 0
    )
    is_bad = bool(nan_bad or bin_bad or skip_bad or insufficient_bin_bad)

    reasons: list[str] = []
    if nan_bad:
        reasons.append("high_nan_ratio")
    if bin_bad:
        reasons.append("bin_standard_not_met")
    if skip_bad:
        reasons.append("high_skip_ratio")
    if insufficient_bin_bad:
        reasons.append("insufficient_bins_any_day")

    return {
        "factor_name": factor_name,
        "factor_col": factor_col,
        "total_days": total_days,
        "skip_days": skip_days,
        "skip_ratio": skip_ratio,
        "bin_standard_fail_days": bin_standard_fail_days,
        "bin_standard_fail_ratio": bin_standard_fail_ratio,
        "bin_ok": bin_ok,
        "bin_fail": bin_fail,
        "bin_fail_ratio": bin_fail_ratio,
        "required_bin_count": required_bin_count,
        "bin_eval_days": bin_eval_days,
        "insufficient_bin_days": insufficient_bin_days,
        "insufficient_bin_ratio": insufficient_bin_ratio,
        "factor_joined_total": joined_total,
        "factor_valid_total": valid_total,
        "nan_ratio": nan_ratio,
        "bad_nan": bool(nan_bad),
        "bad_bin": bool(bin_bad),
        "bad_skip": bool(skip_bad),
        "bad_insufficient_bins": bool(insufficient_bin_bad),
        "is_bad": bool(is_bad),
        "bad_reasons": ",".join(reasons),
    }


def _write_bad_factor_outputs(out_root: Path, *, cfg: dict, rows: list[dict]) -> None:
    bad_root = out_root / "bad_factors"
    bad_root.mkdir(parents=True, exist_ok=True)
    (bad_root / "bad_factor_report_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        all_df = pd.DataFrame(
            columns=[
                "factor_name",
                "factor_col",
                "total_days",
                "skip_days",
                "skip_ratio",
                "bin_standard_fail_days",
                "bin_standard_fail_ratio",
                "bin_ok",
                "bin_fail",
                "bin_fail_ratio",
                "required_bin_count",
                "bin_eval_days",
                "insufficient_bin_days",
                "insufficient_bin_ratio",
                "factor_joined_total",
                "factor_valid_total",
                "nan_ratio",
                "bad_nan",
                "bad_bin",
                "bad_skip",
                "bad_insufficient_bins",
                "is_bad",
                "bad_reasons",
            ]
        )
    else:
        all_df["is_bad"] = all_df["is_bad"].astype(bool)
        all_df = all_df.sort_values(
            by=["is_bad", "nan_ratio", "bin_standard_fail_ratio", "skip_ratio"],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
    all_df.to_csv(bad_root / "factor_bad_quality_all.csv", index=False)

    bad_df = all_df[all_df["is_bad"]].copy() if "is_bad" in all_df.columns else pd.DataFrame()
    bad_df.to_csv(bad_root / "bad_factor_list.csv", index=False)
    (bad_root / "bad_factor_list.txt").write_text(
        "\n".join(bad_df["factor_name"].astype(str).tolist()) if not bad_df.empty else "",
        encoding="utf-8",
    )

    summary = {
        "total_factors": int(len(all_df)),
        "bad_factors": int(len(bad_df)),
        "bad_factor_ratio": float(len(bad_df) / len(all_df)) if len(all_df) > 0 else 0.0,
        "thresholds": {
            "nan_ratio_threshold": float(cfg["nan_ratio_threshold"]),
            "bin_fail_ratio_threshold": float(cfg["bin_fail_ratio_threshold"]),
            "skip_ratio_threshold": float(cfg["skip_ratio_threshold"]),
            "min_days": int(cfg["min_days"]),
            "required_bin_count": int(cfg.get("required_bin_count", 20)),
            "mark_bad_if_any_insufficient_bin_day": bool(
                cfg.get("mark_bad_if_any_insufficient_bin_day", True)
            ),
        },
    }
    (bad_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top_n = int(max(1, cfg.get("top_n_plot", 30)))

    plot_df_nan = all_df.sort_values("nan_ratio", ascending=False).head(top_n) if not all_df.empty else pd.DataFrame()
    if not plot_df_nan.empty:
        colors = ["#E45756" if bool(v) else "#4C78A8" for v in plot_df_nan["bad_nan"].tolist()]
        axes[0].bar(plot_df_nan["factor_name"].astype(str), pd.to_numeric(plot_df_nan["nan_ratio"], errors="coerce"), color=colors)
        axes[0].axhline(float(cfg["nan_ratio_threshold"]), color="#E45756", linestyle="--", linewidth=1.0)
        axes[0].set_title("Top NaN Ratio Factors")
        axes[0].tick_params(axis="x", labelrotation=75, labelsize=7)
        axes[0].grid(True, axis="y", alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No factor rows", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Top NaN Ratio Factors")

    plot_df_bin = (
        all_df.sort_values("bin_standard_fail_ratio", ascending=False).head(top_n)
        if not all_df.empty
        else pd.DataFrame()
    )
    if not plot_df_bin.empty:
        colors = ["#E45756" if bool(v) else "#72B7B2" for v in plot_df_bin["bad_bin"].tolist()]
        axes[1].bar(
            plot_df_bin["factor_name"].astype(str),
            pd.to_numeric(plot_df_bin["bin_standard_fail_ratio"], errors="coerce"),
            color=colors,
        )
        axes[1].axhline(float(cfg["bin_fail_ratio_threshold"]), color="#E45756", linestyle="--", linewidth=1.0)
        axes[1].set_title("Top Bin Standard Fail Ratio Factors")
        axes[1].tick_params(axis="x", labelrotation=75, labelsize=7)
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No factor rows", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Top Bin Standard Fail Ratio Factors")

    fig.tight_layout()
    fig.savefig(bad_root / "bad_factor_metrics.png", dpi=150)
    plt.close(fig)


def _write_screening_outputs(out_root: Path, *, screening_cfg: dict, rows: list[dict]) -> None:
    screened_root = out_root / "screened"
    screened_root.mkdir(parents=True, exist_ok=True)
    (screened_root / "screening_config.json").write_text(
        json.dumps(screening_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        columns = [
            "factor_name",
            "factor_col",
            "ic_metric",
            "ir_metric",
            "ic_metric_value",
            "ir_metric_value",
            "sharpe",
            "passed",
            "failed_rules",
        ]
        all_df = pd.DataFrame(columns=columns)
    else:
        all_df["abs_ic_metric"] = pd.to_numeric(all_df["ic_metric_value"], errors="coerce").abs()
        all_df["abs_ir_metric"] = pd.to_numeric(all_df["ir_metric_value"], errors="coerce").abs()
        all_df["sharpe"] = pd.to_numeric(all_df["sharpe"], errors="coerce")
        all_df["passed"] = all_df["passed"].astype(bool)
        all_df = all_df.sort_values(
            by=["passed", "sharpe", "abs_ic_metric", "abs_ir_metric"],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
    all_df.to_csv(screened_root / "factor_screening_all.csv", index=False)

    shortlist = all_df[all_df["passed"]].copy() if "passed" in all_df.columns else pd.DataFrame()
    shortlist.to_csv(screened_root / "factor_shortlist.csv", index=False)

    if bool(screening_cfg.get("copy_reports", True)) and not shortlist.empty:
        selected_root = screened_root / "selected_reports"
        selected_root.mkdir(parents=True, exist_ok=True)
        for factor_name in shortlist["factor_name"].astype(str).tolist():
            src = out_root / factor_name / "factor_report.png"
            dst = selected_root / f"{factor_name}.png"
            if src.exists():
                shutil.copy2(src, dst)


def _collect_report_plots(out_root: Path, *, bad_factor_names: set[str] | None = None) -> Path:
    # Aggregate current run's factor report images into one folder and split into good/bad sets.
    plot_root = out_root / "plot"
    plot_good_root = out_root / "plot_good"
    plot_bad_root = out_root / "plot_bad"
    plot_root.mkdir(parents=True, exist_ok=True)
    plot_good_root.mkdir(parents=True, exist_ok=True)
    plot_bad_root.mkdir(parents=True, exist_ok=True)

    bad_names = set(bad_factor_names or set())
    copied_rows_all: list[dict[str, str]] = []
    copied_rows_good: list[dict[str, str]] = []
    copied_rows_bad: list[dict[str, str]] = []
    for signal_dir in sorted(out_root.iterdir(), key=lambda p: p.name):
        if not signal_dir.is_dir():
            continue
        src = signal_dir / "factor_report.png"
        if not src.exists():
            continue
        factor_name = signal_dir.name
        status = "bad" if factor_name in bad_names else "good"
        dst_all = plot_root / f"{factor_name}.png"
        shutil.copy2(src, dst_all)
        copied_rows_all.append(
            {
                "factor_name": factor_name,
                "status": status,
                "source": str(src),
                "target": str(dst_all),
            }
        )
        if status == "bad":
            dst_bad = plot_bad_root / f"{factor_name}.png"
            shutil.copy2(src, dst_bad)
            copied_rows_bad.append(
                {
                    "factor_name": factor_name,
                    "status": status,
                    "source": str(src),
                    "target": str(dst_bad),
                }
            )
        else:
            dst_good = plot_good_root / f"{factor_name}.png"
            shutil.copy2(src, dst_good)
            copied_rows_good.append(
                {
                    "factor_name": factor_name,
                    "status": status,
                    "source": str(src),
                    "target": str(dst_good),
                }
            )

    pd.DataFrame(copied_rows_all, columns=["factor_name", "status", "source", "target"]).to_csv(
        plot_root / "index.csv",
        index=False,
    )
    pd.DataFrame(copied_rows_good, columns=["factor_name", "status", "source", "target"]).to_csv(
        plot_good_root / "index.csv",
        index=False,
    )
    pd.DataFrame(copied_rows_bad, columns=["factor_name", "status", "source", "target"]).to_csv(
        plot_bad_root / "index.csv",
        index=False,
    )
    return plot_root


def run_factor_batch(
    cfg: dict,
    *,
    panel_data_root: str | Path,
    factor_data_root: str | Path,
    label_data_root: str | Path,
    raw_data_root: str | Path,
    results_root: str | Path,
    start: date,
    end: date,
    window_minutes: int,
    panel_name: str | None,
    refresh: bool,
    overwrite: bool,
    specs: Sequence[FactorSpec],
) -> Path:
    panel_name_text = str(panel_name or "").strip()
    if not panel_name_text:
        raise ValueError("factor_config.panel_name is required; window_minutes fallback is disabled")
    workers = int(cfg.get("workers", 1))
    factor_workers = int(cfg.get("factor_workers", 1))
    run_factor_pipeline(
        panel_data_root,
        factor_data_root,
        start,
        end,
        window_minutes=window_minutes,
        panel_name=panel_name_text,
        refresh=refresh,
        overwrite=overwrite,
        workers=workers,
        factor_workers=factor_workers,
        raw_data_root=raw_data_root,
        context_cfg=cfg.get("context"),
        compute_cfg=cfg.get("compute"),
        specs=specs,
    )

    results_root = Path(results_root)
    date_label = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = results_root / date_label / "Single_Factor" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    factor_store = FactorStore(Path(factor_data_root), panel_name=panel_name_text, window_minutes=window_minutes)
    backtest_cfg = cfg.get("backtest", {})
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:42"))
    min_count = int(backtest_cfg.get("min_count", 30))
    ic_bins = int(backtest_cfg.get("ic_bins", 5))
    bin_count = backtest_cfg.get("bin_count")
    bin_select = backtest_cfg.get("bin_select")
    bin_source = backtest_cfg.get("bin_source", "manual")
    bin_top_k = int(backtest_cfg.get("bin_top_k", 1))
    bin_lookback_days = int(backtest_cfg.get("bin_lookback_days", 60))
    alpha_significance_window = int(backtest_cfg.get("alpha_significance_window", 40))
    backtest_workers = int(backtest_cfg.get("workers", 1))
    screening_cfg = _load_screening_config(cfg)
    bad_factor_cfg = _load_bad_factor_report_config(cfg)
    screening_rows: list[dict] = []
    bad_factor_rows: list[dict] = []
    trading_days = set(
        list_trading_days_from_raw(
            raw_data_root,
            start,
            end,
            kind="snapshot",
            asset="cbond",
        )
    )

    backtest_enabled = bool(cfg.get("backtest_enabled", True))
    for spec in progress(specs, desc="factor_batch", unit="signal"):
        factor_col = build_factor_col(spec)
        if not backtest_enabled:
            continue
        result = run_intraday_factor_backtest(
            factor_store,
            Path(label_data_root),
            start,
            end,
            factor_col=factor_col,
            factor_time=factor_time,
            label_time=label_time,
            min_count=min_count,
            ic_bins=ic_bins,
            bin_count=bin_count,
            bin_select=bin_select,
            bin_source=bin_source,
            bin_top_k=bin_top_k,
            bin_lookback_days=bin_lookback_days,
            workers=backtest_workers,
        )
        signal_dir = out_root / spec.name
        signal_dir.mkdir(parents=True, exist_ok=True)
        summary = save_single_factor_report(
            result,
            signal_dir,
            factor_name=spec.name,
            factor_col=factor_col,
            trading_days=trading_days,
            alpha_significance_window=alpha_significance_window,
        )
        if bool(screening_cfg.get("enabled", False)):
            screening_rows.append(
                _build_screening_row(
                    factor_name=spec.name,
                    factor_col=factor_col,
                    summary=summary,
                    screening_cfg=screening_cfg,
                )
            )
        if bool(bad_factor_cfg.get("enabled", True)):
            bad_factor_rows.append(
                _build_bad_factor_row(
                    factor_name=spec.name,
                    factor_col=factor_col,
                    result=result,
                    cfg=bad_factor_cfg,
                )
            )
    if bool(screening_cfg.get("enabled", False)):
        _write_screening_outputs(out_root, screening_cfg=screening_cfg, rows=screening_rows)
    bad_names: set[str] = set()
    if bool(bad_factor_cfg.get("enabled", True)):
        _write_bad_factor_outputs(out_root, cfg=bad_factor_cfg, rows=bad_factor_rows)
        bad_df = pd.DataFrame(bad_factor_rows)
        if not bad_df.empty and "is_bad" in bad_df.columns and "factor_name" in bad_df.columns:
            bad_names = set(
                bad_df.loc[bad_df["is_bad"].astype(bool), "factor_name"].astype(str).tolist()
            )
        bad_count = int(bad_df["is_bad"].sum()) if not bad_df.empty and "is_bad" in bad_df.columns else 0
        print(
            "bad factor report:",
            f"enabled=True",
            f"factors={len(bad_factor_rows)}",
            f"bad={bad_count}",
            f"out={(out_root / 'bad_factors').as_posix()}",
        )
    _collect_report_plots(out_root, bad_factor_names=bad_names)
    return out_root


