from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from cbond_on.core.config import load_config_file, resolve_config_file_path
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.core.utils import progress
from cbond_on.factors.pipeline import run_factor_pipeline
from cbond_on.factors.spec import FactorSpec, build_factor_col
from cbond_on.factors.storage import FactorStore
from cbond_on.report.factor_report import save_single_factor_report


@dataclass
class FactorBacktestResult:
    returns: pd.Series
    nav: pd.Series
    ic: pd.Series
    rank_ic: pd.Series
    bin_returns: pd.DataFrame
    daily_stats: pd.DataFrame


@dataclass
class _BacktestDayPrepared:
    trade_day: date
    reason: str
    merged: pd.DataFrame | None = None


def _load_factor_items_from_payload(payload: object, *, source: str) -> list[dict]:
    if isinstance(payload, dict):
        items = payload.get("factors", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError(f"{source} must be list or object with 'factors'")

    if not isinstance(items, list):
        raise TypeError(f"{source}.factors must be a list")

    out: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"{source}.factors[{idx}] must be an object")
        if "name" not in item or "factor" not in item:
            raise KeyError(f"{source}.factors[{idx}] must contain 'name' and 'factor'")
        out.append(dict(item))
    return out


def build_signal_specs(cfg: dict) -> list[FactorSpec]:
    inline = cfg.get("factors", [])
    if not isinstance(inline, list):
        raise TypeError("factor_config.factors must be a list")

    items: list[dict] = _load_factor_items_from_payload(inline, source="factor_config")

    factor_files = cfg.get("factor_files", [])
    if factor_files is None:
        factor_files = []
    if not isinstance(factor_files, list):
        raise TypeError("factor_config.factor_files must be a list of config paths")

    for ref in factor_files:
        ref_text = str(ref).strip()
        if not ref_text:
            continue
        path = resolve_config_file_path(ref_text)
        payload = load_config_file(str(path))
        items.extend(_load_factor_items_from_payload(payload, source=str(path)))

    specs: list[FactorSpec] = []
    seen_names: set[str] = set()
    for item in items:
        name = str(item["name"]).strip()
        if not name:
            raise ValueError("factor spec name must not be empty")
        if name in seen_names:
            raise ValueError(f"duplicate factor spec name: {name}")
        seen_names.add(name)
        specs.append(
            FactorSpec(
                name=name,
                factor=str(item["factor"]),
                params=item.get("params", {}),
                output_col=item.get("output_col"),
            )
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
    factor_df = factor_store.read_day(day)
    if factor_df.empty or factor_col not in factor_df.columns:
        return _BacktestDayPrepared(trade_day=day, reason="missing_factor")

    label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
    if label_df.empty:
        return _BacktestDayPrepared(trade_day=day, reason="missing_label")

    factor_df = factor_df.reset_index()
    merged = factor_df.merge(label_df, on=["dt", "code"], how="inner")
    merged = merged[[factor_col, "y", "dt", "code"]].dropna()
    if merged.empty:
        return _BacktestDayPrepared(trade_day=day, reason="merged_empty")
    return _BacktestDayPrepared(trade_day=day, reason="", merged=merged)


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
    diagnostics: list[dict] = []
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
        result = FactorBacktestResult(empty, empty, empty, empty, pd.DataFrame(), pd.DataFrame())
        result.diagnostics = pd.DataFrame(diagnostics)  # type: ignore[attr-defined]
        return result

    data = pd.concat(rows, ignore_index=True)

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
        if len(g) < min_count:
            diagnostics.append(
                {
                    "trade_date": dt,
                    "status": "skip",
                    "reason": "min_count_not_met",
                    "count": int(len(g)),
                }
            )
            daily_records.append({"dt": dt, "ret": pd.NA, "ic": pd.NA, "rank_ic": pd.NA, "count": len(g)})
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
            daily_records.append({"dt": dt, "ret": pd.NA, "ic": ic, "rank_ic": rank_ic, "count": len(g)})
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
            daily_records.append({"dt": dt, "ret": pd.NA, "ic": ic, "rank_ic": rank_ic, "count": len(g)})
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
            daily_records.append({"dt": dt, "ret": pd.NA, "ic": ic, "rank_ic": rank_ic, "count": len(g)})
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
            daily_records.append({"dt": dt, "ret": pd.NA, "ic": ic, "rank_ic": rank_ic, "count": len(g)})
            continue
        ret = picks["y"].mean()

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
        daily_records.append({"dt": dt, "ret": ret, "ic": ic, "rank_ic": rank_ic, "count": len(g)})

    daily = pd.DataFrame(daily_records).set_index("dt")
    returns = pd.to_numeric(daily["ret"], errors="coerce").dropna()
    ic = pd.to_numeric(daily["ic"], errors="coerce").dropna()
    rank_ic = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    nav = (1.0 + returns.fillna(0.0)).cumprod()

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
    result = FactorBacktestResult(
        returns=returns,
        nav=nav,
        ic=ic,
        rank_ic=rank_ic,
        bin_returns=bin_returns,
        daily_stats=daily_stats,
    )
    result.diagnostics = pd.DataFrame(diagnostics)  # type: ignore[attr-defined]
    result.bin_ok = len(bin_rows)  # type: ignore[attr-defined]
    result.bin_fail = bin_fail  # type: ignore[attr-defined]
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
    backtest_workers = int(backtest_cfg.get("workers", 1))
    screening_cfg = _load_screening_config(cfg)
    screening_rows: list[dict] = []
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
    if bool(screening_cfg.get("enabled", False)):
        _write_screening_outputs(out_root, screening_cfg=screening_cfg, rows=screening_rows)
    return out_root
