from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from cbond_on.core.utils import progress
from cbond_on.data.panel import read_panel_data
from cbond_on.factors.pipeline import run_factor_pipeline
from cbond_on.factors.spec import FactorSpec, build_factor_col
from cbond_on.factors.storage import FactorStore


@dataclass
class FactorBacktestResult:
    returns: pd.Series
    nav: pd.Series
    ic: pd.Series
    rank_ic: pd.Series
    bin_returns: pd.DataFrame
    daily_stats: pd.DataFrame


def build_signal_specs(cfg: dict) -> list[FactorSpec]:
    specs = []
    for item in cfg.get("factors", []):
        specs.append(
            FactorSpec(
                name=item["name"],
                factor=item["factor"],
                params=item.get("params", {}),
                output_col=item.get("output_col"),
            )
        )
    return specs


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current = current + pd.Timedelta(days=1)


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
    current = start
    while current <= end:
        month = f"{current.year:04d}-{current.month:02d}"
        filename = f"{current.strftime('%Y%m%d')}.parquet"
        path = label_root / month / filename
        if path.exists():
            yield current
        current = current + pd.Timedelta(days=1)


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
    panel_root: Path,
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
    use_panel_filter: bool = False,
) -> FactorBacktestResult:
    rows = []
    kept_records: list[dict] = []
    filtered_records: list[dict] = []
    diagnostics: list[dict] = []
    days = list(_iter_existing_label_days(label_root, start, end))
    for day in progress(days, desc="factor_backtest", unit="day", total=len(days)):
        factor_df = factor_store.read_day(day)
        if factor_df.empty or factor_col not in factor_df.columns:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_factor"})
            continue
        label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
        if label_df.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_label"})
            continue
        factor_df = factor_df.reset_index()
        merged = factor_df.merge(label_df, on=["dt", "code"], how="inner")
        merged = merged[[factor_col, "y", "dt", "code"]].dropna()
        if merged.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "merged_empty"})
            continue
        if use_panel_filter:
            tradable = _build_tradable_flags(
                panel_root,
                day,
                panel_name=factor_store.panel_name,
                window_minutes=factor_store.window_minutes,
            )
            if not tradable.empty:
                universe = merged[["dt", "code"]].drop_duplicates()
                merged = merged.merge(tradable, on=["dt", "code"], how="inner")
                merged = merged[merged["tradable"]]
                kept = merged[["dt", "code"]].drop_duplicates()
                filtered = universe.merge(kept, on=["dt", "code"], how="left", indicator=True)
                filtered = filtered[filtered["_merge"] == "left_only"][["dt", "code"]]
                kept_records.extend(kept.to_dict("records"))
                filtered_records.extend(filtered.to_dict("records"))
                if merged.empty:
                    diagnostics.append({"trade_date": day, "status": "skip", "reason": "no_tradable"})
                    continue
        diagnostics.append({"trade_date": day, "status": "ok", "count": len(merged)})
        rows.append(merged)
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
    result.kept_records = kept_records  # type: ignore[attr-defined]
    result.filtered_records = filtered_records  # type: ignore[attr-defined]
    result.diagnostics = pd.DataFrame(diagnostics)  # type: ignore[attr-defined]
    result.bin_ok = len(bin_rows)  # type: ignore[attr-defined]
    result.bin_fail = bin_fail  # type: ignore[attr-defined]
    return result


def _build_tradable_flags(
    panel_root: Path,
    day: date,
    *,
    panel_name: str | None,
    window_minutes: int,
) -> pd.DataFrame:
    panel = read_panel_data(
        panel_root,
        day,
        window_minutes=window_minutes,
        panel_name=panel_name,
        columns=[
            "trade_time",
            "code",
            "seq",
            "last",
            "bid_price1",
            "ask_price1",
            "trading_phase_code",
        ],
    ).data
    if panel is None or panel.empty:
        return pd.DataFrame()
    panel = panel.reset_index()
    panel = panel.sort_values(["dt", "code", "seq"])
    last_rows = panel.groupby(["dt", "code"], sort=False).tail(1)
    tradable = pd.Series(True, index=last_rows.index)
    for col in ("last", "bid_price1", "ask_price1"):
        if col in last_rows.columns:
            tradable &= last_rows[col].notna() & (last_rows[col] > 0)
    out = last_rows.loc[tradable, ["dt", "code"]].copy()
    out["tradable"] = True
    return out


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
    overwrite: bool,
    specs: Sequence[FactorSpec],
) -> Path:
    run_factor_pipeline(
        panel_data_root,
        factor_data_root,
        start,
        end,
        window_minutes=window_minutes,
        panel_name=panel_name,
        overwrite=overwrite,
        specs=specs,
    )

    results_root = Path(results_root)
    date_label = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = results_root / date_label / "Single_Factor" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    factor_store = FactorStore(Path(factor_data_root), panel_name=panel_name, window_minutes=window_minutes)
    backtest_cfg = cfg.get("backtest", {})
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:45"))
    min_count = int(backtest_cfg.get("min_count", 30))
    ic_bins = int(backtest_cfg.get("ic_bins", 5))
    bin_count = backtest_cfg.get("bin_count")
    bin_select = backtest_cfg.get("bin_select")
    bin_source = backtest_cfg.get("bin_source", "manual")
    bin_top_k = int(backtest_cfg.get("bin_top_k", 1))
    bin_lookback_days = int(backtest_cfg.get("bin_lookback_days", 60))
    tradable_cfg = cfg.get("tradable_filter", {})
    use_panel_filter = bool(tradable_cfg.get("use_panel", False))
    record_codes = bool(tradable_cfg.get("record_codes", True))

    backtest_enabled = bool(cfg.get("backtest_enabled", True))
    for spec in progress(specs, desc="factor_batch", unit="signal"):
        factor_col = build_factor_col(spec)
        if not backtest_enabled:
            continue
        result = run_intraday_factor_backtest(
            factor_store,
            Path(label_data_root),
            Path(panel_data_root),
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
            use_panel_filter=use_panel_filter,
        )
        signal_dir = out_root / spec.name
        signal_dir.mkdir(parents=True, exist_ok=True)
        if use_panel_filter and record_codes:
            kept = getattr(result, "kept_records", [])
            filtered = getattr(result, "filtered_records", [])
            if kept:
                pd.DataFrame(kept).to_csv(signal_dir / "tradable_kept.csv", index=False)
            if filtered:
                pd.DataFrame(filtered).to_csv(signal_dir / "tradable_filtered.csv", index=False)
    return out_root
