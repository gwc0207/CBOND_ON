from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from cbond_on.core.utils import progress
from cbond_on.data.panel import read_panel_data
from cbond_on.core.trading_days import list_trading_days_from_raw
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


def _read_label_day(label_root: Path, day: date) -> pd.DataFrame:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = label_root / month / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "trade_time" in df.columns and "dt" not in df.columns:
        df = df.copy()
        df["dt"] = pd.to_datetime(df["trade_time"])
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


def run_intraday_factor_backtest(
    factor_store: FactorStore,
    label_root: Path,
    panel_root: Path,
    start: date,
    end: date,
    *,
    factor_col: str,
    top_k: int = 50,
    min_count: int = 30,
    ic_bins: int = 5,
    use_panel_filter: bool = False,
    allowed_phases: Sequence[str] | None = None,
    strategy: str = "long_short_topk",
    turnover_ratio: float = 0.3,
) -> FactorBacktestResult:
    rows = []
    kept_records: list[dict] = []
    filtered_records: list[dict] = []
    diagnostics: list[dict] = []
    for day in _iter_existing_label_days(label_root, start, end):
        factor_df = factor_store.read_day(day)
        if factor_df.empty or factor_col not in factor_df.columns:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_factor"})
            continue
        label_df = _read_label_day(label_root, day)
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
                allowed_phases=allowed_phases,
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
    strategy = str(strategy or "long_short_topk").lower()
    turnover_ratio = float(turnover_ratio)

    for dt, group in data.groupby("dt", sort=True):
        g = group[[factor_col, "y", "code"]].dropna()
        if len(g) < min_count:
            daily_records.append({"dt": dt, "ret": pd.NA, "ic": pd.NA, "rank_ic": pd.NA, "count": len(g)})
            continue

        ic = g[factor_col].corr(g["y"], method="pearson")
        rank_ic = g[factor_col].corr(g["y"], method="spearman")

        if strategy == "turnover_topk":
            candidates = g.sort_values(factor_col, ascending=False)["code"].tolist()
            candidates = candidates[:top_k]
            candidate_set = set(candidates)
            if not holdings:
                holdings = set(candidates)
            else:
                scores = g.set_index("code")[factor_col].to_dict()
                prev = holdings & set(g["code"])
                keep = prev & candidate_set
                new_holdings = set(keep)

                max_changes = int(round(top_k * turnover_ratio))
                if max_changes < 0:
                    max_changes = 0
                if not prev:
                    max_changes = top_k

                add_list = [c for c in candidates if c not in new_holdings]
                changes = 0

                if len(new_holdings) < top_k:
                    for c in add_list:
                        if changes >= max_changes:
                            break
                        new_holdings.add(c)
                        changes += 1
                else:
                    drop_pool = sorted(list(prev - keep), key=lambda c: scores.get(c, -1e9))
                    add_idx = 0
                    while changes < max_changes and add_idx < len(add_list) and drop_pool:
                        drop = drop_pool.pop(0)
                        new_holdings.remove(drop)
                        new_holdings.add(add_list[add_idx])
                        add_idx += 1
                        changes += 1

                holdings = new_holdings

            if holdings:
                y_map = g.set_index("code")["y"]
                ret = y_map.reindex(list(holdings)).dropna().mean()
            else:
                ret = pd.NA
        else:
            g_sorted = g.sort_values(factor_col)
            top = g_sorted.tail(top_k)
            bottom = g_sorted.head(top_k)
            ret = top["y"].mean() - bottom["y"].mean()

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
    allowed_phases: Sequence[str] | None,
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
    if "trading_phase_code" in last_rows.columns and allowed_phases:
        tradable &= last_rows["trading_phase_code"].isin(set(allowed_phases))
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
    full_refresh: bool,
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
        full_refresh=full_refresh,
        specs=specs,
    )

    results_root = Path(results_root)
    date_label = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = results_root / date_label / "Single_Factor" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    factor_store = FactorStore(Path(factor_data_root), panel_name=panel_name, window_minutes=window_minutes)
    backtest_cfg = cfg.get("backtest", {})
    top_k = int(backtest_cfg.get("top_k", 50))
    min_count = int(backtest_cfg.get("min_count", 30))
    ic_bins = int(backtest_cfg.get("ic_bins", 5))
    strategy = backtest_cfg.get("strategy", "turnover_topk")
    turnover_ratio = float(backtest_cfg.get("turnover_ratio", 0.25))
    tradable_cfg = cfg.get("tradable_filter", {})
    use_panel_filter = bool(tradable_cfg.get("use_panel", False))
    allowed_phases = tradable_cfg.get("allowed_phases")
    record_codes = bool(tradable_cfg.get("record_codes", True))

    for spec in progress(specs, desc="factor_batch", unit="signal"):
        factor_col = build_factor_col(spec)
        result = run_intraday_factor_backtest(
            factor_store,
            Path(label_data_root),
            Path(panel_data_root),
            start,
            end,
            factor_col=factor_col,
            top_k=top_k,
            min_count=min_count,
            ic_bins=ic_bins,
            use_panel_filter=use_panel_filter,
            allowed_phases=allowed_phases,
            strategy=strategy,
            turnover_ratio=turnover_ratio,
        )
        signal_dir = out_root / spec.name
        signal_dir.mkdir(parents=True, exist_ok=True)
        save_single_factor_report(
            result,
            signal_dir,
            factor_name=spec.name,
            factor_col=factor_col,
            trading_days=set(list_trading_days_from_raw(raw_data_root, start, end, kind="snapshot")),
        )
        if use_panel_filter and record_codes:
            kept = getattr(result, "kept_records", [])
            filtered = getattr(result, "filtered_records", [])
            if kept:
                pd.DataFrame(kept).to_csv(signal_dir / "tradable_kept.csv", index=False)
            if filtered:
                pd.DataFrame(filtered).to_csv(signal_dir / "tradable_filtered.csv", index=False)
    return out_root
