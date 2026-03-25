from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.core.utils import progress
from cbond_on.factor_batch.runner import (
    build_signal_specs,
    run_intraday_factor_backtest,
)
from cbond_on.factors import defs  # noqa: F401
from cbond_on.factors.spec import build_factor_col
from cbond_on.factors.storage import FactorStore
from cbond_on.report.factor_report import save_single_factor_report


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> Path:
    paths_cfg = load_config_file("paths")
    factor_cfg = load_config_file("factor")
    backtest_cfg = dict(cfg or factor_cfg)

    start_day = parse_date(start or backtest_cfg.get("start") or factor_cfg.get("start"))
    end_day = parse_date(end or backtest_cfg.get("end") or factor_cfg.get("end"))
    refresh_val = bool(backtest_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(backtest_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if refresh_val:
        overwrite_val = True

    specs = build_signal_specs(factor_cfg)
    panel_name = str(factor_cfg.get("panel_name", "")).strip()
    if not panel_name:
        raise ValueError("factor_config.panel_name is required; window_minutes fallback is disabled")
    factor_store = FactorStore(
        Path(paths_cfg["factor_data_root"]),
        panel_name=panel_name,
        window_minutes=15,
    )

    results_root = Path(paths_cfg["results_root"])
    date_label = f"{start_day:%Y-%m-%d}_{end_day:%Y-%m-%d}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = results_root / date_label / "Single_Factor" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    factor_time = str(factor_cfg.get("factor_time", "14:30"))
    label_time = str(factor_cfg.get("label_time", "14:42"))
    run_cfg = dict(backtest_cfg.get("backtest", {}))
    min_count = int(run_cfg.get("min_count", 30))
    ic_bins = int(run_cfg.get("ic_bins", 5))
    bin_count = run_cfg.get("bin_count")
    bin_select = run_cfg.get("bin_select")
    bin_source = str(run_cfg.get("bin_source", "manual"))
    bin_top_k = int(run_cfg.get("bin_top_k", 1))
    bin_lookback_days = int(run_cfg.get("bin_lookback_days", 60))
    workers = int(run_cfg.get("workers", 1))
    trading_days = set(
        list_trading_days_from_raw(
            paths_cfg["raw_data_root"],
            start_day,
            end_day,
            kind="snapshot",
            asset="cbond",
        )
    )

    for spec in progress(specs, desc="factor_backtest", unit="signal"):
        signal_dir = out_root / spec.name
        if signal_dir.exists() and not overwrite_val:
            continue
        signal_dir.mkdir(parents=True, exist_ok=True)
        factor_col = build_factor_col(spec)
        result = run_intraday_factor_backtest(
            factor_store,
            Path(paths_cfg["label_data_root"]),
            start_day,
            end_day,
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
            workers=workers,
        )
        save_single_factor_report(
            result,
            signal_dir,
            factor_name=spec.name,
            factor_col=factor_col,
            trading_days=trading_days,
        )

    return out_root
