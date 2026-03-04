from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.backtest.runner import run_backtest
from cbond_on.report.backtest_report import render_backtest_report


def _next_run_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / ts


def main() -> None:
    paths_cfg = load_config_file("paths")
    backtest_cfg = load_config_file("backtest")
    live_cfg = load_config_file("live")

    raw_root = paths_cfg["raw_data_root"]
    clean_root = paths_cfg["clean_data_root"]
    start = parse_date(backtest_cfg["start"])
    end = parse_date(backtest_cfg["end"])

    score_path_cfg = backtest_cfg.get("score_path")
    if score_path_cfg:
        score_path = Path(str(score_path_cfg))
    else:
        # Backward compatibility: if not configured in backtest, fallback to model config score_output.
        ms_cfg = backtest_cfg.get("model_score", {})
        model_config = str(ms_cfg.get("model_config", "models/linear/model"))
        model_cfg = load_config_file(model_config)
        score_path = Path(str(model_cfg["score_output"]))
    if not score_path.exists():
        raise FileNotFoundError(f"score file not found: {score_path}")

    batch_id = str(backtest_cfg.get("batch_id", "Backtest"))
    results_root = Path(paths_cfg["results_root"])
    date_dir = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    base_dir = results_root / date_dir / batch_id
    out_dir = _next_run_dir(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_backtest(
        raw_data_root=raw_root,
        clean_data_root=clean_root,
        start=start,
        end=end,
        score_path=str(score_path),
        buy_twap_col=backtest_cfg["buy_twap_col"],
        sell_twap_col=backtest_cfg["sell_twap_col"],
        twap_bps=float(backtest_cfg["twap_bps"]),
        fee_bps=float(backtest_cfg.get("fee_bps", 0.0)),
        bin_source=str(backtest_cfg.get("bin_source", "auto")),
        bin_top_k=int(backtest_cfg.get("bin_top_k", 1)),
        bin_lookback_days=int(backtest_cfg.get("bin_lookback_days", 60)),
        min_count=int(backtest_cfg["min_count"]),
        max_weight=float(backtest_cfg["max_weight"]),
        filter_tradable_flag=bool(live_cfg.get("filter_tradable", True)),
        min_amount=float(live_cfg.get("min_amount", 0)),
        min_volume=float(live_cfg.get("min_volume", 0)),
        ic_bins=int(backtest_cfg.get("ic_bins", 20)),
        live_bin_source=str(live_cfg.get("bin_source", backtest_cfg.get("bin_source", "auto"))),
        live_bin_top_k=int(live_cfg.get("bin_top_k", backtest_cfg.get("bin_top_k", 1))),
        live_bin_lookback_days=int(
            live_cfg.get("bin_lookback_days", backtest_cfg.get("bin_lookback_days", 60))
        ),
    )
    if result.daily_returns is not None:
        result.daily_returns.to_csv(out_dir / "daily_returns.csv", index=False)
    if result.nav_curve is not None:
        result.nav_curve.to_csv(out_dir / "nav_curve.csv", index=False)
    if result.positions is not None:
        result.positions.to_csv(out_dir / "positions.csv", index=False)
    if result.diagnostics is not None:
        result.diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    if result.ic_series is not None:
        result.ic_series.to_csv(out_dir / "ic_series.csv", index=False)
    if result.bin_stats is not None:
        result.bin_stats.to_csv(out_dir / "factor_bins.csv", index=False)
    if result.bin_nav is not None:
        result.bin_nav.to_csv(out_dir / "bin_nav.csv", index=False)
    if result.bin_dir_acc is not None:
        result.bin_dir_acc.to_csv(out_dir / "bin_dir_acc.csv", index=False)
    try:
        render_backtest_report(out_dir)
    except Exception as exc:
        print(f"report skipped: {exc}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
