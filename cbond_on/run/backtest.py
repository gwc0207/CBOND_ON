from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.backtest.runner import run_backtest
from cbond_on.models.impl.lob.score_builder import ScoreConfig, build_scores
from cbond_on.report.backtest_report import render_backtest_report


def main() -> None:
    paths_cfg = load_config_file("paths")
    backtest_cfg = load_config_file("backtest")
    live_cfg = load_config_file("live")
    model_cfg = load_config_file("models/lob/model")

    raw_root = paths_cfg["raw_data_root"]
    clean_root = paths_cfg["clean_data_root"]
    start = parse_date(backtest_cfg["start"])
    end = parse_date(backtest_cfg["end"])

    score_path = Path(model_cfg["score_output"])
    refresh_scores = bool(backtest_cfg.get("refresh_scores", False))
    if refresh_scores or not score_path.exists():
        ds_cfg = load_config_file("models/lob/dataset")
        output_dir = Path(clean_root) / str(ds_cfg.get("output_dir", "LOBDS"))
        train_cfg = model_cfg.get("train", {})
        score_cfg = ScoreConfig(
            device=str(train_cfg.get("device", "cpu")),
            batch_size=int(train_cfg.get("batch_size", 16)),
        )
        build_scores(
            dataset_root=output_dir,
            weights_path=Path(model_cfg["weights_path"]),
            start=start,
            end=end,
            model_params=model_cfg.get("params", {}),
            score_cfg=score_cfg,
            output_path=score_path,
        )

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
        top_n=int(live_cfg.get("top_n", 50)),
        min_count=int(backtest_cfg["min_count"]),
        max_weight=float(backtest_cfg["max_weight"]),
        filter_tradable_flag=bool(live_cfg.get("filter_tradable", True)),
        min_amount=float(live_cfg.get("min_amount", 0)),
        min_volume=float(live_cfg.get("min_volume", 0)),
        ic_bins=int(backtest_cfg.get("ic_bins", 20)),
    )

    results_root = Path(paths_cfg["results_root"])
    out_dir = results_root / "backtest" / f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
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
