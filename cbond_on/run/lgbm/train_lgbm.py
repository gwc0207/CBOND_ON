
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.naming import make_window_label
from cbond_on.factors.storage import FactorStore
from cbond_on.models.score_io import write_scores_by_date
from cbond_on.models.impl.lgbm.trainer import (
    build_dataset,
    evaluate_metrics,
    train_lgbm,
    _iter_existing_label_days,
    _split_days,
)


def _select_factor_cols(sample: pd.DataFrame, cfg: dict) -> list[str]:
    cols = cfg.get("factors")
    if cols:
        return [str(c) for c in cols]
    exclude = {"dt", "code"}
    return [c for c in sample.columns if c not in exclude]


def _format_bins(bin_dir: list[tuple[int, float, int]]) -> str:
    if not bin_dir:
        return "n/a"
    return ",".join([f"{b}:{acc:.3f}({n})" for b, acc, n in bin_dir])


def main(
    *,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> None:
    paths_cfg = load_config_file("paths")
    cfg = load_config_file("models/lgbm/model")

    cfg_start = parse_date(cfg.get("start"))
    cfg_end = parse_date(cfg.get("end"))
    desired_start = parse_date(start) if start else cfg_start
    desired_end = parse_date(end) if end else cfg_end
    cutoff_day = parse_date(label_cutoff) if label_cutoff else None
    if desired_start > desired_end:
        raise ValueError("start date must be <= end date")

    factor_root = Path(paths_cfg["factor_data_root"])
    label_root = Path(paths_cfg["label_data_root"])

    panel_name = cfg.get("panel_name")
    window_minutes = int(cfg.get("window_minutes", 15))
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:45"))

    scan_start = desired_start
    rolling_cfg = cfg.get("rolling", {})
    rolling_enabled = bool(rolling_cfg.get("enabled", False))
    window_days = int(rolling_cfg.get("window_days", 301))
    if rolling_enabled:
        scan_start = desired_start - timedelta(days=window_days * 8)
    days = list(_iter_existing_label_days(label_root, scan_start, desired_end))
    if not days:
        raise RuntimeError("no label days found for range")
    days = sorted(set(days))
    if cutoff_day is not None:
        days = [d for d in days if d <= cutoff_day]
        if not days:
            raise RuntimeError("no label days left after label_cutoff filter")

    def _factor_exists(day: date) -> bool:
        label = panel_name or make_window_label(window_minutes)
        month = f"{day.year:04d}-{day.month:02d}"
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        path = factor_root / "factors" / label / month / filename
        return path.exists()

    # allow scoring for target days without labels (e.g., latest day in live)
    last_label_day = max(days) if days else None
    if last_label_day and desired_end > last_label_day:
        extra_days = []
        cursor = last_label_day + pd.Timedelta(days=1)
        while cursor <= desired_end:
            if _factor_exists(cursor):
                extra_days.append(cursor)
            cursor = cursor + pd.Timedelta(days=1)
        if extra_days:
            days = sorted(set(days + extra_days))

    train_cfg = cfg.get("train", {})
    train_ratio = float(train_cfg.get("train_ratio", 0.7))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))
    test_ratio = float(train_cfg.get("test_ratio", 0.15))

    if window_days < 2:
        raise ValueError("rolling.window_days must be >= 2")

    if not rolling_enabled and abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_days: list[date] = []
    val_days: list[date] = []
    test_days: list[date] = []
    if not rolling_enabled:
        days = [d for d in days if desired_start <= d <= desired_end]
        if not days:
            raise RuntimeError("no label days found for desired range")
        train_days, val_days, test_days = _split_days(days, train_ratio, val_ratio)
        print(f"train days: {len(train_days)}, val days: {len(val_days)}, test days: {len(test_days)}")

    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=window_minutes)
    # pick factor columns from first available day
    sample = pd.DataFrame()
    sample_days = days if rolling_enabled else train_days
    for day in sample_days:
        sample = store.read_day(day)
        if not sample.empty:
            break
    if sample.empty:
        raise RuntimeError("no factor data found")
    if isinstance(sample.index, pd.MultiIndex):
        sample = sample.reset_index()
    factor_cols = _select_factor_cols(sample, cfg)

    winsor = cfg.get("winsor", {})
    winsor_lower = float(winsor.get("lower", 0.01))
    winsor_upper = float(winsor.get("upper", 0.99))
    zscore = bool(cfg.get("zscore", True))
    min_count = int(cfg.get("min_count", 30))
    bins = int(cfg.get("bins", 5))

    lgbm_params = cfg.get("lgbm_params", {})
    grid_cfg = cfg.get("grid_search", {})
    early_rounds = cfg.get("early_stopping_rounds")

    # results output dir
    results_root = Path(paths_cfg["results_root"])
    model_name = cfg.get("model_name", "lgbm_factor")
    date_label = f"{desired_start.strftime('%Y-%m-%d')}_{desired_end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    score_output = Path(cfg.get("score_output", results_root / "scores" / model_name))
    score_overwrite = bool(cfg.get("score_overwrite", False))
    score_dedupe = bool(cfg.get("score_dedupe", True))

    def _best_val_rank_ic(hist: list[dict]) -> float:
        if not hist:
            return float("nan")
        vals = [h.get("val_rank_ic") for h in hist if h.get("val_rank_ic") is not None]
        if not vals:
            return float("nan")
        return float(np.nanmax(vals))

    def _best_iter_from_hist(hist: list[dict]) -> int | None:
        best_val = float("-inf")
        best_it = None
        for h in hist:
            v = h.get("val_rank_ic")
            it = h.get("iteration")
            if v is None or it is None or np.isnan(v):
                continue
            if float(v) > best_val:
                best_val = float(v)
                best_it = int(it)
        return best_it

    if rolling_enabled:
        if len(days) < window_days:
            raise ValueError(
                f"rolling window_days={window_days} exceeds available days={len(days)}"
            )
        rolling_rows: list[dict] = []
        score_rows: list[pd.DataFrame] = []
        desired_days = [d for d in days if desired_start <= d <= desired_end]
        if not desired_days:
            raise RuntimeError("no label days found for desired range")
        first_target_idx = days.index(desired_days[0])
        last_target_idx = days.index(desired_days[-1])
        start_idx = max(window_days - 1, first_target_idx)
        total_rolls = max(0, last_target_idx - start_idx + 1)
        for idx in range(start_idx, last_target_idx + 1):
            window = days[idx - window_days + 1: idx + 1]
            train_pool = window[:-1]
            test_day = window[-1]
            roll_idx = idx - start_idx + 1
            print(f"[rolling] {roll_idx}/{total_rolls} test_day={test_day}")
            if len(train_pool) < 2:
                continue
            n_pool = len(train_pool)
            n_train = max(1, int(n_pool * train_ratio))
            n_val = n_pool - n_train
            if n_val <= 0:
                n_val = 1
                n_train = max(1, n_pool - n_val)
            train_days = list(train_pool[:n_train])
            val_days = list(train_pool[n_train:n_train + n_val])
            train_data = build_dataset(
                factor_store=store,
                label_root=label_root,
                days=train_days,
                factor_cols=factor_cols,
                min_count=min_count,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                zscore=zscore,
                factor_time=factor_time,
                label_time=label_time,
            )
            val_data = build_dataset(
                factor_store=store,
                label_root=label_root,
                days=val_days,
                factor_cols=factor_cols,
                min_count=min_count,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                zscore=zscore,
                factor_time=factor_time,
                label_time=label_time,
            )
            test_data = build_dataset(
                factor_store=store,
                label_root=label_root,
                days=[test_day],
                factor_cols=factor_cols,
                min_count=min_count,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                zscore=zscore,
                factor_time=factor_time,
                label_time=label_time,
                require_label=False,
            )
            if test_data.x.empty or train_data.x.empty:
                continue
            model, params = train_lgbm(
                train=train_data,
                val=val_data,
                lgbm_params=lgbm_params,
                early_stopping_rounds=int(early_rounds) if early_rounds else None,
            )
            test_pred = model.predict(test_data.x)
            if not (desired_start <= test_day <= desired_end):
                continue
            score_df = pd.DataFrame(
                {"trade_date": test_day, "code": test_data.code, "score": test_pred}
            )
            score_rows.append(score_df)
            if np.isfinite(test_data.y).any():
                test_metrics = evaluate_metrics(
                    test_data.x, test_data.y, test_data.dt, test_pred, bins=bins
                )
            else:
                test_metrics = {
                    "mse": float("nan"),
                    "r2": float("nan"),
                    "dir": float("nan"),
                    "ic_mean": float("nan"),
                    "ic_ir": float("nan"),
                    "rank_ic_mean": float("nan"),
                    "rank_ic_ir": float("nan"),
                    "bins": [],
                }
            rolling_rows.append(
                {
                    "trade_date": test_day,
                    "train_days": len(train_days),
                    "val_days": len(val_days),
                    "count": int(len(test_data.y)),
                    "rank_ic": test_metrics["rank_ic_mean"],
                    "ic": test_metrics["ic_mean"],
                    "dir": test_metrics["dir"],
                    "mse": test_metrics["mse"],
                    "r2": test_metrics["r2"],
                }
            )
            print(
                f"rolling {test_day} train_days={len(train_days)} val_days={len(val_days)} "
                f"count={len(test_data.y)} rank_ic={test_metrics['rank_ic_mean']:.4f}"
            )

        if score_rows:
            all_scores = pd.concat(score_rows, ignore_index=True)
            write_scores_by_date(
                score_output,
                all_scores,
                overwrite=score_overwrite,
                dedupe=score_dedupe,
            )
        else:
            raise RuntimeError("rolling produced no scores; check window_days and data range")
        if rolling_rows:
            pd.DataFrame(rolling_rows).to_csv(out_dir / "rolling_metrics.csv", index=False)
        (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "features.json").write_text(json.dumps(factor_cols, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved rolling: {out_dir}")
        print(f"saved scores: {score_output}")
        return

    train_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=train_days,
        factor_cols=factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
    )
    val_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=val_days,
        factor_cols=factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
    )
    test_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=test_days,
        factor_cols=factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
    )

    model = None
    params = {}
    history = []

    grid_enabled = bool(grid_cfg.get("enable", False))
    if grid_enabled:
        alpha_list = grid_cfg.get("reg_alpha", [])
        lambda_list = grid_cfg.get("reg_lambda", [])
        leaves_list = grid_cfg.get("num_leaves", [])
        depth_list = grid_cfg.get("max_depth", [])
        if not alpha_list:
            alpha_list = [lgbm_params.get("reg_alpha", 0.0)]
        if not lambda_list:
            lambda_list = [lgbm_params.get("reg_lambda", 0.0)]
        if not leaves_list:
            leaves_list = [lgbm_params.get("num_leaves", 31)]
        if not depth_list:
            depth_list = [lgbm_params.get("max_depth", -1)]
        grid_rows = []
        best_score = float("-inf")
        best_params = None
        best_model = None
        best_history = None
        best_iter = None
        for alpha in alpha_list:
            for lam in lambda_list:
                for leaves in leaves_list:
                    for depth in depth_list:
                        trial_params = dict(lgbm_params)
                        trial_params["reg_alpha"] = float(alpha)
                        trial_params["reg_lambda"] = float(lam)
                        trial_params["num_leaves"] = int(leaves)
                        trial_params["max_depth"] = int(depth)
                        trial_model, trial_meta = train_lgbm(
                            train=train_data,
                            val=val_data,
                            lgbm_params=trial_params,
                            early_stopping_rounds=int(early_rounds) if early_rounds else None,
                        )
                        trial_hist = trial_meta.get("history", [])
                        trial_best = _best_val_rank_ic(trial_hist)
                        trial_best_iter = _best_iter_from_hist(trial_hist)
                        grid_rows.append(
                            {
                                "reg_alpha": float(alpha),
                                "reg_lambda": float(lam),
                                "num_leaves": int(leaves),
                                "max_depth": int(depth),
                                "best_val_rank_ic": trial_best,
                                "best_iteration": trial_best_iter,
                            }
                        )
                        if not np.isnan(trial_best) and trial_best > best_score:
                            best_score = trial_best
                            best_params = trial_params
                            best_model = trial_model
                            best_history = trial_hist
                            best_iter = trial_best_iter
        pd.DataFrame(grid_rows).to_csv(out_dir / "grid_search.csv", index=False)
        if best_params is None:
            raise RuntimeError("grid search failed to produce valid params")
        (out_dir / "best_params.json").write_text(
            json.dumps(
                {
                    "best_val_rank_ic": best_score,
                    "best_iteration": best_iter,
                    "params": best_params,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model = best_model
        params = best_params | {"history": best_history or []}
        history = best_history or []
    else:
        model, params = train_lgbm(
            train=train_data,
            val=val_data,
            lgbm_params=lgbm_params,
            early_stopping_rounds=int(early_rounds) if early_rounds else None,
        )
        history = params.get("history", [])

    train_pred = model.predict(train_data.x)
    val_pred = model.predict(val_data.x)
    test_pred = model.predict(test_data.x)

    train_metrics = evaluate_metrics(train_data.x, train_data.y, train_data.dt, train_pred, bins=bins)
    val_metrics = evaluate_metrics(val_data.x, val_data.y, val_data.dt, val_pred, bins=bins)
    test_metrics = evaluate_metrics(test_data.x, test_data.y, test_data.dt, test_pred, bins=bins)

    print(
        f"train mse={train_metrics['mse']:.6f} r2={train_metrics['r2']:.4f} dir={train_metrics['dir']:.4f} "
        f"ic={train_metrics['ic_mean']:.4f} ir={train_metrics['ic_ir']:.4f} "
        f"rank_ic={train_metrics['rank_ic_mean']:.4f} rank_ir={train_metrics['rank_ic_ir']:.4f}"
    )
    print(
        f"val mse={val_metrics['mse']:.6f} r2={val_metrics['r2']:.4f} dir={val_metrics['dir']:.4f} "
        f"ic={val_metrics['ic_mean']:.4f} ir={val_metrics['ic_ir']:.4f} "
        f"rank_ic={val_metrics['rank_ic_mean']:.4f} rank_ir={val_metrics['rank_ic_ir']:.4f}"
    )
    print(
        f"test mse={test_metrics['mse']:.6f} r2={test_metrics['r2']:.4f} dir={test_metrics['dir']:.4f} "
        f"ic={test_metrics['ic_mean']:.4f} ir={test_metrics['ic_ir']:.4f} "
        f"rank_ic={test_metrics['rank_ic_mean']:.4f} rank_ir={test_metrics['rank_ic_ir']:.4f}"
    )

    print(f"train bins: {_format_bins(train_metrics['bin_dir'])}")
    print(f"val bins: {_format_bins(val_metrics['bin_dir'])}")
    print(f"test bins: {_format_bins(test_metrics['bin_dir'])}")

    # save model (best iteration if available)
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter and isinstance(best_iter, int):
        model.booster_.save_model(str(out_dir / "model.txt"), num_iteration=best_iter)
    else:
        model.booster_.save_model(str(out_dir / "model.txt"))
    # save config and metrics
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "features.json").write_text(json.dumps(factor_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame([
        {"split": "train", **{k: v for k, v in train_metrics.items() if k != "bin_dir"}},
        {"split": "val", **{k: v for k, v in val_metrics.items() if k != "bin_dir"}},
        {"split": "test", **{k: v for k, v in test_metrics.items() if k != "bin_dir"}},
    ])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    if history:
        pd.DataFrame(history).to_csv(out_dir / "metrics_iter.csv", index=False)

    # save bin dir
    def _bin_df(split, metrics):
        return pd.DataFrame([
            {"split": split, "bin": b, "dir_acc": acc, "count": n} for b, acc, n in metrics["bin_dir"]
        ])
    bin_df = pd.concat([
        _bin_df("train", train_metrics),
        _bin_df("val", val_metrics),
        _bin_df("test", test_metrics),
    ], ignore_index=True)
    bin_df.to_csv(out_dir / "bin_dir.csv", index=False)

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
