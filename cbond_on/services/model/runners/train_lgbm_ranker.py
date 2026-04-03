from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.naming import make_window_label
from cbond_on.core.trading_days import list_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.factors.storage import FactorStore
from cbond_on.models.score_io import load_scores_by_date, write_scores_by_date
from cbond_on.models.impl.lgbm_ranker.trainer import (
    _iter_existing_label_days,
    _split_days,
    build_dataset,
    build_ranker_split_data,
    evaluate_metrics,
    train_lgbm_ranker,
)


def _load_model_config(path: Path | None) -> dict:
    if path is None:
        return load_config_file("models/lgbm_ranker/lgbm_factor_ranker")
    suffix = path.suffix.lower()
    if suffix == ".json5":
        import json5

        with path.open("r", encoding="utf-8") as handle:
            return json5.load(handle) or {}
    if suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) or {}


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


def _load_existing_score_days(score_output: Path) -> set[date]:
    try:
        cache = load_scores_by_date(score_output)
    except FileNotFoundError:
        return set()
    except Exception as exc:
        print(f"[rolling] failed to read existing scores for incremental mode: {exc}")
        return set()
    return set(cache.keys())


def _parse_checkpoint_day(stem: str) -> date | None:
    try:
        return datetime.strptime(stem, "%Y-%m-%d").date()
    except Exception:
        return None


def _checkpoint_path(state_dir: Path, day: date) -> Path:
    return state_dir / f"{day:%Y-%m-%d}.txt"


def _find_previous_checkpoint(state_dir: Path, day: date) -> Path | None:
    if not state_dir.exists():
        return None
    best_day: date | None = None
    best_path: Path | None = None
    for path in state_dir.glob("*.txt"):
        ckpt_day = _parse_checkpoint_day(path.stem)
        if ckpt_day is None or ckpt_day >= day:
            continue
        if best_day is None or ckpt_day > best_day:
            best_day = ckpt_day
            best_path = path
    return best_path


def _best_iter_from_hist(hist: list[dict]) -> int | None:
    best_val = float("-inf")
    best_it = None
    for row in hist:
        val = row.get("val_rank_ic")
        it = row.get("iteration")
        if val is None or it is None:
            continue
        try:
            val_f = float(val)
        except Exception:
            continue
        if not np.isfinite(val_f):
            continue
        if val_f > best_val:
            best_val = val_f
            best_it = int(it)
    return best_it


def _best_val_rank_ic(hist: list[dict]) -> float:
    vals = []
    for row in hist:
        val = row.get("val_rank_ic")
        if val is None:
            continue
        try:
            val_f = float(val)
        except Exception:
            continue
        if np.isfinite(val_f):
            vals.append(val_f)
    return float(np.max(vals)) if vals else float("nan")


def main(
    *,
    config_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
    execution: dict | None = None,
) -> None:
    execution_cfg = dict(execution or {})
    env_refit = os.getenv("CBOND_REFIT_EVERY_N_DAYS")
    env_shards = os.getenv("CBOND_SCORE_PARALLEL_SHARDS")
    env_shard_index = os.getenv("CBOND_SCORE_PARALLEL_SHARD_INDEX")
    refit_every_n_days = max(
        1,
        int(
            execution_cfg.get(
                "refit_every_n_days",
                env_refit if env_refit is not None else 1,
            )
        ),
    )
    parallel_shards = max(
        1,
        int(
            execution_cfg.get(
                "parallel_shards",
                env_shards if env_shards is not None else 1,
            )
        ),
    )
    parallel_shard_index = int(
        execution_cfg.get(
            "parallel_shard_index",
            env_shard_index if env_shard_index is not None else 0,
        )
    )
    if parallel_shard_index < 0 or parallel_shard_index >= parallel_shards:
        raise ValueError(
            f"parallel_shard_index must be in [0, {parallel_shards - 1}], "
            f"got {parallel_shard_index}"
        )
    paths_cfg = load_config_file("paths")
    cfg_file = Path(config_path) if config_path else None
    if cfg_file is None and len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            cfg_file = candidate
    cfg = _load_model_config(cfg_file)

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
    label_time = str(cfg.get("label_time", "14:42"))
    raw_root = paths_cfg["raw_data_root"]

    rolling_cfg = cfg.get("rolling", {})
    rolling_enabled = bool(rolling_cfg.get("enabled", False))
    window_days = int(rolling_cfg.get("window_days", 301))

    scan_start = desired_start
    if rolling_enabled:
        lookback_days = prev_trading_days_from_raw(
            raw_root,
            desired_start,
            window_days,
            kind="snapshot",
            asset="cbond",
        )
        if lookback_days:
            scan_start = lookback_days[0]
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

    # Allow scoring for target days without labels (e.g. latest day in live).
    last_label_day = max(days) if days else None
    if last_label_day and desired_end > last_label_day:
        trade_days = list_trading_days_from_raw(
            raw_root,
            last_label_day,
            desired_end,
            kind="snapshot",
            asset="cbond",
        )
        extra_days = [d for d in trade_days if d > last_label_day and _factor_exists(d)]
        if extra_days:
            days = sorted(set(days + extra_days))

    train_cfg = cfg.get("train", {})
    train_ratio = float(train_cfg.get("train_ratio", 0.7))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))
    test_ratio = float(train_cfg.get("test_ratio", 0.15))
    if not rolling_enabled and abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if window_days < 2:
        raise ValueError("rolling.window_days must be >= 2")

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
    relevance_bins = int(cfg.get("relevance_bins", 20))
    early_rounds = cfg.get("early_stopping_rounds")
    ranker_params = cfg.get("lgbm_ranker_params", {})

    results_root = Path(paths_cfg["results_root"])
    model_name = cfg.get("model_name", "lgbm_ranker_factor_default")
    date_label = f"{desired_start.strftime('%Y-%m-%d')}_{desired_end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    score_output = resolve_output_path(
        cfg.get("score_output"),
        default_path=results_root / "scores" / model_name,
        results_root=results_root,
    )
    score_overwrite = bool(cfg.get("score_overwrite", False))
    score_dedupe = bool(cfg.get("score_dedupe", True))
    incremental_cfg = dict(cfg.get("incremental", {}))
    incremental_enabled = bool(incremental_cfg.get("enabled", True))
    incremental_skip_existing = bool(incremental_cfg.get("skip_existing_scores", True))
    incremental_warm_start = bool(incremental_cfg.get("warm_start", True))
    incremental_save_state = bool(incremental_cfg.get("save_state", True))
    if parallel_shards > 1 and incremental_warm_start:
        print("[rolling] parallel_shards>1: disable warm_start to avoid cross-shard dependency")
        incremental_warm_start = False
    state_dir_raw = str(incremental_cfg.get("state_dir", "")).strip()
    state_dir = resolve_output_path(
        state_dir_raw if state_dir_raw else None,
        default_path=results_root / "model_state" / model_name,
        results_root=results_root,
    )
    if incremental_enabled and (incremental_warm_start or incremental_save_state):
        state_dir.mkdir(parents=True, exist_ok=True)

    if rolling_enabled:
        if len(days) < window_days:
            raise ValueError(f"rolling window_days={window_days} exceeds available days={len(days)}")
        desired_days = [d for d in days if desired_start <= d <= desired_end]
        if not desired_days:
            raise RuntimeError("no label days found for desired range")
        target_days = list(desired_days)
        if incremental_enabled and incremental_skip_existing and not score_overwrite:
            existing_days = _load_existing_score_days(score_output)
            target_days = [d for d in desired_days if d not in existing_days]
            skipped = len(desired_days) - len(target_days)
            if skipped > 0:
                print(
                    f"[rolling] incremental skip_existing_scores=True, "
                    f"skip={skipped}, pending={len(target_days)}"
                )
        if not target_days:
            print("[rolling] incremental: no pending target days, skip training")
            (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "features.json").write_text(json.dumps(factor_cols, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved rolling: {out_dir}")
            print(f"saved scores: {score_output}")
            return
        day_to_idx = {d: i for i, d in enumerate(days)}
        target_indices = [day_to_idx[d] for d in target_days if d in day_to_idx]
        min_idx = window_days - 1
        valid_indices = [i for i in target_indices if i >= min_idx]
        dropped = len(target_indices) - len(valid_indices)
        if dropped > 0:
            print(
                f"[rolling] skip {dropped} target day(s): insufficient history window "
                f"(need window_days={window_days})"
            )
        if not valid_indices:
            raise RuntimeError("rolling has no valid target day after incremental filtering")
        print(
            f"[rolling] execution refit_every_n_days={refit_every_n_days} "
            f"parallel_shards={parallel_shards} parallel_shard_index={parallel_shard_index}"
        )
        if parallel_shards > 1:
            all_count = len(valid_indices)
            valid_indices = [
                idx
                for seq, idx in enumerate(valid_indices)
                if (seq % parallel_shards) == parallel_shard_index
            ]
            print(
                f"[rolling] shard assignment {parallel_shard_index + 1}/{parallel_shards}: "
                f"{len(valid_indices)}/{all_count} target day(s)"
            )
        if not valid_indices:
            print("[rolling] no target day assigned for this shard, skip training")
            (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "features.json").write_text(json.dumps(factor_cols, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved rolling: {out_dir}")
            print(f"saved scores: {score_output}")
            return
        total_rolls = len(valid_indices)

        rolling_rows: list[dict] = []
        score_rows: list[pd.DataFrame] = []
        active_model = None
        active_best_iter: int | None = None
        active_best_val_ic: float = float("nan")
        last_refit_pos: int | None = None
        last_refit_day: date | None = None
        for roll_idx, idx in enumerate(valid_indices, start=1):
            window = days[idx - window_days + 1: idx + 1]
            train_pool = window[:-1]
            test_day = window[-1]
            if len(train_pool) < 2:
                continue

            n_pool = len(train_pool)
            n_train = max(1, int(n_pool * train_ratio))
            n_val = n_pool - n_train
            if n_val <= 0:
                n_val = 1
                n_train = max(1, n_pool - n_val)
            roll_train_days = list(train_pool[:n_train])
            roll_val_days = list(train_pool[n_train:n_train + n_val])
            should_refit = (
                active_model is None
                or refit_every_n_days <= 1
                or last_refit_pos is None
                or (roll_idx - last_refit_pos) >= refit_every_n_days
            )
            print(
                f"[rolling] {roll_idx}/{total_rolls} test_day={test_day} "
                f"refit={'Y' if should_refit else 'N'} cadence={refit_every_n_days}"
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
            if test_data.x.empty:
                continue
            refit_status = "reuse"
            if should_refit:
                train_data = build_dataset(
                    factor_store=store,
                    label_root=label_root,
                    days=roll_train_days,
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
                    days=roll_val_days,
                    factor_cols=factor_cols,
                    min_count=min_count,
                    winsor_lower=winsor_lower,
                    winsor_upper=winsor_upper,
                    zscore=zscore,
                    factor_time=factor_time,
                    label_time=label_time,
                )
                if train_data.x.empty:
                    if active_model is None:
                        print(f"[rolling] skip {test_day}: empty train split and no reusable model")
                        continue
                    refit_status = "reuse_empty_train"
                else:
                    train_ranker = build_ranker_split_data(train_data, relevance_bins=relevance_bins)
                    val_ranker = build_ranker_split_data(val_data, relevance_bins=relevance_bins)
                    if train_ranker.x.empty or not train_ranker.group:
                        if active_model is None:
                            print(f"[rolling] skip {test_day}: empty ranker train split and no reusable model")
                            continue
                        refit_status = "reuse_empty_train_ranker"
                    else:
                        init_model = None
                        if incremental_enabled and incremental_warm_start:
                            prev_ckpt = _find_previous_checkpoint(state_dir, test_day)
                            if prev_ckpt is not None:
                                init_model = str(prev_ckpt)
                                print(f"[rolling] warm-start from checkpoint: {prev_ckpt.name}")
                        try:
                            model, meta = train_lgbm_ranker(
                                train=train_ranker,
                                val=val_ranker,
                                lgbm_ranker_params=ranker_params,
                                early_stopping_rounds=int(early_rounds) if early_rounds else None,
                                init_model=init_model,
                            )
                        except Exception as exc:
                            if init_model is None:
                                raise
                            print(f"[rolling] warm-start failed, fallback cold-start: {type(exc).__name__}: {exc}")
                            model, meta = train_lgbm_ranker(
                                train=train_ranker,
                                val=val_ranker,
                                lgbm_ranker_params=ranker_params,
                                early_stopping_rounds=int(early_rounds) if early_rounds else None,
                                init_model=None,
                            )
                        history = meta.get("history", [])
                        active_best_iter = _best_iter_from_hist(history)
                        active_best_val_ic = _best_val_rank_ic(history)
                        active_model = model
                        last_refit_pos = roll_idx
                        last_refit_day = test_day
                        refit_status = "refit"
                        if incremental_enabled and incremental_save_state:
                            ckpt_path = _checkpoint_path(state_dir, test_day)
                            try:
                                model.booster_.save_model(str(ckpt_path))
                            except Exception as exc:
                                print(f"[rolling] failed to save checkpoint {ckpt_path}: {exc}")
            if active_model is None:
                print(f"[rolling] skip {test_day}: no trained model available")
                continue
            pred_kwargs = {"num_iteration": active_best_iter} if active_best_iter else {}
            test_pred = active_model.predict(test_data.x, **pred_kwargs)
            model_source_day = test_day if refit_status == "refit" else (last_refit_day or test_day)
            best_iter = active_best_iter
            best_val_ic = active_best_val_ic

            if not (desired_start <= test_day <= desired_end):
                continue
            score_rows.append(pd.DataFrame({"trade_date": test_day, "code": test_data.code, "score": test_pred}))

            if np.isfinite(test_data.y).any():
                test_metrics = evaluate_metrics(test_data.x, test_data.y, test_data.dt, test_pred, bins=bins)
            else:
                test_metrics = {
                    "mse": float("nan"),
                    "r2": float("nan"),
                    "dir": float("nan"),
                    "ic_mean": float("nan"),
                    "ic_ir": float("nan"),
                    "rank_ic_mean": float("nan"),
                    "rank_ic_ir": float("nan"),
                    "bin_dir": [],
                }

            rolling_rows.append(
                {
                    "trade_date": test_day,
                    "train_days": len(roll_train_days),
                    "val_days": len(roll_val_days),
                    "count": int(len(test_data.y)),
                    "refit": bool(refit_status == "refit"),
                    "refit_status": refit_status,
                    "model_source_day": model_source_day,
                    "refit_every_n_days": refit_every_n_days,
                    "best_iteration": best_iter,
                    "best_val_rank_ic": best_val_ic,
                    "rank_ic": test_metrics["rank_ic_mean"],
                    "ic": test_metrics["ic_mean"],
                    "dir": test_metrics["dir"],
                    "mse": test_metrics["mse"],
                    "r2": test_metrics["r2"],
                }
            )
            print(
                f"rolling {test_day} refit={refit_status} train_days={len(roll_train_days)} "
                f"val_days={len(roll_val_days)} count={len(test_data.y)} "
                f"rank_ic={test_metrics['rank_ic_mean']:.4f}"
            )

        if not score_rows:
            raise RuntimeError("rolling produced no scores; check window_days and data range")
        all_scores = pd.concat(score_rows, ignore_index=True)
        write_scores_by_date(
            score_output,
            all_scores,
            overwrite=score_overwrite,
            dedupe=score_dedupe,
        )
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

    train_ranker = build_ranker_split_data(train_data, relevance_bins=relevance_bins)
    val_ranker = build_ranker_split_data(val_data, relevance_bins=relevance_bins)
    if train_ranker.x.empty or not train_ranker.group:
        raise RuntimeError("ranker train split is empty after preprocessing")

    model, meta = train_lgbm_ranker(
        train=train_ranker,
        val=val_ranker,
        lgbm_ranker_params=ranker_params,
        early_stopping_rounds=int(early_rounds) if early_rounds else None,
    )
    history = meta.get("history", [])
    best_iter = _best_iter_from_hist(history)
    pred_kwargs = {"num_iteration": best_iter} if best_iter else {}

    train_pred = model.predict(train_data.x, **pred_kwargs)
    val_pred = model.predict(val_data.x, **pred_kwargs)
    test_pred = model.predict(test_data.x, **pred_kwargs)

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

    if best_iter and isinstance(best_iter, int):
        model.booster_.save_model(str(out_dir / "model.txt"), num_iteration=best_iter)
    else:
        model.booster_.save_model(str(out_dir / "model.txt"))

    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "features.json").write_text(json.dumps(factor_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame(
        [
            {"split": "train", **{k: v for k, v in train_metrics.items() if k != "bin_dir"}},
            {"split": "val", **{k: v for k, v in val_metrics.items() if k != "bin_dir"}},
            {"split": "test", **{k: v for k, v in test_metrics.items() if k != "bin_dir"}},
        ]
    )
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    if history:
        pd.DataFrame(history).to_csv(out_dir / "metrics_iter.csv", index=False)

    def _bin_df(split: str, metrics: dict) -> pd.DataFrame:
        return pd.DataFrame(
            [{"split": split, "bin": b, "dir_acc": acc, "count": n} for b, acc, n in metrics["bin_dir"]]
        )

    bin_df = pd.concat(
        [_bin_df("train", train_metrics), _bin_df("val", val_metrics), _bin_df("test", test_metrics)],
        ignore_index=True,
    )
    bin_df.to_csv(out_dir / "bin_dir.csv", index=False)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
