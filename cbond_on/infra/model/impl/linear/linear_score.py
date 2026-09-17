from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cbond_on.core.naming import make_window_label
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.factors.factor_table_resolution import CanonicalFactorTableReader
from cbond_on.infra.model.neutralization import FactorNeutralizer
from cbond_on.infra.model.score_io import write_scores_by_date


@dataclass
class ScoreResult:
    scores: pd.DataFrame
    weights_history: pd.DataFrame


@dataclass(frozen=True)
class LinearFitResult:
    """Raw fitted parameters for a linear refit.

    Scoring normalises the coefficient vector separately.  ElasticNet warm
    start, however, must use sklearn's unnormalised ``coef_`` and
    ``intercept_``.  Keeping the two representations distinct prevents a
    score-time normalisation from becoming the next optimisation initialiser.
    """

    coef: pd.Series
    intercept: float


def _parse_checkpoint_day(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _checkpoint_path(state_dir: Path, day: date) -> Path:
    return state_dir / f"{day:%Y-%m-%d}.json"


def _find_previous_checkpoint(state_dir: Path, day: date) -> Path | None:
    """Return the nearest completed checkpoint strictly before ``day``."""

    if not state_dir.exists():
        return None
    best_day: date | None = None
    best_path: Path | None = None
    for path in state_dir.glob("*.json"):
        checkpoint_day = _parse_checkpoint_day(path.stem)
        if checkpoint_day is None or checkpoint_day >= day:
            continue
        if best_day is None or checkpoint_day > best_day:
            best_day = checkpoint_day
            best_path = path
    return best_path


def _load_elasticnet_warm_start(
    *,
    checkpoint_path: Path,
    expected_fingerprint: str,
    factor_cols: list[str],
    target_day: date,
    label_cutoff: date | None,
) -> LinearFitResult | None:
    """Load one compatible raw ElasticNet state, otherwise fail closed."""

    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[linear] warm-start checkpoint unreadable {checkpoint_path.name}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"[linear] warm-start checkpoint invalid payload: {checkpoint_path.name}")
        return None
    if payload.get("fingerprint") != expected_fingerprint:
        print(f"[linear] warm-start checkpoint contract mismatch: {checkpoint_path.name}")
        return None
    if str(payload.get("regression_kind", "")).strip().lower() != "elasticnet":
        print(f"[linear] warm-start checkpoint is not ElasticNet: {checkpoint_path.name}")
        return None
    if list(payload.get("factor_cols", [])) != list(factor_cols):
        print(f"[linear] warm-start checkpoint factor contract mismatch: {checkpoint_path.name}")
        return None
    filename_day = _parse_checkpoint_day(checkpoint_path.stem)
    checkpoint_day = _parse_checkpoint_day(str(payload.get("date", "")))
    train_end = _parse_checkpoint_day(str(payload.get("train_end", "")))
    if filename_day is None or checkpoint_day is None or checkpoint_day != filename_day:
        print(f"[linear] warm-start checkpoint date provenance mismatch: {checkpoint_path.name}")
        return None
    if checkpoint_day >= target_day:
        print(f"[linear] warm-start checkpoint date is invalid: {checkpoint_path.name}")
        return None
    if train_end is None or train_end >= target_day:
        print(f"[linear] warm-start checkpoint train_end is invalid: {checkpoint_path.name}")
        return None
    # The persisted cutoff is retained for audit.  ``train_end`` is the
    # causal guard: it is the latest label that actually entered the fit.
    if label_cutoff is not None and train_end > label_cutoff:
        print(f"[linear] warm-start checkpoint exceeds label cutoff: {checkpoint_path.name}")
        return None
    try:
        coef = np.asarray(payload["coef"], dtype=float).reshape(-1)
        intercept = float(payload["intercept"])
    except (KeyError, TypeError, ValueError):
        print(f"[linear] warm-start checkpoint parameters are invalid: {checkpoint_path.name}")
        return None
    if coef.size != len(factor_cols) or not np.isfinite(coef).all() or not np.isfinite(intercept):
        print(f"[linear] warm-start checkpoint parameters are non-finite: {checkpoint_path.name}")
        return None
    return LinearFitResult(coef=pd.Series(coef, index=factor_cols, dtype=float), intercept=intercept)


def _save_elasticnet_warm_start(
    *,
    checkpoint_path: Path,
    fit: LinearFitResult,
    factor_cols: list[str],
    score_day: date,
    train_end: date,
    fingerprint: str,
    label_cutoff: date | None,
) -> None:
    """Atomically persist raw ElasticNet parameters for research continuation."""

    coef = fit.coef.reindex(factor_cols).to_numpy(dtype=float)
    if coef.size != len(factor_cols) or not np.isfinite(coef).all() or not np.isfinite(float(fit.intercept)):
        raise ValueError("refusing to save non-finite ElasticNet warm-start state")
    payload = {
        "format_version": 1,
        "date": str(score_day),
        "train_end": str(train_end),
        "label_cutoff": str(label_cutoff) if label_cutoff is not None else None,
        "fingerprint": str(fingerprint),
        "regression_kind": "elasticnet",
        "factor_cols": list(factor_cols),
        "coef": [float(value) for value in coef],
        "intercept": float(fit.intercept),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        # Atomic replacement means a crash cannot leave a partial checkpoint
        # eligible for the next run's directory scan.
        tmp_path.replace(checkpoint_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def linear_contract_fingerprint(payload: dict[str, Any]) -> str:
    """Return a stable fingerprint for the causal linear fitting contract."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _iter_existing_label_days(label_root: Path, start: date, end: date) -> list[date]:
    days: list[date] = []
    if not label_root.exists():
        return days
    for path in label_root.glob("*/*.parquet"):
        stem = path.stem.strip()
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except Exception:
            continue
        if start <= day <= end:
            days.append(day)
    return sorted(set(days))


def _iter_existing_factor_days(
    factor_root: Path,
    *,
    panel_name: str,
    window_minutes: int,
    start: date,
    end: date,
) -> list[date]:
    """Return factor days without consulting labels.

    A score day is defined by the availability of its point-in-time factor
    panel, not by whether its future return label has already materialised.
    This distinction is what allows the same code path to score both
    historical and live target days.
    """
    label = panel_name or make_window_label(window_minutes)
    root = factor_root / "factors" / label
    if not root.exists():
        return []
    days: list[date] = []
    for path in root.glob("*/*.parquet"):
        stem = path.stem.strip()
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except Exception:
            continue
        if start <= day <= end:
            days.append(day)
    return sorted(set(days))


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


def _apply_winsor_zscore(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    lower_q: float | None,
    upper_q: float | None,
    zscore: bool,
) -> pd.DataFrame:
    def _process(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["dt"] = group.name
        for col in factor_cols:
            s = g[col]
            if s.isna().all():
                continue
            if lower_q is not None or upper_q is not None:
                lo = s.quantile(lower_q) if lower_q is not None else None
                hi = s.quantile(upper_q) if upper_q is not None else None
                s = s.clip(lower=lo, upper=hi)
            if zscore:
                mean = s.mean()
                std = s.std(ddof=0)
                if std > 0:
                    s = (s - mean) / std
                else:
                    s = s - mean
            g[col] = s
        return g

    return df.groupby("dt", group_keys=False).apply(_process, include_groups=False)


def _apply_factor_preprocess(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    lower_q: float | None,
    upper_q: float | None,
    zscore: bool,
    neutralizer: FactorNeutralizer | None = None,
) -> pd.DataFrame:
    if neutralizer is None or not neutralizer.enabled:
        return _apply_winsor_zscore(
            df,
            factor_cols,
            lower_q=lower_q,
            upper_q=upper_q,
            zscore=zscore,
        )
    work = df
    if lower_q is not None or upper_q is not None:
        work = _apply_winsor_zscore(
            work,
            factor_cols,
            lower_q=lower_q,
            upper_q=upper_q,
            zscore=False,
        )
    work = neutralizer.apply(work, factor_cols)
    if zscore:
        work = _apply_winsor_zscore(
            work,
            factor_cols,
            lower_q=None,
            upper_q=None,
            zscore=True,
        )
    return work


def _fit_linear_parameters(
    train_df: pd.DataFrame,
    factor_cols: list[str],
    *,
    alpha: float,
    regression_kind: str = "ridge",
    elasticnet_l1_ratio: float = 0.5,
    huber_epsilon: float = 1.35,
    max_iter: int = 1_000,
    device: str = "cpu",
    gpu_fallback_to_cpu: bool = True,
    gpu_state: dict[str, Any] | None = None,
    warm_start: LinearFitResult | None = None,
) -> LinearFitResult | None:
    if train_df.empty:
        return None
    X = train_df[factor_cols]
    y = train_df["y"]
    if X.empty or y.empty:
        return None

    kind = str(regression_kind or "ridge").strip().lower().replace("_", "")
    aliases = {
        "ridge": "ridge",
        "elasticnet": "elasticnet",
        "enet": "elasticnet",
        "huber": "huber",
        "huberregressor": "huber",
    }
    if kind not in aliases:
        raise ValueError(f"unsupported linear regression_kind: {regression_kind}")
    kind = aliases[kind]

    use_gpu = str(device or "cpu").strip().lower() in {"gpu", "cuda"}
    if use_gpu and kind == "ridge":
        try:
            import cupy as cp
            from cuml.linear_model import Ridge as CuRidge

            model = CuRidge(alpha=float(alpha), fit_intercept=True)
            x_gpu = cp.asarray(X.to_numpy(dtype=np.float32))
            y_gpu = cp.asarray(y.to_numpy(dtype=np.float32))
            model.fit(x_gpu, y_gpu)
            coef = model.coef_
            if hasattr(coef, "to_numpy"):
                coef_np = coef.to_numpy()
            elif hasattr(coef, "get"):
                coef_np = coef.get()
            else:
                coef_np = np.asarray(coef)
            return LinearFitResult(
                coef=pd.Series(np.asarray(coef_np).reshape(-1), index=factor_cols, dtype=float),
                # cuML's Ridge adapter does not expose a portable intercept
                # contract here.  It is irrelevant to composite scoring and
                # cannot be used for ElasticNet warm-start in any case.
                intercept=0.0,
            )
        except Exception as exc:
            if not gpu_fallback_to_cpu:
                return None
            if gpu_state is not None and not bool(gpu_state.get("warned", False)):
                print(
                    "[linear] GPU requested but cuML is unavailable; fallback to CPU:",
                    f"{type(exc).__name__}: {exc}",
                )
                gpu_state["warned"] = True
    elif use_gpu and gpu_state is not None and not bool(gpu_state.get("warned", False)):
        print(f"[linear] regression_kind={kind} uses sklearn CPU implementation")
        gpu_state["warned"] = True

    try:
        from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
    except Exception:
        return None
    if kind == "ridge":
        model = Ridge(alpha=float(alpha), fit_intercept=True)
    elif kind == "elasticnet":
        if not 0.0 <= float(elasticnet_l1_ratio) <= 1.0:
            raise ValueError("elasticnet_l1_ratio must be in [0, 1]")
        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(elasticnet_l1_ratio),
            fit_intercept=True,
            max_iter=max(1, int(max_iter)),
            selection="cyclic",
            warm_start=warm_start is not None,
        )
    else:
        if float(huber_epsilon) <= 1.0:
            raise ValueError("huber_epsilon must be > 1")
        model = HuberRegressor(
            epsilon=float(huber_epsilon),
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=max(1, int(max_iter)),
    )
    if warm_start is not None and kind == "elasticnet":
        # A Series rebuilt from a checkpoint can expose a read-only NumPy
        # buffer.  sklearn's coordinate-descent warm-start updates ``coef_``
        # in place, so make an owned writable array before assigning it.
        prior_coef = np.array(
            warm_start.coef.reindex(factor_cols).to_numpy(dtype=float),
            dtype=float,
            copy=True,
        )
        prior_intercept = float(warm_start.intercept)
        if prior_coef.size == len(factor_cols) and np.isfinite(prior_coef).all() and np.isfinite(prior_intercept):
            model.coef_ = prior_coef
            model.intercept_ = prior_intercept
        else:
            warm_start = None
    model.fit(X, y)
    weights = pd.Series(model.coef_, index=factor_cols, dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))
    if not np.isfinite(weights.to_numpy(dtype=float)).all() or not np.isfinite(intercept):
        return None
    return LinearFitResult(coef=weights, intercept=intercept)


def _fit_weights(
    train_df: pd.DataFrame,
    factor_cols: list[str],
    *,
    alpha: float,
    regression_kind: str = "ridge",
    elasticnet_l1_ratio: float = 0.5,
    huber_epsilon: float = 1.35,
    max_iter: int = 1_000,
    device: str = "cpu",
    gpu_fallback_to_cpu: bool = True,
    gpu_state: dict[str, Any] | None = None,
) -> pd.Series | None:
    """Legacy coefficient-only fitting API.

    Existing linear callers intentionally continue to receive exactly the raw
    coefficient Series they received before research-only ElasticNet state was
    added.  The runner below uses ``_fit_linear_parameters`` only when it must
    retain an intercept for an explicit warm start.
    """

    fit = _fit_linear_parameters(
        train_df,
        factor_cols,
        alpha=alpha,
        regression_kind=regression_kind,
        elasticnet_l1_ratio=elasticnet_l1_ratio,
        huber_epsilon=huber_epsilon,
        max_iter=max_iter,
        device=device,
        gpu_fallback_to_cpu=gpu_fallback_to_cpu,
        gpu_state=gpu_state,
    )
    return None if fit is None else fit.coef


def _normalize_weights(weights: pd.Series, method: str, max_weight: float) -> pd.Series:
    w = weights.clip(-max_weight, max_weight)
    if method == "l1":
        denom = w.abs().sum()
        if denom > 0:
            w = w / denom
    return w


def _prepare_factor_day(
    day: date,
    factor_store: FactorStore,
    *,
    factor_cols: list[str],
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    min_count: int,
    factor_time: str,
    label_time: str,
    neutralizer: FactorNeutralizer | None = None,
) -> pd.DataFrame:
    """Build a target-day feature matrix without accessing any label file."""
    fdf = factor_store.read_day(day)
    if fdf.empty:
        return pd.DataFrame()
    if not isinstance(fdf.index, pd.MultiIndex):
        fdf = fdf.reset_index().set_index(["dt", "code"])
    fdf = fdf.reset_index()
    required_cols = ["dt", "code"] + factor_cols
    missing_cols = [col for col in required_cols if col not in fdf.columns]
    if missing_cols:
        return pd.DataFrame()
    work = fdf[required_cols].copy()
    work["dt"] = pd.to_datetime(work["dt"], errors="coerce")
    work["code"] = work["code"].astype(str)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)
    if work.empty:
        return pd.DataFrame()
    counts = work.groupby("dt")["code"].transform("size")
    work = work[counts >= min_count]
    if work.empty:
        return pd.DataFrame()
    work = _apply_factor_preprocess(
        work,
        factor_cols,
        lower_q=winsor_lower,
        upper_q=winsor_upper,
        zscore=zscore,
        neutralizer=neutralizer,
    )
    return work[["dt", "code"] + factor_cols]


def _prepare_labeled_day(
    day: date,
    factor_df: pd.DataFrame,
    label_root: Path,
    *,
    factor_time: str,
    label_time: str,
) -> pd.DataFrame:
    """Attach realised labels to an already point-in-time prepared factor day."""
    if factor_df.empty:
        return pd.DataFrame()
    label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
    if label_df.empty or not {"dt", "code", "y"}.issubset(label_df.columns):
        return pd.DataFrame()
    labels = label_df[["dt", "code", "y"]].copy()
    labels["dt"] = pd.to_datetime(labels["dt"], errors="coerce")
    labels["code"] = labels["code"].astype(str)
    labels["y"] = pd.to_numeric(labels["y"], errors="coerce")
    labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=["dt", "code", "y"])
    if labels.empty:
        return pd.DataFrame()
    return factor_df.merge(labels, on=["dt", "code"], how="inner")


def run_linear_score(
    *,
    factor_root: Path,
    label_root: Path,
    start: date,
    end: date,
    factor_cols: list[str],
    panel_name: str,
    window_minutes: int,
    factor_time: str,
    label_time: str,
    min_count: int,
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    lookback_days: int,
    refit_freq: int,
    regression_alpha: float,
    weight_source: str,
    fallback: str,
    max_weight: float,
    normalize_weights: str,
    manual_weights: pd.Series,
    device: str = "cpu",
    gpu_fallback_to_cpu: bool = True,
    neutralizer: FactorNeutralizer | None = None,
    label_cutoff: date | None = None,
    regression_kind: str = "ridge",
    elasticnet_l1_ratio: float = 0.5,
    huber_epsilon: float = 1.35,
    max_iter: int = 1_000,
    incremental_enabled: bool = False,
    incremental_warm_start: bool = False,
    incremental_save_state: bool = False,
    state_dir: Path | None = None,
    warm_start_fingerprint: str | None = None,
    factor_store: FactorStore | CanonicalFactorTableReader | None = None,
    target_days: Sequence[date] | None = None,
    audit_only: bool = False,
) -> ScoreResult:
    if factor_store is None:
        raise RuntimeError(
            "linear scoring requires an explicit manifest-bound canonical factor reader; "
            "direct factor-root construction is audit-only"
        )
    if isinstance(factor_store, CanonicalFactorTableReader):
        if target_days is None:
            raise RuntimeError(
                "canonical linear scoring requires caller-validated target_days; "
                "do not discover factor days by scanning parquet paths"
            )
        store = factor_store
        resolved_target_days = sorted(set(target_days))
    else:
        if not audit_only:
            raise RuntimeError(
                "direct FactorStore linear scoring is audit-only; normal callers must use "
                "CanonicalFactorTableReader with explicit target_days"
            )
        store = factor_store
        resolved_target_days = (
            sorted(set(target_days))
            if target_days is not None
            else _iter_existing_factor_days(
                factor_root,
                panel_name=panel_name,
                window_minutes=window_minutes,
                start=start,
                end=end,
            )
        )
    if not resolved_target_days:
        return ScoreResult(scores=pd.DataFrame(), weights_history=pd.DataFrame())

    # The label lookup includes pre-start history so a one-day live call still
    # has a complete rolling training window.  Each target later applies the
    # stricter d < target_day boundary, independently of this global cutoff.
    label_end = min(end, label_cutoff) if label_cutoff is not None else end
    label_days = _iter_existing_label_days(label_root, date.min, label_end)
    weights = manual_weights.copy()
    last_refit_idx: int | None = None
    score_rows: list[dict] = []
    weight_rows: list[dict] = []
    gpu_state: dict[str, Any] = {"warned": False}
    factor_cache: dict[date, pd.DataFrame] = {}
    label_cache: dict[date, pd.DataFrame] = {}
    warm_start_active = bool(
        incremental_enabled
        and incremental_warm_start
        and str(regression_kind).strip().lower().replace("_", "") in {"elasticnet", "enet"}
        and state_dir is not None
        and warm_start_fingerprint
    )

    def _factor_day(day: date) -> pd.DataFrame:
        if day not in factor_cache:
            factor_cache[day] = _prepare_factor_day(
                day,
                store,
                factor_cols=factor_cols,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                zscore=zscore,
                min_count=min_count,
                factor_time=factor_time,
                label_time=label_time,
                neutralizer=neutralizer,
            )
        return factor_cache[day]

    def _labeled_day(day: date) -> pd.DataFrame:
        if day not in label_cache:
            label_cache[day] = _prepare_labeled_day(
                day,
                _factor_day(day),
                label_root,
                factor_time=factor_time,
                label_time=label_time,
            )
        return label_cache[day]

    for idx, day in enumerate(resolved_target_days):
        day_df = _factor_day(day)
        if day_df.empty:
            continue

        if weight_source == "regression":
            need_refit = last_refit_idx is None or (idx - last_refit_idx) >= refit_freq
            if need_refit:
                train_days = [d for d in label_days if d < day]
                train_days = train_days[-lookback_days:]
                train_frames = []
                for td in train_days:
                    tdf = _labeled_day(td)
                    if not tdf.empty:
                        train_frames.append(tdf)
                train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
                checkpoint_path: Path | None = None
                prior_fit: LinearFitResult | None = None
                if warm_start_active:
                    checkpoint_path = _find_previous_checkpoint(state_dir, day)
                    if checkpoint_path is not None:
                        prior_fit = _load_elasticnet_warm_start(
                            checkpoint_path=checkpoint_path,
                            expected_fingerprint=str(warm_start_fingerprint),
                            factor_cols=factor_cols,
                            target_day=day,
                            label_cutoff=label_cutoff,
                        )
                        if prior_fit is not None:
                            print(f"[linear] warm-start from checkpoint: {checkpoint_path.name}")
                fit = _fit_linear_parameters(
                    train_df,
                    factor_cols,
                    alpha=regression_alpha,
                    regression_kind=regression_kind,
                    elasticnet_l1_ratio=elasticnet_l1_ratio,
                    huber_epsilon=huber_epsilon,
                    max_iter=max_iter,
                    device=device,
                    gpu_fallback_to_cpu=gpu_fallback_to_cpu,
                    gpu_state=gpu_state,
                    warm_start=prior_fit,
                )
                if fit is None:
                    if fallback == "equal":
                        weights = pd.Series(1.0, index=factor_cols) / len(factor_cols)
                    else:
                        weights = manual_weights.copy()
                else:
                    weights = fit.coef
                weights = _normalize_weights(weights, normalize_weights, max_weight)
                train_end = train_days[-1] if train_days else None
                if (
                    warm_start_active
                    and incremental_save_state
                    and fit is not None
                    and train_end is not None
                ):
                    _save_elasticnet_warm_start(
                        checkpoint_path=_checkpoint_path(state_dir, day),
                        fit=fit,
                        factor_cols=factor_cols,
                        score_day=day,
                        train_end=train_end,
                        fingerprint=str(warm_start_fingerprint),
                        label_cutoff=label_cutoff,
                    )
                for factor, weight in weights.items():
                    weight_rows.append(
                        {
                            "trade_date": day,
                            "factor": factor,
                            "weight": float(weight),
                            "regression_kind": str(regression_kind),
                            "train_start": train_days[0] if train_days else None,
                            "train_end": train_days[-1] if train_days else None,
                            "train_days": int(len(train_days)),
                            "train_rows": int(len(train_df)),
                            "warm_start_checkpoint": str(checkpoint_path) if prior_fit is not None and checkpoint_path is not None else None,
                            "warm_start_active": bool(warm_start_active),
                        }
                    )
                last_refit_idx = idx

        work = day_df[factor_cols].copy()
        valid_mask = work.notna()
        denom = valid_mask.mul(weights.abs(), axis=1).sum(axis=1)
        weighted = work.mul(weights, axis=1).sum(axis=1)
        composite = weighted.where(denom > 0).div(denom)
        for code, score in zip(day_df["code"], composite, strict=False):
            if pd.isna(score):
                continue
            score_rows.append(
                {"trade_date": day, "code": code, "score": float(score)}
            )

    return ScoreResult(
        scores=pd.DataFrame(score_rows),
        weights_history=pd.DataFrame(weight_rows),
    )


def write_linear_outputs(
    *,
    result: ScoreResult,
    score_path: Path,
    weights_path: Path | None,
    meta_path: Path | None,
    meta_payload: dict,
    overwrite: bool,
    dedupe: bool = True,
) -> None:
    write_scores_by_date(
        score_path,
        result.scores,
        overwrite=overwrite,
        dedupe=dedupe,
    )

    if weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and weights_path.exists():
            weights_path.unlink()
        if not result.weights_history.empty:
            weights = result.weights_history.copy()
            if not overwrite and weights_path.exists():
                try:
                    prior = pd.read_csv(weights_path)
                    weights = pd.concat([prior, weights], ignore_index=True, sort=False)
                except Exception:
                    # Keep a newly computed audit trail even if a legacy file
                    # is unreadable; this mirrors score IO's fail-open write.
                    pass
            if "trade_date" in weights.columns:
                weights["trade_date"] = pd.to_datetime(weights["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            if "factor" in weights.columns:
                weights["factor"] = weights["factor"].astype(str)
            key_cols = [col for col in ("trade_date", "factor") if col in weights.columns]
            if key_cols:
                weights = weights.drop_duplicates(subset=key_cols, keep="last")
            weights.to_csv(weights_path, index=False)

    if meta_path is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, ensure_ascii=False, indent=2, default=str)

