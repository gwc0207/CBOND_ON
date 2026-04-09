from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

from cbond_on.infra.model.impl.lgbm.trainer import (
    SplitData,
    _build_day_group_indices,
    _iter_existing_label_days,
    _lgbm_cpu_params,
    _lgbm_gpu_requested,
    _looks_like_lgbm_gpu_error,
    _mean_rank_ic_by_groups,
    _split_days,
    build_dataset,
    evaluate_metrics,
)


@dataclass
class RankerSplitData:
    x: pd.DataFrame
    y: pd.Series
    dt: pd.Series
    code: pd.Series
    relevance: pd.Series
    group: list[int]


def build_ranker_split_data(
    split: SplitData,
    *,
    relevance_bins: int,
) -> RankerSplitData:
    if split.x.empty:
        empty_x = split.x.copy()
        empty_s = pd.Series(dtype=float)
        return RankerSplitData(
            x=empty_x,
            y=empty_s,
            dt=empty_s,
            code=pd.Series(dtype=str),
            relevance=pd.Series(dtype=np.int32),
            group=[],
        )

    work = split.x.copy()
    work["y"] = pd.to_numeric(split.y, errors="coerce")
    work["dt"] = pd.to_datetime(split.dt, errors="coerce")
    work["code"] = split.code.astype(str)
    work = work.dropna(subset=["dt", "y"])
    if work.empty:
        empty_x = split.x.iloc[0:0].copy()
        return RankerSplitData(
            x=empty_x,
            y=pd.Series(dtype=float),
            dt=pd.Series(dtype="datetime64[ns]"),
            code=pd.Series(dtype=str),
            relevance=pd.Series(dtype=np.int32),
            group=[],
        )

    factor_cols = [c for c in split.x.columns]
    work = work.sort_values(["dt", "code"]).reset_index(drop=True)
    pct = work.groupby("dt")["y"].rank(method="average", pct=True)
    rel = np.floor(pct.to_numpy() * float(relevance_bins)).astype(np.int32)
    rel = np.clip(rel, 0, int(relevance_bins) - 1)
    work["relevance"] = rel
    group = work.groupby("dt", sort=True).size().astype(int).tolist()

    return RankerSplitData(
        x=work[factor_cols].copy(),
        y=work["y"].copy(),
        dt=work["dt"].copy(),
        code=work["code"].copy(),
        relevance=work["relevance"].copy(),
        group=group,
    )


def train_lgbm_ranker(
    *,
    train: RankerSplitData,
    val: RankerSplitData,
    lgbm_ranker_params: dict,
    early_stopping_rounds: int | None = None,
    init_model: object | str | None = None,
) -> tuple[object, dict]:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    if train.x.empty or not train.group:
        raise RuntimeError("ranker train split is empty")

    params = dict(lgbm_ranker_params)
    params.setdefault("objective", "lambdarank")
    params.setdefault("metric", "ndcg")
    log_eval_period = max(1, int(params.pop("log_eval_period", 10)))
    gpu_requested = _lgbm_gpu_requested(params)
    model = lgb.LGBMRanker(**params)
    history: list[dict] = []
    train_groups = _build_day_group_indices(train.dt)
    val_groups = _build_day_group_indices(val.dt)

    def _mean_rank_ic(split_data: RankerSplitData, pred: np.ndarray) -> float:
        if split_data.x.empty:
            return float("nan")
        y_true = np.asarray(split_data.y, dtype=float)
        y_pred = np.asarray(pred, dtype=float)
        groups = train_groups if split_data is train else val_groups
        return _mean_rank_ic_by_groups(y_true, y_pred, groups)

    def _eval_rank_ic(y_true, y_pred):
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.shape[0] == train.y.shape[0]:
            score = _mean_rank_ic(train, y_pred_arr)
        elif y_pred_arr.shape[0] == val.y.shape[0]:
            score = _mean_rank_ic(val, y_pred_arr)
        else:
            score = float("nan")
        if not np.isfinite(score):
            score = 0.0
        return ("rank_ic", float(score), True)

    def _record_callback(env):
        iteration = env.iteration + 1
        train_ndcg: list[float] = []
        val_ndcg: list[float] = []
        train_rank_ic: list[float] = []
        val_rank_ic: list[float] = []
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            metric_name = str(metric_name)
            if metric_name.startswith("ndcg"):
                if data_name == "training":
                    train_ndcg.append(float(value))
                elif data_name in {"valid_0", "valid_1", "valid"}:
                    val_ndcg.append(float(value))
                continue
            if metric_name != "rank_ic":
                continue
            if data_name == "training":
                train_rank_ic.append(float(value))
            elif data_name in {"valid_0", "valid_1", "valid"}:
                val_rank_ic.append(float(value))
        train_ndcg_mean = float(np.mean(train_ndcg)) if train_ndcg else float("nan")
        val_ndcg_mean = float(np.mean(val_ndcg)) if val_ndcg else float("nan")
        train_rank_ic_val = float(np.mean(train_rank_ic)) if train_rank_ic else float("nan")
        val_rank_ic_val = float(np.mean(val_rank_ic)) if val_rank_ic else float("nan")

        history.append(
            {
                "iteration": iteration,
                "train_rank_ic": train_rank_ic_val,
                "val_rank_ic": val_rank_ic_val,
                "train_ndcg": train_ndcg_mean,
                "val_ndcg": val_ndcg_mean,
            }
        )
        if iteration % log_eval_period != 0:
            return
        print(
            f"iter {iteration:03d} "
            f"train_rank_ic={train_rank_ic_val:.4f} val_rank_ic={val_rank_ic_val:.4f} "
            f"train_ndcg={train_ndcg_mean:.4f} val_ndcg={val_ndcg_mean:.4f}"
        )

    def _fit_once(estimator) -> None:
        base_fit_kwargs = {}
        if init_model is not None:
            base_fit_kwargs["init_model"] = init_model
        has_val = (not val.x.empty) and bool(val.group)
        if has_val and early_stopping_rounds is not None:
            fit_kwargs = {
                **base_fit_kwargs,
                "group": train.group,
                "eval_set": [(train.x, train.relevance), (val.x, val.relevance)],
                "eval_group": [train.group, val.group],
                "eval_metric": _eval_rank_ic,
            }
            try:
                estimator.fit(
                    train.x,
                    train.relevance,
                    **fit_kwargs,
                    early_stopping_rounds=int(early_stopping_rounds),
                    verbose=False,
                    callbacks=[_record_callback],
                )
                return
            except TypeError:
                callbacks = []
                if hasattr(lgb, "early_stopping"):
                    callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
                callbacks.append(lambda env: _record_callback(env))
                try:
                    estimator.fit(train.x, train.relevance, **fit_kwargs, callbacks=callbacks)
                except TypeError:
                    # Older sklearn wrappers may not accept init_model.
                    fit_kwargs.pop("init_model", None)
                    estimator.fit(train.x, train.relevance, **fit_kwargs, callbacks=callbacks)
                return

        fit_kwargs = {**base_fit_kwargs, "group": train.group}
        if has_val:
            fit_kwargs["eval_set"] = [(train.x, train.relevance), (val.x, val.relevance)]
            fit_kwargs["eval_group"] = [train.group, val.group]
            fit_kwargs["eval_metric"] = _eval_rank_ic
            try:
                estimator.fit(train.x, train.relevance, **fit_kwargs, callbacks=[_record_callback])
            except TypeError:
                fit_kwargs.pop("init_model", None)
                estimator.fit(train.x, train.relevance, **fit_kwargs, callbacks=[_record_callback])
        else:
            try:
                estimator.fit(train.x, train.relevance, **fit_kwargs)
            except TypeError:
                fit_kwargs.pop("init_model", None)
                estimator.fit(train.x, train.relevance, **fit_kwargs)

    try:
        _fit_once(model)
    except Exception as exc:
        if not (gpu_requested and _looks_like_lgbm_gpu_error(exc)):
            raise
        cpu_params = _lgbm_cpu_params(params)
        print(
            "[LightGBM] GPU unavailable, fallback to CPU:",
            f"{type(exc).__name__}: {exc}",
        )
        history.clear()
        model = lgb.LGBMRanker(**cpu_params)
        _fit_once(model)
        params = cpu_params

    return model, params | {"history": history}


