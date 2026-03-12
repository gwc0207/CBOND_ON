from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

from cbond_on.models.impl.lgbm.trainer import (
    SplitData,
    _iter_existing_label_days,
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


def _rank_ic_by_day(df: pd.DataFrame, factor_col: str) -> pd.Series:
    def _calc(group: pd.DataFrame) -> float:
        g = group[[factor_col, "y"]].dropna()
        if len(g) < 2:
            return np.nan
        return g[factor_col].corr(g["y"], method="spearman")

    return df.groupby("dt").apply(_calc, include_groups=False)


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
) -> tuple[object, dict]:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    if train.x.empty or not train.group:
        raise RuntimeError("ranker train split is empty")

    params = dict(lgbm_ranker_params)
    params.setdefault("objective", "lambdarank")
    params.setdefault("metric", "ndcg")
    model = lgb.LGBMRanker(**params)
    history: list[dict] = []

    def _mean_rank_ic(split_data: RankerSplitData, pred: np.ndarray) -> float:
        if split_data.x.empty:
            return float("nan")
        frame = pd.DataFrame({"dt": split_data.dt, "y": split_data.y, "pred": pred})
        rank_ic = _rank_ic_by_day(frame, "pred").dropna()
        if rank_ic.empty:
            return float("nan")
        return float(rank_ic.mean())

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
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            if not str(metric_name).startswith("ndcg"):
                continue
            if data_name == "training":
                train_ndcg.append(float(value))
            elif data_name in {"valid_0", "valid_1", "valid"}:
                val_ndcg.append(float(value))
        train_ndcg_mean = float(np.mean(train_ndcg)) if train_ndcg else float("nan")
        val_ndcg_mean = float(np.mean(val_ndcg)) if val_ndcg else float("nan")

        train_rank_ic = float("nan")
        val_rank_ic = float("nan")
        try:
            current_it = env.model.current_iteration()
            train_pred = env.model.predict(train.x, num_iteration=current_it)
            train_rank_ic = _mean_rank_ic(train, train_pred)
            if not val.x.empty:
                val_pred = env.model.predict(val.x, num_iteration=current_it)
                val_rank_ic = _mean_rank_ic(val, val_pred)
        except Exception:
            pass

        history.append(
            {
                "iteration": iteration,
                "train_rank_ic": train_rank_ic,
                "val_rank_ic": val_rank_ic,
                "train_ndcg": train_ndcg_mean,
                "val_ndcg": val_ndcg_mean,
            }
        )
        print(
            f"iter {iteration:03d} "
            f"train_rank_ic={train_rank_ic:.4f} val_rank_ic={val_rank_ic:.4f} "
            f"train_ndcg={train_ndcg_mean:.4f} val_ndcg={val_ndcg_mean:.4f}"
        )

    has_val = (not val.x.empty) and bool(val.group)
    if has_val and early_stopping_rounds is not None:
        fit_kwargs = {
            "group": train.group,
            "eval_set": [(train.x, train.relevance), (val.x, val.relevance)],
            "eval_group": [train.group, val.group],
            "eval_metric": _eval_rank_ic,
        }
        try:
            model.fit(
                train.x,
                train.relevance,
                **fit_kwargs,
                early_stopping_rounds=int(early_stopping_rounds),
                verbose=False,
                callbacks=[_record_callback],
            )
            return model, params | {"history": history}
        except TypeError:
            callbacks = []
            if hasattr(lgb, "early_stopping"):
                callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
            callbacks.append(lambda env: _record_callback(env))
            model.fit(train.x, train.relevance, **fit_kwargs, callbacks=callbacks)
            return model, params | {"history": history}

    fit_kwargs = {"group": train.group}
    if has_val:
        fit_kwargs["eval_set"] = [(train.x, train.relevance), (val.x, val.relevance)]
        fit_kwargs["eval_group"] = [train.group, val.group]
        fit_kwargs["eval_metric"] = _eval_rank_ic
        model.fit(train.x, train.relevance, **fit_kwargs, callbacks=[_record_callback])
    else:
        model.fit(train.x, train.relevance, **fit_kwargs)
    return model, params | {"history": history}

