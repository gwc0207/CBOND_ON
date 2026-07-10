from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
import json
import math
from pathlib import Path

import pandas as pd


T1430_DISPERSION_FEATURE_SETS: dict[str, list[str]] = {
    "disp_afternoon7": [
        "afternoon1300_1430_std",
        "afternoon1300_1430_iqr",
        "afternoon1300_1430_tail_spread",
        "last30_1330_1430_std",
        "last30_1330_1430_iqr",
        "last30_1330_1430_tail_spread",
        "dispersion_accel",
    ],
}


@dataclass(frozen=True)
class SwitchDecision:
    enabled: bool
    mode: str
    metric: str
    lookback_days: int
    min_periods: int
    threshold: float
    score_day: date
    selected_model_id: str
    selected_name: str
    champion_model_id: str
    champion_name: str
    challenger_model_id: str
    challenger_name: str
    champion_score: float | None
    challenger_score: float | None
    score_diff: float | None
    history_end: date | None
    history_days: int
    reason: str
    regime_state: str | None = None
    regime_observations: int | None = None
    regime_benchmark_sum: float | None = None
    fallback_reason: str | None = None
    candidate_scores: list[dict] | None = None


def _daily_return_frame(
    path: str | Path,
    *,
    return_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not str(path).strip() or not csv_path.exists():
        raise FileNotFoundError(f"model switch return history missing: {csv_path}")
    df = pd.read_csv(csv_path)
    cols = ["trade_date", return_col]
    for col in extra_cols or []:
        if col not in cols:
            cols.append(col)
    required = set(cols)
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise KeyError(f"model switch return history missing columns {missing}: {csv_path}")
    out = df[cols].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    for col in cols:
        if col != "trade_date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["trade_date", return_col])
    out = out.drop_duplicates(subset=["trade_date"], keep="last")
    return out.sort_values("trade_date")


def rolling_sharpe_from_csv(
    path: str | Path,
    *,
    score_day: date,
    lookback_days: int,
    min_periods: int,
    return_col: str = "day_return",
) -> tuple[float | None, date | None, int]:
    history = _daily_return_frame(path, return_col=return_col)
    history = history[history["trade_date"] < score_day].tail(int(lookback_days))
    if history.empty:
        return None, None, 0
    history_days = int(len(history))
    history_end = history["trade_date"].max()
    if history_days < int(min_periods):
        return None, history_end, history_days

    returns = history[return_col].astype(float)
    std = float(returns.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return None, history_end, history_days
    score = float(returns.mean()) / std * math.sqrt(252.0)
    if not math.isfinite(score):
        return None, history_end, history_days
    return score, history_end, history_days


def build_rank_average_scores(
    score_frames: list[tuple[str, pd.DataFrame]],
    *,
    score_day: date,
) -> pd.DataFrame:
    if not score_frames:
        raise ValueError("rankavg challenger requires at least one source score")

    merged: pd.DataFrame | None = None
    rank_cols: list[str] = []
    for idx, (name, raw) in enumerate(score_frames):
        if raw is None or raw.empty:
            raise ValueError(f"rankavg source score is empty: {name}")
        frame = raw[["code", "score"]].copy()
        frame["code"] = frame["code"].astype(str)
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame = frame.dropna(subset=["code", "score"])
        frame = frame.drop_duplicates(subset=["code"], keep="last")
        if frame.empty:
            raise ValueError(f"rankavg source score has no valid rows: {name}")
        rank_col = f"rank_{idx}"
        frame[rank_col] = frame["score"].rank(method="average", pct=True)
        frame = frame[["code", rank_col]]
        rank_cols.append(rank_col)
        merged = frame if merged is None else merged.merge(frame, on="code", how="inner")

    if merged is None or merged.empty:
        raise ValueError("rankavg challenger has no common codes across source scores")

    out = merged[["code"]].copy()
    out["trade_date"] = score_day
    out["score"] = merged[rank_cols].mean(axis=1)
    return out[["trade_date", "code", "score"]]


def _model_switch_groups(cfg: dict) -> tuple[dict, list[dict], str, str]:
    champion = dict(cfg.get("champion", {}))
    champion_model_id = str(champion.get("model_id", "")).strip()
    champion_name = str(champion.get("name") or champion_model_id).strip()

    challengers_raw = cfg.get("challengers")
    if isinstance(challengers_raw, list) and challengers_raw:
        challengers = [dict(item) for item in challengers_raw if isinstance(item, dict)]
    else:
        challenger = dict(cfg.get("challenger", {}))
        challengers = [challenger] if challenger else []
    if not champion_model_id or not challengers:
        raise ValueError("model_switch champion/challenger model_id must not be empty")
    for challenger in challengers:
        if not str(challenger.get("model_id", "")).strip():
            raise ValueError("model_switch champion/challenger model_id must not be empty")
    return champion, challengers, champion_model_id, champion_name


def _candidate_detail(
    *,
    role: str,
    name: str,
    model_id: str,
    score: float | None,
    history_end: date | None,
    history_days: int,
) -> dict:
    return {
        "role": role,
        "name": name,
        "model_id": model_id,
        "score": None if score is None else float(score),
        "history_end": history_end,
        "history_days": int(history_days),
    }


def _best_challenger(candidate_scores: list[dict]) -> dict | None:
    challengers = [
        item
        for item in candidate_scores
        if item.get("role") == "challenger"
        and item.get("score") is not None
        and math.isfinite(float(item.get("score")))
    ]
    if not challengers:
        return None
    return max(challengers, key=lambda item: float(item["score"]))


def _first_challenger(candidate_scores: list[dict]) -> dict:
    for item in candidate_scores:
        if item.get("role") == "challenger":
            return item
    raise ValueError("model_switch has no challenger candidate")


def _history_summary(candidate_scores: list[dict]) -> tuple[date | None, int]:
    ends = [item.get("history_end") for item in candidate_scores if item.get("history_end") is not None]
    days = [int(item.get("history_days", 0)) for item in candidate_scores]
    return (min(ends) if ends else None, min(days) if days else 0)


def decide_single_challenger_by_sharpe(
    cfg: dict,
    *,
    score_day: date,
) -> SwitchDecision:
    metric = str(cfg.get("metric", "rolling_sharpe")).strip().lower()
    if metric not in {"rolling_sharpe", "sharpe"}:
        raise ValueError(f"unsupported live model switch metric: {metric}")

    lookback_days = int(cfg.get("lookback_days", 60))
    min_periods = int(cfg.get("min_periods", lookback_days))
    threshold = float(cfg.get("threshold", 0.0))
    return_col = str(cfg.get("return_col", "day_return")).strip() or "day_return"

    champion, challengers, champion_model_id, champion_name = _model_switch_groups(cfg)

    champion_score, champion_history_end, champion_history_days = rolling_sharpe_from_csv(
        champion.get("return_path") or champion.get("score_return_path") or "",
        score_day=score_day,
        lookback_days=lookback_days,
        min_periods=min_periods,
        return_col=return_col,
    )
    candidate_scores: list[dict] = [
        _candidate_detail(
            role="champion",
            name=champion_name,
            model_id=champion_model_id,
            score=champion_score,
            history_end=champion_history_end,
            history_days=champion_history_days,
        )
    ]
    for challenger in challengers:
        challenger_model_id = str(challenger.get("model_id", "")).strip()
        challenger_name = str(challenger.get("name") or challenger_model_id).strip()
        challenger_score, challenger_history_end, challenger_history_days = rolling_sharpe_from_csv(
            challenger.get("return_path") or challenger.get("score_return_path") or "",
            score_day=score_day,
            lookback_days=lookback_days,
            min_periods=min_periods,
            return_col=return_col,
        )
        candidate_scores.append(
            _candidate_detail(
                role="challenger",
                name=challenger_name,
                model_id=challenger_model_id,
                score=challenger_score,
                history_end=challenger_history_end,
                history_days=challenger_history_days,
            )
        )

    best = _best_challenger(candidate_scores)
    fallback_challenger = best or _first_challenger(candidate_scores)
    history_end, history_days = _history_summary(candidate_scores)

    def _decision(
        *,
        selected_model_id: str,
        selected_name: str,
        challenger: dict,
        reason: str,
        score_diff: float | None,
    ) -> SwitchDecision:
        return SwitchDecision(
            enabled=True,
            mode="single_challenger",
            metric="rolling_sharpe",
            lookback_days=lookback_days,
            min_periods=min_periods,
            threshold=threshold,
            score_day=score_day,
            selected_model_id=selected_model_id,
            selected_name=selected_name,
            champion_model_id=champion_model_id,
            champion_name=champion_name,
            challenger_model_id=str(challenger["model_id"]),
            challenger_name=str(challenger["name"]),
            champion_score=champion_score,
            challenger_score=challenger.get("score"),
            score_diff=score_diff,
            history_end=history_end,
            history_days=history_days,
            reason=reason,
            candidate_scores=candidate_scores,
        )

    if champion_score is None or best is None:
        return _decision(
            selected_model_id=champion_model_id,
            selected_name=champion_name,
            challenger=fallback_challenger,
            reason="insufficient_history_or_zero_volatility",
            score_diff=None,
        )

    score_diff = float(float(best["score"]) - float(champion_score))
    if score_diff > threshold:
        return _decision(
            selected_model_id=str(best["model_id"]),
            selected_name=str(best["name"]),
            challenger=best,
            reason="challenger_sharpe_gt_champion",
            score_diff=score_diff,
        )
    return _decision(
        selected_model_id=champion_model_id,
        selected_name=champion_name,
        challenger=best,
        reason="champion_default",
        score_diff=score_diff,
    )


def _regime_state_from_sum(value: float) -> str:
    return "up" if value >= 0.0 else "down"


def _trim_mean(series: pd.Series, p: float) -> float:
    values = series.dropna().astype(float).sort_values().to_numpy()
    n = int(len(values))
    if n == 0:
        return float("nan")
    cut = int(math.floor(n * float(p)))
    if 2 * cut >= n:
        return float(values.mean())
    return float(values[cut : n - cut].mean())


def _scoreopt_trim20_lcb10(series: pd.Series) -> float:
    values = series.dropna().astype(float)
    if values.empty:
        return float("nan")
    center = _trim_mean(values, 0.20)
    clipped = values.clip(values.quantile(0.10), values.quantile(0.90))
    std = float(clipped.std(ddof=1))
    if not math.isfinite(std):
        std = 0.0
    return float(center - std / math.sqrt(float(len(clipped))))


def _scoreopt_score_sample(sample: pd.DataFrame, *, model_cols: list[str], score_mode: str) -> pd.Series:
    scores: dict[str, float] = {}
    for col in model_cols:
        values = sample[col].dropna().astype(float)
        if score_mode == "trim20_lcb10":
            scores[col] = _scoreopt_trim20_lcb10(values)
        elif score_mode == "lcb10":
            std = float(values.std(ddof=1))
            if not math.isfinite(std):
                std = 0.0
            scores[col] = float(values.mean() - std / math.sqrt(float(len(values)))) if len(values) else float("nan")
        elif score_mode == "mean":
            scores[col] = float(values.mean()) if len(values) else float("nan")
        else:
            raise ValueError(f"unsupported live scoreopt score_mode: {score_mode}")
    return pd.Series(scores)


def _scoreopt_bm_short_features(history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_frame = pd.DataFrame({"trade_date": history["trade_date"]})
    benchmark = history["benchmark_return"].astype(float)
    for window in [1, 3, 5, 10, 20, 40, 60]:
        shifted = benchmark.shift(1)
        if window == 1:
            feature_frame["bm_r1"] = shifted
        else:
            min_periods = max(2, min(window, 5))
            feature_frame[f"bm_sum{window}"] = shifted.rolling(window, min_periods=min_periods).sum()
            feature_frame[f"bm_vol{window}"] = shifted.rolling(window, min_periods=min_periods).std()

    benchmark_nav = (1.0 + benchmark.fillna(0.0)).cumprod().shift(1)
    for window in [20, 60]:
        min_periods = max(5, min(window, 10))
        feature_frame[f"bm_dd{window}"] = benchmark_nav / benchmark_nav.rolling(
            window,
            min_periods=min_periods,
        ).max() - 1.0

    cols = ["bm_r1", "bm_sum3", "bm_sum5", "bm_sum10", "bm_sum20", "bm_vol5", "bm_vol20", "bm_dd20"]
    return feature_frame[["trade_date", *cols]], cols


def _parse_hhmm(value: str) -> time:
    text = str(value).strip()
    hour, minute = text.split(":", 1)
    return time(int(hour), int(minute))


def _clean_snapshot_path(clean_root: str | Path, day: date) -> Path:
    base = Path(clean_root)
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day:%Y%m%d}.parquet"
    canonical = base / "snapshot" / "cbond" / month / filename
    if canonical.exists():
        return canonical
    return base / month / filename


def _snapshot_window_twap(
    snapshot: pd.DataFrame,
    *,
    day: date,
    start: str,
    end: str,
    price_field: str,
) -> pd.Series:
    start_dt = datetime.combine(day, _parse_hhmm(start))
    end_dt = datetime.combine(day, _parse_hhmm(end))
    if end_dt <= start_dt:
        raise ValueError(f"invalid t1430 twap window: {start}-{end}")

    required = {"code", "trade_time", price_field}
    missing = required - set(snapshot.columns)
    if missing:
        raise KeyError(f"snapshot missing columns for t1430 state features: {sorted(missing)}")

    work = snapshot.loc[
        (snapshot["trade_time"] >= start_dt) & (snapshot["trade_time"] <= end_dt),
        ["code", "trade_time", price_field],
    ].copy()
    if work.empty:
        return pd.Series(dtype=float)
    work["code"] = work["code"].astype(str)
    work[price_field] = pd.to_numeric(work[price_field], errors="coerce")
    work = work.dropna(subset=["code", "trade_time", price_field])
    if work.empty:
        return pd.Series(dtype=float)

    work = work.sort_values(["code", "trade_time"])
    work["next_time"] = work.groupby("code")["trade_time"].shift(-1)
    work["next_time"] = work["next_time"].fillna(end_dt)
    work["delta_sec"] = (work["next_time"] - work["trade_time"]).dt.total_seconds().clip(lower=0.0)
    weighted_sum = (work[price_field] * work["delta_sec"]).groupby(work["code"], sort=False).sum()
    weight = work["delta_sec"].groupby(work["code"], sort=False).sum()
    twap = (weighted_sum / weight).mask(weight <= 0)
    return twap.rename(f"twap_{start.replace(':', '')}_{end.replace(':', '')}")


def _safe_div_return(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator.astype(float) / denominator.astype(float) - 1.0
    out = out.replace([math.inf, -math.inf], math.nan)
    return out.dropna()


def _state_return_stats(prefix: str, values: pd.Series) -> dict[str, float]:
    series = values.replace([math.inf, -math.inf], math.nan).dropna().astype(float)
    if series.empty:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_iqr": math.nan,
            f"{prefix}_q10": math.nan,
            f"{prefix}_q90": math.nan,
            f"{prefix}_pos_ratio": math.nan,
            f"{prefix}_tail_spread": math.nan,
        }
    q10 = float(series.quantile(0.10))
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    q90 = float(series.quantile(0.90))
    return {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_median": float(series.median()),
        f"{prefix}_std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_q10": q10,
        f"{prefix}_q90": q90,
        f"{prefix}_pos_ratio": float((series > 0.0).mean()),
        f"{prefix}_tail_spread": float(q90 - q10),
    }


def build_t1430_market_state_feature_row(
    *,
    clean_root: str | Path,
    score_day: date,
    price_field: str = "last",
) -> dict[str, object]:
    snapshot_path = _clean_snapshot_path(clean_root, score_day)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"t1430 state snapshot missing: {snapshot_path}")
    snapshot = pd.read_parquet(snapshot_path)
    if "trade_time" not in snapshot.columns:
        raise KeyError(f"snapshot missing trade_time: {snapshot_path}")
    snapshot = snapshot.copy()
    snapshot["trade_time"] = pd.to_datetime(snapshot["trade_time"], errors="coerce")
    snapshot = snapshot.dropna(subset=["trade_time"])
    cutoff_dt = datetime.combine(score_day, time(14, 30))
    snapshot = snapshot[snapshot["trade_time"] <= cutoff_dt]
    if snapshot.empty:
        raise ValueError(f"t1430 state snapshot has no rows up to 14:30: {snapshot_path}")

    twaps = {
        "open0930_0935": _snapshot_window_twap(
            snapshot,
            day=score_day,
            start="09:30",
            end="09:35",
            price_field=price_field,
        ),
        "early0935_1000": _snapshot_window_twap(
            snapshot,
            day=score_day,
            start="09:35",
            end="10:00",
            price_field=price_field,
        ),
        "morning1100_1130": _snapshot_window_twap(
            snapshot,
            day=score_day,
            start="11:00",
            end="11:30",
            price_field=price_field,
        ),
        "afternoon1300_1330": _snapshot_window_twap(
            snapshot,
            day=score_day,
            start="13:00",
            end="13:30",
            price_field=price_field,
        ),
        "afternoon1330_1400": _snapshot_window_twap(
            snapshot,
            day=score_day,
            start="13:30",
            end="14:00",
            price_field=price_field,
        ),
        "afternoon1400_1430": _snapshot_window_twap(
            snapshot,
            day=score_day,
            start="14:00",
            end="14:30",
            price_field=price_field,
        ),
    }
    twap_frame = pd.concat(twaps.values(), axis=1, keys=twaps.keys())
    returns = {
        "full0935_1430": _safe_div_return(twap_frame["afternoon1400_1430"], twap_frame["open0930_0935"]),
        "morning0935_1130": _safe_div_return(twap_frame["morning1100_1130"], twap_frame["open0930_0935"]),
        "lunch_gap": _safe_div_return(twap_frame["afternoon1300_1330"], twap_frame["morning1100_1130"]),
        "afternoon1300_1430": _safe_div_return(twap_frame["afternoon1400_1430"], twap_frame["afternoon1300_1330"]),
        "last30_1330_1430": _safe_div_return(twap_frame["afternoon1400_1430"], twap_frame["afternoon1330_1400"]),
        "early0935_1000": _safe_div_return(twap_frame["early0935_1000"], twap_frame["open0930_0935"]),
    }

    row: dict[str, object] = {"trade_date": score_day, "valid_count": int(len(returns["full0935_1430"]))}
    for prefix, values in returns.items():
        row.update(_state_return_stats(prefix, values))
    row["trend_accel_median"] = float(row["afternoon1300_1430_median"]) - float(row["morning0935_1130_median"])
    row["trend_accel_mean"] = float(row["afternoon1300_1430_mean"]) - float(row["morning0935_1130_mean"])
    row["dispersion_accel"] = float(row["afternoon1300_1430_iqr"]) - float(row["morning0935_1130_iqr"])
    row["tail_balance_full"] = float(row["full0935_1430_q90"]) + float(row["full0935_1430_q10"])
    return row


def update_t1430_market_state_feature_history(
    *,
    state_feature_path: str | Path,
    clean_root: str | Path,
    score_day: date,
    price_field: str = "last",
) -> pd.DataFrame:
    target = Path(state_feature_path)
    row = build_t1430_market_state_feature_row(
        clean_root=clean_root,
        score_day=score_day,
        price_field=price_field,
    )
    if target.exists():
        history = pd.read_csv(target)
    else:
        history = pd.DataFrame()
    if not history.empty:
        history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce").dt.date
        history = history.dropna(subset=["trade_date"])
        history = history[history["trade_date"] != score_day]
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce").dt.date
    history = history.dropna(subset=["trade_date"]).sort_values("trade_date")
    target.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(target, index=False)
    return history


def decide_scoreopt_bm_short(
    cfg: dict,
    *,
    score_day: date,
) -> SwitchDecision:
    metric = str(cfg.get("score_mode") or cfg.get("metric") or "trim20_lcb10").strip().lower()
    if metric not in {"trim20_lcb10", "lcb10", "mean"}:
        raise ValueError(f"unsupported live scoreopt metric: {metric}")

    feature_set = str(cfg.get("feature_set", "bm_short")).strip().lower()
    if feature_set != "bm_short":
        raise ValueError(f"unsupported live scoreopt feature_set: {feature_set}")

    lookback_days = int(cfg.get("lookback_days", 180))
    nearest_k = int(cfg.get("nearest_k", cfg.get("k", cfg.get("min_periods", 40))))
    min_periods = int(cfg.get("min_periods", nearest_k))
    margin = float(cfg.get("margin", cfg.get("threshold", 0.0)))
    return_col = str(cfg.get("return_col", "day_return")).strip() or "day_return"
    benchmark_col = str(cfg.get("benchmark_col", "benchmark_return")).strip() or "benchmark_return"

    champion, challengers, champion_model_id, champion_name = _model_switch_groups(cfg)
    champion_history = _daily_return_frame(
        champion.get("return_path") or champion.get("score_return_path") or "",
        return_col=return_col,
        extra_cols=[benchmark_col],
    ).rename(columns={return_col: "Champion_HL20", benchmark_col: "benchmark_return"})
    history = champion_history
    model_cols = ["Champion_HL20"]
    model_meta = {
        "Champion_HL20": {
            "role": "champion",
            "name": champion_name,
            "model_id": champion_model_id,
        }
    }
    challenger_cols: list[tuple[dict, str]] = []
    for idx, challenger in enumerate(challengers):
        col = f"challenger_return_{idx}"
        challenger_history = _daily_return_frame(
            challenger.get("return_path") or challenger.get("score_return_path") or "",
            return_col=return_col,
        ).rename(columns={return_col: col})
        history = history.merge(challenger_history[["trade_date", col]], on="trade_date", how="inner")
        model_cols.append(col)
        challenger_cols.append((challenger, col))
        challenger_model_id = str(challenger.get("model_id", "")).strip()
        model_meta[col] = {
            "role": "challenger",
            "name": str(challenger.get("name") or challenger_model_id).strip(),
            "model_id": challenger_model_id,
        }

    history = history[history["trade_date"] < score_day].sort_values("trade_date").reset_index(drop=True)
    history_end = history["trade_date"].max() if not history.empty else None
    current_row = {col: math.nan for col in history.columns}
    current_row["trade_date"] = score_day
    work = pd.concat([history, pd.DataFrame([current_row])], ignore_index=True)
    features, feature_cols = _scoreopt_bm_short_features(work)
    current_features = features.iloc[-1][feature_cols]

    candidate_scores: list[dict]
    first_challenger = challengers[0]
    first_challenger_model_id = str(first_challenger.get("model_id", "")).strip()
    first_challenger_name = str(first_challenger.get("name") or first_challenger_model_id).strip()

    def _decision(
        *,
        selected_model_id: str,
        selected_name: str,
        reason: str,
        observations: int,
        score_gap: float | None,
        scores: pd.Series | None = None,
    ) -> SwitchDecision:
        details: list[dict] = []
        if scores is not None:
            for col in model_cols:
                meta = model_meta[col]
                details.append(
                    _candidate_detail(
                        role=str(meta["role"]),
                        name=str(meta["name"]),
                        model_id=str(meta["model_id"]),
                        score=None if pd.isna(scores.get(col)) else float(scores.get(col)),
                        history_end=history_end,
                        history_days=observations,
                    )
                )
        else:
            details = [
                _candidate_detail(
                    role="champion",
                    name=champion_name,
                    model_id=champion_model_id,
                    score=None,
                    history_end=history_end,
                    history_days=observations,
                )
            ]
            for challenger in challengers:
                challenger_model_id = str(challenger.get("model_id", "")).strip()
                details.append(
                    _candidate_detail(
                        role="challenger",
                        name=str(challenger.get("name") or challenger_model_id).strip(),
                        model_id=challenger_model_id,
                        score=None,
                        history_end=history_end,
                        history_days=observations,
                    )
                )
        best_challenger = _best_challenger(details) or _first_challenger(details)
        champion_score = details[0].get("score") if details else None
        return SwitchDecision(
            enabled=True,
            mode="scoreopt_bm_short",
            metric=metric,
            lookback_days=lookback_days,
            min_periods=min_periods,
            threshold=margin,
            score_day=score_day,
            selected_model_id=selected_model_id,
            selected_name=selected_name,
            champion_model_id=champion_model_id,
            champion_name=champion_name,
            challenger_model_id=str(best_challenger.get("model_id", first_challenger_model_id)),
            challenger_name=str(best_challenger.get("name", first_challenger_name)),
            champion_score=None if champion_score is None else float(champion_score),
            challenger_score=best_challenger.get("score"),
            score_diff=score_gap,
            history_end=history_end,
            history_days=observations,
            reason=reason,
            candidate_scores=details,
        )

    if current_features.isna().any():
        return _decision(
            selected_model_id=champion_model_id,
            selected_name=champion_name,
            reason="feature_na",
            observations=0,
            score_gap=None,
        )

    start = max(0, len(work) - 1 - lookback_days)
    history_features = features.iloc[start:-1][feature_cols]
    history_returns = work.iloc[start:-1][model_cols]
    valid_mask = ~(history_features.isna().any(axis=1) | history_returns.isna().any(axis=1))
    history_features = history_features.loc[valid_mask]
    history_returns = history_returns.loc[valid_mask]
    observations = int(len(history_features))
    if observations < nearest_k or observations < min_periods:
        return _decision(
            selected_model_id=champion_model_id,
            selected_name=champion_name,
            reason="insufficient_history",
            observations=observations,
            score_gap=None,
        )

    mean = history_features.mean(axis=0)
    std = history_features.std(axis=0).replace(0, math.nan).fillna(1.0)
    distances = (((history_features - mean) / std - (current_features - mean) / std) ** 2).sum(axis=1).pow(0.5)
    nearest_index = distances.sort_values().index[:nearest_k]
    sample = history_returns.loc[nearest_index]
    scores = _scoreopt_score_sample(sample, model_cols=model_cols, score_mode=metric).sort_values(ascending=False)
    best_col = str(scores.index[0])
    second_score = float(scores.iloc[1]) if len(scores) > 1 else float("nan")
    best_score = float(scores.iloc[0])
    score_gap = None if not math.isfinite(second_score) else float(best_score - second_score)
    best_meta = model_meta[best_col]
    use_best = score_gap is not None and score_gap > margin
    reason = "score_best" if use_best else "margin_default"
    selected_model_id = str(best_meta["model_id"]) if use_best else champion_model_id
    selected_name = str(best_meta["name"]) if use_best else champion_name
    return _decision(
        selected_model_id=selected_model_id,
        selected_name=selected_name,
        reason=reason,
        observations=int(len(sample)),
        score_gap=score_gap,
        scores=scores,
    )


def decide_scoreopt_t1430_dispersion(
    cfg: dict,
    *,
    score_day: date,
) -> SwitchDecision:
    metric = str(cfg.get("score_mode") or cfg.get("metric") or "lcb10").strip().lower()
    if metric not in {"trim20_lcb10", "lcb10", "mean"}:
        raise ValueError(f"unsupported live scoreopt metric: {metric}")

    feature_set = str(cfg.get("feature_set", "disp_afternoon7")).strip().lower()
    feature_cols = T1430_DISPERSION_FEATURE_SETS.get(feature_set)
    if not feature_cols:
        raise ValueError(f"unsupported live t1430 dispersion feature_set: {feature_set}")

    state_feature_path = str(cfg.get("state_feature_path", "")).strip()
    if not state_feature_path:
        raise ValueError("scoreopt_t1430_dispersion requires state_feature_path")
    state_path = Path(state_feature_path)
    if not state_path.exists():
        raise FileNotFoundError(f"t1430 state feature history missing: {state_path}")

    lookback_days = int(cfg.get("lookback_days", 120))
    nearest_k = int(cfg.get("nearest_k", cfg.get("k", cfg.get("min_periods", 20))))
    min_periods = int(cfg.get("min_periods", nearest_k))
    margin = float(cfg.get("margin", cfg.get("threshold", 0.0)))
    return_col = str(cfg.get("return_col", "day_return")).strip() or "day_return"

    champion, challengers, champion_model_id, champion_name = _model_switch_groups(cfg)
    champion_history = _daily_return_frame(
        champion.get("return_path") or champion.get("score_return_path") or "",
        return_col=return_col,
    ).rename(columns={return_col: "champion_return"})
    returns_history = champion_history
    model_cols = ["champion_return"]
    model_meta = {
        "champion_return": {
            "role": "champion",
            "name": champion_name,
            "model_id": champion_model_id,
        }
    }
    for idx, challenger in enumerate(challengers):
        col = f"challenger_return_{idx}"
        challenger_history = _daily_return_frame(
            challenger.get("return_path") or challenger.get("score_return_path") or "",
            return_col=return_col,
        ).rename(columns={return_col: col})
        returns_history = returns_history.merge(challenger_history[["trade_date", col]], on="trade_date", how="inner")
        model_cols.append(col)
        challenger_model_id = str(challenger.get("model_id", "")).strip()
        model_meta[col] = {
            "role": "challenger",
            "name": str(challenger.get("name") or challenger_model_id).strip(),
            "model_id": challenger_model_id,
        }

    returns_history = returns_history[returns_history["trade_date"] < score_day].sort_values("trade_date")
    history_end = returns_history["trade_date"].max() if not returns_history.empty else None

    state_features = pd.read_csv(state_path)
    if "trade_date" not in state_features.columns:
        raise KeyError(f"t1430 state feature history missing trade_date: {state_path}")
    missing_features = [col for col in feature_cols if col not in state_features.columns]
    if missing_features:
        raise KeyError(f"t1430 state feature history missing columns {missing_features}: {state_path}")
    state_features = state_features.copy()
    state_features["trade_date"] = pd.to_datetime(state_features["trade_date"], errors="coerce").dt.date
    for col in feature_cols:
        state_features[col] = pd.to_numeric(state_features[col], errors="coerce")
    state_features = state_features.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    current_rows = state_features[state_features["trade_date"] == score_day]

    first_challenger = challengers[0]
    first_challenger_model_id = str(first_challenger.get("model_id", "")).strip()
    first_challenger_name = str(first_challenger.get("name") or first_challenger_model_id).strip()

    def _decision(
        *,
        selected_model_id: str,
        selected_name: str,
        reason: str,
        observations: int,
        score_gap: float | None,
        scores: pd.Series | None = None,
    ) -> SwitchDecision:
        details: list[dict] = []
        if scores is not None:
            for col in model_cols:
                meta = model_meta[col]
                details.append(
                    _candidate_detail(
                        role=str(meta["role"]),
                        name=str(meta["name"]),
                        model_id=str(meta["model_id"]),
                        score=None if pd.isna(scores.get(col)) else float(scores.get(col)),
                        history_end=history_end,
                        history_days=observations,
                    )
                )
        else:
            details.append(
                _candidate_detail(
                    role="champion",
                    name=champion_name,
                    model_id=champion_model_id,
                    score=None,
                    history_end=history_end,
                    history_days=observations,
                )
            )
            for challenger in challengers:
                challenger_model_id = str(challenger.get("model_id", "")).strip()
                details.append(
                    _candidate_detail(
                        role="challenger",
                        name=str(challenger.get("name") or challenger_model_id).strip(),
                        model_id=challenger_model_id,
                        score=None,
                        history_end=history_end,
                        history_days=observations,
                    )
                )
        best_challenger = _best_challenger(details) or _first_challenger(details)
        champion_score = details[0].get("score") if details else None
        return SwitchDecision(
            enabled=True,
            mode="scoreopt_t1430_dispersion",
            metric=metric,
            lookback_days=lookback_days,
            min_periods=min_periods,
            threshold=margin,
            score_day=score_day,
            selected_model_id=selected_model_id,
            selected_name=selected_name,
            champion_model_id=champion_model_id,
            champion_name=champion_name,
            challenger_model_id=str(best_challenger.get("model_id", first_challenger_model_id)),
            challenger_name=str(best_challenger.get("name", first_challenger_name)),
            champion_score=None if champion_score is None else float(champion_score),
            challenger_score=best_challenger.get("score"),
            score_diff=score_gap,
            history_end=history_end,
            history_days=observations,
            reason=reason,
            candidate_scores=details,
        )

    if current_rows.empty:
        return _decision(
            selected_model_id=champion_model_id,
            selected_name=champion_name,
            reason="feature_missing",
            observations=0,
            score_gap=None,
        )
    current_features = current_rows.iloc[-1][feature_cols]
    if current_features.isna().any():
        return _decision(
            selected_model_id=champion_model_id,
            selected_name=champion_name,
            reason="feature_na",
            observations=0,
            score_gap=None,
        )

    joined = returns_history.merge(state_features[["trade_date", *feature_cols]], on="trade_date", how="inner")
    joined = joined.sort_values("trade_date").tail(lookback_days)
    history_features = joined[feature_cols]
    history_returns = joined[model_cols]
    valid_mask = ~(history_features.isna().any(axis=1) | history_returns.isna().any(axis=1))
    history_features = history_features.loc[valid_mask]
    history_returns = history_returns.loc[valid_mask]
    observations = int(len(history_features))
    if observations < nearest_k or observations < min_periods:
        return _decision(
            selected_model_id=champion_model_id,
            selected_name=champion_name,
            reason="insufficient_history",
            observations=observations,
            score_gap=None,
        )

    mean = history_features.mean(axis=0)
    std = history_features.std(axis=0).replace(0, math.nan).fillna(1.0)
    distances = (((history_features - mean) / std - (current_features - mean) / std) ** 2).sum(axis=1).pow(0.5)
    nearest_index = distances.sort_values().index[:nearest_k]
    sample = history_returns.loc[nearest_index]
    scores = _scoreopt_score_sample(sample, model_cols=model_cols, score_mode=metric).sort_values(ascending=False)
    best_col = str(scores.index[0])
    second_score = float(scores.iloc[1]) if len(scores) > 1 else float("nan")
    best_score = float(scores.iloc[0])
    score_gap = None if not math.isfinite(second_score) else float(best_score - second_score)
    best_meta = model_meta[best_col]
    use_best = score_gap is not None and score_gap > margin
    reason = "score_best" if use_best else "margin_default"
    selected_model_id = str(best_meta["model_id"]) if use_best else champion_model_id
    selected_name = str(best_meta["name"]) if use_best else champion_name
    return _decision(
        selected_model_id=selected_model_id,
        selected_name=selected_name,
        reason=reason,
        observations=int(len(sample)),
        score_gap=score_gap,
        scores=scores,
    )


def decide_single_challenger_by_regime(
    cfg: dict,
    *,
    score_day: date,
) -> SwitchDecision:
    metric = str(cfg.get("metric", "bm20_sign")).strip().lower()
    if metric not in {"bm20_sign", "regime_bm20_sign"}:
        raise ValueError(f"unsupported live model switch regime metric: {metric}")

    lookback_days = int(cfg.get("lookback_days", 120))
    min_periods = int(cfg.get("min_periods", 40))
    threshold = float(cfg.get("threshold", 0.0))
    benchmark_window_days = int(cfg.get("benchmark_window_days", 20))
    return_col = str(cfg.get("return_col", "day_return")).strip() or "day_return"
    benchmark_col = str(cfg.get("benchmark_col", "benchmark_return")).strip() or "benchmark_return"

    champion, challengers, champion_model_id, champion_name = _model_switch_groups(cfg)
    champion_history = _daily_return_frame(
        champion.get("return_path") or champion.get("score_return_path") or "",
        return_col=return_col,
        extra_cols=[benchmark_col],
    ).rename(columns={return_col: "champion_return", benchmark_col: "benchmark_return"})
    history = champion_history
    challenger_cols: list[tuple[dict, str]] = []
    for idx, challenger in enumerate(challengers):
        challenger_col = f"challenger_return_{idx}"
        challenger_history = _daily_return_frame(
            challenger.get("return_path") or challenger.get("score_return_path") or "",
            return_col=return_col,
        ).rename(columns={return_col: challenger_col})
        history = history.merge(
            challenger_history[["trade_date", challenger_col]],
            on="trade_date",
            how="inner",
        )
        challenger_cols.append((challenger, challenger_col))
    history = history[history["trade_date"] < score_day].sort_values("trade_date").reset_index(drop=True)
    history_end = history["trade_date"].max() if not history.empty else None

    fallback_cfg = dict(cfg)
    fallback_raw = cfg.get("fallback", {})
    if isinstance(fallback_raw, dict):
        fallback_cfg.update(fallback_raw)
    fallback_cfg["metric"] = "rolling_sharpe"

    def _fallback(reason: str, *, state: str | None = None, benchmark_sum: float | None = None, obs: int = 0) -> SwitchDecision:
        fallback = decide_single_challenger_by_sharpe(fallback_cfg, score_day=score_day)
        return SwitchDecision(
            enabled=fallback.enabled,
            mode="regime_bm20_sign",
            metric="bm20_sign",
            lookback_days=lookback_days,
            min_periods=min_periods,
            threshold=threshold,
            score_day=fallback.score_day,
            selected_model_id=fallback.selected_model_id,
            selected_name=fallback.selected_name,
            champion_model_id=fallback.champion_model_id,
            champion_name=fallback.champion_name,
            challenger_model_id=fallback.challenger_model_id,
            challenger_name=fallback.challenger_name,
            champion_score=fallback.champion_score,
            challenger_score=fallback.challenger_score,
            score_diff=fallback.score_diff,
            history_end=fallback.history_end,
            history_days=fallback.history_days,
            reason=f"fallback_rolling_sharpe_{reason}",
            regime_state=state,
            regime_observations=obs,
            regime_benchmark_sum=benchmark_sum,
            fallback_reason=fallback.reason,
            candidate_scores=fallback.candidate_scores,
        )

    if len(history) < benchmark_window_days:
        return _fallback("insufficient_benchmark_history")

    benchmark_returns = history["benchmark_return"].astype(float)
    current_benchmark_sum = float(benchmark_returns.tail(benchmark_window_days).sum())
    current_state = _regime_state_from_sum(current_benchmark_sum)

    work = history.copy()
    work["regime_benchmark_sum"] = (
        work["benchmark_return"].astype(float).rolling(benchmark_window_days, min_periods=benchmark_window_days).sum().shift(1)
    )
    work["regime_state"] = work["regime_benchmark_sum"].map(
        lambda value: _regime_state_from_sum(float(value)) if math.isfinite(float(value)) else None
    )
    candidates = work.tail(lookback_days)
    regime_rows = candidates[candidates["regime_state"] == current_state].copy()
    observations = int(len(regime_rows))
    if observations < min_periods:
        return _fallback(
            "insufficient_regime_observations",
            state=current_state,
            benchmark_sum=current_benchmark_sum,
            obs=observations,
        )

    champion_score = float(regime_rows["champion_return"].mean())
    candidate_scores: list[dict] = [
        _candidate_detail(
            role="champion",
            name=champion_name,
            model_id=champion_model_id,
            score=champion_score,
            history_end=history_end,
            history_days=observations,
        )
    ]
    for challenger, challenger_col in challenger_cols:
        challenger_model_id = str(challenger.get("model_id", "")).strip()
        challenger_name = str(challenger.get("name") or challenger_model_id).strip()
        candidate_scores.append(
            _candidate_detail(
                role="challenger",
                name=challenger_name,
                model_id=challenger_model_id,
                score=float(regime_rows[challenger_col].mean()),
                history_end=history_end,
                history_days=observations,
            )
        )

    best = _best_challenger(candidate_scores)
    fallback_challenger = best or _first_challenger(candidate_scores)
    score_diff = None if best is None else float(float(best["score"]) - champion_score)
    use_challenger = score_diff is not None and score_diff > threshold
    selected_model_id = str(best["model_id"]) if use_challenger and best is not None else champion_model_id
    selected_name = str(best["name"]) if use_challenger and best is not None else champion_name
    reason = "regime_challenger_gt_champion" if use_challenger else "regime_champion_default"

    return SwitchDecision(
        enabled=True,
        mode="regime_bm20_sign",
        metric="bm20_sign",
        lookback_days=lookback_days,
        min_periods=min_periods,
        threshold=threshold,
        score_day=score_day,
        selected_model_id=selected_model_id,
        selected_name=selected_name,
        champion_model_id=champion_model_id,
        champion_name=champion_name,
        challenger_model_id=str(fallback_challenger["model_id"]),
        challenger_name=str(fallback_challenger["name"]),
        champion_score=champion_score,
        challenger_score=fallback_challenger.get("score"),
        score_diff=score_diff,
        history_end=history_end,
        history_days=observations,
        reason=reason,
        regime_state=current_state,
        regime_observations=observations,
        regime_benchmark_sum=current_benchmark_sum,
        candidate_scores=candidate_scores,
    )


def write_switch_decision(path: str | Path, decision: SwitchDecision, *, extra: dict | None = None) -> None:
    out = asdict(decision)
    if extra:
        out.update(extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
