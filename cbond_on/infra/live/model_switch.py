from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
import math
from pathlib import Path

import pandas as pd


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

    champion = dict(cfg.get("champion", {}))
    challenger = dict(cfg.get("challenger", {}))
    champion_model_id = str(champion.get("model_id", "")).strip()
    challenger_model_id = str(challenger.get("model_id", "")).strip()
    champion_name = str(champion.get("name") or champion_model_id).strip()
    challenger_name = str(challenger.get("name") or challenger_model_id).strip()
    if not champion_model_id or not challenger_model_id:
        raise ValueError("model_switch champion/challenger model_id must not be empty")

    champion_score, champion_history_end, champion_history_days = rolling_sharpe_from_csv(
        champion.get("return_path") or champion.get("score_return_path") or "",
        score_day=score_day,
        lookback_days=lookback_days,
        min_periods=min_periods,
        return_col=return_col,
    )
    challenger_score, challenger_history_end, challenger_history_days = rolling_sharpe_from_csv(
        challenger.get("return_path") or challenger.get("score_return_path") or "",
        score_day=score_day,
        lookback_days=lookback_days,
        min_periods=min_periods,
        return_col=return_col,
    )

    history_end_candidates = [x for x in [champion_history_end, challenger_history_end] if x is not None]
    history_end = min(history_end_candidates) if history_end_candidates else None
    history_days = min(champion_history_days, challenger_history_days)

    if champion_score is None or challenger_score is None:
        return SwitchDecision(
            enabled=True,
            mode="single_challenger",
            metric="rolling_sharpe",
            lookback_days=lookback_days,
            min_periods=min_periods,
            threshold=threshold,
            score_day=score_day,
            selected_model_id=champion_model_id,
            selected_name=champion_name,
            champion_model_id=champion_model_id,
            champion_name=champion_name,
            challenger_model_id=challenger_model_id,
            challenger_name=challenger_name,
            champion_score=champion_score,
            challenger_score=challenger_score,
            score_diff=None,
            history_end=history_end,
            history_days=history_days,
            reason="insufficient_history_or_zero_volatility",
        )

    score_diff = float(challenger_score - champion_score)
    use_challenger = score_diff > threshold
    selected_model_id = challenger_model_id if use_challenger else champion_model_id
    selected_name = challenger_name if use_challenger else champion_name
    reason = "challenger_sharpe_gt_champion" if use_challenger else "champion_default"

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
        challenger_model_id=challenger_model_id,
        challenger_name=challenger_name,
        champion_score=float(champion_score),
        challenger_score=float(challenger_score),
        score_diff=score_diff,
        history_end=history_end,
        history_days=history_days,
        reason=reason,
    )


def _model_switch_groups(cfg: dict) -> tuple[dict, dict, str, str, str, str]:
    champion = dict(cfg.get("champion", {}))
    challenger = dict(cfg.get("challenger", {}))
    champion_model_id = str(champion.get("model_id", "")).strip()
    challenger_model_id = str(challenger.get("model_id", "")).strip()
    champion_name = str(champion.get("name") or champion_model_id).strip()
    challenger_name = str(challenger.get("name") or challenger_model_id).strip()
    if not champion_model_id or not challenger_model_id:
        raise ValueError("model_switch champion/challenger model_id must not be empty")
    return champion, challenger, champion_model_id, challenger_model_id, champion_name, challenger_name


def _regime_state_from_sum(value: float) -> str:
    return "up" if value >= 0.0 else "down"


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

    champion, challenger, champion_model_id, challenger_model_id, champion_name, challenger_name = _model_switch_groups(
        cfg
    )
    champion_history = _daily_return_frame(
        champion.get("return_path") or champion.get("score_return_path") or "",
        return_col=return_col,
        extra_cols=[benchmark_col],
    ).rename(columns={return_col: "champion_return", benchmark_col: "benchmark_return"})
    challenger_history = _daily_return_frame(
        challenger.get("return_path") or challenger.get("score_return_path") or "",
        return_col=return_col,
    ).rename(columns={return_col: "challenger_return"})
    history = champion_history.merge(
        challenger_history[["trade_date", "challenger_return"]],
        on="trade_date",
        how="inner",
    )
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
    challenger_score = float(regime_rows["challenger_return"].mean())
    score_diff = float(challenger_score - champion_score)
    use_challenger = score_diff > threshold
    selected_model_id = challenger_model_id if use_challenger else champion_model_id
    selected_name = challenger_name if use_challenger else champion_name
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
        challenger_model_id=challenger_model_id,
        challenger_name=challenger_name,
        champion_score=champion_score,
        challenger_score=challenger_score,
        score_diff=score_diff,
        history_end=history_end,
        history_days=observations,
        reason=reason,
        regime_state=current_state,
        regime_observations=observations,
        regime_benchmark_sum=current_benchmark_sum,
    )


def write_switch_decision(path: str | Path, decision: SwitchDecision, *, extra: dict | None = None) -> None:
    out = asdict(decision)
    if extra:
        out.update(extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
