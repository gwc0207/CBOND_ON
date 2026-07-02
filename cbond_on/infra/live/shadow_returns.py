from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.domain.portfolio.service import normalize_weights, to_prev_positions
from cbond_on.domain.signals.service import SignalSelectionRequest, select_signals
from cbond_on.infra.benchmark.service import (
    build_strict_buy_holdings_from_selection,
    compute_benchmark_breakdowns_for_days,
    compute_strict_sell_detail_for_holdings,
    load_strict_market_day,
)
from cbond_on.infra.model.score_io import load_scores_by_date
from cbond_on.infra.universe.pool_filter import (
    apply_allowlist_filter_to_universe,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)


@dataclass(frozen=True)
class ShadowReturnUpdateResult:
    model_id: str
    score_path: str
    return_path: str
    expected_history_end: date | None
    before_history_end: date | None
    after_history_end: date | None
    appended_rows: int
    status: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_history(df: pd.DataFrame, *, return_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "trade_date" not in df.columns:
        raise KeyError("shadow return history missing trade_date column")
    if return_col not in df.columns:
        raise KeyError(f"shadow return history missing {return_col} column")
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    out[return_col] = pd.to_numeric(out[return_col], errors="coerce")
    out = out.dropna(subset=["trade_date", return_col])
    out = out.drop_duplicates(subset=["trade_date"], keep="last")
    return out.sort_values("trade_date").reset_index(drop=True)


def _read_history(path: str | Path, *, return_col: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=["trade_date", return_col])
    return _normalize_history(pd.read_csv(csv_path), return_col=return_col)


def _history_end(history: pd.DataFrame) -> date | None:
    if history.empty or "trade_date" not in history.columns:
        return None
    values = pd.to_datetime(history["trade_date"], errors="coerce").dt.date.dropna()
    if values.empty:
        return None
    return values.max()


def _open_days(raw_data_root: str, start_day: date, end_day: date) -> list[date]:
    if start_day > end_day:
        return []
    return list_trading_days_from_raw(
        raw_data_root,
        start_day,
        end_day,
        kind="snapshot",
        asset="cbond",
    )


def _build_shadow_daily_returns(
    *,
    raw_data_root: str,
    score_path: str | Path,
    start_day: date,
    end_day: date,
    strategy_id: str,
    strategy_config: dict,
    allowlist_cfg: dict,
) -> pd.DataFrame:
    days = _open_days(raw_data_root, start_day, end_day)
    if not days:
        return pd.DataFrame()

    scores = load_scores_by_date(score_path)
    buy_cost_bps, sell_cost_bps, _ = load_fees_buy_sell_bps()
    pool_cfg = load_upstream_pool_config(allowlist_cfg or None)
    benchmark_daily = compute_benchmark_breakdowns_for_days(
        raw_data_root=raw_data_root,
        trade_days=days,
        buy_bps=buy_cost_bps,
        sell_bps=sell_cost_bps,
        skip_failed_days=True,
    )
    benchmark_by_day: dict[date, pd.Series] = {}
    if not benchmark_daily.empty:
        bench = benchmark_daily.copy()
        bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce").dt.date
        bench = bench.dropna(subset=["trade_date"])
        benchmark_by_day = {row["trade_date"]: row for _, row in bench.iterrows()}

    daily_rows: list[dict] = []
    prev_positions = pd.DataFrame(columns=["code", "weight"])
    strategy_prev_holdings = pd.DataFrame()

    for day in days:
        score_df = scores.get(day, pd.DataFrame())
        if score_df.empty:
            continue
        try:
            merged = load_strict_market_day(
                raw_data_root=raw_data_root,
                trade_day=day,
                buy_bps=buy_cost_bps,
                sell_bps=sell_cost_bps,
            )
        except Exception:
            continue
        merged = merged[
            pd.to_numeric(merged["buy_price"], errors="coerce").notna()
            & pd.to_numeric(merged["buy_close_price"], errors="coerce").notna()
            & (pd.to_numeric(merged["buy_price"], errors="coerce") > 0)
            & (pd.to_numeric(merged["buy_close_price"], errors="coerce") > 0)
        ].copy()
        if merged.empty:
            continue

        pool_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=raw_data_root,
            trade_day=day,
            pool_cfg=pool_cfg,
            enabled=True,
        )
        if bool(pool_info.get("fallback_no_filter", False)):
            raise RuntimeError(
                "[shadow_return] required allowlist is unavailable; "
                f"trade_day={day:%Y-%m-%d} "
                f"expected_pool_day={pool_info.get('pool_day_expected')} "
                f"reason={pool_info.get('fallback_reason')}"
            )
        merged = apply_allowlist_filter_to_universe(merged, allowlist_codes=pool_codes)
        if merged.empty:
            continue
        merged = merged.merge(score_df[["code", "score"]], on="code", how="inner")
        if merged.empty:
            continue

        picks = select_signals(
            SignalSelectionRequest(
                universe=merged[["code", "score"]],
                trade_date=day,
                prev_positions=prev_positions,
                strategy_id=strategy_id,
                strategy_config=strategy_config,
            )
        )
        if picks.empty:
            continue

        picks = build_strict_buy_holdings_from_selection(
            raw_data_root=raw_data_root,
            buy_day=day,
            selection=picks,
            buy_bps=buy_cost_bps,
            normalize=True,
        )
        if picks.empty:
            continue

        picks = normalize_weights(picks, weight_col="weight")
        picks["return"] = pd.to_numeric(picks["buy_leg_ret_net"], errors="coerce")
        picks["full_cycle_ret_net"] = picks["return"]
        day_buy_leg_ret = float(pd.to_numeric(picks["weighted_buy_leg_ret_net"], errors="coerce").sum())
        day_sell_leg_ret = 0.0
        sell_count = 0
        fallback_sell_codes = 0
        fallback_sell_weight = 0.0
        if not strategy_prev_holdings.empty:
            sell_detail = compute_strict_sell_detail_for_holdings(
                raw_data_root=raw_data_root,
                sell_day=day,
                prev_holdings=strategy_prev_holdings,
                sell_bps=sell_cost_bps,
            )
            if not sell_detail.empty:
                day_sell_leg_ret = float(
                    pd.to_numeric(sell_detail["weighted_sell_leg_ret_net"], errors="coerce").sum()
                )
                sell_count = int(sell_detail["code"].nunique())
                fallback_mask = sell_detail["sell_missing_fallback"].astype(bool)
                fallback_sell_codes = int(fallback_mask.sum())
                fallback_sell_weight = float(
                    pd.to_numeric(sell_detail.loc[fallback_mask, "weight"], errors="coerce").sum()
                )

        benchmark_row = benchmark_by_day.get(day)
        if benchmark_row is None:
            continue
        day_return = day_sell_leg_ret + day_buy_leg_ret
        daily_rows.append(
            {
                "trade_date": day,
                "count": int(len(picks)),
                "day_return": day_return,
                "full_cycle_ret_net": day_return,
                "buy_leg_ret_net": day_buy_leg_ret,
                "sell_leg_ret_net": day_sell_leg_ret,
                "benchmark_return": float(benchmark_row["benchmark_return"]),
                "benchmark_full_cycle_ret_net": float(benchmark_row["benchmark_return"]),
                "benchmark_buy_leg_ret_net": float(benchmark_row["buy_leg_ret_net"]),
                "benchmark_sell_leg_ret_net": float(benchmark_row["sell_leg_ret_net"]),
                "benchmark_buy_count": int(benchmark_row.get("buy_count", benchmark_row.get("count", 0))),
                "benchmark_sell_count": int(benchmark_row.get("sell_count", 0)),
                "benchmark_fallback_sell_codes": int(benchmark_row.get("fallback_sell_codes", 0)),
                "benchmark_fallback_sell_weight": float(benchmark_row.get("fallback_sell_weight", 0.0)),
                "benchmark_method": str(
                    benchmark_row.get("benchmark_method", "strict_official_prev_close_split")
                ),
                "avg_return": float(picks["return"].mean()),
                "total_weight": float(picks["weight"].sum()),
                "sell_count": sell_count,
                "fallback_sell_codes": fallback_sell_codes,
                "fallback_sell_weight": fallback_sell_weight,
            }
        )
        prev_positions = to_prev_positions(picks)
        strategy_prev_holdings = picks

    if not daily_rows:
        return pd.DataFrame()
    return pd.DataFrame(daily_rows).sort_values("trade_date").reset_index(drop=True)


def update_shadow_return_history(
    *,
    model_id: str,
    raw_data_root: str,
    score_path: str | Path,
    return_path: str | Path,
    expected_history_end: date | None,
    strategy_id: str,
    strategy_config: dict,
    allowlist_cfg: dict,
    return_col: str = "day_return",
) -> ShadowReturnUpdateResult:
    history = _read_history(return_path, return_col=return_col)
    before_end = _history_end(history)
    if expected_history_end is None:
        return ShadowReturnUpdateResult(
            model_id=model_id,
            score_path=str(score_path),
            return_path=str(return_path),
            expected_history_end=None,
            before_history_end=before_end,
            after_history_end=before_end,
            appended_rows=0,
            status="skipped",
            reason="expected_history_end_missing",
        )
    if before_end is None:
        return ShadowReturnUpdateResult(
            model_id=model_id,
            score_path=str(score_path),
            return_path=str(return_path),
            expected_history_end=expected_history_end,
            before_history_end=None,
            after_history_end=None,
            appended_rows=0,
            status="skipped",
            reason="return_history_empty",
        )
    if before_end >= expected_history_end:
        return ShadowReturnUpdateResult(
            model_id=model_id,
            score_path=str(score_path),
            return_path=str(return_path),
            expected_history_end=expected_history_end,
            before_history_end=before_end,
            after_history_end=before_end,
            appended_rows=0,
            status="up_to_date",
            reason="history_already_current",
        )

    computed = _build_shadow_daily_returns(
        raw_data_root=raw_data_root,
        score_path=score_path,
        start_day=before_end,
        end_day=expected_history_end,
        strategy_id=strategy_id,
        strategy_config=strategy_config,
        allowlist_cfg=allowlist_cfg,
    )
    if computed.empty:
        return ShadowReturnUpdateResult(
            model_id=model_id,
            score_path=str(score_path),
            return_path=str(return_path),
            expected_history_end=expected_history_end,
            before_history_end=before_end,
            after_history_end=before_end,
            appended_rows=0,
            status="skipped",
            reason="no_shadow_rows_computed",
        )

    computed = computed.copy()
    computed["trade_date"] = pd.to_datetime(computed["trade_date"], errors="coerce").dt.date
    computed_days = set(computed["trade_date"].dropna().tolist())
    if before_end not in computed_days:
        return ShadowReturnUpdateResult(
            model_id=model_id,
            score_path=str(score_path),
            return_path=str(return_path),
            expected_history_end=expected_history_end,
            before_history_end=before_end,
            after_history_end=before_end,
            appended_rows=0,
            status="skipped",
            reason="warm_start_score_missing",
        )
    append_rows = computed[
        (computed["trade_date"] > before_end)
        & (computed["trade_date"] <= expected_history_end)
    ].copy()
    if append_rows.empty:
        return ShadowReturnUpdateResult(
            model_id=model_id,
            score_path=str(score_path),
            return_path=str(return_path),
            expected_history_end=expected_history_end,
            before_history_end=before_end,
            after_history_end=before_end,
            appended_rows=0,
            status="skipped",
            reason="no_new_rows_after_warm_start",
        )

    out = pd.concat([history, append_rows], ignore_index=True, sort=False)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    out = out.dropna(subset=["trade_date", return_col])
    out = out.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date")
    after_end = _history_end(out)
    write_df = out.copy()
    write_df["trade_date"] = pd.to_datetime(write_df["trade_date"]).dt.strftime("%Y-%m-%d")
    target = Path(return_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_df.to_csv(target, index=False)

    status = "updated" if after_end and after_end >= expected_history_end else "partial"
    reason = "appended_missing_shadow_returns" if status == "updated" else "history_still_stale_after_update"
    return ShadowReturnUpdateResult(
        model_id=model_id,
        score_path=str(score_path),
        return_path=str(return_path),
        expected_history_end=expected_history_end,
        before_history_end=before_end,
        after_history_end=after_end,
        appended_rows=int(len(append_rows)),
        status=status,
        reason=reason,
    )
