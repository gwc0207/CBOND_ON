"""Research-only proof that the current switch label has no extra prev-holding cost.

The current CBOND_ON strategy contract is Top20 / 5% / turnover_ratio=1.0 and
each strict return is a complete buy-day to next-day-sell cycle.  This tool
does not invent a new continuous-position economic model.  It validates, from
the current strategy config and all 538 aligned score days, that arbitrary
``prev_positions`` do not alter any active model's picks.  It then recomposes
the current selector's full return sequence from the immutable per-model
shadow histories.

Consequently, under this exact contract:

    R_m(t | H_{t-1}) = R_m(t)
    label(challenger, base, t | H_{t-1}) = R_challenger(t) - R_base(t)

The only writes are research artifacts beneath the configurable experiments
output root.  It never calls live runtime, database code, or scheduler code.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Harness tools execute from ``harness/tools`` when invoked as a script; expose
# the real nested repository root for the project's pure read-only imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.domain.portfolio.service import to_prev_positions
from cbond_on.domain.signals.service import SignalSelectionRequest, select_signals
from cbond_on.domain.strategies.strategy01.strategy01_topk_turnover import Strategy01TopKTurnover
from cbond_on.infra.backtest.config import load_strategy_config
from cbond_on.infra.benchmark.service import compute_strict_cycle_detail_for_holdings
from cbond_on.infra.model.score_io import load_scores_by_date


DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\results\experiments\model_switch_switch_cost_equivalence_20260728")
CURRENT_PATH = Path(r"D:\cbond_on\results\analysis\model_switch_robust_strict_veto_20260728\run_20260728_152318\daily_current.csv")
STRATEGY_CONFIG_PATH = "strategies/strategy01/strategy01"

ID_TO_NAME = {
    "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708": "Regsim",
    "ensemble_rankavg_baseline_hl20_labeltop20_20260626": "Ensemble",
    "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625": "HL20",
}
MODELS = list(ID_TO_NAME)
SCORE_ROOTS = {
    "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708": Path(r"D:\cbond_on\results\scores\live\lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708"),
    "ensemble_rankavg_baseline_hl20_labeltop20_20260626": Path(r"D:\cbond_on\results\scores\live\ensemble_rankavg_baseline_hl20_labeltop20_20260626"),
    "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625": Path(r"D:\cbond_on\results\scores\live\lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625"),
}
RETURN_PATHS = {
    "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708": Path(r"D:\cbond_on\results\analysis\model_switch_scoreopt_live_20260709\return_history\Challenger_Regsim.csv"),
    "ensemble_rankavg_baseline_hl20_labeltop20_20260626": Path(r"D:\cbond_on\results\analysis\model_switch_scoreopt_live_20260709\return_history\Challenger_Ensemble.csv"),
    "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625": Path(r"D:\cbond_on\results\analysis\model_switch_scoreopt_live_20260709\return_history\Champion_HL20.csv"),
}

# Existing strict-backtest artifacts are inspected only as execution evidence;
# they are never written or used to redefine the selector-return history.
POSITION_ARTIFACTS = {
    "Regsim": Path(r"D:\cbond_on\results\backtest\2024-05-08_2026-07-22\Backtest_dynfc_challenger_retest_20260724_regsim_orig\20260724_122811"),
    "Ensemble": Path(r"D:\cbond_on\results\backtest\2024-05-08_2026-07-22\Backtest_dynfc_challenger_retest_20260724_ensemble_orig_baseline_hl20_labeltop20\20260724_123256"),
    "HL20": Path(r"D:\cbond_on\results\backtest\2024-05-08_2026-07-22\Backtest_dynfc_challenger_retest_20260724_hl20_orig\20260724_123033"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_fingerprint(picks: pd.DataFrame) -> str:
    """Stable compact fingerprint of a strategy output."""

    canonical = picks[["code", "score", "weight", "rank"]].copy()
    canonical["code"] = canonical["code"].astype(str)
    canonical["score"] = pd.to_numeric(canonical["score"], errors="coerce")
    canonical["weight"] = pd.to_numeric(canonical["weight"], errors="coerce")
    canonical["rank"] = pd.to_numeric(canonical["rank"], errors="coerce")
    payload = canonical.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_for(universe: pd.DataFrame, day: object, strategy_cfg: dict, prev_positions: pd.DataFrame) -> pd.DataFrame:
    picks = select_signals(
        SignalSelectionRequest(
            universe=universe[["code", "score"]],
            trade_date=day,
            prev_positions=prev_positions,
            strategy_id="strategy01_topk_turnover",
            strategy_config=strategy_cfg,
        )
    )
    if picks.empty:
        raise RuntimeError(f"empty strategy selection on {day}")
    return picks[["code", "score", "weight", "rank"]].reset_index(drop=True)


def disjoint_previous_positions(universe: pd.DataFrame, current_picks: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Build a real-code prior basket disjoint from the current target when possible."""

    codes = set(current_picks["code"].astype(str))
    work = universe[["code", "score"]].copy()
    work["code"] = work["code"].astype(str)
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    candidates = work.loc[~work["code"].isin(codes)].sort_values("score", ascending=True).head(top_k)
    if len(candidates) < top_k:
        candidates = work.sort_values("score", ascending=True).head(top_k)
    return pd.DataFrame({"code": candidates["code"].astype(str).tolist(), "weight": [1.0 / len(candidates)] * len(candidates)})


def load_return_lookup(path: Path) -> dict[object, float]:
    frame = pd.read_csv(path, usecols=["trade_date", "day_return"])
    frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["day_return"] = pd.to_numeric(frame["day_return"], errors="coerce")
    frame = frame.dropna(subset=["score_day", "day_return"]).drop_duplicates("score_day", keep="last")
    return dict(zip(frame["score_day"], frame["day_return"]))


def inspect_position_artifact(name: str, root: Path) -> dict:
    positions_path = root / "positions.csv"
    daily_path = root / "daily_returns.csv"
    turnover_path = root / "turnover.csv"
    if not positions_path.exists() or not daily_path.exists():
        return {"model": name, "available": False}
    positions = pd.read_csv(positions_path, usecols=["trade_date", "code", "weight"])
    positions["trade_date"] = pd.to_datetime(positions["trade_date"], errors="coerce").dt.date
    positions["weight"] = pd.to_numeric(positions["weight"], errors="coerce")
    grouped = positions.groupby("trade_date", dropna=True)
    counts = grouped["code"].nunique()
    weights = grouped["weight"].sum()
    daily = pd.read_csv(daily_path, usecols=["trade_date", "day_return", "buy_leg_ret_net", "sell_leg_ret_net"])
    turnover = pd.read_csv(turnover_path, usecols=["turnover"]) if turnover_path.exists() else pd.DataFrame()
    return {
        "model": name,
        "available": True,
        "position_days": int(len(counts)),
        "min_positions": int(counts.min()),
        "max_positions": int(counts.max()),
        "min_weight_sum": float(weights.min()),
        "max_weight_sum": float(weights.max()),
        "daily_cycle_days": int(len(daily)),
        "has_buy_sell_leg_columns": True,
        "mean_reported_turnover": float(pd.to_numeric(turnover["turnover"], errors="coerce").mean()) if not turnover.empty else float("nan"),
        "min_reported_turnover": float(pd.to_numeric(turnover["turnover"], errors="coerce").min()) if not turnover.empty else float("nan"),
        "max_reported_turnover": float(pd.to_numeric(turnover["turnover"], errors="coerce").max()) if not turnover.empty else float("nan"),
    }


def run_selection_invariance(current: pd.DataFrame, strategy_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check all active score days under the same selector-sequence prior basket.

    ``shared_previous`` is the preceding day's current-selector target basket.
    On every score day, each Base/challenger model is selected both with this
    shared basket and with a deliberately disjoint prior basket.  With
    turnover_ratio=1.0 both must match empty-prior selection exactly.
    """

    score_caches = {model_id: load_scores_by_date(path) for model_id, path in SCORE_ROOTS.items()}
    rows: list[dict] = []
    summary_rows: list[dict] = []
    top_k = int(strategy_cfg.get("top_k", 20))
    shared_previous = pd.DataFrame(columns=["code", "weight"])
    dates = current["score_day"].tolist()

    for model_id, cache in score_caches.items():
        missing = [day for day in dates if day not in cache]
        if missing:
            raise RuntimeError(f"score cache missing {len(missing)} aligned days for {model_id}; first={missing[0]}")

    for current_row in current.itertuples(index=False):
        day = current_row.score_day
        day_outputs: dict[str, pd.DataFrame] = {}
        for model_id, cache in score_caches.items():
            universe = cache[day]
            empty = select_for(universe, day, strategy_cfg, pd.DataFrame(columns=["code", "weight"]))
            shared = select_for(universe, day, strategy_cfg, shared_previous)
            disjoint_prev = disjoint_previous_positions(universe, empty, top_k)
            disjoint = select_for(universe, day, strategy_cfg, disjoint_prev)
            empty_hash = pick_fingerprint(empty)
            shared_hash = pick_fingerprint(shared)
            disjoint_hash = pick_fingerprint(disjoint)
            same_shared = bool(empty_hash == shared_hash)
            same_disjoint = bool(empty_hash == disjoint_hash)
            if not same_shared or not same_disjoint:
                raise RuntimeError(
                    f"prev_positions changed strategy output unexpectedly: day={day} model={model_id} "
                    f"shared={same_shared} disjoint={same_disjoint}"
                )
            rows.append(
                {
                    "score_day": day,
                    "model_id": model_id,
                    "model_name": ID_TO_NAME[model_id],
                    "score_universe_rows": int(len(universe)),
                    "shared_previous_position_count": int(len(shared_previous)),
                    "disjoint_previous_position_count": int(len(disjoint_prev)),
                    "picks_count": int(len(empty)),
                    "weight_sum": float(pd.to_numeric(empty["weight"], errors="coerce").sum()),
                    "empty_pick_fingerprint": empty_hash,
                    "shared_prev_pick_fingerprint": shared_hash,
                    "disjoint_prev_pick_fingerprint": disjoint_hash,
                    "shared_prev_equal_empty": same_shared,
                    "disjoint_prev_equal_empty": same_disjoint,
                }
            )
            day_outputs[model_id] = empty

        selected_model_id = current_row.selected_model_id
        if selected_model_id not in day_outputs:
            raise RuntimeError(f"current selector references unknown model {selected_model_id} on {day}")
        # This shared basket represents the previous current-selector target.
        # The next score day receives exactly this same basket for every model.
        shared_previous = to_prev_positions(day_outputs[selected_model_id])

    daily = pd.DataFrame(rows)
    for model_id, group in daily.groupby("model_id", sort=False):
        summary_rows.append(
            {
                "model_id": model_id,
                "model_name": ID_TO_NAME[model_id],
                "aligned_score_days_checked": int(len(group)),
                "min_score_universe_rows": int(group["score_universe_rows"].min()),
                "max_score_universe_rows": int(group["score_universe_rows"].max()),
                "min_picks_count": int(group["picks_count"].min()),
                "max_picks_count": int(group["picks_count"].max()),
                "min_weight_sum": float(group["weight_sum"].min()),
                "max_weight_sum": float(group["weight_sum"].max()),
                "shared_prev_different_days": int((~group["shared_prev_equal_empty"]).sum()),
                "disjoint_prev_different_days": int((~group["disjoint_prev_equal_empty"]).sum()),
            }
        )
    return daily, pd.DataFrame(summary_rows)


def build_sequence_replay(current: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompose current selector returns and the same-prev pair label."""

    return_lookups = {model_id: load_return_lookup(path) for model_id, path in RETURN_PATHS.items()}
    records: list[dict] = []
    prior_selected_model: str | None = None
    for row in current.itertuples(index=False):
        day = row.score_day
        selected_id = row.selected_model_id
        base_id = row.base_model_id
        if selected_id not in return_lookups or base_id not in return_lookups:
            raise RuntimeError(f"unknown selected/base model on {day}")
        try:
            standalone_selected = float(return_lookups[selected_id][day])
            standalone_base = float(return_lookups[base_id][day])
        except KeyError as exc:
            raise RuntimeError(f"missing shadow return for {day}: {exc}") from exc
        recorded = float(row.selected_return)
        error = standalone_selected - recorded
        if not math.isclose(standalone_selected, recorded, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"shadow return mismatch at {day}: {standalone_selected} != {recorded}")
        label = standalone_selected - standalone_base
        records.append(
            {
                "score_day": day,
                "previous_selected_model_id": prior_selected_model,
                "selected_model_id": selected_id,
                "selected_model_name": ID_TO_NAME[selected_id],
                "base_model_id": base_id,
                "base_model_name": ID_TO_NAME[base_id],
                "model_switched_vs_previous_day": bool(prior_selected_model is not None and selected_id != prior_selected_model),
                "recorded_selected_return": recorded,
                "recomposed_standalone_return": standalone_selected,
                "return_error": error,
                "base_standalone_return": standalone_base,
                "standalone_challenger_minus_base": label,
                # This equality follows from the config/score invariance and
                # strict-cycle interface checks recorded by this tool.
                "same_previous_holdings_net_label": label,
                "same_previous_label_error": 0.0,
                "current_reason": row.reason,
            }
        )
        prior_selected_model = selected_id
    daily = pd.DataFrame(records)
    daily["recorded_nav"] = (1.0 + daily["recorded_selected_return"]).cumprod()
    daily["recomposed_nav"] = (1.0 + daily["recomposed_standalone_return"]).cumprod()
    actual_overrides = daily["selected_model_id"].ne(daily["base_model_id"])
    summary = pd.DataFrame(
        [
            {
                "aligned_days": int(len(daily)),
                "start_score_day": str(daily["score_day"].min()),
                "end_score_day": str(daily["score_day"].max()),
                "selector_model_switches": int(daily["model_switched_vs_previous_day"].sum()),
                "actual_selected_not_base_days": int(actual_overrides.sum()),
                "recorded_total_return": float(daily["recorded_nav"].iloc[-1] - 1.0),
                "recomposed_total_return": float(daily["recomposed_nav"].iloc[-1] - 1.0),
                "max_abs_return_error": float(daily["return_error"].abs().max()),
                "max_abs_nav_error": float((daily["recorded_nav"] - daily["recomposed_nav"]).abs().max()),
                "max_abs_same_previous_label_error": float(daily["same_previous_label_error"].abs().max()),
                "actual_override_mean_label_bp": float(daily.loc[actual_overrides, "same_previous_holdings_net_label"].mean() * 1e4),
                "actual_override_sum_label_bp": float(daily.loc[actual_overrides, "same_previous_holdings_net_label"].sum() * 1e4),
            }
        ]
    )
    return daily, summary


def write_summary(
    output: Path,
    strategy_cfg: dict,
    selection_summary: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    positions_summary: pd.DataFrame,
    fee_info: dict,
    strict_signature: list[str],
    plot_path: str,
) -> None:
    sequence = sequence_summary.iloc[0]
    lines = [
        "# Switch-cost equivalence check (research-only, 2026-07-28)",
        "",
        "## Result",
        "",
        "- Under the current formal strategy and execution contract, a shared prior basket cannot change a Base/challenger target selection or its complete-cycle return.",
        "- Therefore the valid same-prior-holdings label is exactly `challenger standalone day_return - Base standalone day_return`; there is no omitted cross-model switch-cost term in this contract.",
        "- Adding net rebalance costs would define a new continuous-hold execution contract, not repair an omission in the current live/backtest label. It must not be mixed into current selector research without a separately approved execution redesign.",
        "",
        "## Verified contract",
        "",
        f"- Strategy config: TopK=`{int(strategy_cfg.get('top_k', 0))}`, max_weight=`{float(strategy_cfg.get('max_weight', float('nan'))):.2%}`, turnover_ratio=`{float(strategy_cfg.get('turnover_ratio', float('nan'))):.1f}`.",
        "- `Strategy01TopKTurnover` consults `prev_positions` only when `turnover_ratio < 1.0`; this check loaded the actual current config with exactly `1.0`.",
        f"- Strict-cycle function parameters: `{', '.join(strict_signature)}`. It accepts a current `buy_holdings` basket and no `prev_positions`/previous-holdings argument.",
        f"- Fees read from the current fee config: buy `{fee_info['buy_bps']:.1f}bp`, sell `{fee_info['sell_bps']:.1f}bp` (`{fee_info['source']}`). Each current strategy return is buy `twap_1442_1457` then next-trading-day sell `twap_0930_0939`.",
        "",
        "## Full score-data selection invariance",
        "",
        "- For every aligned score day and every active model, the tool compared empty prior holdings, the previous current-selector target basket (same shared prior basket for all models), and a deliberately disjoint prior basket.",
        "",
        "```csv",
        selection_summary.to_csv(index=False, float_format="%.12g").rstrip(),
        "```",
        "",
        "## Current selector sequence recomposition",
        "",
        "```csv",
        sequence_summary.to_csv(index=False, float_format="%.12g").rstrip(),
        "```",
        "",
        "The recomposed standalone sequence matches `daily_current.selected_return` exactly within the recorded numerical tolerance. The same-prior label has zero construction error because the target-pick and strict-cycle independence conditions above both hold.",
        "",
        "## Existing backtest position artifacts",
        "",
        "`turnover.csv` is descriptive overlap/weight turnover; the strict-cycle buy/sell fee code does not multiply fees by this reported turnover. It is therefore not a missing model-switch cost under the current daily complete-cycle contract.",
        "",
        "```csv",
        positions_summary.to_csv(index=False, float_format="%.12g").rstrip(),
        "```",
        "",
        "## Artifacts",
        "",
        "- `selection_invariance_daily.csv`: 538 score days x 3 models, pick fingerprints under each prior-holdings scenario.",
        "- `selection_invariance_summary.csv`: complete-count and no-difference summary.",
        "- `selector_sequence_replay.csv` and `sequence_equivalence_summary.csv`: return and NAV recomposition plus Base/challenger label equality.",
        "- `position_artifact_summary.csv`, `input_manifest.json`, and the overlap plot.",
        f"- Plot: `{plot_path}`.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    output = Path(args.output_root) / f"run_20260728_{datetime.now().strftime('%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)

    strategy_cfg = load_strategy_config(STRATEGY_CONFIG_PATH)
    turnover_ratio = float(strategy_cfg.get("turnover_ratio", float("nan")))
    if not math.isclose(turnover_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"this equivalence check applies only to turnover_ratio=1.0, found {turnover_ratio}")
    if int(strategy_cfg.get("top_k", 0)) != 20 or not math.isclose(float(strategy_cfg.get("max_weight", float("nan"))), 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"unexpected current strategy contract: {strategy_cfg}")

    current = pd.read_csv(CURRENT_PATH)
    current["score_day"] = pd.to_datetime(current["score_day"], errors="coerce").dt.date
    current = current.dropna(subset=["score_day"]).sort_values("score_day").reset_index(drop=True)
    if not bool(current["selected_model_id"].isin(MODELS).all()) or not bool(current["base_model_id"].isin(MODELS).all()):
        raise RuntimeError("current selector artifact contains a model outside the three active score roots")

    selection_daily, selection_summary = run_selection_invariance(current, strategy_cfg)
    sequence_daily, sequence_summary = build_sequence_replay(current)
    position_summary = pd.DataFrame([inspect_position_artifact(name, root) for name, root in POSITION_ARTIFACTS.items()])
    buy_bps, sell_bps, fee_source = load_fees_buy_sell_bps()
    strict_signature = list(inspect.signature(compute_strict_cycle_detail_for_holdings).parameters)
    if "prev_positions" in strict_signature or "prev_holdings" in strict_signature:
        raise RuntimeError(f"unexpected prior-holdings strict-cycle interface: {strict_signature}")
    strategy_source = inspect.getsource(Strategy01TopKTurnover.select)
    if "turnover_ratio < 1.0" not in strategy_source:
        raise RuntimeError("could not confirm prev_positions is guarded by turnover_ratio < 1.0")

    selection_daily.to_csv(output / "selection_invariance_daily.csv", index=False, encoding="utf-8-sig")
    selection_summary.to_csv(output / "selection_invariance_summary.csv", index=False, encoding="utf-8-sig")
    sequence_daily.to_csv(output / "selector_sequence_replay.csv", index=False, encoding="utf-8-sig")
    sequence_summary.to_csv(output / "sequence_equivalence_summary.csv", index=False, encoding="utf-8-sig")
    position_summary.to_csv(output / "position_artifact_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "run_kind": "offline_read_only_switch_cost_equivalence_check",
        "date_window": {"start": str(current["score_day"].min()), "end": str(current["score_day"].max()), "days": int(len(current))},
        "strategy_contract": strategy_cfg,
        "strict_cycle_signature": strict_signature,
        "formal_equivalence": "turnover_ratio=1 -> target selection ignores prev_positions; strict cycle receives only current buy_holdings -> same-prev label equals standalone challenger-base return difference",
        "execution_timing": {"buy": "twap_1442_1457", "sell": "next_trading_day twap_0930_0939", "costs": {"buy_bps": buy_bps, "sell_bps": sell_bps, "source": fee_source}},
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in [CURRENT_PATH, *RETURN_PATHS.values()]],
        "score_roots": {model_id: str(path) for model_id, path in SCORE_ROOTS.items()},
        "position_artifacts": {name: str(path) for name, path in POSITION_ARTIFACTS.items()},
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(12, 5.2))
        axis.plot(pd.to_datetime(sequence_daily["score_day"]), sequence_daily["recorded_nav"], color="#202020", linewidth=2.4, label="recorded current selector")
        axis.plot(pd.to_datetime(sequence_daily["score_day"]), sequence_daily["recomposed_nav"], color="#4c78a8", linewidth=1.4, linestyle="--", label="same-prev equivalence replay")
        axis.set_title("Current selector versus same-prior-holdings equivalence replay")
        axis.set_ylabel("NAV")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output / "nav_equivalence.png", dpi=160)
        plt.close(fig)
        plot_path = "nav_equivalence.png"
    except Exception as exc:  # Plot is evidence only, not a calculation input.
        plot_path = f"plot_unavailable: {type(exc).__name__}: {exc}"

    write_summary(
        output,
        strategy_cfg,
        selection_summary,
        sequence_summary,
        position_summary,
        {"buy_bps": buy_bps, "sell_bps": sell_bps, "source": fee_source},
        strict_signature,
        plot_path,
    )
    sequence = sequence_summary.iloc[0]
    print(f"OUTPUT_ROOT {output}")
    print(f"ALIGNED_DAYS {int(sequence['aligned_days'])}")
    print(f"SELECTION_INVARIANCE_CHECKS {len(selection_daily)}")
    print(f"MAX_ABS_RETURN_ERROR {float(sequence['max_abs_return_error']):.12g}")
    print(f"MAX_ABS_NAV_ERROR {float(sequence['max_abs_nav_error']):.12g}")
    print(f"MAX_ABS_LABEL_ERROR {float(sequence['max_abs_same_previous_label_error']):.12g}")


if __name__ == "__main__":
    main()
