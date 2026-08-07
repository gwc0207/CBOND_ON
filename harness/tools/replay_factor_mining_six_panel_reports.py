"""Build CBOND_ON's standard six-panel factor reports for a completed screen.

The project-standard renderer is ``save_single_factor_report``.  Its report is
one ``factor_report.png`` per factor and contains the familiar 2x3 panels:
IC series, IC metrics, bin NAV / walk-forward strategy / benchmark, factor
distribution, return metrics, and bin-alpha rolling-t heatmap.

The completed factor-mining screen intentionally has a different primary
metric: raw same-score-day 14:42 label IC on a fixed T-1 ``o_0005`` universe.
The screen CSVs alone cannot populate all six panels.  This harness therefore
replays the existing standard-report return contract *only after* applying the
screen's immutable T1430/14:42/fixed-pool input contract:

* factor and label files must still match the completed screen's SHA256 audit;
* every T-1 ``o_0005`` pool must resolve exactly as in the screen audit;
* per-code returns are replayed from the project strict execution cycle; and
* the report benchmark is the same-day equal-weight fixed-pool strict return.

Consequently the generated ``factor_report.png`` files are genuine six-panel
fixed-pool strict-execution diagnostics.  They are not a replacement for the
raw-label IC values that originally admitted the factors.  The per-factor
``report_context.json`` and batch manifest record both contracts explicitly.

Only a newly created child of ``D:/cbond_on/research_scratch`` may be written.
The tool never writes the database, live configuration, scheduler, production
FactorStore, or production results root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


# Direct ``py harness/tools/...`` execution must be able to import the project
# and the sibling screen harness without introducing a generic ``run`` entry.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from cbond_on.app.usecases.factor_batch_runtime import (  # noqa: E402
    _build_bad_factor_row,
    _build_screening_row,
    _collect_report_plots,
    _compute_factor_backtest_from_rows,
    _load_bad_factor_report_config,
    _load_screening_config,
    _resolve_factor_backtest_cost_bps,
    _write_bad_factor_outputs,
    _write_screening_outputs,
)
from cbond_on.core.config import load_config_file  # noqa: E402
from cbond_on.core.trading_days import (  # noqa: E402
    list_trading_days_from_raw,
    next_trading_days_from_raw,
)
from cbond_on.infra.benchmark.service import (  # noqa: E402
    compute_strict_cycle_detail_for_holdings,
    load_strict_market_day,
)
from cbond_on.infra.report.factor_report import save_single_factor_report  # noqa: E402
from harness.tools import factor_mining_screen as screen  # noqa: E402


_RESEARCH_SCRATCH = Path(r"D:\cbond_on\research_scratch")
_REPORT_DIR_NAME = "Single_Factor"
_REQUIRED_SCREEN_FILES = (
    "accepted_factors.csv",
    "evaluation_calendar.csv",
    "fixed_pool_audit.csv",
    "input_day_audit.csv",
    "screen_manifest.json",
)
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_STANDARD_REPORT_PARAMETERS = {
    "min_count": 10,
    "ic_bins": 20,
    "bin_count": 20,
    "bin_select": None,
    "bin_source": "walk_forward",
    "bin_top_k": 1,
    "bin_lookback_days": 40,
    "bin_min_train_days": 30,
    "alpha_significance_window": 40,
}
_FACTOR_BATCH_CONFIG = _REPO_ROOT / "cbond_on" / "config" / "factor" / "factor_config.json5"


@dataclass(frozen=True)
class ReplayInputs:
    """Validated immutable screen inputs and derived read-only source paths."""

    screen_dir: Path
    factor_root: Path
    label_root: Path
    raw_data_root: Path
    start: date
    end: date
    accepted: pd.DataFrame
    calendar: pd.DataFrame
    input_audit: pd.DataFrame
    fixed_pool_audit: pd.DataFrame
    manifest: Mapping[str, Any]


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_strict_child(path: Path, root: Path) -> bool:
    return path != root and _is_within(path, root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_evidence(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _parse_day(value: object, *, context: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{context} must be YYYY-MM-DD: {value!r}") from error


def _screen_file_evidence(screen_dir: Path) -> dict[str, dict[str, object]]:
    return {name: _file_evidence(screen_dir / name) for name in _REQUIRED_SCREEN_FILES}


def _screen_contract_paths(manifest: Mapping[str, Any]) -> tuple[Path, Path, Path, date, date]:
    factor_contract = manifest.get("factor_contract")
    fixed_universe = manifest.get("fixed_universe")
    if not isinstance(factor_contract, Mapping):
        raise ValueError("screen_manifest.json missing factor_contract")
    if not isinstance(fixed_universe, Mapping) or not bool(fixed_universe.get("enabled")):
        raise ValueError("screen_manifest.json must declare an enabled fixed_universe")

    factor_root = _resolved(str(factor_contract.get("factor_root", "")))
    label_root = _resolved(str(factor_contract.get("label_root", "")))
    raw_data_root = _resolved(str(fixed_universe.get("raw_data_root", "")))
    if not _is_strict_child(factor_root, _resolved(_RESEARCH_SCRATCH)):
        raise ValueError("screen factor_root must be a scratch FactorStore")
    for path, name in ((factor_root, "factor_root"), (label_root, "label_root"), (raw_data_root, "raw_data_root")):
        if not path.is_dir():
            raise FileNotFoundError(f"screen {name} is missing: {path}")

    panel_name = str(factor_contract.get("panel_name", "")).strip()
    factor_time = str(factor_contract.get("factor_time", "")).strip()
    label_time = str(factor_contract.get("label_time", "")).strip()
    if panel_name != "T1430" or not factor_time.startswith("14:30") or label_time != "same-score-day 14:42":
        raise ValueError(
            "six-panel replay only accepts the completed T1430/14:30/same-day-14:42 screen contract"
        )
    return (
        factor_root,
        label_root,
        raw_data_root,
        _parse_day(factor_contract.get("start"), context="factor_contract.start"),
        _parse_day(factor_contract.get("end"), context="factor_contract.end"),
    )


def _read_inputs(*, screen_dir: str | Path) -> ReplayInputs:
    source = _resolved(screen_dir)
    scratch = _resolved(_RESEARCH_SCRATCH)
    if not _is_strict_child(source, scratch):
        raise ValueError(f"--screen-dir must be below research scratch: {source}")
    if not source.is_dir():
        raise FileNotFoundError(f"completed screen directory missing: {source}")
    for name in _REQUIRED_SCREEN_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"completed screen artifact missing: {source / name}")

    accepted = pd.read_csv(source / "accepted_factors.csv")
    calendar = pd.read_csv(source / "evaluation_calendar.csv")
    input_audit = pd.read_csv(source / "input_day_audit.csv")
    fixed_pool_audit = pd.read_csv(source / "fixed_pool_audit.csv")
    manifest = json.loads((source / "screen_manifest.json").read_text(encoding="utf-8"))

    _require_columns(accepted, {"factor", "family", "selection_status"}, name="accepted_factors.csv")
    _require_columns(calendar, {"score_day", "partition"}, name="evaluation_calendar.csv")
    _require_columns(
        input_audit,
        {
            "score_day",
            "factor_file_path",
            "factor_file_bytes",
            "factor_file_sha256",
            "label_file_path",
            "label_file_bytes",
            "label_file_sha256",
            "pool_label_rows",
            "status",
        },
        name="input_day_audit.csv",
    )
    _require_columns(
        fixed_pool_audit,
        {
            "score_day",
            "pool_day_expected",
            "pool_day_used",
            "pool_codes",
            "fallback_no_filter",
            "allowlist_codes_sha256",
            "pool_file_path",
            "pool_file_bytes",
            "pool_file_sha256",
        },
        name="fixed_pool_audit.csv",
    )

    accepted = accepted.copy()
    accepted["factor"] = accepted["factor"].astype(str).str.strip()
    accepted["family"] = accepted["family"].astype(str).str.strip()
    if accepted.empty or accepted["factor"].eq("").any() or accepted["factor"].duplicated().any():
        raise ValueError("accepted_factors.csv must contain unique, non-empty factor names")
    if not accepted["selection_status"].astype(str).eq("selected").all():
        raise ValueError("accepted_factors.csv contains a factor not marked selected")
    unsafe = [name for name in accepted["factor"] if not _SAFE_FILENAME.fullmatch(name)]
    if unsafe:
        raise ValueError(f"accepted factor names are unsafe for report paths: {unsafe[:5]}")

    for frame, name in ((calendar, "evaluation_calendar.csv"), (input_audit, "input_day_audit.csv"), (fixed_pool_audit, "fixed_pool_audit.csv")):
        frame["score_day"] = frame["score_day"].astype(str)
        if frame["score_day"].duplicated().any():
            raise ValueError(f"{name} has duplicate score_day values")
    expected_days = calendar["score_day"].tolist()
    if expected_days != sorted(expected_days):
        raise ValueError("evaluation_calendar.csv must be sorted by score_day")
    if set(input_audit["score_day"]) != set(expected_days) or set(fixed_pool_audit["score_day"]) != set(expected_days):
        raise ValueError("screen day audits do not exactly match evaluation_calendar.csv")
    if not input_audit["status"].astype(str).eq("ok").all():
        raise ValueError("screen input audit is not fully ok")
    if fixed_pool_audit["fallback_no_filter"].astype(bool).any():
        raise ValueError("screen fixed-pool audit contains a no-filter fallback")

    factor_root, label_root, raw_data_root, start, end = _screen_contract_paths(manifest)
    calendar_start = _parse_day(expected_days[0], context="first evaluation score_day")
    calendar_end = _parse_day(expected_days[-1], context="last evaluation score_day")
    if calendar_start < start or calendar_end > end:
        raise ValueError("evaluation calendar falls outside manifest factor contract")

    return ReplayInputs(
        screen_dir=source,
        factor_root=factor_root,
        label_root=label_root,
        raw_data_root=raw_data_root,
        start=start,
        end=end,
        accepted=accepted,
        calendar=calendar,
        input_audit=input_audit,
        fixed_pool_audit=fixed_pool_audit,
        manifest=manifest,
    )


def _assert_output_dir(*, output_dir: str | Path, screen_dir: Path) -> Path:
    target = _resolved(output_dir)
    scratch = _resolved(_RESEARCH_SCRATCH)
    if not _is_strict_child(target, scratch):
        raise ValueError(f"--output-dir must be a new child of research scratch: {target}")
    if _is_within(target, screen_dir):
        raise ValueError("--output-dir must not be inside the immutable completed screen")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing report output: {target}")
    return target


def _expected_evidence(row: pd.Series, *, kind: str) -> dict[str, object]:
    return {
        "path": str(row[f"{kind}_file_path"]),
        "bytes": int(row[f"{kind}_file_bytes"]),
        "sha256": str(row[f"{kind}_file_sha256"]),
    }


def _assert_evidence(actual: Mapping[str, object], expected: Mapping[str, object], *, context: str) -> None:
    actual_path = _resolved(str(actual["path"]))
    expected_path = _resolved(str(expected["path"]))
    if actual_path != expected_path:
        raise RuntimeError(f"{context} path drifted: expected={expected_path} observed={actual_path}")
    if int(actual["bytes"]) != int(expected["bytes"]) or str(actual["sha256"]) != str(expected["sha256"]):
        raise RuntimeError(f"{context} evidence drifted from completed screen audit")


def _assert_pool_audit_matches(
    *,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    columns = [
        "score_day",
        "pool_day_expected",
        "pool_day_used",
        "pool_codes",
        "fallback_no_filter",
        "allowlist_codes_sha256",
        "pool_file_path",
        "pool_file_bytes",
        "pool_file_sha256",
    ]
    actual = actual[columns].sort_values("score_day", kind="mergesort").reset_index(drop=True)
    expected = expected[columns].sort_values("score_day", kind="mergesort").reset_index(drop=True)
    if not actual.equals(expected):
        raise RuntimeError("current fixed-pool resolution drifted from the completed screen audit")


def _next_day_by_score_day(inputs: ReplayInputs) -> dict[str, date]:
    score_days = [_parse_day(value, context="evaluation score_day") for value in inputs.calendar["score_day"]]
    observed = list_trading_days_from_raw(
        inputs.raw_data_root,
        score_days[0],
        score_days[-1],
        kind="snapshot",
        asset="cbond",
    )
    if observed != score_days:
        raise RuntimeError("DataHub trading calendar drifted from the completed screen calendar")
    next_after_end = next_trading_days_from_raw(
        inputs.raw_data_root,
        score_days[-1],
        1,
        kind="snapshot",
        asset="cbond",
    )
    if not next_after_end:
        raise RuntimeError(f"no next trading day available after {score_days[-1].isoformat()}")
    extended = [*score_days, next_after_end[0]]
    return {day.isoformat(): extended[index + 1] for index, day in enumerate(score_days)}


def _normalise_codes(frame: pd.DataFrame) -> pd.DataFrame:
    if "code" not in frame.columns:
        raise KeyError("strict market data is missing code")
    out = frame.copy()
    out["code"] = screen._normalize_code_series(out["code"])
    out = out.loc[out["code"].ne("")].copy()
    if out["code"].duplicated().any():
        raise ValueError("strict market data has duplicate codes")
    return out


def _strict_returns_for_day(
    *,
    raw_data_root: Path,
    score_day: date,
    sell_day: date,
    codes: pd.Series,
    buy_bps: float,
    sell_bps: float,
) -> pd.DataFrame:
    market = _normalise_codes(
        load_strict_market_day(
            raw_data_root=raw_data_root,
            trade_day=score_day,
            buy_bps=buy_bps,
            sell_bps=sell_bps,
        )
    )
    required = [
        "code",
        "buy_price",
        "buy_close_price",
        "buy_leg_ret_gross",
        "buy_leg_ret_net",
        "buy_cost_bps",
    ]
    missing = sorted(set(required).difference(market.columns))
    if missing:
        raise KeyError(f"strict market data missing {missing} for {score_day.isoformat()}")
    holdings = pd.DataFrame({"code": codes.astype(str).tolist()}).merge(
        market[required], on="code", how="inner", validate="one_to_one"
    )
    holdings["weight"] = 1.0
    holdings["buy_trade_day"] = pd.Timestamp(score_day)
    if holdings.empty:
        return pd.DataFrame(columns=["code", "strict_return_net"])
    detail = compute_strict_cycle_detail_for_holdings(
        raw_data_root=raw_data_root,
        buy_day=score_day,
        sell_day=sell_day,
        buy_holdings=holdings,
        sell_bps=sell_bps,
    )
    if detail.empty or "return_net" not in detail.columns:
        return pd.DataFrame(columns=["code", "strict_return_net"])
    result = _normalise_codes(detail[["code", "return_net"]]).rename(columns={"return_net": "strict_return_net"})
    result["strict_return_net"] = pd.to_numeric(result["strict_return_net"], errors="coerce")
    result.loc[~np.isfinite(result["strict_return_net"]), "strict_return_net"] = np.nan
    return result


def _build_execution_panel(
    *,
    inputs: ReplayInputs,
    factor_names: Sequence[str],
    buy_bps: float,
    sell_bps: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build one immutable fixed-pool panel used by every factor report."""

    score_days = inputs.calendar["score_day"].tolist()
    pool_codes_by_day, current_pool_audit, _pool_manifest = screen._fixed_pool_codes_by_day(
        raw_data_root=inputs.raw_data_root,
        score_days=score_days,
    )
    _assert_pool_audit_matches(actual=current_pool_audit, expected=inputs.fixed_pool_audit)
    if set(pool_codes_by_day) != set(score_days):
        raise RuntimeError("fixed-pool resolver did not produce every score day")

    input_by_day = inputs.input_audit.set_index("score_day", drop=False)
    next_days = _next_day_by_score_day(inputs)
    frames: list[pd.DataFrame] = []
    execution_rows: list[dict[str, object]] = []
    selected = list(factor_names)

    for day_text in score_days:
        input_row = input_by_day.loc[day_text]
        label, label_evidence, status = screen._read_label_1442(label_root=inputs.label_root, day=day_text)
        if label is None or label_evidence is None or status != "ok":
            raise RuntimeError(f"same-score-day 14:42 label unavailable during replay: {day_text} ({status})")
        _assert_evidence(label_evidence, _expected_evidence(input_row, kind="label"), context=f"label {day_text}")
        pool_codes = pool_codes_by_day[day_text]
        label_pool = label.loc[label["code"].isin(pool_codes), ["code", "y"]].copy()
        if len(label_pool) != int(input_row["pool_label_rows"]):
            raise RuntimeError(f"fixed-pool label row count drifted from screen audit: {day_text}")
        if label_pool["code"].duplicated().any():
            raise ValueError(f"duplicate fixed-pool label code during replay: {day_text}")

        factor_path = _resolved(str(input_row["factor_file_path"]))
        factor_frame, factor_evidence = screen._read_factor_frame(
            path=factor_path,
            day=day_text,
            expected_evidence=_expected_evidence(input_row, kind="factor"),
        )
        _assert_evidence(factor_evidence, _expected_evidence(input_row, kind="factor"), context=f"factor {day_text}")
        missing_factors = sorted(set(selected).difference(factor_frame.columns))
        if missing_factors:
            raise KeyError(f"selected FactorStore columns disappeared on {day_text}: {missing_factors[:5]}")
        values = (
            factor_frame.loc[factor_frame["code"].isin(pool_codes), ["code", *selected]]
            .set_index("code")
            .reindex(label_pool["code"].tolist())
            .reset_index()
        )
        for factor in selected:
            values[factor] = pd.to_numeric(values[factor], errors="coerce")
            values.loc[~np.isfinite(values[factor]), factor] = np.nan

        score_day = _parse_day(day_text, context="score_day")
        strict_returns = _strict_returns_for_day(
            raw_data_root=inputs.raw_data_root,
            score_day=score_day,
            sell_day=next_days[day_text],
            codes=label_pool["code"],
            buy_bps=buy_bps,
            sell_bps=sell_bps,
        )
        merged = (
            label_pool.rename(columns={"y": "screen_y_1442"})
            .merge(values, on="code", how="left", validate="one_to_one")
            .merge(strict_returns, on="code", how="left", validate="one_to_one")
        )
        merged["dt"] = pd.Timestamp(score_day)
        frames.append(merged[["dt", "code", "screen_y_1442", "strict_return_net", *selected]])

        strict_values = pd.to_numeric(merged["strict_return_net"], errors="coerce").dropna()
        execution_rows.append(
            {
                "score_day": day_text,
                "sell_day": next_days[day_text].isoformat(),
                "pool_label_rows": int(len(label_pool)),
                "strict_execution_rows": int(len(strict_values)),
                "fixed_pool_benchmark_return": float(strict_values.mean()) if not strict_values.empty else np.nan,
                "status": "ok" if not strict_values.empty else "empty_strict_execution",
            }
        )

    panel = pd.concat(frames, ignore_index=True)
    execution_audit = pd.DataFrame(execution_rows).sort_values("score_day", kind="mergesort").reset_index(drop=True)
    benchmark = pd.Series(
        pd.to_numeric(execution_audit["fixed_pool_benchmark_return"], errors="coerce").values,
        index=[_parse_day(value, context="execution audit score_day") for value in execution_audit["score_day"]],
        dtype=float,
    ).dropna()
    if benchmark.empty:
        raise RuntimeError("fixed-pool strict-execution replay has no benchmark returns")
    return panel, benchmark.sort_index(), execution_audit


def _report_factor(
    *,
    factor: str,
    family: str,
    panel: pd.DataFrame,
    benchmark: pd.Series,
    out_dir: Path,
    trading_days: set[date],
    screen_row: pd.Series,
    cost_context: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], Any]:
    factor_panel = panel[["dt", "code", factor, "strict_return_net"]].rename(columns={"strict_return_net": "y"})
    joined_total = int(len(factor_panel))
    finite_factor = pd.to_numeric(factor_panel[factor], errors="coerce")
    finite_label = pd.to_numeric(factor_panel["y"], errors="coerce")
    valid = np.isfinite(finite_factor) & np.isfinite(finite_label)
    valid_total = int(valid.sum())
    result = _compute_factor_backtest_from_rows(
        rows=[factor_panel],
        benchmark_by_dt=benchmark,
        diagnostics=[],
        raw_data_root=None,
        factor_col=factor,
        min_count=int(_STANDARD_REPORT_PARAMETERS["min_count"]),
        ic_bins=int(_STANDARD_REPORT_PARAMETERS["ic_bins"]),
        bin_count=int(_STANDARD_REPORT_PARAMETERS["bin_count"]),
        bin_select=None,
        bin_source=str(_STANDARD_REPORT_PARAMETERS["bin_source"]),
        bin_top_k=int(_STANDARD_REPORT_PARAMETERS["bin_top_k"]),
        bin_lookback_days=int(_STANDARD_REPORT_PARAMETERS["bin_lookback_days"]),
        bin_min_train_days=int(_STANDARD_REPORT_PARAMETERS["bin_min_train_days"]),
        factor_joined_total=joined_total,
        factor_valid_total=valid_total,
        # ``strict_return_net`` already includes the observed project costs.
        buy_bps=0.0,
        sell_bps=0.0,
    )
    summary = save_single_factor_report(
        result,
        out_dir,
        factor_name=factor,
        factor_col=factor,
        trading_days=trading_days,
        alpha_significance_window=int(_STANDARD_REPORT_PARAMETERS["alpha_significance_window"]),
    )
    report_path = out_dir / "factor_report.png"
    if not report_path.is_file() or report_path.stat().st_size <= 0:
        raise RuntimeError(f"standard six-panel report was not written for {factor}")
    context = {
        "research_only": True,
        "factor": factor,
        "family": family,
        "report_format": "CBOND_ON standard save_single_factor_report 2x3 six-panel PNG",
        "screen_admission_contract": {
            "factor_time": "T1430 / 14:30",
            "label": "same-score-day 14:42 raw y",
            "universe": "fixed T-1 o_0005",
            "screen_overall_mean_pearson_ic": _finite_or_none(screen_row.get("overall_mean_pearson_ic")),
            "screen_overall_mean_rank_ic": _finite_or_none(screen_row.get("overall_mean_rank_ic")),
        },
        "report_return_contract": {
            "factor_universe": "same fixed T-1 o_0005 pool after 14:42 label alignment",
            "per_code_return": "strict execution full-cycle return_net",
            "benchmark": "same-day equal-weight strict return across the fixed-pool label intersection",
            "costs_already_embedded_before_renderer": dict(cost_context),
            "renderer_cost_bps": {"buy": 0.0, "sell": 0.0},
            "note": "six-panel IC and performance fields are strict-execution replay diagnostics, not the raw-label IC used for factor admission",
        },
        "parameters": _STANDARD_REPORT_PARAMETERS,
        "replay_rows": {"joined": joined_total, "finite_factor_and_strict_return": valid_total},
    }
    (out_dir / "report_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record = {
        "factor": factor,
        "family": family,
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "report_bytes": int(report_path.stat().st_size),
        "screen_overall_mean_pearson_ic": _finite_or_none(screen_row.get("overall_mean_pearson_ic")),
        "screen_overall_mean_rank_ic": _finite_or_none(screen_row.get("overall_mean_rank_ic")),
        "replay_sharpe": _finite_or_none(summary.get("sharpe")),
        "replay_ret_total": _finite_or_none(summary.get("ret_total")),
        "replay_alpha_ret_total": _finite_or_none(summary.get("alpha_ret_total")),
        "replay_ic_mean": _finite_or_none(summary.get("ic_mean")),
        "replay_rank_ic_mean": _finite_or_none(summary.get("rank_ic_mean")),
        "replay_valid_rows": valid_total,
    }
    return record, summary, result


def _finite_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _load_factorbatch_screening_configs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Load the current generic FactorBatch screen without touching live config."""

    if not _FACTOR_BATCH_CONFIG.is_file():
        raise FileNotFoundError(f"FactorBatch config is missing: {_FACTOR_BATCH_CONFIG}")
    cfg = load_config_file(str(_FACTOR_BATCH_CONFIG))
    screening_cfg = _load_screening_config(cfg)
    bad_factor_cfg = _load_bad_factor_report_config(cfg)
    if not bool(screening_cfg.get("enabled", False)):
        raise RuntimeError("current generic FactorBatch screening is disabled")
    if str(screening_cfg.get("mode", "")).lower() != "stable_bin_alpha":
        raise RuntimeError(
            "fixed-pool replay supports the project stable_bin_alpha screen only; "
            f"got {screening_cfg.get('mode')!r}"
        )
    wf_screening_cfg = dict(screening_cfg)
    wf_screening_cfg["mode"] = "walk_forward_strategy"
    return screening_cfg, wf_screening_cfg, bad_factor_cfg


def _factorbatch_screening_evidence() -> dict[str, object]:
    return {
        "effective_config_root": _file_evidence(_FACTOR_BATCH_CONFIG),
        "screening_module": _file_evidence(
            _REPO_ROOT / "cbond_on" / "config" / "factor" / "reports" / "screening_stable_bin_alpha.json5"
        ),
        "bad_factor_module": _file_evidence(
            _REPO_ROOT / "cbond_on" / "config" / "factor" / "reports" / "bad_factor_report_default.json5"
        ),
    }


def _json_scalar(value: object) -> object:
    """Normalise pandas/numpy screening scalars before writing the manifest."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _screening_index_fields(row: Mapping[str, object] | None, *, prefix: str) -> dict[str, object]:
    if row is None:
        return {}
    fields = {
        "passed": row.get("passed"),
        "screen_status": row.get("screen_status"),
        "failed_rules": row.get("failed_rules"),
        "best_bin": row.get("best_bin"),
        "best_alpha_t": row.get("best_alpha_t"),
        "best_rolling_score": row.get("best_rolling_score"),
        "ret_total": row.get("ret_total"),
        "alpha_ret_total": row.get("alpha_ret_total"),
        "sharpe": row.get("sharpe"),
        "alpha_sharpe": row.get("alpha_sharpe"),
        "maxdd": row.get("maxdd"),
        "win_rate": row.get("win_rate"),
    }
    return {f"{prefix}{key}": _json_scalar(value) for key, value in fields.items()}


def build_reports(
    *,
    screen_dir: str | Path,
    output_dir: str | Path,
    with_factorbatch_screening: bool = False,
) -> Path:
    """Create a new, atomic six-panel report batch for all selected factors."""

    inputs = _read_inputs(screen_dir=screen_dir)
    target = _assert_output_dir(output_dir=output_dir, screen_dir=inputs.screen_dir)
    work_dir = target.parent / f".{target.name}.partial-{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    reports_root = work_dir / _REPORT_DIR_NAME
    reports_root.mkdir()

    factors = inputs.accepted["factor"].tolist()
    accepted_by_factor = inputs.accepted.set_index("factor", drop=False)
    buy_bps, sell_bps, fee_source = _resolve_factor_backtest_cost_bps()
    cost_context = {"buy_bps": float(buy_bps), "sell_bps": float(sell_bps), "source": str(fee_source)}
    panel, benchmark, execution_audit = _build_execution_panel(
        inputs=inputs,
        factor_names=factors,
        buy_bps=buy_bps,
        sell_bps=sell_bps,
    )
    execution_audit.to_csv(work_dir / "fixed_pool_execution_audit.csv", index=False, encoding="utf-8")

    trading_days = {_parse_day(value, context="report trading day") for value in inputs.calendar["score_day"]}
    records: list[dict[str, object]] = []
    screening_cfg: dict[str, object] | None = None
    wf_screening_cfg: dict[str, object] | None = None
    bad_factor_cfg: dict[str, object] | None = None
    screening_rows: list[dict[str, object]] = []
    wf_screening_rows: list[dict[str, object]] = []
    bad_factor_rows: list[dict[str, object]] = []
    if with_factorbatch_screening:
        screening_cfg, wf_screening_cfg, bad_factor_cfg = _load_factorbatch_screening_configs()
    for accepted in inputs.accepted.itertuples(index=False):
        factor = str(accepted.factor)
        family = str(accepted.family)
        out_dir = reports_root / factor
        record, summary, result = _report_factor(
            factor=factor,
            family=family,
            panel=panel,
            benchmark=benchmark,
            out_dir=out_dir,
            trading_days=trading_days,
            screen_row=accepted_by_factor.loc[factor],
            cost_context=cost_context,
        )
        record["report"] = str(Path(record["report"]).relative_to(work_dir))
        records.append(record)

        if screening_cfg is not None:
            screening_rows.append(
                _build_screening_row(
                    factor_name=factor,
                    factor_col=factor,
                    summary=summary,
                    result=result,
                    screening_cfg=screening_cfg,
                )
            )
            if bool(screening_cfg.get("write_walk_forward_strategy", False)):
                if wf_screening_cfg is None:
                    raise RuntimeError("walk-forward screening config was not resolved")
                wf_screening_rows.append(
                    _build_screening_row(
                        factor_name=factor,
                        factor_col=factor,
                        summary=summary,
                        result=result,
                        screening_cfg=wf_screening_cfg,
                    )
                )
        if bad_factor_cfg is not None and bool(bad_factor_cfg.get("enabled", False)):
            bad_factor_rows.append(
                _build_bad_factor_row(
                    factor_name=factor,
                    factor_col=factor,
                    result=result,
                    cfg=bad_factor_cfg,
                )
            )

    factorbatch_screening_manifest: dict[str, object] | None = None
    if screening_cfg is not None:
        _write_screening_outputs(reports_root, screening_cfg=screening_cfg, rows=screening_rows)
        if bool(screening_cfg.get("write_walk_forward_strategy", False)):
            if wf_screening_cfg is None:
                raise RuntimeError("walk-forward screening config was not resolved")
            _write_screening_outputs(
                reports_root,
                screening_cfg=wf_screening_cfg,
                rows=wf_screening_rows,
                dirname="screened_wf",
            )

        bad_names: set[str] = set()
        if bad_factor_cfg is not None and bool(bad_factor_cfg.get("enabled", False)):
            _write_bad_factor_outputs(reports_root, cfg=bad_factor_cfg, rows=bad_factor_rows)
            bad_names = {
                str(row["factor_name"])
                for row in bad_factor_rows
                if bool(row.get("is_bad", False))
            }
        _collect_report_plots(reports_root, bad_factor_names=bad_names)

        screening_by_factor = {str(row["factor_name"]): row for row in screening_rows}
        wf_by_factor = {str(row["factor_name"]): row for row in wf_screening_rows}
        bad_by_factor = {str(row["factor_name"]): row for row in bad_factor_rows}
        for record in records:
            factor = str(record["factor"])
            record.update(_screening_index_fields(screening_by_factor.get(factor), prefix="screened_"))
            record.update(_screening_index_fields(wf_by_factor.get(factor), prefix="screened_wf_"))
            bad_row = bad_by_factor.get(factor)
            if bad_row is not None:
                record["bad_factor_is_bad"] = bool(bad_row.get("is_bad", False))
                record["bad_factor_reasons"] = _json_scalar(bad_row.get("bad_reasons"))

        screened_root = reports_root / "screened"
        factorbatch_screening_manifest = {
            "source_evidence": _factorbatch_screening_evidence(),
            "screening_config": screening_cfg,
            "walk_forward_screening_config": wf_screening_cfg,
            "bad_factor_config": bad_factor_cfg,
            "primary": {
                "path": str(screened_root.relative_to(work_dir)),
                "all": str((screened_root / "factor_screening_all.csv").relative_to(work_dir)),
                "shortlist": str((screened_root / "factor_shortlist.csv").relative_to(work_dir)),
                "accepted_count": int(sum(bool(row.get("passed", False)) for row in screening_rows)),
            },
            "walk_forward": {
                "enabled": bool(screening_cfg.get("write_walk_forward_strategy", False)),
                "path": str((reports_root / "screened_wf").relative_to(work_dir)),
                "accepted_count": int(sum(bool(row.get("passed", False)) for row in wf_screening_rows)),
            },
            "bad_factor": {
                "enabled": bool(bad_factor_cfg and bad_factor_cfg.get("enabled", False)),
                "path": str((reports_root / "bad_factors").relative_to(work_dir)),
                "bad_count": int(len(bad_names)),
            },
            "plots": {
                "path": str((reports_root / "plot").relative_to(work_dir)),
                "good_path": str((reports_root / "plot_good").relative_to(work_dir)),
                "bad_path": str((reports_root / "plot_bad").relative_to(work_dir)),
                "count": int(len(records)),
            },
        }

    report_index = pd.DataFrame(records).sort_values("factor", kind="mergesort").reset_index(drop=True)
    report_index.to_csv(work_dir / "report_index.csv", index=False, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "purpose": "fixed-pool strict-execution replay rendered with CBOND_ON standard six-panel factor_report.png",
        "screen_dir": str(inputs.screen_dir),
        "screen_inputs": _screen_file_evidence(inputs.screen_dir),
        "sources": {
            "factor_root": str(inputs.factor_root),
            "label_root": str(inputs.label_root),
            "raw_data_root": str(inputs.raw_data_root),
        },
        "screen_contract": {
            "factor_time": "T1430 / 14:30",
            "label": "same-score-day 14:42 raw y",
            "universe": "fixed T-1 o_0005",
        },
        "report_return_contract": {
            "per_code_return": "strict execution full-cycle return_net",
            "benchmark": "same-day equal-weight strict return across fixed-pool label intersection",
            "costs": cost_context,
            "renderer_cost_bps": {"buy": 0.0, "sell": 0.0},
        },
        "parameters": _STANDARD_REPORT_PARAMETERS,
        "factorbatch_screening": factorbatch_screening_manifest,
        "report_count": len(records),
        "execution_audit": {
            "path": "fixed_pool_execution_audit.csv",
            "sha256": _sha256(work_dir / "fixed_pool_execution_audit.csv"),
            "empty_strict_execution_days": int(
                execution_audit["status"].astype(str).ne("ok").sum()
            ),
        },
        "report_index": {
            "path": "report_index.csv",
            "sha256": _sha256(work_dir / "report_index.csv"),
        },
        "reports": records,
    }
    (work_dir / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    work_dir.replace(target)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, help="immutable completed factor-screen directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new report root strictly below D:/cbond_on/research_scratch",
    )
    parser.add_argument(
        "--with-factorbatch-screening",
        action="store_true",
        help="also apply the current FactorBatch stable_bin_alpha / screened_wf / bad-factor logic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = build_reports(
        screen_dir=args.screen_dir,
        output_dir=args.output_dir,
        with_factorbatch_screening=bool(args.with_factorbatch_screening),
    )
    print(json.dumps({"research_only": True, "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
