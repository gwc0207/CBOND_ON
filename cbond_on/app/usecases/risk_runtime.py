from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from cbond_on.config.loader import parse_date
from cbond_on.core.trading_days import list_available_trading_days_from_raw
from cbond_on.domain.risk import RiskModelError, factor_definitions_from_config
from cbond_on.infra.risk.data import LocalDataHubRiskSource
from cbond_on.infra.risk.estimation import (
    estimate_factor_covariance,
    estimate_factor_returns,
    estimate_specific_risk,
)
from cbond_on.infra.risk.exposure import build_risk_exposures, normalize_cbond_codes
from cbond_on.infra.risk.portfolio import attribute_active_return, calculate_portfolio_risk
from cbond_on.infra.risk.reporting import write_risk_report_bundle
from cbond_on.schemas.config.risk import validate_risk_config


def _config_hash(cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(cfg), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_output_root(
    *,
    cfg: Mapping[str, Any],
    paths_cfg: Mapping[str, Any],
    start_text: str,
    end_text: str,
    override: str | Path | None,
) -> Path:
    configured = override or str(dict(cfg.get("output") or {}).get("output_root") or "").strip()
    if configured:
        root = Path(configured)
    else:
        root = Path(paths_cfg["results_root"]) / "risk" / str(cfg["risk_model_id"]) / f"{start_text}_{end_text}"
    norm = root.as_posix().lower()
    if "/results/live" in norm or norm.endswith("/live"):
        raise RiskModelError("CB-Risk output must never write under results/live")
    return root


def _normalize_position_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise RiskModelError(f"positions file is empty: {path}")
    date_candidates = ("score_day", "signal_day", "buy_day", "trade_date")
    date_column = next((column for column in date_candidates if column in frame.columns), None)
    if date_column is None:
        raise RiskModelError(
            f"positions file needs one of {date_candidates}; directory name / target day is not a risk as-of date"
        )
    weight_candidates = ("weight", "target_weight", "buy_weight_base")
    weight_column = next((column for column in weight_candidates if column in frame.columns), None)
    if weight_column is None:
        raise RiskModelError(f"positions file missing one of {weight_candidates}")
    frame = frame.copy()
    frame["risk_trade_date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    frame["weight"] = pd.to_numeric(frame[weight_column], errors="coerce")
    if "code" not in frame.columns and not {"instrument_code", "exchange_code"} <= set(frame.columns):
        raise RiskModelError("positions file missing code or instrument_code/exchange_code")
    frame = frame[frame["risk_trade_date"].notna() & frame["weight"].notna()].copy()
    if frame.empty:
        raise RiskModelError(f"positions file has no usable dated weights: {path}")
    return frame


def _positions_for_day(positions: pd.DataFrame | None, trade_day: object) -> pd.DataFrame | None:
    if positions is None:
        return None
    selected = positions[positions["risk_trade_date"] == trade_day].copy()
    if selected.empty:
        return None
    keep = [column for column in ("code", "instrument_code", "exchange_code", "weight") if column in selected.columns]
    return selected.loc[:, keep]


def _merge_exposure_with_return(exposure: pd.DataFrame, raw_panel: pd.DataFrame, *, trade_day: object, exposure_day: object, sell_day: object) -> pd.DataFrame:
    returns = raw_panel.copy()
    if "code" not in returns.columns:
        returns["code"] = normalize_cbond_codes(returns)
    returns = returns.loc[:, ["code", "gross_return"]].copy()
    output = exposure.merge(returns, on="code", how="inner")
    output["trade_date"] = pd.Timestamp(trade_day)
    output["exposure_date"] = pd.Timestamp(exposure_day)
    output["sell_date"] = pd.Timestamp(sell_day)
    return output


def run(
    *,
    cfg: Mapping[str, Any],
    paths_cfg: Mapping[str, Any],
    start: str | None = None,
    end: str | None = None,
    strategy_positions_path: str | Path | None = None,
    benchmark_positions_path: str | Path | None = None,
    output_root: str | Path | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the isolated offline/shadow CB-Risk core against local DataHub data."""

    risk_cfg = validate_risk_config(cfg)
    inputs = dict(risk_cfg["inputs"])
    raw_root = Path(paths_cfg["raw_data_root"])
    available_days = list_available_trading_days_from_raw(raw_root, kind="snapshot", asset="cbond")
    if not available_days:
        raise RiskModelError(f"no trading calendar/data under raw root: {raw_root}")
    start_day = parse_date(start or risk_cfg.get("start") or available_days[0])
    configured_end = end or risk_cfg.get("end") or available_days[-1]
    end_day = parse_date(configured_end)
    trade_days = [day for day in available_days if start_day <= day <= end_day]
    if not trade_days:
        raise RiskModelError(f"no available trade days in requested range {start_day}..{end_day}")

    definitions = factor_definitions_from_config(risk_cfg["factors"])
    expected_style_factors = tuple(definition.name for definition in definitions)
    industry_cfg = dict(risk_cfg.get("industry") or {})
    if bool(industry_cfg.get("enabled", False)):
        raise RiskModelError(
            "industry risk is not enabled for the local raw source; DataHub PIT industry input is required first"
        )
    source = LocalDataHubRiskSource(raw_root, risk_cfg)
    preprocessing = dict(risk_cfg.get("preprocessing") or {})
    estimation = dict(risk_cfg["estimation"])
    exposures_all: list[pd.DataFrame] = []
    diagnostics_all: list[pd.DataFrame] = []
    factor_rows: list[dict[str, Any]] = []
    residual_rows: list[pd.DataFrame] = []
    daily_status: list[dict[str, Any]] = []

    for trade_day in trade_days:
        source_day = source.build_day(trade_day)
        if source_day is None:
            daily_status.append({"trade_date": trade_day, "status": "calendar_boundary"})
            continue
        status = dict(source_day.diagnostics)
        if source_day.panel.empty or status.get("status") != "ok":
            daily_status.append(status)
            continue
        try:
            built = build_risk_exposures(
                source_day.panel,
                definitions,
                regression_weight_columns=tuple(dict(risk_cfg.get("regression_weight") or {}).get("source_columns") or ("remain_size",)),
                winsor_lower=float(preprocessing.get("winsor_lower", 0.01)),
                winsor_upper=float(preprocessing.get("winsor_upper", 0.99)),
                industry=industry_cfg,
            )
        except Exception as exc:
            status.update({"status": "exposure_error", "message": str(exc)})
            daily_status.append(status)
            continue
        diagnostics = built.diagnostics.copy()
        diagnostics["trade_date"] = pd.Timestamp(trade_day)
        diagnostics["exposure_date"] = pd.Timestamp(source_day.exposure_day)
        diagnostics_all.append(diagnostics)
        if built.factor_columns != expected_style_factors:
            status.update(
                {
                    "status": "factor_contract_not_met",
                    "active_factors": ",".join(built.factor_columns),
                    "expected_factors": ",".join(expected_style_factors),
                }
            )
            daily_status.append(status)
            continue
        panel = _merge_exposure_with_return(
            built.exposures,
            source_day.panel,
            trade_day=trade_day,
            exposure_day=source_day.exposure_day,
            sell_day=source_day.sell_day,
        )
        try:
            fitted = estimate_factor_returns(
                panel,
                expected_style_factors,
                return_column="gross_return",
                weight_column="regression_weight",
                min_samples=int(estimation.get("min_samples", 150)),
                max_condition_number=float(estimation.get("max_condition_number", 100.0)),
                huber_c=float(estimation.get("huber_c", 1.345)),
                huber_iterations=int(estimation.get("huber_iterations", 3)),
            )
        except Exception as exc:
            status.update({"status": "factor_return_error", "message": str(exc), "panel_count": len(panel)})
            daily_status.append(status)
            continue
        exposures_all.append(panel)
        factor_row = {
            "trade_date": pd.Timestamp(trade_day),
            "exposure_date": pd.Timestamp(source_day.exposure_day),
            "sell_date": pd.Timestamp(source_day.sell_day),
            **fitted.factor_returns.to_dict(),
            **{f"diag__{key}": value for key, value in fitted.diagnostics.items()},
        }
        factor_rows.append(factor_row)
        residual = panel.loc[:, ["code", "trade_date"]].copy()
        residual["specific_return"] = fitted.residuals.to_numpy(dtype=float)
        residual_rows.append(residual.dropna(subset=["specific_return"]))
        status.update({"status": fitted.diagnostics["status"], "panel_count": len(panel), **fitted.diagnostics})
        daily_status.append(status)

    exposure_panel = pd.concat(exposures_all, ignore_index=True) if exposures_all else pd.DataFrame()
    exposure_diagnostics = pd.concat(diagnostics_all, ignore_index=True) if diagnostics_all else pd.DataFrame()
    factor_returns = pd.DataFrame(factor_rows)
    specific_returns = pd.concat(residual_rows, ignore_index=True) if residual_rows else pd.DataFrame()
    status_frame = pd.DataFrame(daily_status)
    factor_columns = ("CB_MKT", *expected_style_factors)
    covariance = None
    specific_risk = None
    covariance_error = ""
    if not factor_returns.empty:
        try:
            covariance = estimate_factor_covariance(
                factor_returns,
                factor_columns,
                half_life_days=float(estimation.get("covariance_half_life_days", 60.0)),
                shrinkage=float(estimation.get("covariance_shrinkage", 0.15)),
                min_observations=int(estimation.get("covariance_min_observations", 120)),
                annualization_days=int(estimation.get("annualization_days", 244)),
            )
        except Exception as exc:
            covariance_error = str(exc)
        try:
            specific_risk = estimate_specific_risk(
                specific_returns,
                half_life_days=float(estimation.get("specific_half_life_days", 60.0)),
                min_observations=int(estimation.get("specific_min_observations", 20)),
                shrinkage_observations=float(estimation.get("specific_shrinkage_observations", 60.0)),
                annualization_days=int(estimation.get("annualization_days", 244)),
            )
        except Exception as exc:
            covariance_error = f"{covariance_error}; specific risk: {exc}".strip("; ")

    strategy_path = strategy_positions_path or str(inputs.get("strategy_positions_path") or "").strip() or None
    benchmark_path = benchmark_positions_path or str(inputs.get("benchmark_positions_path") or "").strip() or None
    strategy_positions = _normalize_position_frame(strategy_path) if strategy_path else None
    benchmark_positions = _normalize_position_frame(benchmark_path) if benchmark_path else None
    portfolio_summary_rows: list[dict[str, Any]] = []
    factor_exposure_rows: list[pd.DataFrame] = []
    factor_risk_rows: list[pd.DataFrame] = []
    specific_risk_rows: list[pd.DataFrame] = []
    attribution_rows: list[pd.DataFrame] = []
    if strategy_positions is not None and not factor_returns.empty:
        for _, row in factor_returns.iterrows():
            trade_day = pd.Timestamp(row["trade_date"]).date()
            strategy = _positions_for_day(strategy_positions, trade_day)
            if strategy is None:
                continue
            benchmark = _positions_for_day(benchmark_positions, trade_day)
            previous_factors = factor_returns[pd.to_datetime(factor_returns["trade_date"]).dt.date < trade_day]
            previous_residuals = specific_returns[pd.to_datetime(specific_returns["trade_date"]).dt.date < trade_day]
            risk_snapshot = exposure_panel[pd.to_datetime(exposure_panel["trade_date"]).dt.date == trade_day]
            if risk_snapshot.empty or len(previous_factors) < int(estimation.get("covariance_min_observations", 120)):
                portfolio_summary_rows.append(
                    {"trade_date": trade_day, "status": "risk_state_warmup", "strategy_positions": len(strategy)}
                )
                continue
            try:
                cov_state = estimate_factor_covariance(
                    previous_factors,
                    factor_columns,
                    half_life_days=float(estimation.get("covariance_half_life_days", 60.0)),
                    shrinkage=float(estimation.get("covariance_shrinkage", 0.15)),
                    min_observations=int(estimation.get("covariance_min_observations", 120)),
                    annualization_days=int(estimation.get("annualization_days", 244)),
                )
                specific_state = estimate_specific_risk(
                    previous_residuals,
                    half_life_days=float(estimation.get("specific_half_life_days", 60.0)),
                    min_observations=int(estimation.get("specific_min_observations", 20)),
                    shrinkage_observations=float(estimation.get("specific_shrinkage_observations", 60.0)),
                    annualization_days=int(estimation.get("annualization_days", 244)),
                )
                risk = calculate_portfolio_risk(
                    risk_snapshot,
                    cov_state.covariance_daily,
                    specific_state.specific_risk,
                    strategy,
                    benchmark,
                    factor_columns=factor_columns,
                    annualization_days=int(estimation.get("annualization_days", 244)),
                )
                portfolio_summary_rows.append({"trade_date": trade_day, "status": "ok", **risk.summary})
                factor_exposure_rows.append(risk.factor_exposure.assign(trade_date=trade_day))
                factor_risk_rows.append(risk.factor_risk_contribution.assign(trade_date=trade_day))
                specific_risk_rows.append(risk.specific_risk_contribution.assign(trade_date=trade_day))
                residual_day = specific_returns[pd.to_datetime(specific_returns["trade_date"]).dt.date == trade_day]
                attribution = attribute_active_return(
                    risk_snapshot,
                    strategy,
                    benchmark,
                    row.loc[list(factor_columns)],
                    residual_day,
                    factor_columns=factor_columns,
                )
                attribution_rows.append(attribution.contributions.assign(trade_date=trade_day, **attribution.summary))
            except Exception as exc:
                portfolio_summary_rows.append({"trade_date": trade_day, "status": "portfolio_risk_error", "message": str(exc)})

    portfolio_summary = pd.DataFrame(portfolio_summary_rows)
    portfolio_factor_exposure = pd.concat(factor_exposure_rows, ignore_index=True) if factor_exposure_rows else pd.DataFrame()
    portfolio_factor_risk = pd.concat(factor_risk_rows, ignore_index=True) if factor_risk_rows else pd.DataFrame()
    portfolio_specific_risk = pd.concat(specific_risk_rows, ignore_index=True) if specific_risk_rows else pd.DataFrame()
    attribution = pd.concat(attribution_rows, ignore_index=True) if attribution_rows else pd.DataFrame()

    metadata = {
        "risk_model_id": risk_cfg["risk_model_id"],
        "mode": risk_cfg["mode"],
        "pit_status": dict(risk_cfg["asof"]).get("pit_status"),
        "static_exposure_lag_trading_days": dict(risk_cfg["asof"]).get("static_exposure_lag_trading_days"),
        "raw_data_root": str(raw_root),
        "requested_start": start_day,
        "requested_end": end_day,
        "factor_columns": factor_columns,
        "config_hash": _config_hash(risk_cfg),
        "write_db": False,
        "live_hook_enabled": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "covariance_error": covariance_error,
    }
    out_dir: Path | None = None
    if write_outputs:
        root = _resolve_output_root(
            cfg=risk_cfg,
            paths_cfg=paths_cfg,
            start_text=start_day.isoformat(),
            end_text=end_day.isoformat(),
            override=output_root,
        )
        run_label = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir = write_risk_report_bundle(
            root / run_label,
            metadata=metadata,
            exposure_panel=exposure_panel,
            exposure_diagnostics=exposure_diagnostics,
            factor_returns=factor_returns,
            specific_returns=specific_returns,
            daily_status=status_frame,
            factor_covariance=covariance.covariance_daily if covariance is not None else None,
            factor_correlation=covariance.correlation if covariance is not None else None,
            specific_risk=specific_risk.specific_risk if specific_risk is not None else None,
            portfolio_summary=portfolio_summary,
            portfolio_factor_exposure=portfolio_factor_exposure,
            portfolio_factor_risk=portfolio_factor_risk,
            portfolio_specific_risk=portfolio_specific_risk,
            attribution=attribution,
            write_html_report=bool(dict(risk_cfg["output"]).get("write_html_report", True)),
        )
    non_ok = int((status_frame.get("status", pd.Series(dtype=str)).astype(str) != "ok").sum()) if not status_frame.empty else 0
    return {
        "out_dir": str(out_dir) if out_dir is not None else "",
        "requested_days": int(len(trade_days)),
        "factor_return_days": int(len(factor_returns)),
        "exposure_rows": int(len(exposure_panel)),
        "specific_return_rows": int(len(specific_returns)),
        "non_ok_days": non_ok,
        "covariance_ready": covariance is not None,
        "specific_risk_ready": specific_risk is not None,
        "covariance_error": covariance_error,
        "mode": risk_cfg["mode"],
        "live_isolation": True,
    }
