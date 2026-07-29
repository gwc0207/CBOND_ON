from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_frame(frame: pd.DataFrame | None, path: Path, *, parquet: bool = False) -> None:
    if frame is None:
        return
    if parquet:
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def write_risk_report_bundle(
    out_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
    exposure_panel: pd.DataFrame,
    exposure_diagnostics: pd.DataFrame,
    factor_returns: pd.DataFrame,
    specific_returns: pd.DataFrame,
    daily_status: pd.DataFrame,
    factor_covariance: pd.DataFrame | None,
    factor_correlation: pd.DataFrame | None,
    specific_risk: pd.DataFrame | None,
    portfolio_summary: pd.DataFrame | None = None,
    portfolio_factor_exposure: pd.DataFrame | None = None,
    portfolio_factor_risk: pd.DataFrame | None = None,
    portfolio_specific_risk: pd.DataFrame | None = None,
    attribution: pd.DataFrame | None = None,
    write_html_report: bool = True,
) -> Path:
    """Persist independent CB-Risk artifacts.  No live/DB path is accepted here."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_metadata.json").write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_frame(exposure_panel, root / "exposure_panel.parquet", parquet=True)
    _write_frame(exposure_diagnostics, root / "exposure_quality.csv")
    _write_frame(factor_returns, root / "factor_returns.csv")
    _write_frame(specific_returns, root / "specific_returns.parquet", parquet=True)
    _write_frame(daily_status, root / "daily_status.csv")
    if factor_covariance is not None:
        factor_covariance.to_csv(root / "factor_covariance_daily.csv")
    if factor_correlation is not None:
        factor_correlation.to_csv(root / "factor_correlation.csv")
    _write_frame(specific_risk, root / "specific_risk.parquet", parquet=True)
    _write_frame(portfolio_summary, root / "portfolio_risk_summary.csv")
    _write_frame(portfolio_factor_exposure, root / "portfolio_factor_exposure.csv")
    _write_frame(portfolio_factor_risk, root / "portfolio_factor_risk_contribution.csv")
    _write_frame(portfolio_specific_risk, root / "portfolio_specific_risk_contribution.csv")
    _write_frame(attribution, root / "realized_attribution.csv")

    quality_payload = {
        "daily_status": daily_status.to_dict(orient="records"),
        "exposure_diagnostics": exposure_diagnostics.to_dict(orient="records"),
    }
    (root / "quality_summary.json").write_text(
        json.dumps(quality_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    if write_html_report:
        _write_html_report(
            root / "risk_report.html",
            metadata=metadata,
            daily_status=daily_status,
            portfolio_summary=portfolio_summary,
            factor_exposure=portfolio_factor_exposure,
            factor_risk=portfolio_factor_risk,
        )
    return root


def _write_html_report(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    daily_status: pd.DataFrame,
    portfolio_summary: pd.DataFrame | None,
    factor_exposure: pd.DataFrame | None,
    factor_risk: pd.DataFrame | None,
) -> None:
    sections = [
        "<h1>CB-Risk v1 — Offline Shadow Report</h1>",
        "<p>This report is independent of model selection, trade-list publication, and database writes.</p>",
        "<h2>Run metadata</h2>",
        pd.DataFrame([dict(metadata)]).to_html(index=False, escape=True),
        "<h2>Daily data/model status</h2>",
        daily_status.tail(30).to_html(index=False, escape=True),
    ]
    if portfolio_summary is not None and not portfolio_summary.empty:
        sections.extend(["<h2>Portfolio risk</h2>", portfolio_summary.to_html(index=False, escape=True)])
    if factor_exposure is not None and not factor_exposure.empty:
        sections.extend(["<h2>Portfolio factor exposure</h2>", factor_exposure.to_html(index=False, escape=True)])
    if factor_risk is not None and not factor_risk.empty:
        sections.extend(["<h2>Factor risk contribution</h2>", factor_risk.to_html(index=False, escape=True)])
    html = """<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>CB-Risk v1</title>
<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;color:#202124}table{border-collapse:collapse;margin:12px 0;font-size:13px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f3f6fb}h1{color:#163d73}</style>
</head><body>""" + "\n".join(sections) + "</body></html>"
    path.write_text(html, encoding="utf-8")
