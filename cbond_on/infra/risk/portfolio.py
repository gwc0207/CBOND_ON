from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from cbond_on.domain.risk import RiskModelError
from cbond_on.infra.risk.exposure import normalize_cbond_codes


@dataclass(frozen=True)
class PortfolioRiskResult:
    summary: dict[str, Any]
    factor_exposure: pd.DataFrame
    factor_risk_contribution: pd.DataFrame
    specific_risk_contribution: pd.DataFrame


@dataclass(frozen=True)
class AttributionResult:
    summary: dict[str, Any]
    contributions: pd.DataFrame


def _normalize_weights(
    weights: pd.DataFrame,
    *,
    code_column: str = "code",
    weight_column: str = "weight",
) -> pd.Series:
    if code_column not in weights.columns or weight_column not in weights.columns:
        raise RiskModelError(f"portfolio weights need {code_column!r} and {weight_column!r}")
    work = weights.loc[:, [code_column, weight_column]].copy()
    work["code"] = normalize_cbond_codes(work, code_col=code_column)
    work["weight"] = pd.to_numeric(work[weight_column], errors="coerce")
    work = work[work["code"].notna() & (work["code"] != "") & work["weight"].notna()].copy()
    if work.empty:
        raise RiskModelError("portfolio weights are empty after normalization")
    result = work.groupby("code", sort=True)["weight"].sum()
    total = float(result.sum())
    if not np.isfinite(total) or abs(total) <= 1e-12:
        raise RiskModelError("portfolio weights must have a non-zero sum")
    return result / total


def _align_exposure(
    exposures: pd.DataFrame,
    factor_columns: Iterable[str],
    strategy: pd.Series,
    benchmark: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if "code" not in exposures.columns:
        raise RiskModelError("exposure frame missing code")
    factors = tuple(factor_columns)
    missing_factors = [name for name in factors if name not in exposures.columns]
    if missing_factors:
        raise RiskModelError(f"exposure frame missing factor columns: {missing_factors}")
    work = exposures.loc[:, ["code", *factors]].copy()
    work["code"] = normalize_cbond_codes(work)
    if work["code"].duplicated().any():
        raise RiskModelError("exposure frame has duplicate codes")
    all_codes = strategy.index.union(benchmark.index)
    missing_codes = all_codes.difference(work["code"])
    if len(missing_codes):
        preview = ",".join(missing_codes[:10].tolist())
        raise RiskModelError(f"portfolio has codes with no risk exposure: {preview}")
    indexed = work.set_index("code").loc[all_codes, list(factors)].apply(pd.to_numeric, errors="coerce")
    if indexed.isna().any().any():
        raise RiskModelError("portfolio exposure contains missing factor values")
    return indexed, strategy.reindex(all_codes, fill_value=0.0), benchmark.reindex(all_codes, fill_value=0.0)


def _factor_matrix_with_market(exposure: pd.DataFrame, factor_columns: Iterable[str]) -> pd.DataFrame:
    factors = tuple(factor_columns)
    result = exposure.copy()
    if "CB_MKT" in factors and "CB_MKT" not in result.columns:
        result.insert(0, "CB_MKT", 1.0)
    return result.loc[:, list(factors)]


def calculate_portfolio_risk(
    exposures: pd.DataFrame,
    factor_covariance_daily: pd.DataFrame,
    specific_risk: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame | None = None,
    *,
    factor_columns: Iterable[str] | None = None,
    code_column: str = "code",
    weight_column: str = "weight",
    annualization_days: int = 244,
) -> PortfolioRiskResult:
    """Calculate absolute/active exposure, TE, and exact variance contributions."""

    covariance = factor_covariance_daily.copy()
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise RiskModelError("factor covariance must be a non-empty square matrix")
    if factor_columns is None:
        factors = tuple(covariance.index.astype(str).tolist())
    else:
        factors = tuple(str(value) for value in factor_columns)
    if list(covariance.index.astype(str)) != list(covariance.columns.astype(str)):
        raise RiskModelError("factor covariance row/column labels must match")
    missing_covariance = [name for name in factors if name not in covariance.index]
    if missing_covariance:
        raise RiskModelError(f"factor covariance missing requested factors: {missing_covariance}")
    covariance = covariance.loc[list(factors), list(factors)].apply(pd.to_numeric, errors="coerce")
    if covariance.isna().any().any():
        raise RiskModelError("factor covariance contains missing values")

    strategy = _normalize_weights(strategy_weights, code_column=code_column, weight_column=weight_column)
    has_benchmark = benchmark_weights is not None
    benchmark = (
        _normalize_weights(benchmark_weights, code_column=code_column, weight_column=weight_column)
        if benchmark_weights is not None
        else pd.Series(dtype=float)
    )
    exposure_columns = tuple(name for name in factors if name != "CB_MKT")
    exposure, strategy, benchmark = _align_exposure(exposures, exposure_columns, strategy, benchmark)
    factor_exposure_by_code = _factor_matrix_with_market(exposure, factors)
    strategy_exposure = strategy.to_numpy(dtype=float) @ factor_exposure_by_code.to_numpy(dtype=float)
    benchmark_exposure = benchmark.to_numpy(dtype=float) @ factor_exposure_by_code.to_numpy(dtype=float)
    active_exposure = strategy_exposure - benchmark_exposure

    if "code" not in specific_risk.columns or "specific_variance_daily" not in specific_risk.columns:
        raise RiskModelError("specific risk needs code and specific_variance_daily")
    specific = specific_risk.loc[:, ["code", "specific_variance_daily"]].copy()
    specific["code"] = normalize_cbond_codes(specific)
    specific = specific.drop_duplicates("code", keep="last").set_index("code")
    missing_specific = factor_exposure_by_code.index.difference(specific.index)
    if len(missing_specific):
        preview = ",".join(missing_specific[:10].tolist())
        raise RiskModelError(f"portfolio has codes with no specific risk: {preview}")
    specific_variance = pd.to_numeric(
        specific.loc[factor_exposure_by_code.index, "specific_variance_daily"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(specific_variance).all() or (specific_variance < 0).any():
        raise RiskModelError("specific risk contains invalid variance")

    cov_matrix = covariance.to_numpy(dtype=float)
    factor_marginal = cov_matrix @ active_exposure
    factor_variance_contribution = active_exposure * factor_marginal
    active_weight = strategy.to_numpy(dtype=float) - benchmark.to_numpy(dtype=float)
    specific_variance_contribution = active_weight**2 * specific_variance
    factor_variance = float(factor_variance_contribution.sum())
    specific_variance_total = float(specific_variance_contribution.sum())
    total_variance = max(0.0, factor_variance + specific_variance_total)

    factor_exposure_df = pd.DataFrame(
        {
            "factor": factors,
            "strategy_exposure": strategy_exposure,
            "benchmark_exposure": benchmark_exposure,
            "active_exposure": active_exposure,
        }
    )
    factor_contribution_df = pd.DataFrame(
        {
            "factor": factors,
            "marginal_variance": factor_marginal,
            "variance_contribution": factor_variance_contribution,
            "component_risk_daily": np.sign(factor_variance_contribution)
            * np.sqrt(np.abs(factor_variance_contribution)),
        }
    )
    specific_contribution_df = pd.DataFrame(
        {
            "code": factor_exposure_by_code.index,
            "strategy_weight": strategy.to_numpy(dtype=float),
            "benchmark_weight": benchmark.to_numpy(dtype=float),
            "active_weight": active_weight,
            "specific_variance_daily": specific_variance,
            "variance_contribution": specific_variance_contribution,
        }
    ).sort_values("variance_contribution", ascending=False, ignore_index=True)
    daily_vol = float(np.sqrt(total_variance))
    summary = {
        "risk_kind": "active" if has_benchmark else "absolute",
        "factor_variance_daily": factor_variance,
        "specific_variance_daily": specific_variance_total,
        "total_variance_daily": total_variance,
        "risk_vol_daily": daily_vol,
        "risk_vol_annual": daily_vol * np.sqrt(int(annualization_days)),
        "tracking_error_daily": daily_vol if has_benchmark else float("nan"),
        "tracking_error_annual": daily_vol * np.sqrt(int(annualization_days)) if has_benchmark else float("nan"),
        "factor_contribution_sum": float(factor_contribution_df["variance_contribution"].sum()),
        "specific_contribution_sum": float(specific_contribution_df["variance_contribution"].sum()),
        "reconciliation_error": float(
            total_variance
            - factor_contribution_df["variance_contribution"].sum()
            - specific_contribution_df["variance_contribution"].sum()
        ),
        "strategy_count": int((strategy != 0).sum()),
        "benchmark_count": int((benchmark != 0).sum()),
    }
    return PortfolioRiskResult(
        summary=summary,
        factor_exposure=factor_exposure_df,
        factor_risk_contribution=factor_contribution_df,
        specific_risk_contribution=specific_contribution_df,
    )


def attribute_active_return(
    exposures: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame | None,
    factor_returns: Mapping[str, float] | pd.Series,
    specific_returns: pd.DataFrame,
    *,
    factor_columns: Iterable[str],
    code_column: str = "code",
    weight_column: str = "weight",
    specific_return_column: str = "specific_return",
    cost_active_return: float = 0.0,
) -> AttributionResult:
    """Explain an active gross return as factor plus specific return, with costs separate."""

    factors = tuple(str(value) for value in factor_columns)
    returns = pd.Series(factor_returns, dtype=float).reindex(factors)
    if returns.isna().any():
        missing = returns[returns.isna()].index.tolist()
        raise RiskModelError(f"factor-return attribution missing factors: {missing}")
    has_benchmark = benchmark_weights is not None
    strategy = _normalize_weights(strategy_weights, code_column=code_column, weight_column=weight_column)
    benchmark = (
        _normalize_weights(benchmark_weights, code_column=code_column, weight_column=weight_column)
        if benchmark_weights is not None
        else pd.Series(dtype=float)
    )
    exposure_columns = tuple(name for name in factors if name != "CB_MKT")
    exposure, strategy, benchmark = _align_exposure(exposures, exposure_columns, strategy, benchmark)
    factor_by_code = _factor_matrix_with_market(exposure, factors)
    active_weight = strategy - benchmark
    active_exposure = active_weight.to_numpy(dtype=float) @ factor_by_code.to_numpy(dtype=float)
    factor_contribution = active_exposure * returns.to_numpy(dtype=float)

    if code_column not in specific_returns.columns or specific_return_column not in specific_returns.columns:
        raise RiskModelError("specific-return input missing code or specific_return")
    residuals = specific_returns.loc[:, [code_column, specific_return_column]].copy()
    residuals["code"] = normalize_cbond_codes(residuals, code_col=code_column)
    residuals["specific_return"] = pd.to_numeric(residuals[specific_return_column], errors="coerce")
    residuals = residuals.dropna(subset=["specific_return"]).drop_duplicates("code", keep="last").set_index("code")
    missing = active_weight.index.difference(residuals.index)
    if len(missing):
        preview = ",".join(missing[:10].tolist())
        raise RiskModelError(f"active portfolio has codes with no specific return: {preview}")
    specific_contribution = float(
        np.dot(active_weight.to_numpy(dtype=float), residuals.loc[active_weight.index, "specific_return"].to_numpy(dtype=float))
    )
    gross_active = float(factor_contribution.sum() + specific_contribution)
    net_active = float(gross_active - float(cost_active_return))
    rows = [
        {
            "component_type": "factor",
            "component": factor,
            "active_exposure": float(exposure_value),
            "factor_return": float(factor_return),
            "contribution": float(contribution),
        }
        for factor, exposure_value, factor_return, contribution in zip(
            factors, active_exposure, returns.to_numpy(dtype=float), factor_contribution
        )
    ]
    rows.extend(
        [
            {
                "component_type": "specific",
                "component": "SPECIFIC",
                "active_exposure": float("nan"),
                "factor_return": float("nan"),
                "contribution": specific_contribution,
            },
            {
                "component_type": "cost",
                "component": "COST",
                "active_exposure": float("nan"),
                "factor_return": float("nan"),
                "contribution": -float(cost_active_return),
            },
        ]
    )
    return_kind = "active" if has_benchmark else "portfolio"
    return AttributionResult(
        summary={
            "attribution_kind": return_kind,
            f"gross_{return_kind}_return": gross_active,
            f"net_{return_kind}_return": net_active,
            "factor_contribution": float(factor_contribution.sum()),
            "specific_contribution": specific_contribution,
            "cost_active_return": float(cost_active_return),
            "gross_reconciliation_error": float(gross_active - factor_contribution.sum() - specific_contribution),
        },
        contributions=pd.DataFrame(rows),
    )
