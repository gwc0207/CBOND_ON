from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cbond_on.domain.risk import RiskModelError


@dataclass(frozen=True)
class FactorReturnResult:
    factor_returns: pd.Series
    residuals: pd.Series
    fitted_returns: pd.Series
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CovarianceResult:
    covariance_daily: pd.DataFrame
    covariance_annual: pd.DataFrame
    correlation: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SpecificRiskResult:
    specific_risk: pd.DataFrame
    diagnostics: dict[str, Any]


def _as_numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def _effective_condition_number(x: np.ndarray, *, expected_rank: int) -> float:
    singular = np.linalg.svd(x, compute_uv=False)
    valid = singular[singular > max(1e-12, singular[0] * 1e-12)] if singular.size else np.array([])
    if valid.size < expected_rank or expected_rank <= 0:
        return float("inf")
    return float(valid[0] / valid[expected_rank - 1])


def _constrained_wls(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray | None,
) -> np.ndarray:
    sqrt_w = np.sqrt(weights)
    if constraints is None or constraints.size == 0:
        coef, *_ = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)
        return coef

    hessian = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights * y)
    constraints = np.atleast_2d(constraints)
    zero = np.zeros((constraints.shape[0], constraints.shape[0]), dtype=float)
    kkt = np.block([[hessian, constraints.T], [constraints, zero]])
    target = np.concatenate([rhs, np.zeros(constraints.shape[0], dtype=float)])
    solution, *_ = np.linalg.lstsq(kkt, target, rcond=None)
    return solution[: x.shape[1]]


def _huber_weights(residuals: np.ndarray, *, c: float) -> tuple[np.ndarray, float]:
    center = float(np.nanmedian(residuals))
    mad = float(np.nanmedian(np.abs(residuals - center)))
    scale = mad / 0.6744897501960817 if mad > 0 else float(np.nanstd(residuals))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.ones_like(residuals, dtype=float), 0.0
    clipped = np.minimum(1.0, (float(c) * scale) / np.maximum(np.abs(residuals), 1e-15))
    return clipped, scale


def estimate_factor_returns(
    panel: pd.DataFrame,
    factor_columns: Iterable[str],
    *,
    return_column: str = "gross_return",
    weight_column: str = "regression_weight",
    industry_factor_columns: Iterable[str] = (),
    min_samples: int = 150,
    max_condition_number: float = 100.0,
    huber_c: float = 1.345,
    huber_iterations: int = 3,
) -> FactorReturnResult:
    """Estimate one day's market/style/industry factor returns with robust WLS.

    All industry dummy columns may be supplied simultaneously.  The regression
    then applies the weighted zero-sum industry-return constraint, avoiding an
    arbitrary reference industry in reports.
    """

    factors = tuple(str(name) for name in factor_columns)
    if not factors:
        raise RiskModelError("factor-return regression needs at least one factor")
    missing = [name for name in (*factors, return_column, weight_column) if name not in panel.columns]
    if missing:
        raise RiskModelError(f"factor-return panel missing columns: {missing}")

    y_all = _as_numeric(panel[return_column])
    x_all = np.column_stack([_as_numeric(panel[name]) for name in factors])
    w_all = _as_numeric(panel[weight_column])
    valid = np.isfinite(y_all) & np.isfinite(w_all) & (w_all > 0) & np.isfinite(x_all).all(axis=1)
    n_obs = int(valid.sum())
    industry = tuple(name for name in industry_factor_columns if name in factors)
    parameter_names = ("CB_MKT", *factors)
    expected_rank = len(parameter_names) - (1 if industry else 0)
    if n_obs < max(int(min_samples), expected_rank + 2):
        raise RiskModelError(
            f"insufficient factor-return observations: n={n_obs}, "
            f"need>={max(int(min_samples), expected_rank + 2)}"
        )

    y = y_all[valid]
    x = np.column_stack([np.ones(n_obs, dtype=float), x_all[valid]])
    base_w = w_all[valid]
    base_w = base_w / float(np.mean(base_w))
    condition_number = _effective_condition_number(x * np.sqrt(base_w)[:, None], expected_rank=expected_rank)

    constraints: np.ndarray | None = None
    if industry:
        industry_positions = [parameter_names.index(name) for name in industry]
        industry_weight = np.average(x[:, industry_positions], axis=0, weights=base_w)
        constraint = np.zeros(len(parameter_names), dtype=float)
        constraint[industry_positions] = industry_weight
        constraints = constraint.reshape(1, -1)

    robust = np.ones(n_obs, dtype=float)
    beta = np.zeros(len(parameter_names), dtype=float)
    scale = float("nan")
    for _ in range(max(1, int(huber_iterations))):
        beta = _constrained_wls(x, y, base_w * robust, constraints)
        residuals = y - x @ beta
        robust, scale = _huber_weights(residuals, c=huber_c)
    residuals = y - x @ beta
    fitted = x @ beta

    weighted_mean = float(np.average(y, weights=base_w))
    sse = float(np.sum(base_w * residuals**2))
    sst = float(np.sum(base_w * (y - weighted_mean) ** 2))
    weighted_r2 = float(1.0 - sse / sst) if sst > 1e-18 else float("nan")
    all_residuals = pd.Series(np.nan, index=panel.index, dtype=float)
    all_fitted = pd.Series(np.nan, index=panel.index, dtype=float)
    all_residuals.loc[panel.index[valid]] = residuals
    all_fitted.loc[panel.index[valid]] = fitted

    status = "ok" if np.isfinite(condition_number) and condition_number <= float(max_condition_number) else "condition_warning"
    diagnostics = {
        "status": status,
        "sample_count": n_obs,
        "parameter_count": len(parameter_names),
        "effective_rank": expected_rank,
        "condition_number": condition_number,
        "weighted_r2": weighted_r2,
        "huber_scale": scale,
        "huber_downweighted_fraction": float((robust < 0.999999).mean()),
        "industry_zero_sum_constraint": bool(industry),
    }
    return FactorReturnResult(
        factor_returns=pd.Series(beta, index=parameter_names, dtype=float),
        residuals=all_residuals,
        fitted_returns=all_fitted,
        diagnostics=diagnostics,
    )


def _ewma_weights(count: int, half_life_days: float) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=float)
    ages = np.arange(count - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / float(half_life_days))
    return weights / float(weights.sum())


def _nearest_psd(matrix: np.ndarray, *, floor: float = 1e-18) -> tuple[np.ndarray, float]:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    min_eigen = float(eigenvalues.min()) if eigenvalues.size else float("nan")
    clipped = np.maximum(eigenvalues, float(floor))
    psd = (vectors * clipped) @ vectors.T
    return (psd + psd.T) / 2.0, min_eigen


def estimate_factor_covariance(
    factor_returns: pd.DataFrame,
    factor_columns: Iterable[str] | None = None,
    *,
    half_life_days: float = 60.0,
    shrinkage: float = 0.15,
    min_observations: int = 120,
    annualization_days: int = 244,
) -> CovarianceResult:
    """Estimate a PSD EWMA covariance matrix from factor-return history."""

    if half_life_days <= 0:
        raise RiskModelError("covariance half life must be positive")
    if not 0.0 <= shrinkage <= 1.0:
        raise RiskModelError("covariance shrinkage must be between zero and one")
    if factor_columns is None:
        factors = tuple(column for column in factor_returns.columns if column != "trade_date")
    else:
        factors = tuple(str(column) for column in factor_columns)
    if not factors:
        raise RiskModelError("factor covariance needs factor columns")
    missing = [column for column in factors if column not in factor_returns.columns]
    if missing:
        raise RiskModelError(f"factor covariance missing columns: {missing}")
    matrix = factor_returns.loc[:, list(factors)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    matrix = matrix.dropna(how="any")
    if len(matrix) < max(2, int(min_observations)):
        raise RiskModelError(f"insufficient factor-return history: {len(matrix)} < {min_observations}")
    values = matrix.to_numpy(dtype=float)
    weights = _ewma_weights(len(values), half_life_days)
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    covariance = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * np.diag(np.diag(covariance))
    covariance, min_eigen_before_clip = _nearest_psd(covariance)
    diag = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
    correlation = covariance / np.outer(diag, diag)
    covariance_df = pd.DataFrame(covariance, index=factors, columns=factors)
    correlation_df = pd.DataFrame(correlation, index=factors, columns=factors)
    annual = covariance_df * int(annualization_days)
    return CovarianceResult(
        covariance_daily=covariance_df,
        covariance_annual=annual,
        correlation=correlation_df,
        diagnostics={
            "observations": int(len(matrix)),
            "half_life_days": float(half_life_days),
            "shrinkage": float(shrinkage),
            "min_eigen_before_psd_clip": min_eigen_before_clip,
            "annualization_days": int(annualization_days),
        },
    )


def _ewma_second_moment(values: np.ndarray, half_life_days: float) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    weights = _ewma_weights(len(clean), half_life_days)
    return float(np.average(clean**2, weights=weights))


def estimate_specific_risk(
    residual_history: pd.DataFrame,
    *,
    code_column: str = "code",
    residual_column: str = "specific_return",
    half_life_days: float = 60.0,
    min_observations: int = 20,
    shrinkage_observations: float = 60.0,
    annualization_days: int = 244,
) -> SpecificRiskResult:
    """Estimate diagonal specific risk with a transparent global shrinkage prior."""

    if code_column not in residual_history.columns or residual_column not in residual_history.columns:
        raise RiskModelError("specific-risk history missing code or residual columns")
    work = residual_history.loc[:, [code_column, residual_column]].copy()
    work[code_column] = work[code_column].astype(str).str.strip()
    work[residual_column] = pd.to_numeric(work[residual_column], errors="coerce")
    work = work[(work[code_column] != "") & work[residual_column].notna()].copy()
    if work.empty:
        raise RiskModelError("specific-risk history is empty")
    local_rows: list[dict[str, Any]] = []
    local_variances: list[float] = []
    for code, group in work.groupby(code_column, sort=True):
        values = group[residual_column].to_numpy(dtype=float)
        variance = _ewma_second_moment(values, half_life_days)
        n_obs = int(np.isfinite(values).sum())
        if np.isfinite(variance):
            local_variances.append(variance)
        local_rows.append({"code": code, "local_variance": variance, "observations": n_obs})
    prior = float(np.nanmedian(np.asarray(local_variances, dtype=float)))
    if not np.isfinite(prior) or prior <= 0:
        prior = 1e-8
    output = pd.DataFrame(local_rows)
    alpha = output["observations"].astype(float) / (
        output["observations"].astype(float) + float(shrinkage_observations)
    )
    local = pd.to_numeric(output["local_variance"], errors="coerce").fillna(prior)
    output["specific_variance_daily"] = alpha * local + (1.0 - alpha) * prior
    output["specific_variance_daily"] = output["specific_variance_daily"].clip(lower=max(prior * 0.05, 1e-12))
    output["specific_vol_daily"] = np.sqrt(output["specific_variance_daily"])
    output["specific_vol_annual"] = output["specific_vol_daily"] * np.sqrt(int(annualization_days))
    output["risk_source"] = np.where(
        output["observations"] >= int(min_observations),
        "ewma_shrunk",
        "insufficient_history_shrunk",
    )
    return SpecificRiskResult(
        specific_risk=output.drop(columns=["local_variance"]),
        diagnostics={
            "codes": int(len(output)),
            "global_variance_prior": prior,
            "half_life_days": float(half_life_days),
            "min_observations": int(min_observations),
        },
    )
