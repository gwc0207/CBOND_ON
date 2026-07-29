from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from cbond_on.domain.risk import FactorDefinition, RiskModelError


_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SSE": "SH",
    "SH": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "SZ": "SZ",
}

_RATING_ORDER = {
    "AAA": 1.0,
    "AAA-": 2.0,
    "AA+": 3.0,
    "AA": 4.0,
    "AA-": 5.0,
    "A+": 6.0,
    "A": 7.0,
    "A-": 8.0,
    "BBB+": 9.0,
    "BBB": 10.0,
    "BBB-": 11.0,
    "BB+": 12.0,
    "BB": 13.0,
    "BB-": 14.0,
    "B+": 15.0,
    "B": 16.0,
    "B-": 17.0,
    "CCC": 18.0,
    "CC": 19.0,
    "C": 20.0,
}


@dataclass(frozen=True)
class ExposureBuildResult:
    exposures: pd.DataFrame
    factor_columns: tuple[str, ...]
    industry_columns: tuple[str, ...]
    diagnostics: pd.DataFrame


def _clean_identifier(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\.0$", "", text)
    return text


def normalize_cbond_codes(frame: pd.DataFrame, *, code_col: str = "code") -> pd.Series:
    """Return project-standard ``instrument.exchange`` codes without mutating input."""

    if code_col in frame.columns:
        raw = frame[code_col].map(_clean_identifier)
        return raw.str.upper()
    if "instrument_code" not in frame.columns:
        raise RiskModelError("risk input needs code or instrument_code")
    instrument = frame["instrument_code"].map(_clean_identifier)
    if "exchange_code" not in frame.columns:
        return instrument.str.upper()
    exchange = frame["exchange_code"].map(_clean_identifier).str.upper().map(
        lambda value: _EXCHANGE_ALIASES.get(value, value)
    )
    return instrument.str.upper() + "." + exchange


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(valid.any()):
        return float("nan")
    x = values[valid]
    w = weights[valid]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    return float(x[min(int(np.searchsorted(cumulative, cutoff, side="left")), len(x) - 1)])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_std(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(valid.any()) or not np.isfinite(mean):
        return float("nan")
    return float(np.sqrt(np.average((values[valid] - mean) ** 2, weights=weights[valid])))


def _build_regression_weight(
    raw: pd.DataFrame,
    *,
    source_columns: Iterable[str],
    lower_quantile: float,
    upper_quantile: float,
) -> pd.Series:
    source: pd.Series | None = None
    for column in source_columns:
        if column in raw.columns:
            candidate = _numeric(raw[column])
            if candidate.notna().any():
                source = candidate
                break
    if source is None:
        return pd.Series(1.0, index=raw.index, dtype=float)
    values = source.where(source > 0)
    weights = np.sqrt(values.to_numpy(dtype=float))
    finite = weights[np.isfinite(weights) & (weights > 0)]
    if finite.size == 0:
        return pd.Series(1.0, index=raw.index, dtype=float)
    low, high = np.quantile(finite, [lower_quantile, upper_quantile])
    clipped = np.clip(weights, low, high)
    fallback = float(np.nanmedian(finite))
    clipped = np.where(np.isfinite(clipped) & (clipped > 0), clipped, fallback)
    clipped = clipped / float(np.mean(clipped))
    return pd.Series(clipped, index=raw.index, dtype=float)


def _rating_to_numeric(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.replace("（", "(", regex=False)
        .str.replace("）", ")", regex=False)
        .str.replace(r"\(.*?\)", "", regex=True)
    )
    normalized = normalized.replace({"NAN": "", "NONE": "", "NULL": ""})
    return normalized.map(_RATING_ORDER).astype(float)


def _transform(values: pd.Series, kind: str) -> pd.Series:
    key = str(kind).strip().lower()
    if key == "rating_ordinal":
        return _rating_to_numeric(values)
    out = _numeric(values)
    if key == "identity":
        return out
    if key == "positive":
        return out.where(out > 0)
    if key == "log":
        return np.log(out.where(out > 0))
    if key == "log1p":
        return np.log1p(out.where(out >= 0))
    if key == "binary":
        return out.where(out.isin([0.0, 1.0]))
    raise RiskModelError(f"unsupported risk factor transform: {kind}")


def _resolve_factor_source(raw: pd.DataFrame, definition: FactorDefinition) -> tuple[pd.Series | None, str]:
    for column in definition.source_columns:
        if column not in raw.columns:
            continue
        values = _transform(raw[column], definition.transform)
        if values.notna().any():
            return values, column
    return None, ""


def _standardize(
    values: pd.Series,
    weights: pd.Series,
    *,
    winsor_lower: float,
    winsor_upper: float,
) -> tuple[pd.Series, float, float, float, float]:
    raw = _numeric(values)
    w = _numeric(weights).fillna(0.0)
    x = raw.to_numpy(dtype=float)
    weight_values = w.to_numpy(dtype=float)
    median = _weighted_quantile(x, weight_values, 0.5)
    if not np.isfinite(median):
        raise RiskModelError("cannot standardize an all-missing factor")
    filled = np.where(np.isfinite(x), x, median)
    low = _weighted_quantile(filled, weight_values, winsor_lower)
    high = _weighted_quantile(filled, weight_values, winsor_upper)
    clipped = np.clip(filled, low, high)
    mean = _weighted_mean(clipped, weight_values)
    std = _weighted_std(clipped, weight_values, mean)
    if not np.isfinite(std) or std <= 1e-12:
        raise RiskModelError("factor has zero cross-sectional dispersion")
    standardized = (clipped - mean) / std
    return pd.Series(standardized, index=values.index), median, low, high, std


def _weighted_residualize(
    target: pd.Series,
    anchors: list[pd.Series],
    weights: pd.Series,
) -> pd.Series:
    y = _numeric(target).to_numpy(dtype=float)
    x_columns = [np.ones(len(target), dtype=float)]
    x_columns.extend(_numeric(anchor).to_numpy(dtype=float) for anchor in anchors)
    x = np.column_stack(x_columns)
    w = _numeric(weights).to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(x).all(axis=1)
    if int(valid.sum()) <= x.shape[1]:
        raise RiskModelError("insufficient observations for factor orthogonalization")
    sqrt_w = np.sqrt(w[valid])
    beta, *_ = np.linalg.lstsq(x[valid] * sqrt_w[:, None], y[valid] * sqrt_w, rcond=None)
    residual = y - x @ beta
    residual[~valid] = np.nan
    return pd.Series(residual, index=target.index)


def _weighted_rescale_without_winsor(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Center/scale an already residualized factor without breaking orthogonality."""

    x = _numeric(values).to_numpy(dtype=float)
    w = _numeric(weights).to_numpy(dtype=float)
    median = _weighted_quantile(x, w, 0.5)
    if not np.isfinite(median):
        raise RiskModelError("cannot rescale an all-missing residualized factor")
    filled = np.where(np.isfinite(x), x, median)
    mean = _weighted_mean(filled, w)
    std = _weighted_std(filled, w, mean)
    if not np.isfinite(std) or std <= 1e-12:
        raise RiskModelError("residualized factor has zero cross-sectional dispersion")
    return pd.Series((filled - mean) / std, index=values.index)


def _safe_factor_suffix(value: Any) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or "UNKNOWN"


def _add_industry_exposures(
    out: pd.DataFrame,
    source: pd.Series,
    *,
    min_group_size: int,
    prefix: str,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, int]]:
    groups = source.fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    counts = groups.value_counts(dropna=False)
    collapsed = groups.where(groups.map(counts) >= int(min_group_size), "OTHER")
    columns: list[str] = []
    for group in sorted(collapsed.unique().tolist()):
        column = f"{prefix}__{_safe_factor_suffix(group)}"
        out[column] = (collapsed == group).astype(float)
        columns.append(column)
    return out, tuple(columns), {str(key): int(value) for key, value in collapsed.value_counts().items()}


def build_risk_exposures(
    raw: pd.DataFrame,
    definitions: Iterable[FactorDefinition],
    *,
    code_col: str = "code",
    regression_weight_columns: Iterable[str] = ("remain_size",),
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    industry: Mapping[str, Any] | None = None,
) -> ExposureBuildResult:
    """Build standardized, auditable risk exposures from one static daily panel.

    This function is intentionally pure: callers must enforce PIT selection of
    ``raw`` before invoking it.  It records factor-level coverage and never
    silently substitutes a missing factor with zero exposure.
    """

    if raw.empty:
        raise RiskModelError("cannot build risk exposures from an empty panel")
    if not 0.0 <= winsor_lower < winsor_upper <= 1.0:
        raise RiskModelError("invalid winsor quantiles")

    work = raw.copy()
    work["code"] = normalize_cbond_codes(work, code_col=code_col)
    work = work[work["code"].notna() & (work["code"] != "")].copy()
    if work["code"].duplicated().any():
        raise RiskModelError("risk exposure input has duplicate codes")
    work["regression_weight"] = _build_regression_weight(
        work,
        source_columns=regression_weight_columns,
        lower_quantile=0.05,
        upper_quantile=0.95,
    )

    diagnostics: list[dict[str, Any]] = []
    factor_columns: list[str] = []
    pending_orthogonalization: list[FactorDefinition] = []
    for definition in definitions:
        source, source_column = _resolve_factor_source(work, definition)
        if source is None:
            diagnostics.append(
                {
                    "factor": definition.name,
                    "status": "disabled_missing_source",
                    "source_column": "",
                    "coverage": 0.0,
                    "missing_count": len(work),
                    "message": "none of configured source columns is usable",
                }
            )
            continue
        coverage = float(source.notna().mean())
        if coverage < definition.min_coverage:
            diagnostics.append(
                {
                    "factor": definition.name,
                    "status": "disabled_low_coverage",
                    "source_column": source_column,
                    "coverage": coverage,
                    "missing_count": int(source.isna().sum()),
                    "message": f"coverage below minimum {definition.min_coverage:.3f}",
                }
            )
            continue
        try:
            standardized, impute, low, high, std = _standardize(
                source,
                work["regression_weight"],
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
            )
        except RiskModelError as exc:
            diagnostics.append(
                {
                    "factor": definition.name,
                    "status": "disabled_no_dispersion",
                    "source_column": source_column,
                    "coverage": coverage,
                    "missing_count": int(source.isna().sum()),
                    "message": str(exc),
                }
            )
            continue
        work[definition.name] = standardized
        work[f"missing__{definition.name}"] = source.isna().astype(int)
        factor_columns.append(definition.name)
        if definition.orthogonalize_against:
            pending_orthogonalization.append(definition)
        diagnostics.append(
            {
                "factor": definition.name,
                "status": "enabled",
                "source_column": source_column,
                "coverage": coverage,
                "missing_count": int(source.isna().sum()),
                "impute_value": impute,
                "winsor_low": low,
                "winsor_high": high,
                "cross_sectional_std": std,
                "message": "",
            }
        )

    for definition in pending_orthogonalization:
        if definition.name not in factor_columns:
            continue
        anchors: list[pd.Series] = []
        missing_anchor = [name for name in definition.orthogonalize_against if name not in factor_columns]
        if missing_anchor:
            factor_columns.remove(definition.name)
            work.drop(columns=[definition.name], inplace=True)
            diagnostics.append(
                {
                    "factor": definition.name,
                    "status": "disabled_missing_orthogonal_anchor",
                    "source_column": "",
                    "coverage": 0.0,
                    "missing_count": len(work),
                    "message": ",".join(missing_anchor),
                }
            )
            continue
        for name in definition.orthogonalize_against:
            anchor = work[name]
            anchors.append(anchor)
            if definition.orthogonalize_square:
                anchors.append(anchor.pow(2))
        residual = _weighted_residualize(work[definition.name], anchors, work["regression_weight"])
        work[definition.name] = _weighted_rescale_without_winsor(residual, work["regression_weight"])

    industry_columns: tuple[str, ...] = ()
    if industry and bool(industry.get("enabled", False)):
        industry_column = str(industry.get("source_column") or "").strip()
        if not industry_column or industry_column not in work.columns:
            diagnostics.append(
                {
                    "factor": "STOCK_INDUSTRY",
                    "status": "disabled_missing_source",
                    "source_column": industry_column,
                    "coverage": 0.0,
                    "missing_count": len(work),
                    "message": "industry input unavailable",
                }
            )
        else:
            work, industry_columns, group_counts = _add_industry_exposures(
                work,
                work[industry_column],
                min_group_size=int(industry.get("min_group_size", 15)),
                prefix=str(industry.get("prefix", "IND")),
            )
            diagnostics.append(
                {
                    "factor": "STOCK_INDUSTRY",
                    "status": "enabled",
                    "source_column": industry_column,
                    "coverage": float(work[industry_column].notna().mean()),
                    "missing_count": int(work[industry_column].isna().sum()),
                    "message": str(group_counts),
                }
            )

    if not factor_columns:
        raise RiskModelError("no risk style factor remains after data-quality checks")
    keep = ["code", "regression_weight", *factor_columns, *industry_columns]
    keep.extend(column for column in work.columns if column.startswith("missing__"))
    return ExposureBuildResult(
        exposures=work.loc[:, list(dict.fromkeys(keep))].copy(),
        factor_columns=tuple(factor_columns),
        industry_columns=industry_columns,
        diagnostics=pd.DataFrame(diagnostics),
    )
