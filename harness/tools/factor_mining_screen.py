"""Screen a scratch-only T1430 factor store under the fixed T-1 ``o_0005`` universe.

This is deliberately a research harness tool rather than factor-engine code.
It reads already-built factor parquet files, applies the existing causal
``o_0005`` allowlist, then opens the same-score-day 14:42 labels.  It never
imports factor implementations, writes a database, changes a configuration,
or writes anywhere other than the explicit ``--output-dir``.

The selection order is deterministic: absolute *mean daily Pearson IC* first,
then factor name.  A retained factor must have ``abs(mean Pearson IC)`` above
the explicit threshold and satisfy the pairwise redundancy rule against every
previously retained factor:

* members of the same family: redundancy < 0.80;
* members of different families: redundancy < 0.70.

Pairwise redundancy is the larger of the mean daily absolute Pearson and
Spearman correlations on that day's common, finite, label-aligned and
T-1-``o_0005``-filtered factor observations.  It is not a full-period
concatenated correlation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ``py harness/tools/factor_mining_screen.py`` adds harness/tools rather than
# the checkout root to sys.path.  Keep the standalone invocation documented by
# the harness working without introducing a new run entrypoint.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cbond_on.infra.universe.pool_filter import (  # noqa: E402
    UpstreamPoolConfig,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.core.trading_days import list_trading_days_from_raw  # noqa: E402


_BASE_START = "2025-01-01"
_POOL_TABLE = "quant_factor_dev.researcher_xuvb.o_0005"
_IC_THRESHOLD = 0.02
_WITHIN_FAMILY_THRESHOLD = 0.80
_CROSS_FAMILY_THRESHOLD = 0.70
_MAX_SELECTED = 100
_MIN_VALID_DAYS = 250
_MIN_VALID_DAYS_PER_PARTITION = 50
_MIN_REDUNDANCY_DAYS = 200
_PANEL_FACTOR_TIMESTAMP = "14:30:00"
_LABEL_TIMESTAMP = "14:42:00"
_FIXED_POOL_CONTRACT = {
    "pool_table": _POOL_TABLE,
    "positive_field": "factor_value",
    "positive_fallback_field": "weight",
    "positive_threshold": 0.0,
    "pool_lag_trading_days": 1,
    "pool_asset": "cbond",
}
_SHA256_CHUNK_BYTES = 1024 * 1024
_PROTECTED_OUTPUT_ROOTS = (
    Path(r"D:\cbond_on\factor_data"),
    Path(r"D:\cbond_on\results\live"),
    Path(r"D:\cbond_on\results\model_state"),
    Path(r"D:\cbond_on\results\backtest"),
    Path(r"D:\cbond_on\results\analysis"),
)


def _parse_day(value: str, *, context: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{context} must be YYYY-MM-DD: {value!r}") from exc


def _as_day_text(value: object, *, context: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    _parse_day(text, context=context)
    return text


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_research_only_paths(
    *,
    factor_root: Path,
    label_root: Path,
    raw_data_root: Path,
    output_dir: Path,
) -> None:
    """Reject destinations that could turn this read-only screen into a live write."""
    production_factor_root = _resolved(r"D:\cbond_on\factor_data")
    if _is_within(factor_root, production_factor_root):
        raise ValueError(
            "--factor-root must be a scratch FactorStore, not D:/cbond_on/factor_data"
        )
    for protected in (_resolved(item) for item in _PROTECTED_OUTPUT_ROOTS):
        if _is_within(output_dir, protected):
            raise ValueError(
                "--output-dir must be a research scratch directory; "
                f"refusing protected destination: {output_dir}"
            )
    for input_root, name in (
        (factor_root, "factor-root"),
        (label_root, "label-root"),
        (raw_data_root, "raw-data-root"),
    ):
        if _is_within(output_dir, input_root):
            raise ValueError(
                f"--output-dir must not be inside the read-only --{name}: {output_dir}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_evidence(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"input file missing: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _same_evidence(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return (
        str(left.get("path")) == str(right.get("path"))
        and int(left.get("bytes", -1)) == int(right.get("bytes", -1))
        and str(left.get("sha256")) == str(right.get("sha256"))
    )


def _assert_expected_evidence(
    path: Path,
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    actual = _file_evidence(path)
    if not _same_evidence(actual, expected):
        raise RuntimeError(f"{context} changed while screening: {path}")


def _normalize_code_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _canonical_code_set_sha256(codes: set[str]) -> str:
    payload = "\n".join(sorted({str(code) for code in codes})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _factor_day_paths(
    *,
    factor_root: Path,
    panel_name: str,
    start: str,
    end: str,
) -> dict[str, Path]:
    base = factor_root / "factors" / panel_name
    if not base.is_dir():
        raise FileNotFoundError(f"scratch FactorStore panel root missing: {base}")
    paths: dict[str, Path] = {}
    for path in sorted(base.glob("*/*.parquet")):
        stem = path.stem
        if len(stem) != 8 or not stem.isdigit():
            raise ValueError(f"unexpected FactorStore daily filename: {path}")
        try:
            day = datetime.strptime(stem, "%Y%m%d").date().isoformat()
        except ValueError as exc:
            raise ValueError(f"invalid FactorStore daily filename: {path}") from exc
        if not (start <= day <= end):
            continue
        if day in paths:
            raise ValueError(f"duplicate FactorStore day {day}: {paths[day]} and {path}")
        paths[day] = path
    if not paths:
        raise RuntimeError(
            f"no T1430 scratch FactorStore parquet in requested range {start}..{end}: {base}"
        )
    return paths


def _expected_score_days(*, raw_data_root: Path, start: str, end: str) -> list[str]:
    """Freeze the full score-day calendar independently of FactorStore files."""

    days = list_trading_days_from_raw(
        raw_data_root,
        _parse_day(start, context="start"),
        _parse_day(end, context="end"),
        kind="snapshot",
        asset="cbond",
    )
    if not days:
        raise RuntimeError(
            "no score days resolved from the authoritative cbond trading calendar; "
            "refusing to define the IC window from surviving FactorStore files"
        )
    return [day.isoformat() for day in days]


def _load_family_catalog(path: Path) -> pd.DataFrame:
    """Load a strict, auditable family-to-factor declaration.

    Supported JSON forms are deliberately small and explicit:

    ``{"family_name": ["factor_a", "factor_b"]}``
        The preferred family-to-factor hierarchy.

    ``{"factor_a": "family_name"}``
        Accepted for an external generator that emits factor-to-family mapping.

    ``{"families": {"family_name": ["factor_a"]}}``
        A wrapped version of the preferred hierarchy.
    """
    if not path.is_file():
        raise FileNotFoundError(f"factor family catalog missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"factor family catalog is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("factor family catalog must be a JSON object")
    if set(raw) == {"families"}:
        raw = raw["families"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("factor family catalog is empty or malformed")

    rows: list[dict[str, str]] = []
    values = list(raw.values())
    if all(isinstance(value, list) for value in values):
        for family_raw, factors_raw in raw.items():
            family = str(family_raw).strip()
            if not family:
                raise ValueError("factor family names must be non-empty")
            if not factors_raw:
                raise ValueError(f"factor family has no factors: {family}")
            for factor_raw in factors_raw:
                factor = str(factor_raw).strip()
                if not factor:
                    raise ValueError(f"empty factor name in family: {family}")
                rows.append({"family": family, "factor": factor})
    elif all(isinstance(value, str) for value in values):
        for factor_raw, family_raw in raw.items():
            factor = str(factor_raw).strip()
            family = str(family_raw).strip()
            if not factor or not family:
                raise ValueError("factor-to-family catalog entries must be non-empty strings")
            rows.append({"family": family, "factor": factor})
    else:
        raise ValueError(
            "factor family catalog must map family->list[factor] or factor->family; "
            "mixed JSON shapes are not accepted"
        )
    catalog = pd.DataFrame(rows, columns=["family", "factor"])
    if catalog.empty or catalog["factor"].duplicated().any():
        duplicates = sorted(catalog.loc[catalog["factor"].duplicated(), "factor"].tolist())
        raise ValueError(f"factor family catalog has duplicate factor names: {duplicates[:10]}")
    return catalog.sort_values(["family", "factor"], kind="mergesort").reset_index(drop=True)


def _pool_day_path(
    *,
    raw_data_root: Path,
    pool_cfg: UpstreamPoolConfig,
    pool_day: str,
) -> Path:
    day = _parse_day(pool_day, context="resolved T-1 pool day")
    return (
        raw_data_root
        / pool_cfg.pool_table.replace(".", "__")
        / f"{day.year:04d}-{day.month:02d}"
        / f"{day:%Y%m%d}.parquet"
    )


def _fixed_pool_codes_by_day(
    *,
    raw_data_root: Path,
    score_days: Sequence[str],
) -> tuple[dict[str, set[str]], pd.DataFrame, dict[str, object]]:
    """Resolve the existing T-1 pool for every day before any label access.

    The explicit no-fallback check is intentional.  A missing historical pool
    cannot be replaced by a broader score universe without invalidating the
    requested IC contract.
    """
    pool_cfg = load_upstream_pool_config()
    observed_pool_contract = asdict(pool_cfg)
    if observed_pool_contract != _FIXED_POOL_CONTRACT:
        raise ValueError(
            "fixed universe contract drifted; expected "
            f"{_FIXED_POOL_CONTRACT!r}, got {observed_pool_contract!r}"
        )

    codes_by_day: dict[str, set[str]] = {}
    rows: list[dict[str, object]] = []
    for day_text in sorted({str(day) for day in score_days}):
        trade_day = _parse_day(day_text, context="score day")
        codes, info = resolve_pool_codes_for_trade_day(
            raw_data_root=raw_data_root,
            trade_day=trade_day,
            pool_cfg=pool_cfg,
            enabled=True,
        )
        if codes is None or bool(info.get("fallback_no_filter", False)):
            raise RuntimeError(
                "fixed T-1 o_0005 pool unavailable; refusing no-filter fallback: "
                f"score_day={day_text} expected_pool_day={info.get('pool_day_expected')} "
                f"reason={info.get('fallback_reason')}"
            )
        normalized = {
            str(code).strip().removesuffix(".0")
            for code in codes
            if str(code).strip()
        }
        if not normalized:
            raise RuntimeError(f"fixed T-1 o_0005 pool resolved empty: {day_text}")
        pool_day_used = _as_day_text(
            info.get("pool_day_used"), context=f"pool_day_used for {day_text}"
        )
        evidence = _file_evidence(
            _pool_day_path(
                raw_data_root=raw_data_root,
                pool_cfg=pool_cfg,
                pool_day=pool_day_used,
            )
        )
        codes_by_day[day_text] = normalized
        rows.append(
            {
                "score_day": day_text,
                "pool_day_expected": _as_day_text(
                    info.get("pool_day_expected"), context=f"pool_day_expected for {day_text}"
                ),
                "pool_day_used": pool_day_used,
                "pool_codes": int(len(normalized)),
                "fallback_no_filter": False,
                "allowlist_codes_sha256": _canonical_code_set_sha256(normalized),
                "pool_file_path": evidence["path"],
                "pool_file_bytes": evidence["bytes"],
                "pool_file_sha256": evidence["sha256"],
            }
        )
    return (
        codes_by_day,
        pd.DataFrame(rows),
        {
            "enabled": True,
            "kind": "existing_tminus1_o0005_allowlist",
            "raw_data_root": str(raw_data_root),
            "pool_config": asdict(pool_cfg),
            "fallback_policy": "fail_closed_no_no_filter_fallback",
            "label_access": "none before all pool days resolve",
        },
    )


def _read_label_1442(
    *,
    label_root: Path,
    day: str,
) -> tuple[pd.DataFrame | None, dict[str, object] | None, str]:
    path = label_root / day[:7] / f"{day.replace('-', '')}.parquet"
    if not path.is_file():
        return None, None, "missing_label_file"
    before = _file_evidence(path)
    frame = pd.read_parquet(path)
    after = _file_evidence(path)
    if not _same_evidence(before, after):
        raise RuntimeError(f"14:42 label changed while being read: {path}")
    required = {"code", "trade_time", "y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"14:42 label missing columns {missing}: {path}")
    trade_time = pd.to_datetime(frame["trade_time"], errors="coerce")
    expected_time = pd.Timestamp(f"{day} {_LABEL_TIMESTAMP}")
    filtered = frame.loc[trade_time == expected_time, ["code", "y"]].copy()
    if filtered.empty:
        return None, after, "empty_1442_label"
    filtered["code"] = _normalize_code_series(filtered["code"])
    filtered["y"] = pd.to_numeric(filtered["y"], errors="coerce")
    filtered = filtered.loc[
        filtered["code"].ne("") & np.isfinite(filtered["y"]), ["code", "y"]
    ].copy()
    if filtered.empty:
        return None, after, "no_finite_1442_label"
    if filtered["code"].duplicated().any():
        raise ValueError(f"duplicate code in same-day 14:42 label: {path}")
    return filtered, after, "ok"


def _read_factor_frame(
    *,
    path: Path,
    day: str,
    expected_evidence: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if expected_evidence is not None:
        _assert_expected_evidence(
            path, expected_evidence, context=f"T1430 factor source for score_day={day}"
        )
    before = _file_evidence(path)
    frame = pd.read_parquet(path)
    after = _file_evidence(path)
    if not _same_evidence(before, after):
        raise RuntimeError(f"T1430 factor source changed while being read: {path}")
    if expected_evidence is not None and not _same_evidence(after, expected_evidence):
        raise RuntimeError(f"T1430 factor source drifted after initial audit: {path}")
    if frame.columns.duplicated().any():
        raise ValueError(f"duplicate factor column in T1430 source: {path}")
    if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
        frame = frame.reset_index()
    required = {"dt", "code"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"T1430 FactorStore file missing {missing}: {path}")
    dt = pd.to_datetime(frame["dt"], errors="coerce")
    expected_time = pd.Timestamp(f"{day} {_PANEL_FACTOR_TIMESTAMP}")
    if dt.isna().any() or not (dt == expected_time).all():
        observed = sorted({str(item) for item in dt.dropna().unique()})
        raise ValueError(
            "T1430 FactorStore timestamp mismatch for "
            f"{path}: expected={expected_time.isoformat()} observed={observed[:5]}"
        )
    frame = frame.copy()
    frame["code"] = _normalize_code_series(frame["code"])
    frame = frame.loc[frame["code"].ne("")].copy()
    if frame.empty:
        raise ValueError(f"T1430 FactorStore contains no non-empty codes: {path}")
    if frame["code"].duplicated().any():
        raise ValueError(f"duplicate code in T1430 FactorStore source: {path}")
    return frame, after


def _finite_or_none(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _correlation(
    frame: pd.DataFrame,
    *,
    method: str,
    min_cross_section: int,
) -> float | None:
    usable = frame.dropna()
    if len(usable) < min_cross_section:
        return None
    left = usable.iloc[:, 0]
    right = usable.iloc[:, 1]
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    return _finite_or_none(left.corr(right, method=method))


def _daily_factor_metric(
    *,
    day: str,
    factor: str,
    family: str,
    factor_frame: pd.DataFrame,
    label_pool: pd.DataFrame,
    pool_codes: set[str],
    min_cross_section: int,
    top_k: int,
) -> dict[str, object]:
    """Calculate raw (not sign-flipped) daily factor diagnostics."""
    source_rows = int(len(factor_frame))
    factor_pool = factor_frame.loc[factor_frame["code"].isin(pool_codes), ["code"]].copy()
    base: dict[str, object] = {
        "score_day": day,
        "factor": factor,
        "family": family,
        "factor_source_rows": source_rows,
        "pool_codes": int(len(pool_codes)),
        "pool_label_rows": int(len(label_pool)),
        "pool_factor_rows": int(len(factor_pool)),
        "factor_label_rows": 0,
        "finite_rows": 0,
        "coverage": 0.0 if len(label_pool) else None,
        "finite_rate": 0.0 if len(factor_pool) else None,
        "n": 0,
        "pearson_ic": None,
        "rank_ic": None,
        "top20_mean_y": None,
        "valid_for_ic": False,
        "status": "missing_factor_column",
    }
    if factor not in factor_frame.columns:
        return base
    values = factor_frame.loc[
        factor_frame["code"].isin(pool_codes), ["code", factor]
    ].copy()
    values[factor] = pd.to_numeric(values[factor], errors="coerce")
    merged = label_pool.merge(values, on="code", how="left", validate="one_to_one")
    observed = merged[factor].notna()
    finite = observed & np.isfinite(merged[factor])
    factor_label_rows = int(observed.sum())
    finite_rows = int(finite.sum())
    base.update(
        {
            "factor_label_rows": factor_label_rows,
            "finite_rows": finite_rows,
            "coverage": _finite_or_none(finite_rows / len(label_pool)) if len(label_pool) else None,
            "finite_rate": _finite_or_none(finite_rows / len(values)) if len(values) else None,
        }
    )
    usable = merged.loc[finite, ["code", factor, "y"]].copy()
    base["n"] = int(len(usable))
    if len(usable) >= top_k:
        top = usable.sort_values([factor, "code"], ascending=[False, True], kind="mergesort").head(top_k)
        base["top20_mean_y"] = _finite_or_none(top["y"].mean())
    if len(usable) < min_cross_section:
        base["status"] = "insufficient_finite_rows"
        return base
    if usable[factor].nunique() < 2:
        base["status"] = "constant_factor"
        return base
    if usable["y"].nunique() < 2:
        base["status"] = "constant_label"
        return base
    base["pearson_ic"] = _correlation(
        usable[[factor, "y"]], method="pearson", min_cross_section=min_cross_section
    )
    base["rank_ic"] = _correlation(
        usable[[factor, "y"]], method="spearman", min_cross_section=min_cross_section
    )
    if base["pearson_ic"] is None or base["rank_ic"] is None:
        base["status"] = "invalid_correlation"
        return base
    base["valid_for_ic"] = True
    base["status"] = "ok"
    return base


def _columnwise_pearson(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return one Pearson correlation per value column against ``target``.

    Inputs must already be finite and aligned.  This mirrors the correlation
    algebra used by the two-column ``np.corrcoef`` path reached from
    ``Series.corr`` while avoiding one Python/Pandas call for every factor.
    """

    result = np.full(values.shape[1], np.nan, dtype=float)
    if values.shape[0] == 0 or values.shape[1] == 0:
        return result
    centered_values = values - values.mean(axis=0, keepdims=True)
    centered_target = target - target.mean()
    numerator = centered_values.T @ centered_target
    value_norm = np.einsum("ij,ij->j", centered_values, centered_values)
    target_norm = float(centered_target @ centered_target)
    denominator = np.sqrt(value_norm * target_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def _daily_factor_metrics(
    *,
    day: str,
    catalog: pd.DataFrame,
    factor_frame: pd.DataFrame,
    label_pool: pd.DataFrame,
    pool_codes: set[str],
    min_cross_section: int,
    top_k: int,
) -> list[dict[str, object]]:
    """Calculate every catalogued daily metric from one code-aligned panel.

    This is deliberately equivalent to calling ``_daily_factor_metric`` for
    every catalog row.  The readers have already proved that both source sides
    have unique codes, so a left ``merge(..., validate='one_to_one')`` is
    exactly a factor-code reindex in the existing ``label_pool`` order.

    Pearson and Spearman are still calculated per score day and per factor on
    that factor's common finite observations.  Columns sharing a finite-mask
    share the aligned target vector and its rank; no cross-day or global-rank
    shortcut is permitted.
    """

    source_rows = int(len(factor_frame))
    pool_mask = factor_frame["code"].isin(pool_codes)
    pool_factor_rows = int(pool_mask.sum())
    catalog_rows = [(str(row.family), str(row.factor)) for row in catalog.itertuples(index=False)]
    present_factors = [factor for _family, factor in catalog_rows if factor in factor_frame.columns]

    def base_row(*, family: str, factor: str) -> dict[str, object]:
        return {
            "score_day": day,
            "factor": factor,
            "family": family,
            "factor_source_rows": source_rows,
            "pool_codes": int(len(pool_codes)),
            "pool_label_rows": int(len(label_pool)),
            "pool_factor_rows": pool_factor_rows,
            "factor_label_rows": 0,
            "finite_rows": 0,
            "coverage": 0.0 if len(label_pool) else None,
            "finite_rate": 0.0 if pool_factor_rows else None,
            "n": 0,
            "pearson_ic": None,
            "rank_ic": None,
            "top20_mean_y": None,
            "valid_for_ic": False,
            "status": "missing_factor_column",
        }

    if not present_factors:
        return [base_row(family=family, factor=factor) for family, factor in catalog_rows]

    # This single alignment is equivalent to the existing per-factor left
    # merge: values outside the fixed pool are excluded and label-only codes
    # become NaN.  ``_read_factor_frame`` and ``_read_label_1442`` enforce the
    # one-to-one code contract before this helper is reached.
    values = (
        factor_frame.loc[pool_mask, ["code", *present_factors]]
        .set_index("code")
        .reindex(label_pool["code"].tolist())
    )
    numeric = values.apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=float, na_value=np.nan, copy=True)
    target = pd.to_numeric(label_pool["y"], errors="coerce").to_numpy(dtype=float, copy=True)
    observed = ~np.isnan(matrix)
    finite = observed & np.isfinite(matrix)
    factor_label_rows = observed.sum(axis=0, dtype=np.int64)
    finite_rows = finite.sum(axis=0, dtype=np.int64)
    finite_for_rank = np.where(finite, matrix, np.nan)

    # ``sort_values([factor, code], ascending=[False, True], kind='mergesort')``
    # is reproduced by first imposing code order, then using the stable
    # ``first`` tie policy on descending factor ranks.
    code_order = np.argsort(label_pool["code"].astype(str).to_numpy(), kind="stable")
    top_rank = pd.DataFrame(
        finite_for_rank[code_order], columns=present_factors
    ).rank(method="first", ascending=False, na_option="keep")
    top_mask = top_rank.to_numpy(dtype=float, copy=False) <= float(top_k)
    top_mean = (top_mask.T @ target[code_order]) / float(top_k)

    count = len(present_factors)
    pearson = np.full(count, np.nan, dtype=float)
    rank_ic = np.full(count, np.nan, dtype=float)
    valid_for_ic = np.zeros(count, dtype=bool)
    status = np.full(count, "ok", dtype=object)
    sufficient = finite_rows >= int(min_cross_section)
    status[~sufficient] = "insufficient_finite_rows"
    low = np.min(np.where(finite, matrix, np.inf), axis=0)
    high = np.max(np.where(finite, matrix, -np.inf), axis=0)
    factor_nonconstant = low != high
    constant_factor = sufficient & ~factor_nonconstant
    status[constant_factor] = "constant_factor"

    # Cache by the exact finite observation mask.  Target ranks cannot be
    # shared across different masks: each factor's RankIC must exclude only
    # that factor's non-finite observations, exactly like ``spearmanr``.
    group_members: dict[bytes, list[int]] = {}
    for index in np.flatnonzero(sufficient & factor_nonconstant):
        key = np.packbits(finite[:, index]).tobytes()
        group_members.setdefault(key, []).append(int(index))
    for indexes in group_members.values():
        mask = finite[:, indexes[0]]
        group_values = matrix[mask][:, indexes]
        group_target = target[mask]
        if np.min(group_target) == np.max(group_target):
            status[indexes] = "constant_label"
            continue
        group_pearson = _columnwise_pearson(group_values, group_target)
        ranked_values = rankdata(group_values, axis=0)
        ranked_target = rankdata(group_target)
        group_rank_ic = _columnwise_pearson(ranked_values, ranked_target)
        for offset, index in enumerate(indexes):
            pearson_value = _finite_or_none(group_pearson[offset])
            rank_value = _finite_or_none(group_rank_ic[offset])
            if pearson_value is None or rank_value is None:
                status[index] = "invalid_correlation"
                continue
            pearson[index] = pearson_value
            rank_ic[index] = rank_value
            valid_for_ic[index] = True

    factor_index = {factor: index for index, factor in enumerate(present_factors)}
    rows: list[dict[str, object]] = []
    for family, factor in catalog_rows:
        row = base_row(family=family, factor=factor)
        index = factor_index.get(factor)
        if index is None:
            rows.append(row)
            continue
        finite_count = int(finite_rows[index])
        row.update(
            {
                "factor_label_rows": int(factor_label_rows[index]),
                "finite_rows": finite_count,
                "coverage": _finite_or_none(finite_count / len(label_pool))
                if len(label_pool)
                else None,
                "finite_rate": _finite_or_none(finite_count / pool_factor_rows)
                if pool_factor_rows
                else None,
                "n": finite_count,
                "pearson_ic": _finite_or_none(pearson[index]),
                "rank_ic": _finite_or_none(rank_ic[index]),
                "top20_mean_y": _finite_or_none(top_mean[index])
                if finite_count >= top_k
                else None,
                "valid_for_ic": bool(valid_for_ic[index]),
                "status": str(status[index]),
            }
        )
        rows.append(row)
    return rows


def _chronological_partitions(days: Sequence[str]) -> tuple[dict[str, str], dict[str, int]]:
    ordered = sorted({str(day) for day in days})
    if len(ordered) < 5:
        raise RuntimeError(
            "need at least five globally label/pool-aligned FactorStore days for a 60/20/20 split; "
            f"got {len(ordered)}"
        )
    discovery_count = int(math.floor(len(ordered) * 0.60))
    validation_count = int(math.floor(len(ordered) * 0.20))
    holdout_count = len(ordered) - discovery_count - validation_count
    if min(discovery_count, validation_count, holdout_count) <= 0:
        raise RuntimeError(f"cannot form non-empty 60/20/20 chronology from {len(ordered)} days")
    assignment: dict[str, str] = {}
    for day in ordered[:discovery_count]:
        assignment[day] = "discovery"
    for day in ordered[discovery_count : discovery_count + validation_count]:
        assignment[day] = "validation"
    for day in ordered[discovery_count + validation_count :]:
        assignment[day] = "holdout"
    return assignment, {
        "total": len(ordered),
        "discovery": discovery_count,
        "validation": validation_count,
        "holdout": holdout_count,
    }


def _mean_and_t(values: pd.Series) -> tuple[float | None, float | None]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return None, None
    mean = _finite_or_none(finite.mean())
    if mean is None or len(finite) < 2:
        return mean, None
    std = float(finite.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return mean, None
    return mean, _finite_or_none(float(mean / std * math.sqrt(len(finite))))


def _summarize_daily_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for partition in ("overall", "discovery", "validation", "holdout"):
        scoped = daily if partition == "overall" else daily.loc[daily["partition"] == partition]
        for (family, factor), group in scoped.groupby(["family", "factor"], sort=True):
            pearson_mean, pearson_t = _mean_and_t(group["pearson_ic"])
            rank_mean, rank_t = _mean_and_t(group["rank_ic"])
            top20_mean, top20_t = _mean_and_t(group["top20_mean_y"])
            rows.append(
                {
                    "factor": factor,
                    "family": family,
                    "partition": partition,
                    "calendar_days": int(len(group)),
                    "valid_days": int(group["valid_for_ic"].astype(bool).sum()),
                    "top20_days": int(pd.to_numeric(group["top20_mean_y"], errors="coerce").notna().sum()),
                    "first_score_day": str(group["score_day"].min()),
                    "last_score_day": str(group["score_day"].max()),
                    "mean_n": _finite_or_none(pd.to_numeric(group["n"], errors="coerce").mean()),
                    "mean_coverage": _finite_or_none(
                        pd.to_numeric(group["coverage"], errors="coerce").mean()
                    ),
                    "mean_finite_rate": _finite_or_none(
                        pd.to_numeric(group["finite_rate"], errors="coerce").mean()
                    ),
                    "mean_pearson_ic": pearson_mean,
                    "pearson_ic_t": pearson_t,
                    "mean_rank_ic": rank_mean,
                    "rank_ic_t": rank_t,
                    "mean_top20_y": top20_mean,
                    "top20_y_t": top20_t,
                }
            )
    return pd.DataFrame(rows).sort_values(["factor", "partition"], kind="mergesort").reset_index(drop=True)


def _build_screen_table(
    *,
    catalog: pd.DataFrame,
    summary: pd.DataFrame,
    ic_threshold: float,
    min_valid_days: int,
    min_valid_days_per_partition: int,
) -> pd.DataFrame:
    result = catalog.copy()
    metrics = (
        "calendar_days",
        "valid_days",
        "top20_days",
        "mean_n",
        "mean_coverage",
        "mean_finite_rate",
        "mean_pearson_ic",
        "pearson_ic_t",
        "mean_rank_ic",
        "rank_ic_t",
        "mean_top20_y",
        "top20_y_t",
    )
    for partition in ("overall", "discovery", "validation", "holdout"):
        part = summary.loc[summary["partition"] == partition, ["factor", *metrics]].copy()
        part = part.rename(columns={metric: f"{partition}_{metric}" for metric in metrics})
        result = result.merge(part, on="factor", how="left", validate="one_to_one")
    overall_ic = pd.to_numeric(result["overall_mean_pearson_ic"], errors="coerce")
    result["abs_overall_mean_pearson_ic"] = overall_ic.abs()
    result["quality_gate"] = "eligible_for_redundancy"
    result["eligible_for_redundancy"] = True
    for index, row in result.iterrows():
        reason: str | None = None
        ic = _finite_or_none(row["overall_mean_pearson_ic"])
        if ic is None:
            reason = "no_valid_overall_pearson_ic"
        elif abs(ic) <= ic_threshold:
            reason = f"abs_mean_pearson_ic_not_above_{ic_threshold:.6f}"
        elif int(row["overall_valid_days"] or 0) < min_valid_days:
            reason = f"valid_days_below_{min_valid_days}"
        else:
            for partition in ("discovery", "validation", "holdout"):
                valid_days = int(row[f"{partition}_valid_days"] or 0)
                if valid_days < min_valid_days_per_partition:
                    reason = f"{partition}_valid_days_below_{min_valid_days_per_partition}"
                    break
        if reason is not None:
            result.at[index, "quality_gate"] = reason
            result.at[index, "eligible_for_redundancy"] = False
    result = result.sort_values(
        ["eligible_for_redundancy", "abs_overall_mean_pearson_ic", "factor"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return result


def _pairwise_redundancy(
    *,
    candidates: pd.DataFrame,
    factor_paths: Mapping[str, Path],
    evaluation_codes_by_day: Mapping[str, set[str]],
    source_evidence_by_day: Mapping[str, Mapping[str, object]],
    min_cross_section: int,
    within_family_threshold: float,
    cross_family_threshold: float,
) -> pd.DataFrame:
    names = candidates["factor"].astype(str).tolist()
    families = candidates.set_index("factor")["family"].astype(str).to_dict()
    columns = [
        "factor_a",
        "family_a",
        "factor_b",
        "family_b",
        "relation",
        "threshold",
        "common_valid_days",
        "mean_common_observations",
        "mean_abs_pearson",
        "mean_abs_spearman",
        "redundancy",
    ]
    if len(names) < 2:
        return pd.DataFrame(columns=columns)
    count = len(names)
    pearson_sum = np.zeros((count, count), dtype=float)
    spearman_sum = np.zeros((count, count), dtype=float)
    common_days = np.zeros((count, count), dtype=np.int64)
    common_n_sum = np.zeros((count, count), dtype=float)
    for day in sorted(evaluation_codes_by_day):
        frame, _ = _read_factor_frame(
            path=factor_paths[day],
            day=day,
            expected_evidence=source_evidence_by_day[day],
        )
        present = [name for name in names if name in frame.columns]
        numeric = frame.loc[frame["code"].isin(evaluation_codes_by_day[day]), ["code", *present]].copy()
        numeric = numeric.set_index("code").reindex(sorted(evaluation_codes_by_day[day]))
        numeric = numeric.reindex(columns=names)
        numeric = numeric.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = numeric.notna().to_numpy(dtype=np.int64)
        shared_n = finite.T @ finite
        pearson = numeric.corr(method="pearson", min_periods=min_cross_section).to_numpy(dtype=float)
        spearman = numeric.corr(method="spearman", min_periods=min_cross_section).to_numpy(dtype=float)
        valid = (shared_n >= min_cross_section) & np.isfinite(pearson) & np.isfinite(spearman)
        np.fill_diagonal(valid, False)
        pearson_sum += np.where(valid, np.abs(pearson), 0.0)
        spearman_sum += np.where(valid, np.abs(spearman), 0.0)
        common_days += valid.astype(np.int64)
        common_n_sum += np.where(valid, shared_n, 0.0)
    rows: list[dict[str, object]] = []
    for left in range(count):
        for right in range(left + 1, count):
            days = int(common_days[left, right])
            mean_pearson = _finite_or_none(pearson_sum[left, right] / days) if days else None
            mean_spearman = _finite_or_none(spearman_sum[left, right] / days) if days else None
            redundancy = (
                _finite_or_none(max(float(mean_pearson), float(mean_spearman)))
                if mean_pearson is not None and mean_spearman is not None
                else None
            )
            relation = "within_family" if families[names[left]] == families[names[right]] else "cross_family"
            rows.append(
                {
                    "factor_a": names[left],
                    "family_a": families[names[left]],
                    "factor_b": names[right],
                    "family_b": families[names[right]],
                    "relation": relation,
                    "threshold": within_family_threshold
                    if relation == "within_family"
                    else cross_family_threshold,
                    "common_valid_days": days,
                    "mean_common_observations": _finite_or_none(common_n_sum[left, right] / days)
                    if days
                    else None,
                    "mean_abs_pearson": mean_pearson,
                    "mean_abs_spearman": mean_spearman,
                    "redundancy": redundancy,
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["factor_a", "factor_b"], kind="mergesort"
    ).reset_index(drop=True)


def _apply_redundancy_selection(
    *,
    screen: pd.DataFrame,
    pairs: pd.DataFrame,
    max_selected: int,
    min_redundancy_days: int,
) -> pd.DataFrame:
    result = screen.copy()
    result["priority_rank"] = np.nan
    result["selection_status"] = "rejected"
    result["selection_reason"] = result["quality_gate"].astype(str)
    eligible = result.loc[result["eligible_for_redundancy"].astype(bool)].copy()
    eligible = eligible.sort_values(
        ["abs_overall_mean_pearson_ic", "factor"], ascending=[False, True], kind="mergesort"
    )
    pair_lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for _, row in pairs.iterrows():
        pair_lookup[tuple(sorted((str(row["factor_a"]), str(row["factor_b"]))))] = row
    selected: list[str] = []
    for rank, (index, row) in enumerate(eligible.iterrows(), start=1):
        result.at[index, "priority_rank"] = rank
        factor = str(row["factor"])
        if len(selected) >= max_selected:
            result.at[index, "selection_reason"] = f"selected_limit_reached_{max_selected}"
            continue
        conflicts: list[str] = []
        for previous in selected:
            pair = pair_lookup.get(tuple(sorted((factor, previous))))
            if pair is None or int(pair["common_valid_days"]) < min_redundancy_days:
                observed = 0 if pair is None else int(pair["common_valid_days"])
                conflicts.append(
                    f"redundancy_unavailable_with={previous};common_days={observed};required={min_redundancy_days}"
                )
                continue
            threshold = float(pair["threshold"])
            value = _finite_or_none(pair["redundancy"])
            if value is None:
                conflicts.append(f"redundancy_unavailable_with={previous};common_days={int(pair['common_valid_days'])}")
            elif value >= threshold:
                conflicts.append(
                    f"redundancy_{value:.6f}_ge_{threshold:.6f}_with={previous};{pair['relation']}"
                )
        if conflicts:
            result.at[index, "selection_reason"] = " | ".join(conflicts)
            continue
        selected.append(factor)
        result.at[index, "selection_status"] = "selected"
        result.at[index, "selection_reason"] = "selected_abs_ic_priority_and_redundancy_pass"
    return result.sort_values(
        ["selection_status", "priority_rank", "factor"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite_or_none(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    factor_root = _resolved(args.factor_root)
    label_root = _resolved(args.label_root)
    raw_data_root = _resolved(args.raw_data_root)
    output_dir = _resolved(args.output_dir)
    catalog_path = _resolved(args.factor_catalog)
    if str(args.panel_name) != "T1430":
        raise ValueError("factor mining screen only accepts --panel-name T1430")
    if str(args.start) != _BASE_START:
        raise ValueError(
            f"IC contract is unified from {_BASE_START}; refusing --start {args.start!r}"
        )
    start = _parse_day(str(args.start), context="start")
    end = _parse_day(str(args.end), context="end")
    if start > end:
        raise ValueError("start must be <= end")
    if int(args.top_k) != 20:
        raise ValueError("this contract requires --top-k 20")
    if int(args.min_cross_section) < 2:
        raise ValueError("--min-cross-section must be at least 2")
    if int(args.min_valid_days) != _MIN_VALID_DAYS:
        raise ValueError(f"this contract requires --min-valid-days {_MIN_VALID_DAYS}")
    if int(args.min_valid_days_per_partition) != _MIN_VALID_DAYS_PER_PARTITION:
        raise ValueError(
            "this contract requires --min-valid-days-per-partition "
            f"{_MIN_VALID_DAYS_PER_PARTITION}"
        )
    if int(args.min_redundancy_days) != _MIN_REDUNDANCY_DAYS:
        raise ValueError(
            f"this contract requires --min-redundancy-days {_MIN_REDUNDANCY_DAYS}"
        )
    if int(args.max_selected) != _MAX_SELECTED:
        raise ValueError(f"this contract requires --max-selected {_MAX_SELECTED}")
    if not math.isclose(float(args.ic_threshold), _IC_THRESHOLD, abs_tol=1e-12):
        raise ValueError(f"this contract requires --ic-threshold {_IC_THRESHOLD}")
    if not math.isclose(float(args.within_family_threshold), _WITHIN_FAMILY_THRESHOLD, abs_tol=1e-12):
        raise ValueError(
            f"this contract requires --within-family-threshold {_WITHIN_FAMILY_THRESHOLD}"
        )
    if not math.isclose(float(args.cross_family_threshold), _CROSS_FAMILY_THRESHOLD, abs_tol=1e-12):
        raise ValueError(
            f"this contract requires --cross-family-threshold {_CROSS_FAMILY_THRESHOLD}"
        )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing --output-dir: {output_dir}")
    for root, name in (
        (factor_root, "factor-root"),
        (label_root, "label-root"),
        (raw_data_root, "raw-data-root"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"--{name} does not exist or is not a directory: {root}")
    _assert_research_only_paths(
        factor_root=factor_root,
        label_root=label_root,
        raw_data_root=raw_data_root,
        output_dir=output_dir,
    )
    return factor_root, label_root, raw_data_root, output_dir, catalog_path


def run_screen(args: argparse.Namespace) -> Path:
    """Run an isolated factor screen and return the newly created output root."""
    factor_root, label_root, raw_data_root, output_dir, catalog_path = _validate_args(args)
    catalog = _load_family_catalog(catalog_path)
    factor_paths = _factor_day_paths(
        factor_root=factor_root,
        panel_name=str(args.panel_name),
        start=str(args.start),
        end=str(args.end),
    )
    expected_score_days = _expected_score_days(
        raw_data_root=raw_data_root,
        start=str(args.start),
        end=str(args.end),
    )
    observed_score_days = sorted(factor_paths)
    if observed_score_days != expected_score_days:
        expected_set = set(expected_score_days)
        observed_set = set(observed_score_days)
        missing = sorted(expected_set.difference(observed_set))
        unexpected = sorted(observed_set.difference(expected_set))
        raise RuntimeError(
            "FactorStore score-day coverage does not match the frozen calendar: "
            f"expected={len(expected_score_days)} observed={len(observed_score_days)} "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    # Resolve every historical T-1 pool before _read_label_1442 is reachable.
    # This protects the fixed-universe contract even if one late day is missing.
    pool_codes_by_day, pool_audit, pool_manifest = _fixed_pool_codes_by_day(
        raw_data_root=raw_data_root,
        score_days=expected_score_days,
    )
    if set(pool_codes_by_day) != set(expected_score_days):
        raise RuntimeError("fixed pool resolver did not return every FactorStore score day")

    input_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    evaluation_codes_by_day: dict[str, set[str]] = {}
    factor_evidence_by_day: dict[str, Mapping[str, object]] = {}
    for day in sorted(factor_paths):
        factor_path = factor_paths[day]
        label, label_evidence, label_status = _read_label_1442(label_root=label_root, day=day)
        row: dict[str, object] = {
            "score_day": day,
            "factor_file_path": str(factor_path),
            "factor_file_bytes": None,
            "factor_file_sha256": None,
            "label_file_path": str(label_root / day[:7] / f"{day.replace('-', '')}.parquet"),
            "label_file_bytes": None if label_evidence is None else label_evidence["bytes"],
            "label_file_sha256": None if label_evidence is None else label_evidence["sha256"],
            "pool_codes": int(len(pool_codes_by_day[day])),
            "label_1442_rows": 0 if label is None else int(len(label)),
            "pool_label_rows": 0,
            "status": label_status,
        }
        if label is None:
            raise RuntimeError(
                "strict same-score-day 14:42 label unavailable: "
                f"score_day={day} status={label_status} path={row['label_file_path']}"
            )
        label_pool = label.loc[label["code"].isin(pool_codes_by_day[day]), ["code", "y"]].copy()
        row["pool_label_rows"] = int(len(label_pool))
        if len(label_pool) < int(args.min_cross_section):
            raise RuntimeError(
                "strict same-score-day pool/label coverage insufficient: "
                f"score_day={day} pool_label_rows={len(label_pool)} "
                f"min_cross_section={int(args.min_cross_section)}"
            )
        factor_frame, evidence = _read_factor_frame(path=factor_path, day=day)
        factor_evidence_by_day[day] = evidence
        row.update(
            {
                "factor_file_bytes": evidence["bytes"],
                "factor_file_sha256": evidence["sha256"],
                "status": "ok",
            }
        )
        input_rows.append(row)
        evaluation_codes_by_day[day] = set(label_pool["code"].tolist())
        daily_rows.extend(
            _daily_factor_metrics(
                day=day,
                catalog=catalog,
                factor_frame=factor_frame,
                label_pool=label_pool,
                pool_codes=pool_codes_by_day[day],
                min_cross_section=int(args.min_cross_section),
                top_k=int(args.top_k),
            )
        )
    if set(evaluation_codes_by_day) != set(expected_score_days):
        missing = sorted(set(expected_score_days).difference(evaluation_codes_by_day))
        raise RuntimeError(
            "evaluation calendar lost a required score day after strict input checks: "
            f"missing={missing[:10]}"
        )
    partition_by_day, partition_counts = _chronological_partitions(sorted(evaluation_codes_by_day))
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("factor catalog produced no daily metrics")
    daily["partition"] = daily["score_day"].map(partition_by_day)
    if daily["partition"].isna().any():
        raise RuntimeError("daily metrics contain a day outside the fixed chronological calendar")
    daily = daily.sort_values(["factor", "score_day"], kind="mergesort").reset_index(drop=True)
    summary = _summarize_daily_metrics(daily)
    screen = _build_screen_table(
        catalog=catalog,
        summary=summary,
        ic_threshold=float(args.ic_threshold),
        min_valid_days=int(args.min_valid_days),
        min_valid_days_per_partition=int(args.min_valid_days_per_partition),
    )
    redundancy_candidates = screen.loc[screen["eligible_for_redundancy"].astype(bool), ["factor", "family"]]
    redundancy = _pairwise_redundancy(
        candidates=redundancy_candidates,
        factor_paths=factor_paths,
        evaluation_codes_by_day=evaluation_codes_by_day,
        source_evidence_by_day=factor_evidence_by_day,
        min_cross_section=int(args.min_cross_section),
        within_family_threshold=float(args.within_family_threshold),
        cross_family_threshold=float(args.cross_family_threshold),
    )
    screen = _apply_redundancy_selection(
        screen=screen,
        pairs=redundancy,
        max_selected=int(args.max_selected),
        min_redundancy_days=int(args.min_redundancy_days),
    )
    accepted = screen.loc[screen["selection_status"] == "selected"].copy()
    rejected = screen.loc[screen["selection_status"] != "selected"].copy()

    output_dir.mkdir(parents=True, exist_ok=False)
    input_audit = pd.DataFrame(input_rows).sort_values("score_day", kind="mergesort")
    calendar = pd.DataFrame(
        [
            {
                "score_day": day,
                "partition": partition_by_day[day],
                "pool_codes": int(len(pool_codes_by_day[day])),
                "pool_label_codes": int(len(evaluation_codes_by_day[day])),
            }
            for day in sorted(evaluation_codes_by_day)
        ]
    )
    catalog.to_csv(output_dir / "factor_family_catalog.csv", index=False, encoding="utf-8")
    pool_audit.to_csv(output_dir / "fixed_pool_audit.csv", index=False, encoding="utf-8")
    input_audit.to_csv(output_dir / "input_day_audit.csv", index=False, encoding="utf-8")
    calendar.to_csv(output_dir / "evaluation_calendar.csv", index=False, encoding="utf-8")
    daily.to_csv(output_dir / "daily_factor_metrics.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "factor_summary_metrics.csv", index=False, encoding="utf-8")
    redundancy.to_csv(output_dir / "factor_pair_redundancy.csv", index=False, encoding="utf-8")
    screen.to_csv(output_dir / "factor_screen.csv", index=False, encoding="utf-8")
    accepted.to_csv(output_dir / "accepted_factors.csv", index=False, encoding="utf-8")
    rejected.to_csv(output_dir / "rejected_factors.csv", index=False, encoding="utf-8")

    summary_json = {
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "eligible_for_redundancy_count": int(len(redundancy_candidates)),
        "factor_catalog_count": int(len(catalog)),
        "evaluation_days": int(len(evaluation_codes_by_day)),
        "partitions": partition_counts,
        "accepted_by_family": {
            str(family): int(count)
            for family, count in accepted.groupby("family", sort=True).size().items()
        },
        "rejected_by_reason": {
            str(reason): int(count)
            for reason, count in rejected.groupby("selection_reason", sort=True).size().items()
        },
    }
    (output_dir / "screen_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "research-only scratch FactorStore screen; no live admission or model change",
        "side_effect_boundary": {
            "database_writes": "none",
            "production_output_writes": "none",
            "configuration_writes": "none",
            "factor_code_label_or_mask_access": "none; labels and masks are opened only by this harness tool",
            "output_dir_only": str(output_dir),
        },
        "factor_contract": {
            "factor_root": str(factor_root),
            "panel_name": "T1430",
            "factor_time": "14:30 via T1430 FactorStore contract",
            "label_root": str(label_root),
            "label_time": "same-score-day 14:42",
            "start": str(args.start),
            "end": str(args.end),
            "family_catalog": str(catalog_path),
        },
        "fixed_universe": pool_manifest,
        "metric_contract": {
            "primary_metric": "raw mean daily cross-sectional Pearson IC",
            "secondary_metrics": ["raw mean daily RankIC", "raw Top20 mean y", "coverage", "finite rate"],
            "daily_min_cross_section": int(args.min_cross_section),
            "top_k": int(args.top_k),
            "chronology": "earliest 60% discovery, next 20% validation, final 20% holdout",
            "partition_counts": partition_counts,
            "ic_threshold_strictly_greater_than": float(args.ic_threshold),
        },
        "redundancy_contract": {
            "method": "max(mean daily abs Pearson, mean daily abs Spearman) on common finite factor observations",
            "within_family_threshold_strictly_less_than": float(args.within_family_threshold),
            "cross_family_threshold_strictly_less_than": float(args.cross_family_threshold),
            "minimum_common_valid_days": int(args.min_redundancy_days),
            "selection_order": "descending abs overall mean Pearson IC, then factor name",
            "max_selected": int(args.max_selected),
        },
        "input_integrity": {
            "hash_algorithm": "sha256",
            "factor_and_label_day_audit": "input_day_audit.csv",
            "pool_audit": "fixed_pool_audit.csv",
        },
        "outputs": [
            "factor_family_catalog.csv",
            "fixed_pool_audit.csv",
            "input_day_audit.csv",
            "evaluation_calendar.csv",
            "daily_factor_metrics.csv",
            "factor_summary_metrics.csv",
            "factor_pair_redundancy.csv",
            "factor_screen.csv",
            "accepted_factors.csv",
            "rejected_factors.csv",
            "screen_summary.json",
        ],
        "summary": summary_json,
    }
    (output_dir / "screen_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-root", required=True, help="scratch FactorStore root (contains factors/T1430)")
    parser.add_argument("--label-root", required=True, help="read-only same-day 14:42 label root")
    parser.add_argument("--raw-data-root", required=True, help="read-only DataHub raw root used for o_0005")
    parser.add_argument("--factor-catalog", "--family-map", dest="factor_catalog", required=True)
    parser.add_argument("--output-dir", required=True, help="new explicit research scratch output directory")
    parser.add_argument("--panel-name", required=True, choices=["T1430"])
    parser.add_argument("--start", required=True, help=f"must be {_BASE_START}")
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--min-valid-days", type=int, default=_MIN_VALID_DAYS)
    parser.add_argument(
        "--min-valid-days-per-partition",
        type=int,
        default=_MIN_VALID_DAYS_PER_PARTITION,
    )
    parser.add_argument("--min-redundancy-days", type=int, default=_MIN_REDUNDANCY_DAYS)
    parser.add_argument("--ic-threshold", type=float, default=0.02)
    parser.add_argument("--within-family-threshold", type=float, default=0.80)
    parser.add_argument("--cross-family-threshold", type=float, default=0.70)
    parser.add_argument("--max-selected", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output = run_screen(build_parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
