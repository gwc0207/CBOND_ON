from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class TailFeatureSpec:
    source: str
    kind: str
    output_col: str
    upper_q: float = 0.9
    lower_q: float = 0.1


def tail_features_enabled(cfg: dict[str, Any] | None) -> bool:
    return isinstance(cfg, dict) and bool(cfg.get("enabled", False))


def _normalize_source_columns(raw: Any, fallback: Sequence[str]) -> list[str]:
    if raw is None:
        return [str(c) for c in fallback]
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"*", "all"}:
            return [str(c) for c in fallback]
        return [raw.strip()] if raw.strip() else []
    if not isinstance(raw, (list, tuple)):
        raise TypeError("tail_features.source_factors must be a list, '*', or 'all'")
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        col = str(item).strip()
        if not col or col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def _normalize_feature_defs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        raw = ["rank_pct", "upper_tail_flag", "lower_tail_flag", "tail_strength"]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise TypeError("tail_features.features must be a list")

    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            kind = item.strip()
            payload: dict[str, Any] = {"kind": kind}
        elif isinstance(item, dict):
            payload = dict(item)
            kind = str(payload.get("kind", "")).strip()
        else:
            raise TypeError("tail_features.features entries must be strings or objects")
        if not kind:
            raise ValueError("tail_features.features entry has empty kind")
        payload["kind"] = kind
        out.append(payload)
    return out


def _feature_suffix(kind: str, item: dict[str, Any]) -> str:
    raw = item.get("suffix")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    if kind == "rank_pct":
        return "__rank_pct"
    if kind == "upper_tail_flag":
        q = int(round(float(item.get("upper_q", item.get("quantile", 0.9))) * 100))
        return f"__top{100 - q}"
    if kind == "lower_tail_flag":
        q = int(round(float(item.get("lower_q", item.get("quantile", 0.1))) * 100))
        return f"__bottom{q}"
    if kind == "tail_strength":
        upper_q = float(item.get("upper_q", 0.9))
        lower_q = float(item.get("lower_q", 0.1))
        if abs(upper_q - 0.9) < 1e-12 and abs(lower_q - 0.1) < 1e-12:
            return "__tail10_strength"
        return "__tail_strength"
    return f"__{kind}"


def build_tail_feature_specs(
    cfg: dict[str, Any] | None,
    *,
    source_columns: Sequence[str],
) -> list[TailFeatureSpec]:
    if not tail_features_enabled(cfg):
        return []
    cfg = dict(cfg or {})
    sources = _normalize_source_columns(cfg.get("source_factors"), source_columns)
    if not sources:
        raise ValueError("tail_features enabled but source_factors is empty")
    features = _normalize_feature_defs(cfg.get("features"))
    specs: list[TailFeatureSpec] = []
    seen: set[str] = set()
    for source in sources:
        for item in features:
            kind = str(item["kind"]).strip()
            if kind not in {"rank_pct", "upper_tail_flag", "lower_tail_flag", "tail_strength"}:
                raise ValueError(f"unsupported tail feature kind: {kind}")
            upper_q = float(item.get("upper_q", item.get("quantile", cfg.get("upper_q", 0.9))))
            lower_q = float(item.get("lower_q", item.get("quantile", cfg.get("lower_q", 0.1))))
            if not 0.0 < lower_q < upper_q < 1.0:
                raise ValueError(
                    f"tail feature quantiles must satisfy 0 < lower_q < upper_q < 1, "
                    f"got lower_q={lower_q}, upper_q={upper_q}"
                )
            output_col = str(item.get("output_col", "")).strip()
            if not output_col:
                output_col = f"{source}{_feature_suffix(kind, item)}"
            if output_col in seen:
                raise ValueError(f"duplicate tail feature output column: {output_col}")
            seen.add(output_col)
            specs.append(
                TailFeatureSpec(
                    source=source,
                    kind=kind,
                    output_col=output_col,
                    upper_q=upper_q,
                    lower_q=lower_q,
                )
            )
    return specs


def tail_feature_output_columns(
    cfg: dict[str, Any] | None,
    *,
    source_columns: Sequence[str],
) -> list[str]:
    return [spec.output_col for spec in build_tail_feature_specs(cfg, source_columns=source_columns)]


def _group_key(frame: pd.DataFrame, groupby: str) -> pd.Series:
    if groupby in frame.columns:
        return pd.to_datetime(frame[groupby], errors="coerce")
    if isinstance(frame.index, pd.MultiIndex):
        names = list(frame.index.names)
        if groupby in names:
            return pd.Series(frame.index.get_level_values(groupby), index=frame.index)
        if "dt" in names:
            return pd.Series(frame.index.get_level_values("dt"), index=frame.index)
    if frame.index.name == groupby:
        return pd.Series(frame.index, index=frame.index)
    raise ValueError(f"tail_features groupby key not found: {groupby}")


def _rank_pct_by_group(values: pd.Series, group_key: pd.Series, min_count: int) -> pd.Series:
    valid = values.notna() & group_key.notna()
    out = pd.Series(pd.NA, index=values.index, dtype="Float64")
    if not bool(valid.any()):
        return out.astype(float)
    counts = valid.groupby(group_key).transform("sum")
    eligible = valid & (counts >= min_count)
    if not bool(eligible.any()):
        return out.astype(float)
    ranks = values[eligible].groupby(group_key[eligible]).rank(pct=True, method="average")
    out.loc[eligible] = ranks.astype(float)
    return out.astype(float)


def compute_tail_features(
    frame: pd.DataFrame,
    cfg: dict[str, Any] | None,
    *,
    source_columns: Sequence[str],
) -> pd.DataFrame:
    specs = build_tail_feature_specs(cfg, source_columns=source_columns)
    if not specs:
        return pd.DataFrame(index=frame.index)
    cfg = dict(cfg or {})
    groupby = str(cfg.get("groupby", "dt")).strip() or "dt"
    min_count = max(1, int(cfg.get("min_count", 1)))
    missing_policy = str(cfg.get("missing_policy", "raise")).strip().lower()
    group_key = _group_key(frame, groupby)

    rank_cache: dict[str, pd.Series] = {}
    out: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.source not in frame.columns:
            if missing_policy in {"nan", "keep_nan"}:
                out[spec.output_col] = pd.Series(pd.NA, index=frame.index, dtype="Float64").astype(float)
                continue
            raise KeyError(f"tail_features source column missing: {spec.source}")
        rank_pct = rank_cache.get(spec.source)
        if rank_pct is None:
            values = pd.to_numeric(frame[spec.source], errors="coerce")
            rank_pct = _rank_pct_by_group(values, group_key, min_count)
            rank_cache[spec.source] = rank_pct
        if spec.kind == "rank_pct":
            series = rank_pct
        elif spec.kind == "upper_tail_flag":
            series = (rank_pct >= spec.upper_q).astype(float)
            series[rank_pct.isna()] = pd.NA
        elif spec.kind == "lower_tail_flag":
            series = (rank_pct <= spec.lower_q).astype(float)
            series[rank_pct.isna()] = pd.NA
        else:
            upper_strength = (rank_pct - spec.upper_q) / max(1e-12, 1.0 - spec.upper_q)
            lower_strength = (spec.lower_q - rank_pct) / max(1e-12, spec.lower_q)
            series = pd.concat([upper_strength, lower_strength], axis=1).max(axis=1).clip(lower=0.0, upper=1.0)
            series[rank_pct.isna()] = pd.NA
        out[spec.output_col] = series.astype(float)
    return pd.DataFrame(out, index=frame.index)
