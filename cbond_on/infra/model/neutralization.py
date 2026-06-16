from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

import numpy as np
import pandas as pd


_FACTOR_SOURCES = {"", "factor", "factors", "self"}
_RAW_ALIASES: dict[str, tuple[str, str, str, str | None]] = {
    "market_cbond.daily_base": ("market_cbond.daily_base", "trade_date", "instrument_code", "exchange_code"),
    "market_cbond.daily_price": ("market_cbond.daily_price", "trade_date", "instrument_code", "exchange_code"),
    "market_cbond.daily_twap": ("market_cbond.daily_twap", "trade_date", "instrument_code", "exchange_code"),
    "market_cbond.daily_vwap": ("market_cbond.daily_vwap", "trade_date", "instrument_code", "exchange_code"),
    "market_cbond.daily_deriv": ("market_cbond.daily_deriv", "trade_date", "instrument_code", "exchange_code"),
    "market_cbond.daily_rating": ("market_cbond.daily_rating", "trade_date", "instrument_code", "exchange_code"),
    "cbond_daily_base": ("market_cbond.daily_base", "trade_date", "instrument_code", "exchange_code"),
    "cbond_daily_price": ("market_cbond.daily_price", "trade_date", "instrument_code", "exchange_code"),
    "cbond_daily_twap": ("market_cbond.daily_twap", "trade_date", "instrument_code", "exchange_code"),
    "cbond_daily_vwap": ("market_cbond.daily_vwap", "trade_date", "instrument_code", "exchange_code"),
    "cbond_daily_deriv": ("market_cbond.daily_deriv", "trade_date", "instrument_code", "exchange_code"),
    "cbond_daily_rating": ("market_cbond.daily_rating", "trade_date", "instrument_code", "exchange_code"),
}


@dataclass(frozen=True)
class NeutralizationExposure:
    name: str
    source: str
    column: str
    table: str | None = None
    date_col: str = "trade_date"
    code_col: str = "code"
    exchange_col: str | None = None
    transform: str = "none"
    encoding: str = "numeric"
    drop_first: bool = True
    dummy_na: bool = False


@dataclass(frozen=True)
class NeutralizationConfig:
    enabled: bool
    factors: tuple[str, ...] | None = None
    exclude_factors: tuple[str, ...] = ()
    exposures: tuple[NeutralizationExposure, ...] = ()
    method: str = "ridge"
    ridge_alpha: float = 1e-6
    add_intercept: bool = True
    min_count: int = 30
    standardize_exposures: bool = True
    missing_policy: str = "keep_original"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _source_key(source: str) -> str:
    return str(source or "").strip().lower()


def _resolve_raw_source(source: str, item: dict[str, Any]) -> tuple[str | None, str, str, str | None]:
    table = _norm_text(item.get("table"))
    date_col = _norm_text(item.get("date_col"))
    code_col = _norm_text(item.get("code_col"))
    exchange_col_raw = item.get("exchange_col", None)
    exchange_col = _norm_text(exchange_col_raw) if exchange_col_raw is not None else ""

    key = _source_key(source)
    if not table:
        default = _RAW_ALIASES.get(key)
        if default is not None:
            table, default_date_col, default_code_col, default_exchange_col = default
            date_col = date_col or default_date_col
            code_col = code_col or default_code_col
            if exchange_col_raw is None:
                exchange_col = default_exchange_col or ""
        elif "." in source:
            table = source
    if not table:
        return None, date_col or "trade_date", code_col or "code", exchange_col or None
    return table, date_col or "trade_date", code_col or "code", exchange_col or None


def _parse_exposure(raw: Any) -> NeutralizationExposure:
    if isinstance(raw, str):
        name = _norm_text(raw)
        if not name:
            raise ValueError("neutralization exposure string must not be empty")
        return NeutralizationExposure(name=name, source="factor", column=name)
    if not isinstance(raw, dict):
        raise TypeError(f"neutralization exposure must be string or object, got {type(raw).__name__}")

    column = _norm_text(raw.get("column", raw.get("col", "")))
    if not column:
        raise ValueError("neutralization exposure.column is required")
    name = _norm_text(raw.get("name")) or column
    source = _norm_text(raw.get("source", raw.get("from", "factor")))
    table: str | None = None
    date_col = "trade_date"
    code_col = "code"
    exchange_col: str | None = None
    if _source_key(source) not in _FACTOR_SOURCES:
        table, date_col, code_col, exchange_col = _resolve_raw_source(source, raw)
        if table is None:
            raise ValueError(
                f"unknown neutralization exposure source={source!r}; "
                "use source='factor', a known daily source, or set table/date_col/code_col"
            )

    encoding = _norm_text(raw.get("encoding", "numeric")).lower() or "numeric"
    transform = _norm_text(raw.get("transform", "none")).lower() or "none"
    if transform == "onehot":
        encoding = "onehot"
        transform = "none"
    if encoding not in {"numeric", "onehot"}:
        raise ValueError(f"neutralization exposure.encoding must be numeric or onehot, got {encoding!r}")
    return NeutralizationExposure(
        name=name,
        source=source,
        column=column,
        table=table,
        date_col=date_col,
        code_col=code_col,
        exchange_col=exchange_col,
        transform=transform,
        encoding=encoding,
        drop_first=bool(raw.get("drop_first", True)),
        dummy_na=bool(raw.get("dummy_na", False)),
    )


def parse_neutralization_config(raw: Any) -> NeutralizationConfig:
    if raw is None or raw is False:
        return NeutralizationConfig(enabled=False)
    if raw is True:
        raise ValueError("neutralization=true is ambiguous; provide exposures")
    if not isinstance(raw, dict):
        raise TypeError(f"neutralization config must be object, got {type(raw).__name__}")
    enabled = bool(raw.get("enabled", False))
    exposures = tuple(_parse_exposure(item) for item in _as_list(raw.get("exposures")))
    factors_raw = raw.get("factors", None)
    factors: tuple[str, ...] | None
    if factors_raw is None or str(factors_raw).strip().lower() in {"", "all", "*"}:
        factors = None
    else:
        factors = tuple(_norm_text(x) for x in _as_list(factors_raw) if _norm_text(x))
    method = _norm_text(raw.get("method", "ridge")).lower() or "ridge"
    if method not in {"ols", "ridge"}:
        raise ValueError(f"neutralization.method must be ols or ridge, got {method!r}")
    missing_policy = _norm_text(raw.get("missing_policy", "keep_original")).lower() or "keep_original"
    if missing_policy not in {"keep_original", "nan"}:
        raise ValueError("neutralization.missing_policy must be 'keep_original' or 'nan'")
    return NeutralizationConfig(
        enabled=enabled,
        factors=factors,
        exclude_factors=tuple(_norm_text(x) for x in _as_list(raw.get("exclude_factors")) if _norm_text(x)),
        exposures=exposures,
        method=method,
        ridge_alpha=float(raw.get("ridge_alpha", 1e-6) or 0.0),
        add_intercept=bool(raw.get("add_intercept", True)),
        min_count=max(2, int(raw.get("min_count", 30) or 30)),
        standardize_exposures=bool(raw.get("standardize_exposures", True)),
        missing_policy=missing_policy,
    )


def _table_day_path(raw_data_root: Path, table: str, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day:%Y%m%d}.parquet"
    return raw_data_root / table.replace(".", "__") / month / filename


def _normalize_exchange_value(exchange: Any) -> str:
    exch = str(exchange or "").strip().upper()
    aliases = {
        "XSHG": "SH",
        "SSE": "SH",
        "SHSE": "SH",
        "XSHE": "SZ",
        "SZSE": "SZ",
    }
    return aliases.get(exch, exch)


def _normalize_code_value(code: Any, exchange: Any = None) -> str:
    if isinstance(code, (int, np.integer)):
        text = str(int(code))
    elif isinstance(code, (float, np.floating)) and np.isfinite(code) and float(code).is_integer():
        text = str(int(code))
    else:
        text = str(code or "").strip().upper()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
    if not text:
        return ""
    if "." in text:
        return text
    exch = _normalize_exchange_value(exchange)
    if exch:
        return f"{text}.{exch}"
    return text


def _normalize_codes(code: pd.Series, exchange: pd.Series | None = None) -> pd.Series:
    if exchange is None:
        return code.map(_normalize_code_value)
    return pd.Series(
        (_normalize_code_value(c, e) for c, e in zip(code, exchange, strict=False)),
        index=code.index,
        dtype="object",
    )


def _as_day(value: Any) -> date | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _numeric_transform(series: pd.Series, transform: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    if transform in {"", "none", "identity"}:
        return s
    if transform == "log":
        return np.log(s.where(s > 0.0))
    if transform == "log1p":
        return np.log1p(s.where(s > -1.0))
    if transform == "sqrt":
        return np.sqrt(s.where(s >= 0.0))
    if transform == "abs":
        return s.abs()
    if transform == "rank_pct":
        return s.rank(pct=True, method="average")
    raise ValueError(f"unsupported neutralization exposure transform={transform!r}")


def _standardize_matrix_columns(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce").astype("float64")
        finite = np.isfinite(s.to_numpy(dtype=float))
        if finite.sum() <= 1:
            out[col] = np.nan
            continue
        mean = float(np.nanmean(s.to_numpy(dtype=float)))
        std = float(np.nanstd(s.to_numpy(dtype=float), ddof=0))
        if std > 0.0:
            out[col] = (s - mean) / std
        else:
            out[col] = s - mean
    return out


class FactorNeutralizer:
    def __init__(self, cfg: NeutralizationConfig, *, raw_data_root: str | Path | None = None):
        self.cfg = cfg
        self.raw_data_root = Path(raw_data_root).expanduser() if raw_data_root is not None else None
        self._raw_cache: dict[tuple[str, date], pd.DataFrame] = {}
        self._lock = RLock()

    @classmethod
    def from_config(
        cls,
        raw: Any,
        *,
        raw_data_root: str | Path | None = None,
    ) -> FactorNeutralizer | None:
        cfg = parse_neutralization_config(raw)
        if not cfg.enabled:
            return None
        if not cfg.exposures:
            raise ValueError("neutralization.enabled=true requires non-empty exposures")
        return cls(cfg, raw_data_root=raw_data_root)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.cfg.enabled),
            "method": self.cfg.method,
            "ridge_alpha": float(self.cfg.ridge_alpha),
            "min_count": int(self.cfg.min_count),
            "factors": "all" if self.cfg.factors is None else list(self.cfg.factors),
            "exclude_factors": list(self.cfg.exclude_factors),
            "exposures": [item.name for item in self.cfg.exposures],
        }

    def target_factors(self, factor_cols: Iterable[str]) -> list[str]:
        source = [str(c) for c in factor_cols]
        include = set(self.cfg.factors) if self.cfg.factors is not None else None
        exclude = set(self.cfg.exclude_factors)
        out = []
        for col in source:
            if include is not None and col not in include:
                continue
            if col in exclude:
                continue
            out.append(col)
        return out

    def _read_raw_source_day(self, table: str, day: date) -> pd.DataFrame:
        if self.raw_data_root is None:
            return pd.DataFrame()
        key = (table, day)
        with self._lock:
            cached = self._raw_cache.get(key)
        if cached is not None:
            return cached
        path = _table_day_path(self.raw_data_root, table, day)
        if not path.exists():
            out = pd.DataFrame()
        else:
            out = pd.read_parquet(path)
        with self._lock:
            self._raw_cache[key] = out
        return out

    def _raw_exposure(self, spec: NeutralizationExposure, day: date, codes: pd.Series) -> pd.Series:
        if not spec.table:
            return pd.Series(index=codes.index, dtype="object")
        raw = self._read_raw_source_day(spec.table, day)
        if raw.empty or spec.column not in raw.columns or spec.code_col not in raw.columns:
            return pd.Series(index=codes.index, dtype="object")
        cols = [spec.code_col, spec.column]
        if spec.exchange_col and spec.exchange_col in raw.columns:
            cols.append(spec.exchange_col)
        work = raw.loc[:, cols].copy()
        exchange = work[spec.exchange_col] if spec.exchange_col and spec.exchange_col in work.columns else None
        work["_code"] = _normalize_codes(work[spec.code_col], exchange)
        work = work.dropna(subset=["_code"]).drop_duplicates(subset=["_code"], keep="last")
        mapped = codes.astype(str).str.upper().map(work.set_index("_code")[spec.column])
        mapped.index = codes.index
        return mapped

    def _factor_exposure(self, spec: NeutralizationExposure, group: pd.DataFrame) -> pd.Series:
        if spec.column not in group.columns:
            return pd.Series(index=group.index, dtype="object")
        return group[spec.column]

    def _build_exposure_matrix(self, group: pd.DataFrame, day: date) -> pd.DataFrame:
        codes = _normalize_codes(group["code"])
        parts: list[pd.DataFrame] = []
        for spec in self.cfg.exposures:
            if _source_key(spec.source) in _FACTOR_SOURCES:
                raw = self._factor_exposure(spec, group)
            else:
                raw = self._raw_exposure(spec, day, codes)
            if spec.encoding == "onehot":
                cat = raw.astype("string")
                if spec.dummy_na:
                    cat = cat.fillna("__missing__")
                dummies = pd.get_dummies(cat, prefix=spec.name, dummy_na=False, dtype=float)
                if spec.drop_first and dummies.shape[1] > 1:
                    dummies = dummies.iloc[:, 1:]
                dummies.index = group.index
                parts.append(dummies)
            else:
                parts.append(pd.DataFrame({spec.name: _numeric_transform(raw, spec.transform)}, index=group.index))
        if not parts:
            return pd.DataFrame(index=group.index)
        x = pd.concat(parts, axis=1)
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.loc[:, ~x.columns.duplicated()]
        if self.cfg.standardize_exposures and not x.empty:
            x = _standardize_matrix_columns(x)
        return x

    def _residualize_one(self, y: pd.Series, x: pd.DataFrame) -> pd.Series:
        out = y.copy()
        x_np = x.to_numpy(dtype=float)
        y_np = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_np) & np.isfinite(x_np).all(axis=1)
        min_count = max(int(self.cfg.min_count), x_np.shape[1] + (2 if self.cfg.add_intercept else 1))
        if int(valid.sum()) < min_count or x_np.shape[1] == 0:
            if self.cfg.missing_policy == "nan":
                out.loc[:] = np.nan
            return out

        x_valid = x_np[valid]
        y_valid = y_np[valid]
        if self.cfg.add_intercept:
            x_design = np.column_stack([np.ones(x_valid.shape[0], dtype=float), x_valid])
            penalty = np.ones(x_design.shape[1], dtype=float)
            penalty[0] = 0.0
        else:
            x_design = x_valid
            penalty = np.ones(x_design.shape[1], dtype=float)

        try:
            if self.cfg.method == "ridge" and self.cfg.ridge_alpha > 0.0:
                xtx = x_design.T @ x_design
                xtx = xtx + np.diag(penalty * float(self.cfg.ridge_alpha))
                beta = np.linalg.solve(xtx, x_design.T @ y_valid)
            else:
                beta, *_ = np.linalg.lstsq(x_design, y_valid, rcond=None)
            fitted = x_design @ beta
            resid = y_valid - fitted
        except np.linalg.LinAlgError:
            if self.cfg.missing_policy == "nan":
                out.loc[:] = np.nan
            return out

        out_np = out.to_numpy(dtype=float, copy=True)
        if self.cfg.missing_policy == "nan":
            out_np[~valid] = np.nan
        out_np[valid] = resid
        return pd.Series(out_np, index=out.index, dtype="float64")

    def _apply_group(self, group: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
        if group.empty:
            return group
        group_dt = group["dt"].iloc[0] if "dt" in group.columns else group.name
        day = _as_day(group_dt)
        if day is None:
            return group
        x = self._build_exposure_matrix(group, day)
        x = x.dropna(axis=1, how="all")
        if x.empty:
            return group
        out = group.copy()
        if "dt" not in out.columns:
            out["dt"] = group_dt
        for col in factor_cols:
            if col not in out.columns:
                continue
            out[col] = self._residualize_one(out[col], x)
        return out

    def apply(self, df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
        if not self.enabled or df.empty:
            return df
        target_cols = self.target_factors(factor_cols)
        if not target_cols or "dt" not in df.columns or "code" not in df.columns:
            return df
        return df.groupby("dt", group_keys=False).apply(
            lambda group: self._apply_group(group, target_cols),
            include_groups=False,
        )


def build_neutralizer(
    raw: Any,
    *,
    raw_data_root: str | Path | None = None,
) -> FactorNeutralizer | None:
    return FactorNeutralizer.from_config(raw, raw_data_root=raw_data_root)
