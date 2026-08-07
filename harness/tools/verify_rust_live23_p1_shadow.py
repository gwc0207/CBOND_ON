"""Read-only exact-parity verifier for the seven daily live23 P1 factors.

The command always loads a separately unpacked scratch wheel.  It never
imports the repository's active ``cbond_on_rust`` binary, writes a FactorStore,
or changes the live hybrid route.  The live50 FactorStore is treated as the
Python-side golden output because these seven columns remain in
``python_columns`` in the active configuration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Support direct `py harness/tools/...py` invocation without relying on an
# environment-specific PYTHONPATH.  The scratch extension is inserted ahead of
# this root later and is asserted separately before use.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors.rust_legacy_reference import (
    build_factor_frame_rust_legacy_reference,
)


_COLUMNS = (
    "base_debt_premium_floor_gap",
    "dredemption_bondpremium_interaction",
    "dredemption_premium_z20",
    "dret_volatility_20",
    "dliq_volume_return_corr20",
    "dtwap_morning_slope20",
    "drt_rebound_from_low20",
)
_SPECS = (
    FactorSpec(
        name="base_debt_premium_floor_gap",
        factor="factor_mining_daily_catalog_v1",
        params={"signal": "base_debt_premium_floor_gap"},
    ),
    FactorSpec(
        name="dredemption_bondpremium_interaction",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dredemption_bondpremium_interaction"},
    ),
    FactorSpec(
        name="dredemption_premium_z20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dredemption_premium_z20"},
    ),
    FactorSpec(
        name="dret_volatility_20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dret_volatility_20"},
    ),
    FactorSpec(
        name="dliq_volume_return_corr20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dliq_volume_return_corr20"},
    ),
    FactorSpec(
        name="dtwap_morning_slope20",
        factor="factor_mining_daily_expansion_v1",
        params={"signal": "dtwap_morning_slope20"},
    ),
    FactorSpec(
        name="drt_rebound_from_low20",
        factor="factor_mining_daily_incremental_v1",
        params={"signal": "drt_rebound_from_low20"},
    ),
)
_SOURCE_STEMS = {
    "market_cbond.daily_base": "market_cbond__daily_base",
    "market_cbond.daily_price": "market_cbond__daily_price",
    "market_cbond.daily_twap": "market_cbond__daily_twap",
}


@dataclass(frozen=True)
class ColumnResult:
    name: str
    nan_mask_mismatch: int
    finite_exact_mismatch: int
    finite_pairs: int
    max_abs_difference: float


@dataclass(frozen=True)
class DayResult:
    score_day: str
    rows: int
    keys_equal: bool
    columns_equal: bool
    source_rows: dict[str, int]
    columns: tuple[ColumnResult, ...]

    @property
    def passed(self) -> bool:
        return self.keys_equal and self.columns_equal and all(
            result.nan_mask_mismatch == 0 and result.finite_exact_mismatch == 0
            for result in self.columns
        )


def _parse_day(value: str) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day):
        raise argparse.ArgumentTypeError(f"invalid date: {value!r}")
    return day.normalize()


def _iter_score_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    for day in pd.bdate_range(start=start, end=end):
        yield pd.Timestamp(day).normalize()


def _load_scratch_module(site: Path) -> object:
    site = site.resolve()
    package = site / "cbond_on_rust"
    if not package.is_dir():
        raise RuntimeError(f"shadow site does not contain cbond_on_rust: {site}")
    for name in tuple(sys.modules):
        if name == "cbond_on_rust" or name.startswith("cbond_on_rust."):
            del sys.modules[name]
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    module = importlib.import_module("cbond_on_rust")
    extension = importlib.import_module("cbond_on_rust.cbond_on_rust")
    extension_path = Path(extension.__file__).resolve()
    if not extension_path.is_relative_to(site):
        raise RuntimeError(
            f"refusing non-scratch extension: {extension_path}; expected below {site}"
        )
    if not hasattr(module, "compute_typed_factor_frame"):
        raise RuntimeError("scratch extension does not expose compute_typed_factor_frame")
    return module


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cursor = start.replace(day=1)
    final = end.replace(day=1)
    while cursor <= final:
        yield cursor
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()


def _load_daily_source(
    raw_root: Path,
    stem: str,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in _month_starts(start, end_exclusive - pd.Timedelta(days=1)):
        directory = raw_root / stem / month.strftime("%Y-%m")
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.parquet")):
            try:
                day = pd.Timestamp(path.stem).normalize()
            except ValueError:
                continue
            if start <= day < end_exclusive:
                pieces.append(pd.read_parquet(path))
    if not pieces:
        raise FileNotFoundError(f"no daily source files for {stem} from {start.date()} to {end_exclusive.date()}")
    return pd.concat(pieces, ignore_index=True).rename(columns={"instrument_code": "code"})


def _daily_context(raw_root: Path, score_day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    # The longest P1 requirement is 75 DataHub lookback days.  Use a larger
    # calendar buffer so weekends/holidays never truncate the strict history.
    start = score_day - pd.Timedelta(days=140)
    return {
        source: _load_daily_source(raw_root, stem, start=start, end_exclusive=score_day)
        for source, stem in _SOURCE_STEMS.items()
    }


def _output_panel(index: pd.MultiIndex, score_day: pd.Timestamp) -> pd.DataFrame:
    panel_index = pd.MultiIndex.from_tuples(
        [(dt, code, 0) for dt, code in index], names=["dt", "code", "seq"]
    )
    panel = pd.DataFrame(
        {"trade_time": [score_day + pd.Timedelta(hours=14, minutes=29)] * len(panel_index)},
        index=panel_index,
    )
    panel.attrs["__build_day__"] = score_day.date().isoformat()
    return panel


def _compare(score_day: pd.Timestamp, expected: pd.DataFrame, actual: pd.DataFrame, source_rows: dict[str, int]) -> DayResult:
    results: list[ColumnResult] = []
    for column in _COLUMNS:
        left = actual[column].to_numpy(dtype="float64")
        right = expected[column].to_numpy(dtype="float64")
        nan_mask_mismatch = int(np.count_nonzero(np.isnan(left) != np.isnan(right)))
        finite = np.isfinite(left) & np.isfinite(right)
        exact_mismatch = int(np.count_nonzero(left[finite] != right[finite]))
        max_abs = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        results.append(
            ColumnResult(
                name=column,
                nan_mask_mismatch=nan_mask_mismatch,
                finite_exact_mismatch=exact_mismatch,
                finite_pairs=int(finite.sum()),
                max_abs_difference=max_abs,
            )
        )
    return DayResult(
        score_day=score_day.date().isoformat(),
        rows=len(expected),
        keys_equal=actual.index.equals(expected.index),
        columns_equal=list(actual.columns) == list(expected.columns),
        source_rows=source_rows,
        columns=tuple(results),
    )


def _verify_day(
    *,
    score_day: pd.Timestamp,
    factor_root: Path,
    raw_root: Path,
    rust_module: object,
) -> DayResult | None:
    factor_path = factor_root / score_day.strftime("%Y-%m") / f"{score_day:%Y%m%d}.parquet"
    if not factor_path.is_file():
        return None
    expected = pd.read_parquet(factor_path, columns=list(_COLUMNS)).sort_index()
    daily_data = _daily_context(raw_root, score_day)
    actual = build_factor_frame_rust_legacy_reference(
        _output_panel(expected.index, score_day),
        _SPECS,
        rust_module=rust_module,
        daily_data=daily_data,
    )
    return _compare(
        score_day,
        expected,
        actual,
        {source: len(frame) for source, frame in daily_data.items()},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-site", required=True, type=Path)
    parser.add_argument("--start", required=True, type=_parse_day)
    parser.add_argument("--end", required=True, type=_parse_day)
    parser.add_argument(
        "--factor-root",
        type=Path,
        default=Path(r"D:/cbond_on/factor_data_live50_20260805/factors/T1430"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(r"D:/cbond_data_hub/raw_data"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be on or after --start")
    rust_module = _load_scratch_module(args.shadow_site)
    results: list[DayResult] = []
    skipped: list[str] = []
    for score_day in _iter_score_days(args.start, args.end):
        result = _verify_day(
            score_day=score_day,
            factor_root=args.factor_root,
            raw_root=args.raw_root,
            rust_module=rust_module,
        )
        if result is None:
            skipped.append(score_day.date().isoformat())
        else:
            results.append(result)
    payload = {
        "read_only": True,
        "shadow_site": str(args.shadow_site.resolve()),
        "days": [{**asdict(result), "passed": result.passed} for result in results],
        "skipped_missing_factor_store": skipped,
        "passed": bool(results) and all(result.passed for result in results),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
