from __future__ import annotations

import hashlib
from importlib import import_module
import json
from time import perf_counter
from typing import Sequence

import pandas as pd

from cbond_on.domain.factors.base import ensure_panel_index
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col


# Profile-neutral ABI for the loaded Rust factor-contract capability surface.
RUST_FIRST_CAPABILITY_ABI = "rust_factor_contracts_20260806_r1"


def _import_rust_module():
    try:
        return import_module("cbond_on_rust")
    except Exception as exc:
        raise RuntimeError(
            "factor engine=rust but module 'cbond_on_rust' is not available. "
            "Build/install the Rust extension first (see rust/factor_engine/README.md)."
        ) from exc


def _specs_payload(specs: Sequence[FactorSpec]) -> list[dict]:
    payload: list[dict] = []
    for spec in specs:
        payload.append(
            {
                "name": str(spec.name),
                "factor": str(spec.factor),
                "params": dict(spec.params or {}),
                "output_col": spec.output_col,
                "rust_contract_id": spec.rust_contract_id,
            }
        )
    return payload


def _rust_first_enabled(compute_backend_params: dict | None) -> bool:
    root = dict(compute_backend_params or {})
    backend = root.get("__compute_backend__")
    policy = root.get("execution_policy")
    if policy is None and isinstance(backend, dict):
        policy = backend.get("execution_policy")
    return str(policy or "").strip().lower() == "rust_first"


def _canonical_params_sha256(params: dict) -> str:
    """Hash the JSON configuration value, not a lossy string representation."""

    try:
        canonical = json.dumps(
            params,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "rust_first factor params must be finite JSON-compatible values "
            "to validate an exact Rust instance contract"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_rust_first_capabilities(module: object, specs_payload: list[dict]) -> None:
    """Validate exact instance contracts against the loaded Rust extension.

    Class-level factor keys are deliberately insufficient: a single research
    factor class can expose many ``params.signal`` variants, and legacy factor
    keys can have materially different parameter contracts.  A rust-first
    caller therefore must name a binary-advertised instance contract ID; the
    factor/output/signal tuple is checked as a second independent guard.
    """

    try:
        capabilities = getattr(module, "factor_capabilities")()
    except AttributeError as exc:
        raise RuntimeError(
            "rust_first factor execution requires a rebuilt cbond_on_rust extension "
            "with factor_capabilities()"
        ) from exc
    if not isinstance(capabilities, dict):
        raise RuntimeError("cbond_on_rust.factor_capabilities() must return a dict")
    if str(capabilities.get("abi_revision", "")).strip() != RUST_FIRST_CAPABILITY_ABI:
        raise RuntimeError(
            "cbond_on_rust factor capability ABI does not match rust_first contract: "
            f"expected={RUST_FIRST_CAPABILITY_ABI!r}, "
            f"actual={capabilities.get('abi_revision')!r}"
        )
    if str(capabilities.get("compute_api", capabilities.get("api", ""))).strip() != "compute_factor_frame":
        raise RuntimeError("cbond_on_rust capability compute_api must be compute_factor_frame")
    if capabilities.get("python_fallback") is not False:
        raise RuntimeError("cbond_on_rust rust_first capability must declare python_fallback=false")

    raw_contracts = capabilities.get("factor_contracts")
    if not isinstance(raw_contracts, list):
        raise RuntimeError("cbond_on_rust capability payload is missing factor_contracts")
    contracts_by_id: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for item in raw_contracts:
        if not isinstance(item, dict):
            continue
        contract_id = str(item.get("id", "")).strip()
        if not contract_id:
            continue
        if contract_id in contracts_by_id:
            duplicate_ids.add(contract_id)
        contracts_by_id[contract_id] = item
    if duplicate_ids:
        raise RuntimeError(
            "cbond_on_rust capability payload has duplicate contract IDs: "
            + ", ".join(sorted(duplicate_ids))
        )
    advertised_ids = {str(item).strip() for item in capabilities.get("contract_ids", [])}
    if advertised_ids and advertised_ids != set(contracts_by_id):
        raise RuntimeError(
            "cbond_on_rust capability contract_ids do not match factor_contracts"
        )

    failures: list[str] = []
    seen_ids: set[str] = set()
    for spec in specs_payload:
        contract_id = str(spec.get("rust_contract_id") or "").strip()
        factor = str(spec.get("factor", "")).strip()
        params = dict(spec.get("params") or {})
        output = str(spec.get("output_col") or spec.get("name") or factor).strip()
        signal = str(params.get("signal", "")).strip() or None
        if not contract_id:
            failures.append(f"{output}: missing rust_contract_id")
            continue
        if contract_id in seen_ids:
            failures.append(f"{output}: duplicate rust_contract_id={contract_id!r}")
            continue
        seen_ids.add(contract_id)
        contract = contracts_by_id.get(contract_id)
        if contract is None:
            failures.append(f"{output}: unknown rust_contract_id={contract_id!r}")
            continue
        actual_factor = str(contract.get("factor", "")).strip()
        actual_output = str(contract.get("output_col", "")).strip()
        actual_signal = str(contract.get("signal") or "").strip() or None
        actual_params_sha256 = str(contract.get("params_sha256", "")).strip().lower()
        expected_params_sha256 = _canonical_params_sha256(params)
        if (
            (factor, output, signal)
            != (actual_factor, actual_output, actual_signal)
            or expected_params_sha256 != actual_params_sha256
        ):
            failures.append(
                f"{output}: contract={contract_id!r} expects "
                f"({actual_factor}, {actual_output}, {actual_signal!r}, "
                f"params_sha256={actual_params_sha256!r}), got "
                f"({factor}, {output}, {signal!r}, "
                f"params_sha256={expected_params_sha256!r})"
            )
    if failures:
        raise RuntimeError(
            "rust_first factor execution contract mismatch: " + "; ".join(failures)
        )


def validate_rust_first_contracts(specs: Sequence[FactorSpec]) -> None:
    """Fail before panel/context I/O unless every spec has a loaded Rust contract.

    ``run_factor_pipeline`` calls this public helper at its rust-first gate;
    the adapter calls it again immediately before compute as defence in depth
    for direct adapter callers.
    """

    spec_list = list(specs)
    if not spec_list:
        return
    module = _import_rust_module()
    _validate_rust_first_capabilities(module, _specs_payload(spec_list))


def _materialize_factor_time_metadata(frame: pd.DataFrame) -> None:
    """Preserve Pandas-labelled score-day semantics across the Rust boundary.

    Every Rust factor contract gets the same labelled-versus-physical score-day
    predicates, including when panel timestamps are timezone-aware while
    ``__build_day__`` is naive. These extra columns preserve exact daily and
    intraday kernel semantics without creating a profile-specific adapter.
    """

    label_day = pd.to_datetime(frame["dt"], errors="coerce").dt.normalize()
    raw_build_day = frame.attrs.get("__build_day__")
    parsed_build_day = (
        pd.to_datetime(raw_build_day, errors="coerce")
        if raw_build_day is not None
        else pd.NaT
    )
    if pd.notna(parsed_build_day):
        target_day = pd.Timestamp(parsed_build_day).normalize()
        label_score_day_match = label_day.eq(target_day)
    else:
        target_day = None
        label_score_day_match = label_day.eq(label_day)
    frame["__label_matches_score_day__"] = label_score_day_match.fillna(False).astype(
        "int64"
    )

    # Do not invent a trade_time when the input is malformed: typed Rust
    # intraday kernels deliberately retain their own missing-column failure
    # precedence.  The normal factor panel always carries this column.
    if "trade_time" not in frame.columns:
        return
    trade_time = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame["trade_time"] = trade_time
    frame["__trade_time_ns__"] = trade_time.astype("int64")
    frame["__trade_time_date__"] = trade_time.dt.strftime("%Y-%m-%d")
    hour = trade_time.dt.hour.fillna(0).astype("int64")
    minute = trade_time.dt.minute.fillna(0).astype("int64")
    second = trade_time.dt.second.fillna(0).astype("int64")
    microsecond = trade_time.dt.microsecond.fillna(0).astype("int64")
    nanosecond = trade_time.dt.nanosecond.fillna(0).astype("int64")
    frame["__trade_time_clock_ns__"] = (
        ((hour * 60 + minute) * 60 + second) * 1_000_000_000
        + microsecond * 1_000
        + nanosecond
    )
    if target_day is not None:
        trade_score_day_match = trade_time.dt.normalize().eq(target_day)
    else:
        trade_score_day_match = trade_time.dt.normalize().eq(label_day)
    frame["__trade_time_matches_score_day__"] = trade_score_day_match.fillna(False).astype(
        "int64"
    )


def _prepare_panel_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Create one lossless adapter frame for every ordinary Rust invocation."""

    attrs = dict(getattr(panel, "attrs", {}) or {})
    indexed = ensure_panel_index(panel)
    prepared = indexed.reset_index().copy()
    # pandas operations do not promise to retain attrs.  __build_day__ is part
    # of the factor score-date contract, rather than incidental DataFrame metadata.
    prepared.attrs = attrs
    prepared = prepared.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(
        drop=True
    )
    prepared.attrs = attrs
    prepared["dt"] = pd.to_datetime(prepared["dt"], errors="coerce")
    prepared["code"] = prepared["code"].astype(str)
    _materialize_factor_time_metadata(prepared)
    return prepared


def _call_compute_factor_frame(
    module: object,
    *,
    panel_df: pd.DataFrame,
    specs_payload: list[dict],
    stock_df: pd.DataFrame | None,
    map_df: pd.DataFrame | None,
    daily_payload: dict[str, pd.DataFrame] | None,
    compute_backend_params: dict,
):
    fn = getattr(module, "compute_factor_frame")
    try:
        return fn(
            panel_df,
            specs_payload,
            stock_df,
            map_df,
            daily_payload or None,
            compute_backend_params,
        )
    except TypeError as exc:
        # A rust-first run has already checked the current ABI and must never
        # reinterpret a runtime TypeError as permission to retry an older API.
        if _rust_first_enabled(compute_backend_params):
            raise RuntimeError(
                "rust_first cbond_on_rust.compute_factor_frame call failed; "
                "the extension is not eligible for legacy-signature retry"
            ) from exc
        # Backward-compatibility for a non-rust-first built extension with a
        # historical 5-arg signature.
        if daily_payload:
            raise RuntimeError(
                "rust extension is outdated for daily factor context; rebuild cbond_on_rust first"
            ) from exc
        return fn(
            panel_df,
            specs_payload,
            stock_df,
            map_df,
            compute_backend_params,
        )


def build_factor_frame_rust(
    panel: pd.DataFrame,
    specs: Sequence[FactorSpec],
    *,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
    compute_backend_params: dict | None = None,
) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame()

    t_total = perf_counter()
    t_prepare_panel = perf_counter()
    panel_df = _prepare_panel_frame(panel)
    t_prepare_panel = perf_counter() - t_prepare_panel

    t_prepare_context = perf_counter()
    if stock_panel is not None and not isinstance(stock_panel, pd.DataFrame):
        raise TypeError(
            f"rust_backend stock_panel must be pandas.DataFrame, got {type(stock_panel).__name__}"
        )
    if bond_stock_map is not None and not isinstance(bond_stock_map, pd.DataFrame):
        raise TypeError(
            f"rust_backend bond_stock_map must be pandas.DataFrame, got {type(bond_stock_map).__name__}"
        )

    stock_df = None
    if stock_panel is not None and not stock_panel.empty:
        stock_df = _prepare_panel_frame(stock_panel)

    map_df = None
    if bond_stock_map is not None and not bond_stock_map.empty:
        map_df = bond_stock_map.copy()

    daily_payload: dict[str, pd.DataFrame] = {}
    for source, raw_df in dict(daily_data or {}).items():
        if isinstance(raw_df, dict) and isinstance(raw_df.get("data"), pd.DataFrame):
            raw_df = raw_df["data"]
        if raw_df is not None and not isinstance(raw_df, pd.DataFrame):
            raise TypeError(
                f"rust_backend daily_data[{source!r}] must be pandas.DataFrame, got {type(raw_df).__name__}"
            )
        if raw_df is None or raw_df.empty:
            continue
        df = raw_df.copy()
        # The legacy adapter used to retain numeric columns only.  That loses
        # exchange_code and breaks the strict canonicalisation contract of the
        # unified Rust kernels. Pass the complete daily frame through;
        # legacy Rust parsing still consumes only its numeric subset.
        if "trade_date" not in df.columns or "code" not in df.columns:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df["code"] = df["code"].astype(str)
        daily_payload[str(source)] = df.reset_index(drop=True)
    t_prepare_context = perf_counter() - t_prepare_context

    t_payload = perf_counter()
    payload = _specs_payload(specs)
    t_payload = perf_counter() - t_payload
    module = _import_rust_module()
    if not hasattr(module, "compute_factor_frame"):
        raise RuntimeError("module 'cbond_on_rust' missing function: compute_factor_frame")
    if _rust_first_enabled(compute_backend_params):
        _validate_rust_first_capabilities(module, payload)
    runtime_params = dict(compute_backend_params or {})
    build_day = panel_df.attrs.get("__build_day__")
    if build_day is not None and "__factor_score_date" not in runtime_params:
        parsed_build_day = pd.to_datetime(build_day, errors="coerce")
        if pd.notna(parsed_build_day):
            runtime_params["__factor_score_date"] = (
                pd.Timestamp(parsed_build_day).normalize().date().isoformat()
            )

    t_rust = perf_counter()
    out = _call_compute_factor_frame(
        module,
        panel_df=panel_df,
        specs_payload=payload,
        stock_df=stock_df,
        map_df=map_df,
        daily_payload=daily_payload or None,
        compute_backend_params=runtime_params,
    )
    t_rust = perf_counter() - t_rust
    if not isinstance(out, pd.DataFrame):
        raise RuntimeError("cbond_on_rust.compute_factor_frame must return pandas.DataFrame")
    if "dt" not in out.columns or "code" not in out.columns:
        raise RuntimeError("cbond_on_rust output must include columns: dt, code")

    t_post = perf_counter()
    out = out.copy()
    out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
    out["code"] = out["code"].astype(str)
    out = out.set_index(["dt", "code"]).sort_index()

    cols = [build_factor_col(spec) for spec in specs]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise RuntimeError(f"cbond_on_rust output missing factor columns: {missing}")
    t_post = perf_counter() - t_post
    print(
        "rust_engine:",
        f"panel_rows={len(panel_df)}",
        f"specs={len(specs)}",
        f"t_prepare_panel={t_prepare_panel:.2f}s",
        f"t_prepare_context={t_prepare_context:.2f}s",
        f"t_payload={t_payload:.2f}s",
        f"t_rust_call={t_rust:.2f}s",
        f"t_post={t_post:.2f}s",
        f"total={perf_counter() - t_total:.2f}s",
        flush=True,
    )
    return out[cols]

