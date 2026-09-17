from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors import rust_backend


class _CapabilityModule:
    def __init__(self, capabilities: dict) -> None:
        self._capabilities = capabilities

    def factor_capabilities(self) -> dict:
        return self._capabilities


class _CaptureModule:
    def __init__(self) -> None:
        self.panel: pd.DataFrame | None = None
        self.daily: dict[str, pd.DataFrame] | None = None

    def compute_factor_frame(
        self,
        panel_df: pd.DataFrame,
        specs_payload: list[dict],
        _stock_df: pd.DataFrame | None,
        _map_df: pd.DataFrame | None,
        daily_data: dict[str, pd.DataFrame] | None,
        _params: dict,
    ) -> pd.DataFrame:
        self.panel = panel_df
        self.daily = daily_data
        out = panel_df.loc[:, ["dt", "code"]].drop_duplicates().copy()
        for spec in specs_payload:
            out[str(spec["output_col"] or spec["name"])] = 1.0
        return out


def _panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-08-06 14:30:00"), "110001.SH", 0),
            (pd.Timestamp("2026-08-06 14:30:00"), "110001.SH", 1),
            (pd.Timestamp("2026-08-06 14:30:00"), "110001.SH", 2),
        ],
        names=["dt", "code", "seq"],
    )
    frame = pd.DataFrame(
        {
            "trade_time": pd.to_datetime(
                [
                    "2026-08-06 14:28:00",
                    "2026-08-06 14:29:00",
                    "2026-08-06 14:30:00",
                ]
            ),
            "last": [100.0, 100.2, 100.4],
            "amount": [1000.0, 1200.0, 1400.0],
            "ask_price1": [100.2, 100.4, 100.6],
            "bid_price1": [99.8, 100.0, 100.2],
            "num_trades": [1.0, 2.0, 3.0],
        },
        index=index,
    )
    frame.attrs["__build_day__"] = "2026-08-06"
    return frame


def _capabilities_for(spec: FactorSpec) -> dict:
    output = spec.output_col or spec.name
    params_hash = rust_backend._canonical_params_sha256(dict(spec.params))
    contract = {
        "id": str(spec.rust_contract_id),
        "output_col": output,
        "factor": spec.factor,
        "signal": str(spec.params.get("signal", "")).strip() or None,
        "params_sha256": params_hash,
    }
    return {
        "abi_revision": rust_backend.RUST_FIRST_CAPABILITY_ABI,
        "compute_api": "compute_factor_frame",
        "python_fallback": False,
        "contract_ids": [contract["id"]],
        "factor_contracts": [contract],
    }


def test_rust_first_contract_validation_rejects_full_params_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FactorSpec(
        name="candidate",
        factor="candidate_factor_v1",
        params={"window": 20, "weights": [5, 4, 3]},
        rust_contract_id="research/candidate/v1",
    )
    module = _CapabilityModule(_capabilities_for(spec))
    monkeypatch.setattr(rust_backend, "_import_rust_module", lambda: module)

    rust_backend.validate_rust_first_contracts([spec])

    drifted = FactorSpec(
        name="candidate",
        factor="candidate_factor_v1",
        params={"window": 21, "weights": [5, 4, 3]},
        rust_contract_id="research/candidate/v1",
    )
    with pytest.raises(RuntimeError, match="params_sha256"):
        rust_backend.validate_rust_first_contracts([drifted])


def test_common_adapter_preserves_factor_time_metadata_and_daily_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _CaptureModule()
    monkeypatch.setattr(rust_backend, "_import_rust_module", lambda: module)
    spec = FactorSpec(name="amount_30m", factor="amount_sum", params={"window_minutes": 30})
    daily = {
        "market_cbond.daily_base": pd.DataFrame(
            {
                "trade_date": ["2026-08-05"],
                "code": ["110001"],
                "exchange_code": ["XSHG"],
                "rating": ["AA+"],
            }
        )
    }

    out = rust_backend.build_factor_frame_rust(_panel(), [spec], daily_data=daily)

    assert list(out.columns) == ["amount_30m"]
    assert module.panel is not None
    assert module.daily is not None
    assert module.panel.attrs["__build_day__"] == "2026-08-06"
    assert module.panel["__label_matches_score_day__"].eq(1).all()
    assert {"__trade_time_ns__", "__trade_time_clock_ns__"}.issubset(module.panel.columns)
    observed_daily = module.daily["market_cbond.daily_base"]
    assert observed_daily.loc[0, "exchange_code"] == "XSHG"
    assert observed_daily.loc[0, "rating"] == "AA+"


def _unified_module() -> object:
    raw = os.environ.get("CBOND_ON_RUST50_UNIFIED_SITE", "").strip()
    if not raw:
        pytest.skip("set CBOND_ON_RUST50_UNIFIED_SITE to run against an isolated Rust wheel")
    site = Path(raw).resolve()
    if not (site / "cbond_on_rust").is_dir():
        raise AssertionError(f"unified site does not contain cbond_on_rust: {site}")
    for name in tuple(sys.modules):
        if name == "cbond_on_rust" or name.startswith("cbond_on_rust."):
            del sys.modules[name]
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    module = importlib.import_module("cbond_on_rust")
    extension = importlib.import_module("cbond_on_rust.cbond_on_rust")
    assert Path(extension.__file__).resolve().is_relative_to(site)
    return module


def test_isolated_wheel_runs_mixed_specs_through_common_compute_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _unified_module()
    monkeypatch.setattr(rust_backend, "_import_rust_module", lambda: module)
    specs = [
        FactorSpec(
            name="amount_30m",
            factor="amount_sum",
            params={"window_minutes": 30, "amount_col": "amount"},
        ),
        FactorSpec(
            name="qed_prior_quote_lag2_agreement",
            factor="factor_mining_quote_execution_dynamics_v1",
            params={"signal": "qed_prior_quote_lag2_agreement"},
        ),
    ]

    out = rust_backend.build_factor_frame_rust(_panel(), specs)

    assert list(out.columns) == [spec.name for spec in specs]
    assert out.index.names == ["dt", "code"]
    assert len(out) == 1
    capabilities = module.factor_capabilities()
    assert capabilities["compute_api"] == "compute_factor_frame"
    assert capabilities["python_fallback"] is False
    contracts = capabilities["factor_contracts"]
    live50_contracts = [item for item in contracts if item["id"].startswith("live50_r5/")]
    research_r88_contracts = [
        item for item in contracts if item["id"].startswith("research_r88_20260825/")
    ]
    assert len(live50_contracts) == 50
    assert len(research_r88_contracts) == 38
    assert len(contracts) == 88
    assert len({item["id"] for item in contracts}) == 88
    assert all(len(item["params_sha256"]) == 64 for item in contracts)
