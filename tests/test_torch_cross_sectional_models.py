from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cbond_on.infra.model.impl.torch_cross_section import build_cross_section_model
from cbond_on.infra.model.impl.lgbm.trainer import SplitData
from cbond_on.infra.model.runners import train_torch_cross_section as cross_section_runner
from cbond_on.infra.model.runners.train_torch_cross_section import (
    CrossSectionSplit,
    _config_fingerprint,
    _listnet_loss,
    _mean_daily_listnet_loss,
    _resolve_model_input_factors,
)


def _r88_factors() -> list[str]:
    return [f"r88_factor_{index:03d}" for index in range(1, 89)]


def _r88_specs(factors: list[str]) -> list[dict[str, object]]:
    return [
        {
            "name": factor,
            "factor": f"research::{factor}",
            "params": {"window": index},
            "output_col": None,
            "rust_contract_id": f"research_r88_20260825/{factor}",
        }
        for index, factor in enumerate(factors, start=1)
    ]


def _r88_specs_sha256(specs: list[dict[str, object]]) -> str:
    canonical = [
        {
            "name": spec["name"],
            "factor": spec["factor"],
            "params": spec["params"],
            "output_col": spec.get("output_col"),
            "rust_contract_id": spec.get("rust_contract_id"),
        }
        for spec in specs
    ]
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("architecture", ["linear", "mlp", "deepsets", "settransformer"])
def test_cross_section_models_are_permutation_equivariant(architecture: str) -> None:
    torch.manual_seed(7)
    model = build_cross_section_model(
        architecture,
        n_features=6,
        model_params={"hidden_size": 16, "num_heads": 4, "dropout": 0.0},
    ).eval()
    x = torch.randn(9, 6)
    order = torch.tensor([7, 1, 8, 0, 4, 2, 6, 3, 5])
    with torch.no_grad():
        baseline = model(x)
        shuffled = model(x[order])
    assert torch.allclose(shuffled, baseline[order], atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("architecture", ["deepsets", "settransformer"])
def test_set_models_ignore_invalid_padding(architecture: str) -> None:
    torch.manual_seed(11)
    model = build_cross_section_model(
        architecture,
        n_features=4,
        model_params={"hidden_size": 16, "num_heads": 4, "dropout": 0.0},
    ).eval()
    x = torch.randn(5, 4)
    mask = torch.tensor([True, True, True, False, False])
    altered = x.clone()
    altered[~mask] = 1e6
    with torch.no_grad():
        baseline = model(x, mask)
        changed = model(altered, mask)
    assert torch.allclose(changed[mask], baseline[mask], atol=1e-6, rtol=0.0)
    assert torch.equal(changed[~mask], torch.zeros_like(changed[~mask]))


def test_listnet_is_day_permutation_invariant() -> None:
    pred = torch.tensor([0.2, -0.1, 0.4, 0.0])
    target = torch.tensor([-0.03, 0.01, 0.05, -0.02])
    order = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(
        _listnet_loss(pred, target, temperature=1.0),
        _listnet_loss(pred[order], target[order], temperature=1.0),
        atol=1e-7,
        rtol=0.0,
    )


def test_daily_loss_is_equal_weighted_not_row_weighted() -> None:
    class _ZeroModel(torch.nn.Module):
        def forward(self, x, valid_mask=None):
            return x[:, 0] * 0.0

    split = CrossSectionSplit(
        x=np.ones((5, 1), dtype=np.float32),
        y=np.asarray([-1.0, 1.0, -1.0, 0.0, 1.0], dtype=np.float32),
        dt=np.asarray(["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06", "2026-01-06"], dtype=object),
        code=np.asarray(["a", "b", "c", "d", "e"], dtype=object),
        groups=[np.asarray([0, 1]), np.asarray([2, 3, 4])],
    )
    model = _ZeroModel()
    actual = _mean_daily_listnet_loss(model=model, split=split, device=torch.device("cpu"), temperature=1.0)
    first = _listnet_loss(torch.zeros(2), torch.tensor([-1.0, 1.0]), temperature=1.0)
    second = _listnet_loss(torch.zeros(3), torch.tensor([-1.0, 0.0, 1.0]), temperature=1.0)
    assert torch.allclose(actual, (first + second) / 2.0, atol=1e-7, rtol=0.0)


def test_model_input_factors_must_be_an_ordered_unique_live50_subset() -> None:
    full = ["fast_a", "fast_b", "slow_a", "slow_b"]
    assert _resolve_model_input_factors(full_factors=full, configured_subset=None) == full
    assert _resolve_model_input_factors(full_factors=full, configured_subset=["fast_a", "slow_a"]) == ["fast_a", "slow_a"]
    with pytest.raises(ValueError, match="duplicates"):
        _resolve_model_input_factors(full_factors=full, configured_subset=["fast_a", "fast_a"])
    with pytest.raises(ValueError, match="outside"):
        _resolve_model_input_factors(full_factors=full, configured_subset=["missing"])
    with pytest.raises(ValueError, match="preserve"):
        _resolve_model_input_factors(full_factors=full, configured_subset=["slow_a", "fast_a"])


def test_subset_changes_fingerprint_but_full50_default_does_not() -> None:
    common = {
        "architecture": "deepsets",
        "factors": ["a", "b", "c"],
        "model_params": {"hidden_size": 16},
        "input_missingness": {"add_missing_mask": True},
        "objective": {"name": "listnet"},
        "neutralization": {"method": "ridge"},
        "contract": {"window_days": 60},
    }
    historical = _config_fingerprint(**common)
    assert _config_fingerprint(**common, model_input_factors=["a", "b", "c"]) == historical
    assert _config_fingerprint(**common, model_input_factors=["a", "c"]) != historical


def test_r88_contract_accepts_only_the_frozen_ordered_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = _r88_factors()
    specs = _r88_specs(factors)
    profile_path = (
        tmp_path
        / "cbond_on"
        / "factor_contracts"
        / "profiles"
        / "research_r88_rust88_20260825.json5"
    )
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        json.dumps(
                {
                    "research_only": True,
                    "admission_profile": "research_r88_rust88_20260825",
                    "execution_policy": "rust_first",
                    "compute": {"engine": "rust", "execution_policy": "rust_first"},
                "factors": factors,
                "factor_specs": specs,
                "specs_sha256": _r88_specs_sha256(specs),
                "time_contract": {
                    "panel_name": "T1430",
                    "factor_time": "14:30",
                    "label_time": "14:42",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cross_section_runner, "PROJECT_ROOT", tmp_path)

    admission = cross_section_runner._require_r88_factor_contract(
        contract_ref="factor_contracts/profiles/research_r88_rust88_20260825.json5",
        factors=factors,
        panel_name="T1430",
        factor_time="14:30",
        label_time="14:42",
    )
    assert admission["admission_profile"] == "research_r88_rust88_20260825"
    assert admission["research_only"] is True

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["factor_specs"][0]["params"]["window"] = 999
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="specs_sha256"):
        cross_section_runner._require_r88_factor_contract(
            contract_ref="factor_contracts/profiles/research_r88_rust88_20260825.json5",
            factors=factors,
            panel_name="T1430",
            factor_time="14:30",
            label_time="14:42",
        )


def test_r88_research_profile_uses_explicit_isolated_root_and_coverage_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cross_section_runner, "_R88_RESEARCH_SCRATCH_ROOT", tmp_path)
    factor_root = tmp_path / "r88_backfill" / "factor_data"
    factor_root.mkdir(parents=True)
    results_root = tmp_path / "r88_models" / "results"
    score_output = results_root / "scores" / "model"
    state_dir = results_root / "model_state" / "model"
    neutralization_cache_root = results_root / "neutralization_cache" / "model"
    cfg = {
        "research_only": {
            "profile": "research_r88_rust88_20260825",
            "factor_store_root": str(factor_root),
            "min_available_fraction": 0.75,
        },
        "experiment": {"research_only": True},
    }
    missing_values = {
        "enabled": True,
        "keep_nan": True,
        "min_available_factors": 66,
        "add_valid_count_features": False,
    }

    admission = cross_section_runner._require_r88_research_profile(
        cfg,
        factor_root=factor_root,
        results_root=results_root,
        score_output=score_output,
        state_dir=state_dir,
        neutralization_cache_root=neutralization_cache_root,
        factors=_r88_factors(),
        missing_values=missing_values,
    )
    assert admission["min_available_factors"] == 66
    assert admission["min_available_fraction"] == 0.75
    assert cross_section_runner._r88_min_available_factors(fraction=1.0, factor_count=88) == 88

    missing_values["min_available_factors"] = 27
    with pytest.raises(ValueError, match="must equal ceil"):
        cross_section_runner._require_r88_research_profile(
            cfg,
            factor_root=factor_root,
            results_root=results_root,
            score_output=score_output,
            state_dir=state_dir,
            neutralization_cache_root=neutralization_cache_root,
            factors=_r88_factors(),
            missing_values=missing_values,
        )


def test_r88_normal_reader_requires_canonical_experiment_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_root = tmp_path / "factor_store" / "experiment"
    factor_root.mkdir(parents=True)
    results_root = tmp_path / "research_scratch" / "models" / "results"
    monkeypatch.setattr(cross_section_runner, "_R88_RESEARCH_SCRATCH_ROOT", tmp_path / "research_scratch")
    cfg = {
        "research_only": {
            "profile": "research_r88_rust88_20260825",
            "factor_table": {"table_id": "experiment", "root": str(tmp_path / "factor_store")},
            "min_available_fraction": 0.75,
        },
        "experiment": {"research_only": True},
    }
    paths_cfg = {
        "factor_table": {"table_id": "experiment", "root": str(tmp_path / "factor_store")},
        "read_only_input_roots": {"panel_data_root": str(tmp_path / "panel"), "label_data_root": str(tmp_path / "label")},
    }
    missing_values = {"enabled": True, "keep_nan": True, "min_available_factors": 66, "add_valid_count_features": False}

    admitted = cross_section_runner._require_r88_research_profile(
        cfg,
        paths_cfg=paths_cfg,
        factor_root=factor_root,
        results_root=results_root,
        score_output=results_root / "scores" / "model",
        state_dir=results_root / "model_state" / "model",
        neutralization_cache_root=results_root / "neutralization_cache" / "model",
        factors=_r88_factors(),
        missing_values=missing_values,
    )
    assert admitted["factor_table"] == cfg["research_only"]["factor_table"]
    assert "factor_store_root" not in admitted

    cfg["research_only"]["factor_store_root"] = str(factor_root)
    with pytest.raises(ValueError, match="retired"):
        cross_section_runner._require_r88_research_profile(
            cfg,
            paths_cfg=paths_cfg,
            factor_root=factor_root,
            results_root=results_root,
            score_output=results_root / "scores" / "model",
            state_dir=results_root / "model_state" / "model",
            neutralization_cache_root=results_root / "neutralization_cache" / "model",
            factors=_r88_factors(),
            missing_values=missing_values,
        )


def test_r88_rejects_implicit_fraction_and_out_of_order_subset() -> None:
    factors = _r88_factors()
    with pytest.raises(ValueError, match="must be one of"):
        cross_section_runner._r88_min_available_factors(fraction=0.74, factor_count=88)
    with pytest.raises(ValueError, match="preserve frozen R88"):
        _resolve_model_input_factors(
            full_factors=factors,
            configured_subset=[factors[2], factors[1]],
            contract_label="frozen R88 research contract",
        )


def test_r88_requires_completed_full_coverage_backfill_before_model_input_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_root = tmp_path / "factor_data"
    factor_root.mkdir()
    frozen_root = tmp_path / "frozen_r50_factor_data"
    raw_root = tmp_path / "raw_data"
    clean_root = tmp_path / "clean_data"
    factors = _r88_factors()
    full_days = [date(2024, 1, 3), date(2026, 7, 30)]
    factor_contract = {
        "execution_status": "eligible_for_fresh_rust_backfill",
        "specs_sha256": "a" * 64,
    }
    for root in (frozen_root, factor_root):
        store = cross_section_runner.FactorStore(root, panel_name="T1430")
        for day in full_days:
            store.write_day(day, pd.DataFrame({"test": [1.0]}))
    monkeypatch.setattr(cross_section_runner, "_R88_FROZEN_R50_FACTOR_ROOT", frozen_root)

    manifest = {
        "schema_version": "r88_factor_backfill_manifest/v1",
        "research_only": True,
        "execution_status": "completed_rust88_backfill",
        "model_training_ready": True,
        "profile": {
            "path": "cbond_on/factor_contracts/profiles/research_r88_rust88_20260825.json5",
            "admission_profile": "research_r88_rust88_20260825",
            "specs_sha256": "a" * 64,
        },
        "factor_root": str(factor_root.resolve()),
        "factor_count": 88,
        "factors": factors,
        "time_contract": {"panel_name": "T1430", "factor_time": "14:30", "label_time": "14:42"},
        "range_contract": {"start": "2024-01-01", "end": "2026-07-30"},
        "source_inputs": {"raw_data_root": str(raw_root), "clean_data_root": str(clean_root)},
        "execution": {
            "mode": "r38_only_plus_frozen_r50_merge",
            "frozen_r50_factor_root": str(frozen_root),
            "r88_factor_root": str(factor_root),
            "days": [{"day": day.isoformat()} for day in full_days],
        },
        "completion": {
            "definition": "requested_approved_range_and_exact_frozen_r50_day_coverage",
            "approved_range": {"start": "2024-01-01", "end": "2026-07-30"},
            "profile_range": {"start": "2024-01-01", "end": "2026-07-30"},
            "requested_range": {"start": "2024-01-01", "end": "2026-07-30"},
            "frozen_r50_factor_root": str(frozen_root),
            "expected_frozen_r50_days": [day.isoformat() for day in full_days],
            "execution_days": [day.isoformat() for day in full_days],
            "r88_factor_store_days": [day.isoformat() for day in full_days],
            "full_completion": True,
            "blocking_reasons": [],
        },
    }
    manifest_path = factor_root / "r88_backfill_manifest.json"

    # An old smoke manifest could falsely claim the legacy completed/ready
    # status but has no modern completion proof.  It must not reach model I/O.
    manifest_path.write_text(json.dumps(manifest | {"completion": None}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing completion evidence"):
        cross_section_runner._require_r88_backfill_readiness(
            factor_root=factor_root,
            factor_contract=factor_contract,
            factors=factors,
            panel_name="T1430",
            factor_time="14:30",
            label_time="14:42",
            raw_root=raw_root,
            clean_root=clean_root,
        )

    incomplete_claim = json.loads(json.dumps(manifest))
    incomplete_claim["completion"]["full_completion"] = False
    manifest_path.write_text(json.dumps(incomplete_claim), encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion.full_completion must be true"):
        cross_section_runner._require_r88_backfill_readiness(
            factor_root=factor_root,
            factor_contract=factor_contract,
            factors=factors,
            panel_name="T1430",
            factor_time="14:30",
            label_time="14:42",
            raw_root=raw_root,
            clean_root=clean_root,
        )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    readiness = cross_section_runner._require_r88_backfill_readiness(
        factor_root=factor_root,
        factor_contract=factor_contract,
        factors=factors,
        panel_name="T1430",
        factor_time="14:30",
        label_time="14:42",
        raw_root=raw_root,
        clean_root=clean_root,
    )
    assert readiness["model_training_ready"] is True

    partial_claim = json.loads(json.dumps(manifest))
    partial_claim["completion"]["requested_range"] = {"start": "2026-07-30", "end": "2026-07-30"}
    manifest_path.write_text(json.dumps(partial_claim), encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion.requested_range must equal the approved R88 range"):
        cross_section_runner._require_r88_backfill_readiness(
            factor_root=factor_root,
            factor_contract=factor_contract,
            factors=factors,
            panel_name="T1430",
            factor_time="14:30",
            label_time="14:42",
            raw_root=raw_root,
            clean_root=clean_root,
        )

    coverage_claim = json.loads(json.dumps(manifest))
    coverage_claim["completion"]["r88_factor_store_days"] = [full_days[0].isoformat()]
    manifest_path.write_text(json.dumps(coverage_claim), encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion coverage is not exact"):
        cross_section_runner._require_r88_backfill_readiness(
            factor_root=factor_root,
            factor_contract=factor_contract,
            factors=factors,
            panel_name="T1430",
            factor_time="14:30",
            label_time="14:42",
            raw_root=raw_root,
            clean_root=clean_root,
        )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    factor_contract["execution_status"] = "blocked_pending_exact_rust_contracts"
    with pytest.raises(RuntimeError, match="not eligible"):
        cross_section_runner._require_r88_backfill_readiness(
            factor_root=factor_root,
            factor_contract=factor_contract,
            factors=factors,
            panel_name="T1430",
            factor_time="14:30",
            label_time="14:42",
            raw_root=raw_root,
            clean_root=clean_root,
        )


def test_model_subset_is_selected_only_after_full50_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A factor slice cannot weaken the shared live50 admission contract."""

    full = ["fast_a", "fast_b", "slow_a", "slow_b"]
    observed: dict[str, object] = {}

    def _fake_build_dataset(**kwargs) -> SplitData:
        observed.update(kwargs)
        return SplitData(
            x=pd.DataFrame(
                {
                    "fast_a": [1.0, np.nan],
                    "fast_b": [2.0, 3.0],
                    "slow_a": [4.0, 5.0],
                    "slow_b": [6.0, 7.0],
                }
            ),
            y=pd.Series([0.01, -0.02]),
            dt=pd.Series(pd.to_datetime(["2024-07-02", "2024-07-02"])),
            code=pd.Series(["a", "b"]),
        )

    monkeypatch.setattr(cross_section_runner, "build_dataset", _fake_build_dataset)
    split = cross_section_runner._build_data(
        store=object(),
        label_root=object(),
        days=[date(2024, 7, 2)],
        factors=full,
        model_input_factors=["fast_a", "slow_a"],
        min_count=2,
        winsor_lower=None,
        winsor_upper=None,
        zscore=True,
        factor_time="14:30",
        label_time="14:42",
        require_label=True,
        tradable_code_map={},
        neutralizer=None,
        missing_values={"enabled": True, "min_available_factors": 2},
        input_missingness={"fill_value": 0.0, "add_missing_mask": True},
    )

    assert observed["factor_cols"] == full
    assert observed["raw_factor_cols"] == full
    assert observed["preprocess_factor_cols"] == full
    # Two selected factors followed by their missingness mask; fast_b and
    # slow_b never enter the model input matrix.
    np.testing.assert_allclose(split.x, [[1.0, 4.0, 0.0, 0.0], [0.0, 5.0, 1.0, 0.0]])
