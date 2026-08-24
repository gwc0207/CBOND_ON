from __future__ import annotations

from datetime import date

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
