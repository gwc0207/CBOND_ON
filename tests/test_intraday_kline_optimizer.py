from __future__ import annotations

import numpy as np
import pytest
import torch

from cbond_on.infra.model.impl.torch_image import KlineImageCNN
from cbond_on.infra.model.impl.torch_sequence import build_intraday_sequence_model
from cbond_on.infra.model.runners.optimize_intraday_kline import (
    _compute_loss,
    _parse_folds,
    build_intraday_feature_batch,
)


def test_intraday_features_are_causal_within_the_morning_window() -> None:
    first = torch.zeros((1, 120, 4), dtype=torch.float32)
    second = first.clone()
    second[:, 60:, :] = 0.02

    first_features = build_intraday_feature_batch(first, {})
    second_features = build_intraday_feature_batch(second, {})

    assert first_features.shape == (1, 120, 10)
    assert torch.allclose(first_features[:, :60], second_features[:, :60])
    assert not torch.allclose(first_features[:, 60:], second_features[:, 60:])


@pytest.mark.parametrize("architecture", ["cnn1d", "inception", "tcn", "gru", "transformer"])
def test_intraday_sequence_models_return_one_score_per_sample(architecture: str) -> None:
    model = build_intraday_sequence_model(
        architecture,
        n_features=10,
        sequence_length=120,
        model_params={},
    )
    output = model(torch.zeros((2, 120, 10), dtype=torch.float32))
    assert output.shape == (2,)


def test_intraday_image_model_contract() -> None:
    model = KlineImageCNN(base_channels=8)
    output = model(torch.zeros((2, 3, 96, 120), dtype=torch.float32))
    assert output.shape == (2,)


@pytest.mark.parametrize("loss_name", ["huber_z", "huber_rank", "listnet", "hybrid_listnet", "tail_pairwise"])
def test_intraday_losses_are_finite(loss_name: str) -> None:
    prediction = torch.linspace(-0.5, 0.5, 40)
    z_target = torch.linspace(-2.0, 2.0, 40)
    rank_target = torch.linspace(-1.0, 1.0, 40)
    loss = _compute_loss(
        prediction,
        z_target,
        rank_target,
        loss_cfg={"name": loss_name, "top_n": 10},
    )
    assert torch.isfinite(loss)


def test_walk_forward_fold_rejects_training_test_overlap() -> None:
    with pytest.raises(ValueError):
        _parse_folds(
            [
                {
                    "name": "bad",
                    "train_start": "2024-01-02",
                    "train_end": "2024-07-01",
                    "test_start": "2024-07-01",
                    "test_end": "2024-09-30",
                }
            ]
        )


def test_intraday_feature_scaling_is_finite() -> None:
    rng = np.random.default_rng(42)
    ohlc = torch.from_numpy(rng.normal(0.0, 0.01, size=(4, 120, 4)).astype(np.float32))
    features = build_intraday_feature_batch(ohlc, {"clip": 6.0})
    assert torch.isfinite(features).all()
    assert float(features.abs().max()) <= 6.0


def test_intraday_body_first_return_removes_overnight_gap() -> None:
    ohlc = torch.zeros((1, 120, 4), dtype=torch.float32)
    ohlc[:, 0, 0] = 0.02
    ohlc[:, 0, 3] = 0.021

    gap_features = build_intraday_feature_batch(
        ohlc,
        {"include_levels": False, "first_return_mode": "gap"},
    )
    body_features = build_intraday_feature_batch(
        ohlc,
        {"include_levels": False, "first_return_mode": "body"},
    )

    assert torch.isclose(gap_features[0, 0, 0], torch.tensor(8.0))
    assert torch.isclose(body_features[0, 0, 0], torch.tensor(0.5), atol=1e-5)
