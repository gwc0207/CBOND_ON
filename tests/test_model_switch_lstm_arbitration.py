from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from harness.model_switch_lstm_arbitration import (
    LSTMTrial,
    assert_targets_before,
    build_causal_sequence_samples,
    fit_feature_scaler,
    fit_lstm_with_early_stopping,
    make_lstm_forecast,
    predict_lstm_contrasts,
    small_lstm_grid,
    split_chronological_samples,
    transform_samples,
)
from harness.tools.model_switch_lstm_arbitration_replay import (
    EXPERIMENT_ROOT,
    LSTM_PLAN,
    ValidationTrialResult,
    _assert_experiment_output_root,
    _grid_table,
)


def _panel(days: int = 48, features: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(days, dtype=float)
    state = np.column_stack([np.sin(index / (column + 2.0)) + column * 0.1 for column in range(features)])
    targets = np.column_stack([0.0004 * np.sin(index / 5.0), 0.0003 * np.cos(index / 7.0)])
    return state, targets, np.ones(days, dtype=bool)


def test_endpoint_return_is_not_an_lstm_input() -> None:
    state, targets, present = _panel(days=12)
    original = build_causal_sequence_samples(state, targets, present, sequence_length=4, target_indices=[6], require_target=False)
    poisoned = targets.copy()
    poisoned[6] = np.asarray([99.0, -99.0])
    repeated = build_causal_sequence_samples(state, poisoned, present, sequence_length=4, target_indices=[6], require_target=False)
    assert len(original) == len(repeated) == 1
    assert_allclose(original.inputs, repeated.inputs, atol=0.0, rtol=0.0)
    # A missing endpoint label may block a *training* row, but it must not
    # block construction of the state-only prediction window.
    poisoned[6] = np.asarray([np.nan, np.nan])
    prediction_window = build_causal_sequence_samples(state, poisoned, present, sequence_length=4, target_indices=[6], require_target=False)
    training_window = build_causal_sequence_samples(state, poisoned, present, sequence_length=4, target_indices=[6], require_target=True)
    assert len(prediction_window) == 1
    assert len(training_window) == 0


def test_missing_state_breaks_sequence_without_zero_fill() -> None:
    state, targets, present = _panel(days=12)
    present[5] = False
    samples = build_causal_sequence_samples(state, targets, present, sequence_length=4, target_indices=[6, 7, 8, 9])
    # Endpoints 6--8 cross the missing state at day 5; only day 9 can begin
    # after it with a complete four-day state sequence.
    assert samples.target_indices.tolist() == [9]
    assert samples.input_start_indices.tolist() == [6]


def test_scaler_uses_only_the_callers_training_samples_and_future_poison_does_not_change_it() -> None:
    state, targets, present = _panel(days=16)
    train = build_causal_sequence_samples(state, targets, present, sequence_length=4, target_indices=[3, 4, 5, 6])
    scaler = fit_feature_scaler(train)
    expected_mean = train.inputs.reshape(-1, train.feature_count).mean(axis=0)
    assert_allclose(scaler.mean, expected_mean, atol=0.0, rtol=0.0)

    future_poisoned = state.copy()
    future_poisoned[7:] = 10_000.0
    poisoned_train = build_causal_sequence_samples(future_poisoned, targets, present, sequence_length=4, target_indices=[3, 4, 5, 6])
    poisoned_scaler = fit_feature_scaler(poisoned_train)
    assert_allclose(scaler.mean, poisoned_scaler.mean, atol=0.0, rtol=0.0)
    assert_allclose(scaler.scale, poisoned_scaler.scale, atol=0.0, rtol=0.0)


def test_training_targets_must_precede_a_prediction_day() -> None:
    state, targets, present = _panel(days=16)
    train = build_causal_sequence_samples(state, targets, present, sequence_length=4, target_indices=[3, 4, 5, 6])
    assert_targets_before(train, 7)
    with pytest.raises(ValueError, match="not strictly before"):
        assert_targets_before(train, 6)


def test_fixed_cpu_seed_is_deterministic() -> None:
    state, targets, present = _panel(days=42)
    raw = build_causal_sequence_samples(state, targets, present, sequence_length=5, target_indices=range(4, 34))
    fit_raw, stop_raw = split_chronological_samples(raw, validation_fraction=0.2)
    scaler = fit_feature_scaler(fit_raw)
    trial = LSTMTrial(sequence_length=5, hidden_size=4, dropout=0.1, max_epochs=10, patience=4, batch_size=8)
    first = fit_lstm_with_early_stopping(transform_samples(fit_raw, scaler), transform_samples(stop_raw, scaler), trial=trial)
    second = fit_lstm_with_early_stopping(transform_samples(fit_raw, scaler), transform_samples(stop_raw, scaler), trial=trial)
    prediction_samples = transform_samples(raw, scaler)
    assert first.best_epoch == second.best_epoch
    assert_allclose(
        predict_lstm_contrasts(first.model, prediction_samples),
        predict_lstm_contrasts(second.model, prediction_samples),
        atol=0.0,
        rtol=0.0,
    )


def test_two_contrast_output_is_coherent() -> None:
    forecast = make_lstm_forecast(
        trial=LSTMTrial(sequence_length=20, hidden_size=8, dropout=0.0),
        contrast_prediction=np.asarray([0.0004, -0.0002]),
        history_days=300,
    )
    assert abs(float(forecast.utilities.sum())) < 1e-12
    assert_allclose(forecast.pair_mean + forecast.pair_mean.T, np.zeros((3, 3)), atol=1e-12, rtol=0.0)
    assert forecast.forecaster_id == "lstm_seq20_hidden8_drop0"


def test_grid_is_exactly_the_predeclared_eight_trials() -> None:
    grid = small_lstm_grid()
    assert len(grid) == 8
    assert {trial.sequence_length for trial in grid} == {20, 40}
    assert {trial.hidden_size for trial in grid} == {8, 16}
    assert {trial.dropout for trial in grid} == {0.0, 0.1}


def _validation_stub(trial: LSTMTrial, *, rmse: float, hit: float, selector_sharpe: float) -> ValidationTrialResult:
    return ValidationTrialResult(
        trial=trial,
        sample_sets={},
        early_stop_result=SimpleNamespace(best_epoch=3, best_validation_loss=1.0),
        validation_refit_result=SimpleNamespace(epochs_ran=3),
        validation_scaler=SimpleNamespace(),
        validation_forecasts=None,
        validation_arbitration=None,
        calibration_errors={},
        metrics={
            "two_contrast_rmse_bp": rmse,
            "two_contrast_direction_hit_rate": hit,
            "validation_selector_total_return": 0.0,
            "validation_selector_sharpe": selector_sharpe,
        },
        checkpoint={"sha256": "fixture"},
        training_events=[],
    )


def test_hyperparameter_selection_uses_validation_forecast_rmse_not_selector_or_holdout_return() -> None:
    low_error = _validation_stub(LSTMTrial(sequence_length=20, hidden_size=8, dropout=0.0), rmse=2.0, hit=0.50, selector_sharpe=-4.0)
    high_error = _validation_stub(LSTMTrial(sequence_length=20, hidden_size=16, dropout=0.0), rmse=3.0, hit=0.99, selector_sharpe=99.0)
    table, selected = _grid_table([high_error, low_error])
    assert selected.trial.trial_id == low_error.trial.trial_id
    assert bool(table.loc[table["trial_id"] == low_error.trial.trial_id, "selected_for_final_holdout"].iloc[0])
    assert "final holdout returns" in LSTM_PLAN["selection"]["not_used"]


def test_replay_rejects_output_paths_outside_the_approved_experiment_root(tmp_path) -> None:
    assert EXPERIMENT_ROOT.resolve() == _assert_experiment_output_root(EXPERIMENT_ROOT)
    with pytest.raises(ValueError, match="output root must stay"):
        _assert_experiment_output_root(tmp_path)
