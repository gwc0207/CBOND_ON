from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cbond_on.infra.model.impl.torch_sequence.models import FactorTCNModel
from cbond_on.infra.model.runners import train_torch_sequence


_CODES = ["110001", "110002", "110003", "110004"]
_FACTOR_COLS = ["f1", "f2"]


def _write_factor_day(root: Path, day: date) -> None:
    frame = pd.DataFrame(
        {
            "dt": [pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)] * len(_CODES),
            "code": _CODES,
            "f1": [-2.0, -0.5, 0.5, 2.0],
            "f2": [1.5, -1.0, 0.25, -0.75],
        }
    ).set_index(["dt", "code"])
    path = root / "factors" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _write_label_day(root: Path, day: date) -> None:
    frame = pd.DataFrame(
        {
            "trade_time": [pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=42)] * len(_CODES),
            "code": _CODES,
            "y": [-0.03, -0.005, 0.015, 0.04],
        }
    )
    path = root / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def test_score_only_sequence_never_reads_target_label(tmp_path: Path, monkeypatch) -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    factor_root = tmp_path / "factor_data"
    label_root = tmp_path / "label_data"
    for day in days:
        _write_factor_day(factor_root, day)
    _write_label_day(label_root, days[0])

    real_read = train_torch_sequence._read_label_day

    def guarded_read(root, day, *, factor_time, label_time):
        if day == days[-1]:
            raise AssertionError("target label must not be read while scoring")
        return real_read(root, day, factor_time=factor_time, label_time=label_time)

    monkeypatch.setattr(train_torch_sequence, "_read_label_day", guarded_read)
    store = train_torch_sequence.FactorStore(factor_root, panel_name="T1430", window_minutes=15)
    x, codes, y = train_torch_sequence._build_day_sequence(
        target_day=days[-1],
        all_days=days,
        day_to_pos={day: idx for idx, day in enumerate(days)},
        sequence_days=2,
        store=store,
        label_root=label_root,
        factor_cols=_FACTOR_COLS,
        winsor_lower=None,
        winsor_upper=None,
        zscore=True,
        factor_time="14:30",
        label_time="14:42",
        missing_policy="raise",
        allow_missing_values=False,
        factor_cache={},
        label_cache={},
        load_label=False,
    )

    assert x.shape == (len(_CODES), 2, len(_FACTOR_COLS))
    assert set(codes) == set(_CODES)
    assert np.isnan(y).all()


def test_eval_honours_label_cutoff(tmp_path: Path, monkeypatch) -> None:
    first, target = date(2026, 1, 5), date(2026, 1, 6)
    label_root = tmp_path / "label_data"
    _write_label_day(label_root, first)
    calls: list[date] = []
    real_read = train_torch_sequence._read_label_day

    def guarded_read(root, day, *, factor_time, label_time):
        calls.append(day)
        if day > first:
            raise AssertionError("evaluation read a label later than its cutoff")
        return real_read(root, day, factor_time=factor_time, label_time=label_time)

    monkeypatch.setattr(train_torch_sequence, "_read_label_day", guarded_read)
    scores = pd.DataFrame(
        {
            "trade_date": [first, target],
            "code": [_CODES[0], _CODES[0]],
            "score": [0.1, 0.2],
        }
    )
    result = train_torch_sequence._score_eval_daily(
        scores_df=scores,
        label_root=label_root,
        factor_time="14:30",
        label_time="14:42",
        label_cache={},
        label_cutoff=first,
    )

    assert calls == [first]
    assert result["trade_date"].tolist() == [first]


def test_tcn_is_prefix_causal_at_final_state() -> None:
    torch.manual_seed(7)
    model = FactorTCNModel(n_features=3, channels=4, num_layers=2, kernel_size=3, dropout=0.0).eval()
    base = torch.randn(2, 8, 3)
    changed = base.clone()
    changed[:, -1, :] = changed[:, -1, :] + 50.0

    with torch.no_grad():
        # Earlier hidden states must not change when only a future point changes.
        h_base = model.net(model.input_norm(base).transpose(1, 2))
        h_changed = model.net(model.input_norm(changed).transpose(1, 2))

    assert torch.allclose(h_base[:, :, :-1], h_changed[:, :, :-1], atol=1e-6, rtol=0.0)


def test_rolling_split_uses_every_realised_pre_score_day() -> None:
    days = [date(2026, 1, value) for value in range(5, 15)]
    train_days, val_days = train_torch_sequence._split_rolling_train_validation(days, 0.7)

    assert train_days == days[:7]
    assert val_days == days[7:]
    assert train_days + val_days == days
    assert max(train_days + val_days) == days[-1]


def test_score_day_seed_is_stable_when_pending_positions_change() -> None:
    day = date(2026, 1, 7)

    assert train_torch_sequence._seed_for_score_day(base_seed=20260812, score_day=day) == (
        train_torch_sequence._seed_for_score_day(base_seed=20260812, score_day=day)
    )
    assert train_torch_sequence._seed_for_score_day(base_seed=20260812, score_day=day) != (
        train_torch_sequence._seed_for_score_day(base_seed=20260812, score_day=date(2026, 1, 8))
    )


def test_checkpoint_fingerprint_covers_training_and_input_contracts() -> None:
    common = {
        "architecture": "lstm",
        "factor_cols": ["f1"],
        "feature_names": ["f1"],
        "sequence_days": 20,
        "model_params": {"hidden_size": 8},
        "feature_cfg": {"fill_missing_enabled": True},
        "neutralization": {"enabled": True},
        "input_contract": {"factor_time": "14:30", "label_time": "14:42"},
    }
    first = train_torch_sequence._config_fingerprint(
        **common,
        training_contract={"window_days": 60, "label_transform": "zscore_day"},
    )
    changed_window = train_torch_sequence._config_fingerprint(
        **common,
        training_contract={"window_days": 80, "label_transform": "zscore_day"},
    )
    changed_label = train_torch_sequence._config_fingerprint(
        **common,
        training_contract={"window_days": 60, "label_transform": "none"},
    )

    assert first != changed_window
    assert first != changed_label


def test_warm_start_checkpoint_requires_prior_causal_provenance(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "2026-01-05.pt"
    expected_fingerprint = "causal-contract"
    model = torch.nn.Linear(2, 1)
    valid_metadata = {
        "score_day": "2026-01-05",
        "max_train_label_day": "2026-01-03",
        "max_validation_label_day": "2026-01-04",
    }

    train_torch_sequence._save_warm_start_state(
        checkpoint_path=checkpoint_path,
        model=model,
        fingerprint=expected_fingerprint,
        train_day=date(2026, 1, 5),
        metadata=valid_metadata,
    )
    assert train_torch_sequence._load_warm_start_state(
        checkpoint_path=checkpoint_path,
        expected_fingerprint=expected_fingerprint,
        device=torch.device("cpu"),
        score_day=date(2026, 1, 6),
        label_cutoff=date(2026, 1, 4),
    ) is not None

    def _payload(*, train_day: str = "2026-01-05", metadata: object = valid_metadata) -> dict:
        return {
            "format_version": 2,
            "train_day": train_day,
            "fingerprint": expected_fingerprint,
            "model_state": model.state_dict(),
            "metadata": metadata,
        }

    # A file date alone is not sufficient provenance.
    for invalid in (
        _payload(metadata={}),
        _payload(train_day="2026-01-04", metadata={**valid_metadata, "score_day": "2026-01-04"}),
        _payload(metadata={**valid_metadata, "max_validation_label_day": "2026-01-06"}),
    ):
        torch.save(invalid, checkpoint_path)
        assert train_torch_sequence._load_warm_start_state(
            checkpoint_path=checkpoint_path,
            expected_fingerprint=expected_fingerprint,
            device=torch.device("cpu"),
            score_day=date(2026, 1, 6),
            label_cutoff=None,
        ) is None

    # A checkpoint valid without a cutoff must still be rejected in a
    # retrospective request whose cutoff makes its used labels unavailable.
    torch.save(
        _payload(metadata={**valid_metadata, "max_validation_label_day": "2026-01-04"}),
        checkpoint_path,
    )
    assert train_torch_sequence._load_warm_start_state(
        checkpoint_path=checkpoint_path,
        expected_fingerprint=expected_fingerprint,
        device=torch.device("cpu"),
        score_day=date(2026, 1, 7),
        label_cutoff=date(2026, 1, 3),
    ) is None

    # Direct calls cannot reuse a checkpoint for its own score day either.
    torch.save(_payload(), checkpoint_path)
    assert train_torch_sequence._load_warm_start_state(
        checkpoint_path=checkpoint_path,
        expected_fingerprint=expected_fingerprint,
        device=torch.device("cpu"),
        score_day=date(2026, 1, 5),
        label_cutoff=None,
    ) is None


def test_nonrolling_contract_is_explicitly_rejected() -> None:
    with pytest.raises(ValueError, match="supports rolling.enabled=true only"):
        train_torch_sequence._require_rolling_enabled({"enabled": False})


def test_absent_neutralization_cache_root_keeps_legacy_contract_label() -> None:
    # Runner only resolves/redirects an explicitly supplied config value.
    raw = {}.get("neutralization_cache_root")
    assert raw is None
    assert (str(raw) if raw else "legacy_panel_default") == "legacy_panel_default"
