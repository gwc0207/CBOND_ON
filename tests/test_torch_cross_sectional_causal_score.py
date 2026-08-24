from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cbond_on.infra.model.impl.lgbm import trainer
from cbond_on.infra.model.runners import train_torch_cross_section as runner


def test_score_only_uses_allowlist_before_zscore_and_never_reads_label(monkeypatch) -> None:
    score_day = date(2026, 1, 5)
    score_dt = pd.Timestamp("2026-01-05 14:30")

    class _Store:
        def read_day(self, requested_day):
            assert requested_day == score_day
            return pd.DataFrame(
                {"f1": [0.0, 1.0, 1000.0]},
                index=pd.MultiIndex.from_arrays(
                    [[score_dt] * 3, ["keep_a", "keep_b", "drop"]], names=["dt", "code"]
                ),
            )

    monkeypatch.setattr(trainer, "_read_label_day", lambda *_a, **_k: pytest.fail("score-only read target label"))
    raw = trainer.build_dataset(
        factor_store=_Store(), label_root=Path("unused"), days=[score_day], factor_cols=["f1"],
        min_count=2, winsor_lower=None, winsor_upper=None, zscore=True, factor_time="14:30", label_time="14:42",
        require_label=False, tradable_code_map={score_day: {"keep_a", "keep_b"}}, tradable_strict=True,
        read_label_when_not_required=False, apply_tradable_filter_when_label_not_required=True,
    )
    assert raw.code.tolist() == ["keep_a", "keep_b"]
    assert np.allclose(raw.x["f1"].to_numpy(dtype=float), [-1.0, 1.0])


def test_research_contract_rejects_live_paths_and_incremental_skip(tmp_path: Path) -> None:
    cfg = {
        "rolling": {"enabled": True}, "refit_every_n_days": 1,
        "incremental": {"skip_existing_scores": True, "warm_start": True, "save_state": True},
        "score_only_no_target_label_read": True, "score_only_apply_tradable_filter": True,
        "tradable_filter": {"strict": True},
    }
    with pytest.raises(ValueError, match="skip_existing_scores"):
        runner._require_research_contract(cfg, results_root=tmp_path, score_output=tmp_path / "scores", state_dir=tmp_path / "state")


def test_listnet_standardizes_only_its_completed_daily_target() -> None:
    pred = torch.tensor([-0.4, 0.1, 0.6])
    raw = torch.tensor([-0.002, 0.001, 0.005])
    # A positive affine rescaling of a completed day's return labels preserves
    # the listwise target distribution.  This prevents raw 1bp-scale labels
    # from degenerating to a nearly uniform softmax.
    assert torch.allclose(
        runner._listnet_loss(pred, raw, temperature=1.0),
        runner._listnet_loss(pred, raw * 1000.0 + 3.0, temperature=1.0),
        atol=1e-7,
        rtol=0.0,
    )


def test_first_successful_score_day_is_allowed_to_cold_start() -> None:
    # In a 60-day rolling chain the requested start can predate its first
    # scoreable day.  Checkpoint ancestry must use successful days, never the
    # absolute index inside the requested calendar.
    state_dir = Path("scratch-state")
    previous, predecessor = runner._warm_start_chain_paths(state_dir, [])
    assert previous is None
    assert predecessor is None

    previous, predecessor = runner._warm_start_chain_paths(state_dir, [date(2024, 5, 8)])
    assert previous == state_dir / "2024-05-08.pt"
    assert predecessor is None

    previous, predecessor = runner._warm_start_chain_paths(
        state_dir,
        [date(2024, 5, 8), date(2024, 5, 9)],
    )
    assert previous == state_dir / "2024-05-09.pt"
    assert predecessor == state_dir / "2024-05-08.pt"


def test_label_cutoff_cannot_shorten_the_natural_rolling_history() -> None:
    history = [date(2024, 4, 8), date(2024, 4, 9), date(2024, 4, 10)]
    error = runner._strict_history_window_error(
        history,
        expected_history_days=3,
        label_cutoff=date(2024, 4, 9),
    )
    assert error is not None
    assert "does not cover" in error

    assert runner._strict_history_window_error(
        history,
        expected_history_days=3,
        label_cutoff=date(2024, 4, 10),
    ) is None


def test_warm_start_checkpoint_parent_chain_is_validated_from_successful_days(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    model = torch.nn.Linear(2, 1)
    fingerprint = "r3-test-fingerprint"
    first_day = date(2024, 5, 8)
    second_day = date(2024, 5, 9)

    first_path = state_dir / "2024-05-08.pt"
    runner._save_warm_start_state(
        checkpoint_path=first_path,
        model=model,
        fingerprint=fingerprint,
        score_day=first_day,
        max_train_label_day=date(2024, 4, 8),
        max_validation_label_day=date(2024, 5, 7),
        previous_checkpoint=None,
    )
    loaded_first = runner._load_warm_start_state(
        checkpoint_path=first_path,
        expected_fingerprint=fingerprint,
        score_day=second_day,
        label_cutoff=None,
        device=torch.device("cpu"),
        expected_checkpoint_predecessor=None,
    )
    assert loaded_first is not None

    second_path = state_dir / "2024-05-09.pt"
    runner._save_warm_start_state(
        checkpoint_path=second_path,
        model=model,
        fingerprint=fingerprint,
        score_day=second_day,
        max_train_label_day=date(2024, 4, 9),
        max_validation_label_day=first_day,
        previous_checkpoint=first_path,
    )
    loaded_second = runner._load_warm_start_state(
        checkpoint_path=second_path,
        expected_fingerprint=fingerprint,
        score_day=date(2024, 5, 10),
        label_cutoff=None,
        device=torch.device("cpu"),
        expected_checkpoint_predecessor=first_path,
    )
    assert loaded_second is not None


@pytest.mark.parametrize("kind", ["checkpoint", "temporary", "score"])
def test_fresh_research_outputs_reject_resume_or_overwrite(tmp_path: Path, kind: str) -> None:
    state_dir = tmp_path / "state"
    score_output = tmp_path / "scores"
    if kind == "checkpoint":
        state_dir.mkdir()
        (state_dir / "2024-05-08.pt").write_bytes(b"old-state")
    elif kind == "temporary":
        state_dir.mkdir()
        (state_dir / "2024-05-08.abcd.tmp").write_bytes(b"partial-state")
    else:
        score_output.mkdir()
        (score_output / "2024-05-08.csv").write_text("trade_date,code,score\\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fresh run refuses existing"):
        runner._require_fresh_research_outputs(state_dir=state_dir, score_output=score_output)


def _checkpoint_for_resume_test(
    *,
    state_dir: Path,
    score_day: date,
    parent: Path | None,
    fingerprint: str,
) -> Path:
    model = torch.nn.Linear(2, 1)
    checkpoint = state_dir / f"{score_day:%Y-%m-%d}.pt"
    runner._save_warm_start_state(
        checkpoint_path=checkpoint,
        model=model,
        fingerprint=fingerprint,
        score_day=score_day,
        max_train_label_day=date(2024, 1, 2),
        max_validation_label_day=date(2024, 1, 3),
        previous_checkpoint=parent,
    )
    return checkpoint


def _atomic_score_frame(score_day: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [score_day, score_day],
            "code": ["110001", "110002"],
            "score": [0.1, 0.2],
        }
    )


def test_orphan_checkpoint_is_recoverable_only_as_the_single_next_day(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    score_root = tmp_path / "scores"
    state_dir.mkdir()
    fingerprint = "r3-test-fingerprint"
    source_terminal = tmp_path / "source" / "2024-01-04.pt"
    source_terminal.parent.mkdir()
    _checkpoint_for_resume_test(
        state_dir=source_terminal.parent,
        score_day=date(2024, 1, 4),
        parent=None,
        fingerprint=fingerprint,
    )
    orphan = _checkpoint_for_resume_test(
        state_dir=state_dir,
        score_day=date(2024, 1, 5),
        parent=source_terminal,
        fingerprint=fingerprint,
    )

    record = runner._find_resume_orphan_checkpoint(
        state_dir=state_dir,
        score_output=score_root,
        committed_continuation_days=[],
        committed_score_days=[],
        source_terminal=source_terminal,
        source_terminal_day=date(2024, 1, 4),
        calendar_days=[date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)],
        expected_fingerprint=fingerprint,
        label_cutoff=None,
    )
    assert record is not None
    assert record.path == orphan
    assert record.previous_checkpoint == source_terminal.resolve(strict=False)

    _checkpoint_for_resume_test(
        state_dir=state_dir,
        score_day=date(2024, 1, 8),
        parent=orphan,
        fingerprint=fingerprint,
    )
    with pytest.raises(RuntimeError, match="single next natural trading day"):
        runner._find_resume_orphan_checkpoint(
            state_dir=state_dir,
            score_output=score_root,
            committed_continuation_days=[],
            committed_score_days=[],
            source_terminal=source_terminal,
            source_terminal_day=date(2024, 1, 4),
            calendar_days=[date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)],
            expected_fingerprint=fingerprint,
            label_cutoff=None,
        )


def test_atomic_commit_can_repair_score_only_without_retraining(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    score_root = tmp_path / "scores"
    commit_root = tmp_path / "commits"
    score_day = date(2024, 1, 5)
    fingerprint = "r3-test-fingerprint"
    checkpoint = _checkpoint_for_resume_test(
        state_dir=state_dir,
        score_day=score_day,
        parent=None,
        fingerprint=fingerprint,
    )
    frame = _atomic_score_frame(score_day)
    # Simulate the interruption after the atomic score write and before JSON commit.
    score_path = runner.write_scores_for_day_atomic(score_root, frame, score_day=score_day)
    payload = runner._durably_commit_score_day(
        score_output=score_root,
        commit_root=commit_root,
        score_day=score_day,
        score_frame=frame,
        checkpoint_path=checkpoint,
        checkpoint_origin="continuation_checkpoint_v5",
        warm_start_from=None,
        max_train_label_day=date(2024, 1, 2),
        max_validation_label_day=date(2024, 1, 3),
        contract_fingerprint=fingerprint,
        score_provenance="reconstructed_from_checkpoint_current_inputs",
        history=[],
    )
    assert score_path.exists()
    assert runner._commit_path(commit_root, score_day).exists()
    assert payload["score_provenance"] == "reconstructed_from_checkpoint_current_inputs"
    # A second entry validates and returns the durable record; it never overwrites either artifact.
    repeated = runner._durably_commit_score_day(
        score_output=score_root,
        commit_root=commit_root,
        score_day=score_day,
        score_frame=frame,
        checkpoint_path=checkpoint,
        checkpoint_origin="continuation_checkpoint_v5",
        warm_start_from=None,
        max_train_label_day=date(2024, 1, 2),
        max_validation_label_day=date(2024, 1, 3),
        contract_fingerprint=fingerprint,
        score_provenance="reconstructed_from_checkpoint_current_inputs",
        history=[],
    )
    assert repeated == payload


def test_commit_semantics_rejects_training_metadata_on_reconstruction(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="reconstructed commit contains training evidence"):
        runner._validate_daily_commit_semantics(
            checkpoint_origin="continuation_checkpoint_v5",
            score_provenance="reconstructed_from_checkpoint_current_inputs",
            history=[{"epoch": 1}],
            train_days=41,
            val_days=18,
            source=tmp_path / "commit.json",
        )
