from __future__ import annotations

from datetime import date

import pandas as pd

from cbond_on.infra.model.impl.lgbm.trainer import SplitData
from cbond_on.infra.model.runners.train_lgbm import (
    _nested_factor_selection_contribution,
    _resolve_strict_warm_start_config,
    _resolve_nested_factor_selection_config,
    _verify_warm_start_feature_schema,
)


def _split() -> SplitData:
    rows: list[dict[str, float]] = []
    labels: list[float] = []
    dates: list[pd.Timestamp] = []
    codes: list[str] = []
    for day_index, day in enumerate(pd.date_range("2026-01-01", periods=30, freq="D")):
        for rank in range(12):
            label = float(rank)
            rows.append(
                {
                    "strong": label,
                    "strong_copy": label * 2.0,
                    "other": float((rank * 7 + day_index) % 12),
                    "weak": float((rank * 5 + day_index * 3) % 12),
                }
            )
            labels.append(label)
            dates.append(day)
            codes.append(f"113{day_index:03d}{rank:02d}.SZ")
    return SplitData(
        x=pd.DataFrame(rows),
        y=pd.Series(labels, dtype=float),
        dt=pd.Series(dates),
        code=pd.Series(codes),
    )


def test_nested_selector_uses_only_fit_history_and_applies_diversity() -> None:
    selector = _resolve_nested_factor_selection_config(
        {
            "feature_engineering": {
                "nested_factor_selection": {
                    "enabled": True,
                    "score": "abs_icir",
                    "top_k": 2,
                    "min_selected": 2,
                    "lookback_days": 30,
                    "min_days": 20,
                    "min_samples_per_day": 8,
                    "correlation_max_abs": 0.8,
                    "candidate_features": ["strong", "strong_copy", "other", "weak"],
                }
            }
        }
    )

    contribution, rows, summary = _nested_factor_selection_contribution(
        {},
        selector,
        _split(),
        target_day=date(2026, 2, 1),
        feature_cols=["strong", "strong_copy", "other", "weak"],
    )

    selected = {row["feature"] for row in rows if bool(row["selected"])}
    assert "strong" in selected
    assert "strong_copy" not in selected
    assert len(selected) == 2
    assert summary["source_split"] == "fit_train_only"
    assert summary["latest_train_day"] == date(2026, 1, 30)
    assert contribution["values"]["strong"] == 1.0
    assert contribution["values"]["strong_copy"] == 0.0


def test_nested_selector_rejects_non_prior_fit_history() -> None:
    selector = _resolve_nested_factor_selection_config(
        {
            "feature_engineering": {
                "nested_factor_selection": {
                    "enabled": True,
                    "top_k": 1,
                    "min_selected": 1,
                    "lookback_days": 30,
                    "min_days": 20,
                    "min_samples_per_day": 8,
                }
            }
        }
    )
    split = _split()
    try:
        _nested_factor_selection_contribution(
            {},
            selector,
            split,
            target_day=date(2026, 1, 30),
            feature_cols=list(split.x.columns),
        )
    except RuntimeError as exc:
        assert "strictly before score day" in str(exc)
    else:  # pragma: no cover - protects the PIT contract above.
        raise AssertionError("expected non-prior fitting history to be rejected")


def test_warm_start_feature_schema_rejects_reordered_columns(tmp_path) -> None:
    state_dir = tmp_path / "state"
    _verify_warm_start_feature_schema(state_dir, ["factor_a", "factor_b"])
    _verify_warm_start_feature_schema(state_dir, ["factor_a", "factor_b"])
    try:
        _verify_warm_start_feature_schema(state_dir, ["factor_b", "factor_a"])
    except RuntimeError as exc:
        assert "feature schema mismatch" in str(exc)
    else:  # pragma: no cover - protects warm-start position semantics.
        raise AssertionError("expected reordered feature schema to be rejected")


def test_strict_warm_start_requires_an_explicit_seed_day_for_report_mode() -> None:
    resolved = _resolve_strict_warm_start_config(
        {
            "strict_warm_start": {
                "enabled": True,
                "require_initial_checkpoint": True,
                "initial_checkpoint_day": "2025-12-31",
            }
        }
    )

    assert resolved["enabled"] is True
    assert resolved["require_initial_checkpoint"] is True
    assert resolved["initial_checkpoint_day"] == date(2025, 12, 31)

    try:
        _resolve_strict_warm_start_config(
            {"strict_warm_start": {"enabled": True, "require_initial_checkpoint": True}}
        )
    except ValueError as exc:
        assert "requires initial_checkpoint_day" in str(exc)
    else:  # pragma: no cover - protects report-only warm-start provenance.
        raise AssertionError("expected a strict report chain without its seed date to be rejected")
