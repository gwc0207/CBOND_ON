from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from harness.tools import r88_trio_score_fusion_b1 as b1


def _codes() -> list[str]:
    return [f"C{index:02d}" for index in range(21)]


def test_neutral_rank_fusion_preserves_target_universe_and_uses_midrank() -> None:
    target = _codes()
    scores = {
        "P6": pd.Series(np.arange(21, dtype=float), index=target),
        "P3": pd.Series(np.arange(20, -1, -1, dtype=float), index=target),
        "Regsim": pd.Series(np.arange(20, dtype=float), index=target[:-1]),
    }
    fused, audit = b1.neutral_rank_fusion(scores, target_codes=target, score_day=pd.Timestamp("2025-01-02"))
    assert fused["code"].tolist() == target
    assert len(fused) == len(target)
    assert audit["regsim_neutral_fill_count"] == 1
    assert audit["p6_neutral_fill_count"] == 0
    assert audit["p3_neutral_fill_count"] == 0
    expected_last = (1.0 + (1.0 / 21.0) + b1.NEUTRAL_PERCENTILE_RANK) / 3.0
    assert math.isclose(float(fused.loc[fused["code"].eq("C20"), "score"].iloc[0]), expected_last, abs_tol=1e-12)


def test_neutral_rank_fusion_is_positive_affine_invariant_on_available_scores() -> None:
    target = _codes()
    base = {
        "P6": pd.Series(np.arange(21, dtype=float), index=target),
        "P3": pd.Series(np.roll(np.arange(21, dtype=float), 5), index=target),
        "Regsim": pd.Series(np.arange(20, -1, -1, dtype=float), index=target),
    }
    original, _ = b1.neutral_rank_fusion(base, target_codes=target, score_day=pd.Timestamp("2025-01-02"))
    transformed = {model: values * (index + 2.0) - index for index, (model, values) in enumerate(base.items())}
    changed, _ = b1.neutral_rank_fusion(transformed, target_codes=target, score_day=pd.Timestamp("2025-01-02"))
    pd.testing.assert_frame_equal(original, changed, check_exact=True)


def test_neutral_rank_fusion_rejects_non_simplex_weights() -> None:
    codes = _codes()
    scores = {model: pd.Series(np.arange(21, dtype=float), index=codes) for model in b1.MODELS}
    with pytest.raises(b1.B1Error, match="sum to one"):
        b1.neutral_rank_fusion(scores, target_codes=codes, score_day=pd.Timestamp("2025-01-02"), weights=[0.4, 0.4, 0.4])
