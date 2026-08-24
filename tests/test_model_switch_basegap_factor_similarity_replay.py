from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from harness.tools.model_switch_basegap_factor_similarity_replay import (
    build_factor_distance_cache,
    combine_neighbour_distances,
    factor_rank_distance,
    make_factor_rank_slice,
    run_factor_similarity_variant,
)
from harness.tools.model_switch_basegap_tuning_replay import MODELS, run_basegap_variant


FACTOR_NAMES = [f"f{index:02d}" for index in range(50)]


def _factor_frame(*, rows: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(rows, len(FACTOR_NAMES))),
        index=pd.Index([f"bond_{index:03d}" for index in range(rows)], name="code"),
        columns=FACTOR_NAMES,
    )


def _panel(days: int = 65) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=days)
    values = {"score_day": dates, "state_available": True, "path": rng.normal(size=days)}
    for offset, model in enumerate(MODELS):
        values[model] = rng.normal(loc=0.0003 * (offset + 1), scale=0.01, size=days)
    values["execution_metadata_complete"] = True
    return pd.DataFrame(values)


def test_factor_rank_distance_is_code_aligned_and_scale_invariant() -> None:
    frame = _factor_frame()
    shuffled = frame.sample(frac=1.0, random_state=17) * 37.0 + 5.0
    same = factor_rank_distance(make_factor_rank_slice(frame), make_factor_rank_slice(shuffled))
    assert same.shared_codes == 100
    assert same.valid_factors == 50
    assert same.median_rho is not None and abs(same.median_rho - 1.0) < 1e-12
    assert same.distance is not None and abs(same.distance) < 1e-12


def test_factor_rank_distance_treats_reversed_ordering_as_dissimilar() -> None:
    frame = _factor_frame()
    reversed_slice = factor_rank_distance(make_factor_rank_slice(frame), make_factor_rank_slice(-frame))
    assert reversed_slice.valid_factors == 50
    assert reversed_slice.median_rho is not None and abs(reversed_slice.median_rho + 1.0) < 1e-12
    assert reversed_slice.distance is not None and abs(reversed_slice.distance - 2.0) < 1e-12


def test_factor_rank_distance_fails_closed_when_shared_universe_is_too_small() -> None:
    frame = _factor_frame(rows=79)
    result = factor_rank_distance(make_factor_rank_slice(frame), make_factor_rank_slice(frame))
    assert result.distance is None
    assert result.valid_factors == 0
    assert result.shared_codes == 79


def test_rank_fusion_preserves_single_distance_endpoints() -> None:
    path = pd.Series([0.1, 0.2, 0.3], index=[10, 11, 12])
    factor = pd.Series([0.3, 0.2, 0.1], index=[10, 11, 12])
    assert_frame_equal(
        combine_neighbour_distances(path, factor, path_weight=1.0, factor_weight=0.0).to_frame(),
        path.to_frame(),
    )
    assert_frame_equal(
        combine_neighbour_distances(path, factor, path_weight=0.0, factor_weight=1.0).to_frame(),
        factor.to_frame(),
    )


def test_factor_distance_cache_uses_strict_predecessors_only() -> None:
    panel = _panel(days=4)
    slices = {pd.Timestamp(day).date(): make_factor_rank_slice(_factor_frame(seed=index)) for index, day in enumerate(panel["score_day"])}
    cache, audit = build_factor_distance_cache(panel, path_features=("path",), factor_slices=slices)
    assert cache[0].empty
    assert not cache[3].empty
    assert (pd.to_datetime(audit["history_day"]) < pd.to_datetime(audit["score_day"])).all()


def test_path_only_variant_exactly_matches_stage_a_in_memory_contract() -> None:
    panel = _panel(days=65)
    direct = run_basegap_variant(
        panel,
        feature_cols=("path",),
        lookback_days=240,
        nearest_k=60,
        min_periods=60,
        metric="trim20_lcb10",
        variant="stage_a_reference",
    )
    replay = run_factor_similarity_variant(
        panel,
        path_features=("path",),
        factor_cache={},
        variant="P0_path44_only",
        path_weight=1.0,
        factor_weight=0.0,
    )
    fields = [
        "score_day",
        "basegap_selected_name",
        "basegap_reason",
        "basegap_history_days",
        "basegap_history_end",
        "basegap_score_gap",
        *(f"basegap_score_{model}" for model in MODELS),
    ]
    assert_frame_equal(
        replay[fields].reset_index(drop=True),
        direct[fields].reset_index(drop=True),
        check_dtype=False,
        check_like=False,
    )
