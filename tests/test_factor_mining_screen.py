from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import harness.tools.factor_mining_screen as screen


def _codes(count: int = 30) -> list[str]:
    return [f"110{index:03d}.SH" for index in range(count)]


def _write_factor_day(root: Path, day: str, values: dict[str, np.ndarray]) -> None:
    path = root / "factors" / "T1430" / day[:7] / f"{day.replace('-', '')}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "dt": [pd.Timestamp(f"{day} 14:30:00")] * len(_codes()),
            "code": _codes(),
            **values,
        }
    ).set_index(["dt", "code"])
    frame.to_parquet(path)


def _write_label(root: Path, day: str, y: np.ndarray) -> None:
    path = root / day[:7] / f"{day.replace('-', '')}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "code": _codes(),
            "trade_time": [pd.Timestamp(f"{day} 14:42:00")] * len(y),
            "y": y,
        }
    ).to_parquet(path, index=False)


def _args(
    *,
    factor_root: Path,
    label_root: Path,
    raw_data_root: Path,
    catalog: Path,
    output_dir: Path,
    end: str,
) -> Namespace:
    return Namespace(
        factor_root=str(factor_root),
        label_root=str(label_root),
        raw_data_root=str(raw_data_root),
        factor_catalog=str(catalog),
        output_dir=str(output_dir),
        panel_name="T1430",
        start="2025-01-01",
        end=end,
        top_k=20,
        min_cross_section=30,
        min_valid_days=screen._MIN_VALID_DAYS,
        min_valid_days_per_partition=screen._MIN_VALID_DAYS_PER_PARTITION,
        min_redundancy_days=screen._MIN_REDUNDANCY_DAYS,
        ic_threshold=0.02,
        within_family_threshold=0.80,
        cross_family_threshold=0.70,
        max_selected=100,
    )


def _fake_fixed_pool(codes: list[str]):
    def fake(*, raw_data_root: Path, score_days):
        days = sorted(str(day) for day in score_days)
        allowed = set(codes)
        return (
            {day: allowed for day in days},
            pd.DataFrame(
                {
                    "score_day": days,
                    "pool_day_expected": ["2024-12-31"] * len(days),
                    "pool_day_used": ["2024-12-31"] * len(days),
                    "pool_codes": [len(allowed)] * len(days),
                    "fallback_no_filter": [False] * len(days),
                    "allowlist_codes_sha256": [screen._canonical_code_set_sha256(allowed)] * len(days),
                    "pool_file_path": [str(raw_data_root / "pool.parquet")] * len(days),
                    "pool_file_bytes": [1] * len(days),
                    "pool_file_sha256": ["test"] * len(days),
                }
            ),
            {
                "enabled": True,
                "kind": "existing_tminus1_o0005_allowlist",
                "fallback_policy": "fail_closed_no_no_filter_fallback",
            },
        )

    return fake


def _assert_metric_dicts_match(
    actual: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert list(actual) == list(expected)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, (float, np.floating)):
            assert actual_value is not None, key
            assert float(actual_value) == pytest.approx(float(expected_value), abs=1e-12), key
        else:
            assert actual_value == expected_value, key


def test_bulk_daily_metrics_match_scalar_reference_for_alignment_masks_and_ties() -> None:
    pool_codes = {f"110{index:03d}.SH" for index in range(1, 9)}
    source_codes = [
        "110008.SH",
        "110007.SH",
        "110006.SH",
        "110005.SH",
        "110004.SH",
        "110003.SH",
        "110002.SH",
        "110001.SH",
        "110999.SH",
    ]
    y_by_code = {f"110{index:03d}.SH": float(index * 10) for index in range(1, 8)}
    label_pool = pd.DataFrame(
        {
            "code": [
                "110004.SH",
                "110002.SH",
                "110006.SH",
                "110001.SH",
                "110007.SH",
                "110003.SH",
                "110005.SH",
            ],
            "y": [
                y_by_code["110004.SH"],
                y_by_code["110002.SH"],
                y_by_code["110006.SH"],
                y_by_code["110001.SH"],
                y_by_code["110007.SH"],
                y_by_code["110003.SH"],
                y_by_code["110005.SH"],
            ],
        }
    )
    factor_frame = pd.DataFrame(
        {
            "code": source_codes,
            "factor_good": ["8", "7", "6", "5", "4", "3", "2", "1", "999"],
            "factor_ties": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 999.0],
            "factor_sparse": [np.nan, np.nan, 3.0, np.nan, 2.0, np.nan, np.nan, np.nan, 999.0],
            "factor_constant": [1.0] * len(source_codes),
            "factor_inf": [8.0, np.inf, 6.0, -np.inf, 4.0, 3.0, 2.0, 1.0, 999.0],
        }
    )
    catalog = pd.DataFrame(
        {
            "family": ["alpha", "alpha", "beta", "beta", "gamma", "gamma"],
            "factor": [
                "factor_good",
                "factor_ties",
                "factor_sparse",
                "factor_constant",
                "factor_inf",
                "factor_missing",
            ],
        }
    )

    actual_rows = screen._daily_factor_metrics(
        day="2025-01-02",
        catalog=catalog,
        factor_frame=factor_frame,
        label_pool=label_pool,
        pool_codes=pool_codes,
        min_cross_section=3,
        top_k=3,
    )
    expected_rows = [
        screen._daily_factor_metric(
            day="2025-01-02",
            factor=str(row.factor),
            family=str(row.family),
            factor_frame=factor_frame,
            label_pool=label_pool,
            pool_codes=pool_codes,
            min_cross_section=3,
            top_k=3,
        )
        for row in catalog.itertuples(index=False)
    ]

    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        _assert_metric_dicts_match(actual, expected)

    actual_by_factor = {str(row["factor"]): row for row in actual_rows}
    assert actual_by_factor["factor_ties"]["top20_mean_y"] == pytest.approx(20.0)
    assert actual_by_factor["factor_sparse"]["status"] == "insufficient_finite_rows"
    assert actual_by_factor["factor_constant"]["status"] == "constant_factor"
    assert actual_by_factor["factor_missing"]["status"] == "missing_factor_column"


def test_bulk_daily_metrics_matches_scalar_reference_for_constant_label() -> None:
    codes = ["110001.SH", "110002.SH", "110003.SH", "110004.SH"]
    factor_frame = pd.DataFrame(
        {
            "code": list(reversed(codes)),
            "factor_a": [4.0, 3.0, 2.0, 1.0],
        }
    )
    label_pool = pd.DataFrame({"code": [codes[2], codes[0], codes[3], codes[1]], "y": [1.0] * 4})
    catalog = pd.DataFrame({"family": ["family"], "factor": ["factor_a"]})

    actual = screen._daily_factor_metrics(
        day="2025-01-02",
        catalog=catalog,
        factor_frame=factor_frame,
        label_pool=label_pool,
        pool_codes=set(codes),
        min_cross_section=3,
        top_k=3,
    )[0]
    expected = screen._daily_factor_metric(
        day="2025-01-02",
        factor="factor_a",
        family="family",
        factor_frame=factor_frame,
        label_pool=label_pool,
        pool_codes=set(codes),
        min_cross_section=3,
        top_k=3,
    )

    _assert_metric_dicts_match(actual, expected)
    assert actual["status"] == "constant_label"
    assert actual["valid_for_ic"] is False


def test_screen_uses_fixed_pool_chronology_and_redundancy_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_root = tmp_path / "scratch_factor_store"
    label_root = tmp_path / "labels"
    raw_data_root = tmp_path / "raw"
    label_root.mkdir()
    raw_data_root.mkdir()
    catalog = tmp_path / "families.json"
    output_dir = tmp_path / "screen_output"
    catalog.write_text(
        json.dumps(
            {
                "price_path": ["factor_a", "factor_a_duplicate"],
                "liquidity": ["factor_b", "factor_c"],
            }
        ),
        encoding="utf-8",
    )
    y = np.linspace(-1.0, 1.0, len(_codes()))
    residual = np.cos(np.linspace(0.0, 8.0 * np.pi, len(_codes())))
    residual = residual - residual.mean()
    residual = residual - y * float(np.dot(residual, y) / np.dot(y, y))
    residual = residual / np.std(residual)
    low_redundancy_signal = 0.10 * y + residual
    days = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    for day in days:
        _write_factor_day(
            factor_root,
            day,
            {
                "factor_a": y,
                "factor_a_duplicate": 2.0 * y,
                "factor_b": -y,
                "factor_c": low_redundancy_signal,
            },
        )
        _write_label(label_root, day, y)
    monkeypatch.setattr(screen, "_expected_score_days", lambda **_kwargs: days)
    monkeypatch.setattr(screen, "_fixed_pool_codes_by_day", _fake_fixed_pool(_codes()))
    # The fixture has five score days.  Production code keeps the fixed
    # 250/50/200 admission contract; this test only exercises the mechanics
    # with an explicitly local reduced contract.
    monkeypatch.setattr(screen, "_MIN_VALID_DAYS", 1)
    monkeypatch.setattr(screen, "_MIN_VALID_DAYS_PER_PARTITION", 1)
    monkeypatch.setattr(screen, "_MIN_REDUNDANCY_DAYS", 1)
    monkeypatch.setattr(
        screen,
        "_daily_factor_metric",
        lambda **_kwargs: pytest.fail("run_screen must use the bulk daily metric path"),
    )

    output = screen.run_screen(
        _args(
            factor_root=factor_root,
            label_root=label_root,
            raw_data_root=raw_data_root,
            catalog=catalog,
            output_dir=output_dir,
            end=days[-1],
        )
    )

    assert output == output_dir.resolve()
    accepted = pd.read_csv(output / "accepted_factors.csv")
    assert set(accepted["factor"]) == {"factor_a", "factor_c"}
    rejected = pd.read_csv(output / "rejected_factors.csv").set_index("factor")
    assert "within_family" in rejected.loc["factor_a_duplicate", "selection_reason"]
    assert "cross_family" in rejected.loc["factor_b", "selection_reason"]

    factor_a = pd.read_csv(output / "factor_summary_metrics.csv").query(
        "factor == 'factor_a' and partition == 'overall'"
    ).iloc[0]
    assert factor_a["mean_pearson_ic"] == pytest.approx(1.0)
    assert factor_a["mean_rank_ic"] == pytest.approx(1.0)
    assert factor_a["mean_coverage"] == pytest.approx(1.0)
    assert factor_a["valid_days"] == 5
    daily = pd.read_csv(output / "daily_factor_metrics.csv")
    assert len(daily) == len(days) * 4
    assert daily.columns.tolist() == [
        "score_day",
        "factor",
        "family",
        "factor_source_rows",
        "pool_codes",
        "pool_label_rows",
        "pool_factor_rows",
        "factor_label_rows",
        "finite_rows",
        "coverage",
        "finite_rate",
        "n",
        "pearson_ic",
        "rank_ic",
        "top20_mean_y",
        "valid_for_ic",
        "status",
        "partition",
    ]
    calendar = pd.read_csv(output / "evaluation_calendar.csv")
    assert calendar["partition"].value_counts().to_dict() == {
        "discovery": 3,
        "validation": 1,
        "holdout": 1,
    }
    pairs = pd.read_csv(output / "factor_pair_redundancy.csv")
    assert set(pairs["relation"]) == {"within_family", "cross_family"}
    manifest = json.loads((output / "screen_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fixed_universe"]["fallback_policy"] == "fail_closed_no_no_filter_fallback"
    assert manifest["summary"]["accepted_count"] == 2


def test_screen_refuses_pool_fallback_before_any_label_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_root = tmp_path / "scratch_factor_store"
    label_root = tmp_path / "labels"
    raw_data_root = tmp_path / "raw"
    label_root.mkdir()
    raw_data_root.mkdir()
    catalog = tmp_path / "families.json"
    output_dir = tmp_path / "screen_output"
    catalog.write_text(json.dumps({"family": ["factor_a"]}), encoding="utf-8")
    _write_factor_day(factor_root, "2025-01-01", {"factor_a": np.linspace(-1.0, 1.0, len(_codes()))})

    config = screen.UpstreamPoolConfig(
        pool_table="quant_factor_dev.researcher_xuvb.o_0005",
        positive_field="factor_value",
        positive_fallback_field="weight",
        positive_threshold=0.0,
        pool_lag_trading_days=1,
        pool_asset="cbond",
    )
    monkeypatch.setattr(screen, "load_upstream_pool_config", lambda: config)
    monkeypatch.setattr(screen, "_expected_score_days", lambda **_kwargs: ["2025-01-01"])
    monkeypatch.setattr(
        screen,
        "resolve_pool_codes_for_trade_day",
        lambda **_kwargs: (
            None,
            {
                "fallback_no_filter": True,
                "pool_day_expected": "2024-12-31",
                "fallback_reason": "missing_pool_day_file",
            },
        ),
    )
    monkeypatch.setattr(screen, "_read_label_1442", lambda **_kwargs: pytest.fail("labels must stay closed"))

    with pytest.raises(RuntimeError, match="refusing no-filter fallback"):
        screen.run_screen(
            _args(
                factor_root=factor_root,
                label_root=label_root,
                raw_data_root=raw_data_root,
                catalog=catalog,
                output_dir=output_dir,
                end="2025-01-01",
            )
        )
    assert not output_dir.exists()


def test_screen_locks_user_requested_threshold_contract(tmp_path: Path) -> None:
    factor_root = tmp_path / "scratch_factor_store"
    label_root = tmp_path / "labels"
    raw_data_root = tmp_path / "raw"
    catalog = tmp_path / "families.json"
    output_dir = tmp_path / "screen_output"
    factor_root.mkdir()
    label_root.mkdir()
    raw_data_root.mkdir()
    catalog.write_text(json.dumps({"family": ["factor_a"]}), encoding="utf-8")
    args = _args(
        factor_root=factor_root,
        label_root=label_root,
        raw_data_root=raw_data_root,
        catalog=catalog,
        output_dir=output_dir,
        end="2025-01-02",
    )
    args.ic_threshold = 0.01

    with pytest.raises(ValueError, match="requires --ic-threshold 0.02"):
        screen._validate_args(args)


@pytest.mark.parametrize(
    ("attribute", "bad_value", "message"),
    [
        ("min_valid_days", 249, "requires --min-valid-days 250"),
        (
            "min_valid_days_per_partition",
            49,
            "requires --min-valid-days-per-partition 50",
        ),
        ("min_redundancy_days", 199, "requires --min-redundancy-days 200"),
    ],
)
def test_screen_locks_validity_and_redundancy_contract(
    tmp_path: Path,
    attribute: str,
    bad_value: int,
    message: str,
) -> None:
    factor_root = tmp_path / "scratch_factor_store"
    label_root = tmp_path / "labels"
    raw_data_root = tmp_path / "raw"
    catalog = tmp_path / "families.json"
    output_dir = tmp_path / "screen_output"
    factor_root.mkdir()
    label_root.mkdir()
    raw_data_root.mkdir()
    catalog.write_text(json.dumps({"family": ["factor_a"]}), encoding="utf-8")
    args = _args(
        factor_root=factor_root,
        label_root=label_root,
        raw_data_root=raw_data_root,
        catalog=catalog,
        output_dir=output_dir,
        end="2025-01-02",
    )
    setattr(args, attribute, bad_value)

    with pytest.raises(ValueError, match=message):
        screen._validate_args(args)


def test_screen_refuses_missing_factor_day_from_frozen_calendar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_root = tmp_path / "scratch_factor_store"
    label_root = tmp_path / "labels"
    raw_data_root = tmp_path / "raw"
    label_root.mkdir()
    raw_data_root.mkdir()
    catalog = tmp_path / "families.json"
    output_dir = tmp_path / "screen_output"
    catalog.write_text(json.dumps({"family": ["factor_a"]}), encoding="utf-8")
    _write_factor_day(factor_root, "2025-01-02", {"factor_a": np.linspace(-1.0, 1.0, len(_codes()))})
    monkeypatch.setattr(
        screen,
        "_expected_score_days",
        lambda **_kwargs: ["2025-01-02", "2025-01-03"],
    )
    monkeypatch.setattr(
        screen,
        "_fixed_pool_codes_by_day",
        lambda **_kwargs: pytest.fail("pool resolution must follow coverage validation"),
    )

    with pytest.raises(RuntimeError, match="FactorStore score-day coverage"):
        screen.run_screen(
            _args(
                factor_root=factor_root,
                label_root=label_root,
                raw_data_root=raw_data_root,
                catalog=catalog,
                output_dir=output_dir,
                end="2025-01-03",
            )
        )


def test_screen_requires_exact_factor_and_label_timestamps(tmp_path: Path) -> None:
    factor_path = tmp_path / "factor.parquet"
    label_root = tmp_path / "labels"
    day = "2025-01-02"
    pd.DataFrame(
        {
            "dt": [pd.Timestamp(f"{day} 14:29:00")] * len(_codes()),
            "code": _codes(),
            "factor_a": np.linspace(-1.0, 1.0, len(_codes())),
        }
    ).set_index(["dt", "code"]).to_parquet(factor_path)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        screen._read_factor_frame(path=factor_path, day=day)

    _write_label(label_root, day, np.linspace(-1.0, 1.0, len(_codes())))
    label_path = label_root / "2025-01" / "20250102.parquet"
    label = pd.read_parquet(label_path)
    label["trade_time"] = pd.Timestamp("2025-01-03 14:42:00")
    label.to_parquet(label_path, index=False)
    filtered, _evidence, status = screen._read_label_1442(label_root=label_root, day=day)
    assert filtered is None
    assert status == "empty_1442_label"
