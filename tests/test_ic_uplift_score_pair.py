from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import harness.tools.ic_uplift_score_pair as score_pair


def _write_scores(root: Path, day: str, scores: np.ndarray) -> None:
    path = root / day[:7] / f"{day}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": [day] * len(scores),
            "code": [f"110{i:03d}.SH" for i in range(len(scores))],
            "score": scores,
        }
    ).to_csv(path, index=False)


def _write_label(root: Path, day: str, values: np.ndarray) -> None:
    path = root / day[:7] / f"{day.replace('-', '')}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "code": [f"110{i:03d}.SH" for i in range(len(values))],
            "trade_time": [pd.Timestamp(f"{day} 14:42:00")] * len(values),
            "y": values,
        }
    ).to_parquet(path, index=False)


def test_score_pair_audit_freezes_universe_before_evaluation(tmp_path: Path) -> None:
    day = "2026-01-05"
    values = np.linspace(-1.0, 1.0, 30)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    label_root = tmp_path / "labels"
    output_root = tmp_path / "pair"
    _write_scores(baseline_root, day, values)
    _write_scores(candidate_root, day, np.linspace(1.0, -1.0, 31))

    score_pair.audit(
        Namespace(
            baseline_score_root=str(baseline_root),
            candidate_score_root=str(candidate_root),
            output_root=str(output_root),
            start=day,
            end=day,
            validation_start=day,
            baseline_name="regsim",
            candidate_name="candidate",
        )
    )

    audit_manifest = json.loads((output_root / "score_audit_manifest.json").read_text(encoding="utf-8"))
    assert audit_manifest["schema_version"] == 2
    assert audit_manifest["label_access"] == "none in audit stage"
    assert audit_manifest["integrity"]["hash_algorithm"] == "sha256"
    assert audit_manifest["integrity"]["source_score_files"]["baseline"][day]["sha256"]
    assert audit_manifest["integrity"]["source_score_files"]["candidate"][day]["sha256"]
    assert audit_manifest["frozen_score_pairs"]["sha256"]
    audit = pd.read_csv(output_root / "score_universe_audit.csv")
    assert audit["status"].tolist() == ["ok_candidate_superset"]
    frozen = pd.read_csv(output_root / "frozen_pair_scores.csv")
    assert len(frozen) == 30

    _write_label(label_root, day, values)
    score_pair.evaluate(Namespace(output_root=str(output_root), label_root=str(label_root)))

    daily = pd.read_csv(output_root / "daily_metrics.csv")
    assert set(daily["prediction"]) == {"regsim", "candidate"}
    assert set(daily["n"]) == {30}
    candidate_ic = daily.loc[daily["prediction"] == "candidate", "pearson_ic"].iloc[0]
    assert candidate_ic == pytest.approx(-1.0)
    paired = pd.read_csv(output_root / "paired_vs_baseline.csv")
    assert set(paired["metric"]) == {"pearson_ic", "rank_ic", "top20_mean_y"}


def test_score_pair_evaluate_fails_closed_on_source_score_drift_before_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = "2026-01-05"
    values = np.linspace(-1.0, 1.0, 30)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    output_root = tmp_path / "pair"
    _write_scores(baseline_root, day, values)
    _write_scores(candidate_root, day, values[::-1])
    score_pair.audit(
        Namespace(
            baseline_score_root=str(baseline_root),
            candidate_score_root=str(candidate_root),
            output_root=str(output_root),
            start=day,
            end=day,
            validation_start=day,
            baseline_name="regsim",
            candidate_name="candidate",
        )
    )

    _write_scores(candidate_root, day, values)
    monkeypatch.setattr(score_pair, "_read_label", lambda *_args: pytest.fail("label should not be read"))
    with pytest.raises(ValueError, match="candidate source score"):
        score_pair.evaluate(Namespace(output_root=str(output_root), label_root="unused"))


def test_score_pair_evaluate_fails_closed_on_frozen_pair_drift_before_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = "2026-01-05"
    values = np.linspace(-1.0, 1.0, 30)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    output_root = tmp_path / "pair"
    _write_scores(baseline_root, day, values)
    _write_scores(candidate_root, day, values[::-1])
    score_pair.audit(
        Namespace(
            baseline_score_root=str(baseline_root),
            candidate_score_root=str(candidate_root),
            output_root=str(output_root),
            start=day,
            end=day,
            validation_start=day,
            baseline_name="regsim",
            candidate_name="candidate",
        )
    )

    frozen_path = output_root / "frozen_pair_scores.csv"
    frozen_path.write_text(frozen_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(score_pair, "_read_label", lambda *_args: pytest.fail("label should not be read"))
    with pytest.raises(ValueError, match="frozen pair score artifact"):
        score_pair.evaluate(Namespace(output_root=str(output_root), label_root="unused"))


def test_score_pair_evaluate_refuses_to_open_labels_before_audit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(score_pair, "_read_label", lambda *_args: pytest.fail("label should not be read"))

    with pytest.raises(FileNotFoundError):
        score_pair.evaluate(Namespace(output_root=str(tmp_path), label_root="unused"))


def test_score_pair_audit_applies_existing_fixed_pool_before_pairing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = "2026-01-05"
    values = np.linspace(-1.0, 1.0, 30)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    output_root = tmp_path / "pair"
    raw_root = tmp_path / "raw"
    pool_path = raw_root / "quant_factor_dev__researcher_xuvb__o_0005" / "2026-01" / "20260102.parquet"
    pool_path.parent.mkdir(parents=True)
    pool_path.write_bytes(b"original fixed pool snapshot")
    pool_evidence = score_pair._file_evidence(pool_path)
    _write_scores(baseline_root, day, values)
    _write_scores(candidate_root, day, values[:29])
    allowed = {f"110{i:03d}.SH" for i in range(29)}
    allowlist_hash = score_pair._canonical_code_set_sha256(allowed)

    def fake_pool(*, raw_data_root, score_days):
        assert raw_data_root == str(raw_root)
        assert list(score_days) == [day]
        return (
            {day: allowed},
            pd.DataFrame(
                [
                    {
                        "score_day": day,
                        "pool_day_expected": "2026-01-02",
                        "pool_day_used": "2026-01-02",
                        "pool_codes": 29,
                        "fallback_no_filter": False,
                        "allowlist_codes_sha256": allowlist_hash,
                        "pool_file_path": pool_evidence["path"],
                        "pool_file_bytes": pool_evidence["bytes"],
                        "pool_file_sha256": pool_evidence["sha256"],
                    }
                ]
            ),
            {
                "enabled": True,
                "kind": "existing_tminus1_o0005_allowlist",
                "raw_data_root": str(raw_root),
                "label_access": "none",
            },
        )

    monkeypatch.setattr(score_pair, "_fixed_pool_codes_by_day", fake_pool)
    score_pair.audit(
        Namespace(
            baseline_score_root=str(baseline_root),
            candidate_score_root=str(candidate_root),
            output_root=str(output_root),
            start=day,
            end=day,
            validation_start=day,
            baseline_name="regsim",
            candidate_name="candidate",
            fixed_pool_raw_root=str(raw_root),
        )
    )

    audit = pd.read_csv(output_root / "score_universe_audit.csv")
    assert audit["baseline_raw_codes"].tolist() == [30]
    assert audit["baseline_codes"].tolist() == [29]
    assert audit["status"].tolist() == ["ok_exact_universe"]
    assert (output_root / "fixed_pool_audit.csv").is_file()
    fixed_codes = pd.read_csv(output_root / "fixed_pool_codes.csv")
    assert len(fixed_codes) == 29
    assert score_pair._canonical_code_set_sha256({"a", "b"}) == score_pair._canonical_code_set_sha256({"b", "a"})

    pool_path.write_bytes(b"mutated fixed pool snapshot")
    monkeypatch.setattr(score_pair, "_read_label", lambda *_args: pytest.fail("label should not be read"))
    with pytest.raises(ValueError, match="T-1 pool parquet"):
        score_pair.evaluate(Namespace(output_root=str(output_root), label_root="unused"))
