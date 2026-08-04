from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from harness.tools.similar60_compose_score_root import compose_score_root


def _write_score(root: Path, day: str, payload: bytes) -> Path:
    path = root / day[:7] / f"{day}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_similar_audit(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compose_uses_baseline_bytes_for_explicit_fallback(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    candidate_first = _write_score(
        candidate,
        "2026-08-03",
        b"trade_date,code,score\n2026-08-03,110001.SH,1.0\n",
    )
    candidate_second = _write_score(
        candidate,
        "2026-08-04",
        b"trade_date,code,score\n2026-08-04,110001.SH,2.0\n",
    )
    baseline_first = _write_score(
        baseline,
        "2026-08-03",
        b"trade_date,code,score\n2026-08-03,110001.SH,11.0\n",
    )
    baseline_second = _write_score(
        baseline,
        "2026-08-04",
        b"trade_date,code,score\n2026-08-04,110001.SH,12.0\n",
    )
    similar = _write_similar_audit(
        tmp_path / "rolling_similar_days.csv",
        [
            {"target_day": "2026-08-03", "reason": "ok"},
            {"target_day": "2026-08-04", "reason": "insufficient_candidates_72_lt_80"},
        ],
    )

    output = compose_score_root(
        candidate_score_root=candidate,
        baseline_score_root=baseline,
        rolling_similar_days_path=similar,
        output_root=tmp_path / "composed",
    )

    ordinary = output / "2026-08" / "2026-08-03.csv"
    fallback = output / "2026-08" / "2026-08-04.csv"
    assert ordinary.read_bytes() == candidate_first.read_bytes()
    assert fallback.read_bytes() == baseline_second.read_bytes()
    assert fallback.read_bytes() != candidate_second.read_bytes()
    audit = pd.read_csv(output / "compose_audit.csv")
    assert audit.loc[audit["score_day"] == "2026-08-03", "selection"].item() == "candidate_similar60"
    assert audit.loc[audit["score_day"] == "2026-08-04", "selection"].item() == "baseline_exact_fallback"
    assert audit.loc[audit["score_day"] == "2026-08-04", "destination_sha256"].item() == _sha256(baseline_second)
    manifest = json.loads((output / "compose_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fallback_days"] == ["2026-08-04"]
    assert manifest["compose_audit"]["sha256"] == _sha256(output / "compose_audit.csv")
    assert _sha256(baseline_first) != _sha256(candidate_first)


def test_compose_rejects_unemitted_fallback(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _write_score(candidate, "2026-08-03", b"trade_date,code,score\n2026-08-03,110001.SH,1.0\n")
    _write_score(baseline, "2026-08-03", b"trade_date,code,score\n2026-08-03,110001.SH,2.0\n")
    similar = _write_similar_audit(
        tmp_path / "rolling_similar_days.csv",
        [{"target_day": "2026-08-04", "reason": "current_state_missing"}],
    )

    with pytest.raises(ValueError, match="has no candidate score file"):
        compose_score_root(
            candidate_score_root=candidate,
            baseline_score_root=baseline,
            rolling_similar_days_path=similar,
            output_root=tmp_path / "composed",
        )
    assert not (tmp_path / "composed").exists()


def test_compose_range_ignores_valid_fallbacks_outside_the_selected_days(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _write_score(candidate, "2026-08-03", b"trade_date,code,score\n2026-08-03,110001.SH,1.0\n")
    _write_score(candidate, "2026-08-04", b"trade_date,code,score\n2026-08-04,110001.SH,2.0\n")
    _write_score(baseline, "2026-08-03", b"trade_date,code,score\n2026-08-03,110001.SH,3.0\n")
    _write_score(baseline, "2026-08-04", b"trade_date,code,score\n2026-08-04,110001.SH,4.0\n")
    similar = _write_similar_audit(
        tmp_path / "rolling_similar_days.csv",
        [
            {"target_day": "2026-08-03", "reason": "ok"},
            {"target_day": "2026-08-04", "reason": "current_state_missing"},
        ],
    )

    output = compose_score_root(
        candidate_score_root=candidate,
        baseline_score_root=baseline,
        rolling_similar_days_path=similar,
        output_root=tmp_path / "composed",
        start="2026-08-03",
        end="2026-08-03",
    )

    assert (output / "2026-08" / "2026-08-03.csv").is_file()
    assert not (output / "2026-08" / "2026-08-04.csv").exists()
    manifest = json.loads((output / "compose_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fallback_days"] == []
