from __future__ import annotations

from pathlib import Path

import pytest

import cbond_on.core.config as config


def _reset_path_env(monkeypatch) -> None:
    for name in (
        "CBOND_ON_RUNTIME_ROOT",
        "CBOND_ON_PATHS_PROFILE",
        "CBOND_ON_RAW_ROOT",
        "CBOND_ON_CLEAN_ROOT",
        "CBOND_ON_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    # Mark this as an explicit profile so the supplied scratch results root is
    # used even when the test runtime has a normal local D:/cbond_on tree.
    monkeypatch.setenv("CBOND_ON_PATHS_CONFIG", "research_profile.json5")


def test_read_only_input_roots_keep_model_inputs_outside_scratch(monkeypatch, tmp_path: Path) -> None:
    _reset_path_env(monkeypatch)
    scratch_runtime = tmp_path / "scratch_runtime"
    input_runtime = tmp_path / "production_inputs"
    resolved = config._apply_runtime_paths_profile(
        {
            "raw_data_root": str(tmp_path / "raw"),
            "clean_data_root": str(tmp_path / "clean"),
            "results_root": str(scratch_runtime / "results"),
            "read_only_input_roots": {
                "panel_data_root": str(input_runtime / "panel_data"),
                "label_data_root": str(input_runtime / "label_data"),
                "factor_data_root": str(input_runtime / "factor_data"),
            },
        }
    )

    assert resolved["panel_data_root"] == (input_runtime / "panel_data").as_posix()
    assert resolved["label_data_root"] == (input_runtime / "label_data").as_posix()
    assert resolved["factor_data_root"] == (input_runtime / "factor_data").as_posix()
    assert resolved["results_root"] == (scratch_runtime / "results").as_posix()
    assert resolved["model_root"] == (scratch_runtime / "results" / "models").as_posix()
    assert resolved["score_root"] == (scratch_runtime / "results" / "scores").as_posix()
    assert resolved["logs_root"] == (scratch_runtime / "logs").as_posix()


def test_legacy_profile_keeps_runtime_derived_input_roots(monkeypatch, tmp_path: Path) -> None:
    _reset_path_env(monkeypatch)
    scratch_runtime = tmp_path / "scratch_runtime"
    resolved = config._apply_runtime_paths_profile(
        {
            "raw_data_root": str(tmp_path / "raw"),
            "clean_data_root": str(tmp_path / "clean"),
            "results_root": str(scratch_runtime / "results"),
        }
    )

    assert resolved["panel_data_root"] == (scratch_runtime / "panel_data").as_posix()
    assert resolved["label_data_root"] == (scratch_runtime / "label_data").as_posix()
    assert resolved["factor_data_root"] == (scratch_runtime / "factor_data").as_posix()


def test_read_only_input_roots_reject_partial_profile(monkeypatch, tmp_path: Path) -> None:
    _reset_path_env(monkeypatch)
    with pytest.raises(ValueError, match="must declare"):
        config._apply_runtime_paths_profile(
            {
                "raw_data_root": str(tmp_path / "raw"),
                "clean_data_root": str(tmp_path / "clean"),
                "results_root": str(tmp_path / "scratch" / "results"),
                "read_only_input_roots": {"factor_data_root": str(tmp_path / "factors")},
            }
        )


@pytest.mark.parametrize(
    ("env_name", "bad_value"),
    [
        ("CBOND_ON_RUNTIME_ROOT", "C:/unrelated_runtime"),
        ("CBOND_ON_RAW_ROOT", "C:/unrelated_raw"),
        ("CBOND_ON_CLEAN_ROOT", "C:/unrelated_clean"),
    ],
)
def test_read_only_input_roots_reject_conflicting_path_override(
    monkeypatch,
    tmp_path: Path,
    env_name: str,
    bad_value: str,
) -> None:
    _reset_path_env(monkeypatch)
    monkeypatch.setenv(env_name, bad_value)

    with pytest.raises(ValueError, match=env_name):
        config._apply_runtime_paths_profile(
            {
                "raw_data_root": str(tmp_path / "raw"),
                "clean_data_root": str(tmp_path / "clean"),
                "results_root": str(tmp_path / "scratch_runtime" / "results"),
                "read_only_input_roots": {
                    "panel_data_root": str(tmp_path / "production_inputs" / "panel_data"),
                    "label_data_root": str(tmp_path / "production_inputs" / "label_data"),
                    "factor_data_root": str(tmp_path / "production_inputs" / "factor_data"),
                },
            }
        )


def test_read_only_input_roots_allow_matching_path_overrides(monkeypatch, tmp_path: Path) -> None:
    _reset_path_env(monkeypatch)
    raw_root = tmp_path / "raw"
    clean_root = tmp_path / "clean"
    scratch_runtime = tmp_path / "scratch_runtime"
    monkeypatch.setenv("CBOND_ON_RUNTIME_ROOT", str(scratch_runtime))
    monkeypatch.setenv("CBOND_ON_RAW_ROOT", str(raw_root))
    monkeypatch.setenv("CBOND_ON_CLEAN_ROOT", str(clean_root))

    resolved = config._apply_runtime_paths_profile(
        {
            "raw_data_root": str(raw_root),
            "clean_data_root": str(clean_root),
            "results_root": str(scratch_runtime / "results"),
            "read_only_input_roots": {
                "panel_data_root": str(tmp_path / "production_inputs" / "panel_data"),
                "label_data_root": str(tmp_path / "production_inputs" / "label_data"),
                "factor_data_root": str(tmp_path / "production_inputs" / "factor_data"),
            },
        }
    )

    assert resolved["results_root"] == (scratch_runtime / "results").as_posix()
