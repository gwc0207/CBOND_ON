from __future__ import annotations

from cbond_on.app.usecases import model_score_runtime
from cbond_on.bootstrap.research import load_model_score_config


def test_target_zscore_registry_resolves_to_isolated_lgbm_candidate(monkeypatch) -> None:
    config_name = "score/model/model_score_target_zscore_day_equal_day_mass_20260731"
    cfg = load_model_score_config(config_name)
    model_id = "research_regsim_target_zscore_day_equal_day_mass_20260731"
    calls: list[tuple[str, str, str]] = []

    class _Adapter:
        def fit(self, *, start: str, end: str, label_cutoff: str | None, execution: dict):
            _ = (label_cutoff, execution)
            calls.append(("fit", start, end))
            return object()

        def predict(self, *, start: str, end: str, artifact: object, label_cutoff: str | None, execution: dict):
            _ = (artifact, label_cutoff, execution)
            calls.append(("predict", start, end))

    monkeypatch.setattr(model_score_runtime, "build_adapter", lambda *_args, **_kwargs: _Adapter())

    result = model_score_runtime.run(cfg=cfg)

    assert cfg["model_id"] == model_id
    assert set(cfg["models"]) == {model_id}
    assert result["model_id"] == model_id
    assert result["model_type"] == "lgbm"
    assert "results/experiments/ic_uplift_oos_20260731" in result["score_output"].replace("\\", "/")
    assert "results/scores/live" not in result["score_output"].replace("\\", "/")
    assert calls == [("fit", "2024-05-08", "2026-05-05"), ("predict", "2024-05-08", "2026-05-05")]
