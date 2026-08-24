from __future__ import annotations

from pathlib import Path

from cbond_on.infra.model.adapters import TorchCrossSectionAdapter, build_adapter


def test_cross_section_adapter_is_a_distinct_model_type() -> None:
    adapter = build_adapter("torch_cross_section", model_config_path=Path("research.json5"))
    assert isinstance(adapter, TorchCrossSectionAdapter)
    assert adapter.fit(start="2026-01-01", end="2026-01-02").meta["model_type"] == "torch_cross_section"
