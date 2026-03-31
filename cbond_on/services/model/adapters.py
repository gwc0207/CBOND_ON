from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelArtifact:
    meta: dict


class ModelAdapter(ABC):
    @abstractmethod
    def fit(
        self,
        *,
        start: str,
        end: str,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> ModelArtifact:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        *,
        start: str,
        end: str,
        artifact: ModelArtifact,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> None:
        raise NotImplementedError

    def evaluate(self, **kwargs) -> dict:
        return {}


class LinearAdapter(ModelAdapter):
    def __init__(self, model_config_path: Path | None = None) -> None:
        self.model_config_path = model_config_path

    def fit(
        self,
        *,
        start: str,
        end: str,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> ModelArtifact:
        _ = label_cutoff
        _ = execution
        return ModelArtifact(meta={"mode": "script", "model_type": "linear"})

    def predict(
        self,
        *,
        start: str,
        end: str,
        artifact: ModelArtifact,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> None:
        _ = artifact
        _ = label_cutoff
        from cbond_on.services.model.runners import train_linear

        train_linear.main(
            config_path=self.model_config_path,
            start=start,
            end=end,
            execution=execution,
        )


class LgbmAdapter(ModelAdapter):
    def __init__(self, model_config_path: Path | None = None) -> None:
        self.model_config_path = model_config_path

    def fit(
        self,
        *,
        start: str,
        end: str,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> ModelArtifact:
        _ = start
        _ = end
        _ = label_cutoff
        _ = execution
        return ModelArtifact(meta={"mode": "script", "model_type": "lgbm"})

    def predict(
        self,
        *,
        start: str,
        end: str,
        artifact: ModelArtifact,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> None:
        _ = artifact
        from cbond_on.services.model.runners import train_lgbm

        train_lgbm.main(
            config_path=self.model_config_path,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            execution=execution,
        )


class LgbmRankerAdapter(ModelAdapter):
    def __init__(self, model_config_path: Path | None = None) -> None:
        self.model_config_path = model_config_path

    def fit(
        self,
        *,
        start: str,
        end: str,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> ModelArtifact:
        _ = start
        _ = end
        _ = label_cutoff
        _ = execution
        return ModelArtifact(meta={"mode": "script", "model_type": "lgbm_ranker"})

    def predict(
        self,
        *,
        start: str,
        end: str,
        artifact: ModelArtifact,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> None:
        _ = artifact
        from cbond_on.services.model.runners import train_lgbm_ranker

        train_lgbm_ranker.main(
            config_path=self.model_config_path,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            execution=execution,
        )


class LobAdapter(ModelAdapter):
    def __init__(self, model_config_path: Path | None = None) -> None:
        self.model_config_path = model_config_path

    def fit(
        self,
        *,
        start: str,
        end: str,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> ModelArtifact:
        _ = start
        _ = end
        _ = label_cutoff
        _ = execution
        return ModelArtifact(meta={"mode": "script", "model_type": "lob"})

    def predict(
        self,
        *,
        start: str,
        end: str,
        artifact: ModelArtifact,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ) -> None:
        _ = artifact
        from cbond_on.services.model.runners import train_lob

        train_lob.main(
            config_path=self.model_config_path,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            execution=execution,
        )


def build_adapter(model_type: str, *, model_config_path: Path | None = None) -> ModelAdapter:
    kind = str(model_type).strip().lower()
    if kind == "linear":
        return LinearAdapter(model_config_path=model_config_path)
    if kind == "lgbm":
        return LgbmAdapter(model_config_path=model_config_path)
    if kind == "lgbm_ranker":
        return LgbmRankerAdapter(model_config_path=model_config_path)
    if kind in {"lob", "lob_st"}:
        return LobAdapter(model_config_path=model_config_path)
    raise ValueError(f"unsupported model_type: {model_type}")
