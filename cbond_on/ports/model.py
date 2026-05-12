from __future__ import annotations

from typing import Any, Protocol


class ModelScorer(Protocol):
    def score(self, features: Any, *, model_ref: str, **kwargs: Any) -> Any: ...


class ModelTrainer(Protocol):
    def train(self, features: Any, labels: Any, **kwargs: Any) -> Any: ...

