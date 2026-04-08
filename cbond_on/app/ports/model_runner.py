from __future__ import annotations

from typing import Protocol


class ModelRunnerPort(Protocol):
    def fit(self, *, start: str, end: str, label_cutoff: str | None = None, execution: dict | None = None): ...

    def predict(
        self,
        *,
        start: str,
        end: str,
        artifact,
        label_cutoff: str | None = None,
        execution: dict | None = None,
    ): ...

