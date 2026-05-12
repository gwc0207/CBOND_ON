from __future__ import annotations

from typing import Protocol, TypeVar


T = TypeVar("T")


class ScoringStep(Protocol[T]):
    def __call__(self) -> T: ...


def run_scoring_step(step: ScoringStep[T]) -> T:
    return step()

