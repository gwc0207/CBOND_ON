from __future__ import annotations

from datetime import date
from typing import Any, Protocol


class PanelRepository(Protocol):
    def load(self, *, start: date | str, end: date | str, **kwargs: Any) -> Any: ...


class LabelRepository(Protocol):
    def load(self, *, start: date | str, end: date | str, **kwargs: Any) -> Any: ...

