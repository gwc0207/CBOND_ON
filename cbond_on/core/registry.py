from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Type


class RegistryError(RuntimeError):
    pass


@dataclass
class _Registry:
    """Runtime registry for executable operators.

    The storage is intentionally shared by the public operator name and its
    legacy factor name below. Existing factor decorators keep using the same
    object and therefore retain their registration behaviour unchanged.
    """

    _items: Dict[str, Type] = field(default_factory=dict)

    def register(self, name: str) -> Callable[[Type], Type]:
        def _decorator(cls: Type) -> Type:
            if name in self._items:
                raise RegistryError(f"duplicate operator registration: {name}")
            self._items[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> Type:
        if name not in self._items:
            raise RegistryError(f"operator is not registered: {name}")
        return self._items[name]

    def names(self) -> Iterable[str]:
        return self._items.keys()

    def clear(self) -> None:
        self._items.clear()


# `FactorRegistry` remains a strict compatibility alias: it is not a wrapper,
# proxy, or copied mapping. Imports through either public name therefore see
# the identical runtime registration state.
OperatorRegistry = _Registry()
FactorRegistry = OperatorRegistry
FilterRegistry = _Registry()
