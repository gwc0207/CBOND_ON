from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Type


class RegistryError(RuntimeError):
    pass


@dataclass
class _Registry:
    _items: Dict[str, Type] = field(default_factory=dict)

    def register(self, name: str) -> Callable[[Type], Type]:
        def _decorator(cls: Type) -> Type:
            if name in self._items:
                raise RegistryError(f"重复注册: {name}")
            self._items[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> Type:
        if name not in self._items:
            raise RegistryError(f"未找到注册项: {name}")
        return self._items[name]

    def names(self) -> Iterable[str]:
        return self._items.keys()

    def clear(self) -> None:
        self._items.clear()


FactorRegistry = _Registry()
FilterRegistry = _Registry()
