from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
