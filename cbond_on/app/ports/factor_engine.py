from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

from cbond_on.domain.factors.spec import FactorSpec


class FactorEnginePort(Protocol):
    def compute(
        self,
        panel: pd.DataFrame,
        specs: Sequence[FactorSpec],
        *,
        stock_panel: pd.DataFrame | None = None,
        bond_stock_map: pd.DataFrame | None = None,
        compute_backend_params: dict | None = None,
    ) -> pd.DataFrame: ...


