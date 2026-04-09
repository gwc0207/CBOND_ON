from __future__ import annotations

from typing import Sequence

import pandas as pd

from cbond_on.infra.factors.rust_shm_backend import build_factor_frame_rust_shm
from cbond_on.domain.factors.spec import FactorSpec


def compute(
    panel: pd.DataFrame,
    specs: Sequence[FactorSpec],
    *,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    compute_backend_params: dict | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    return build_factor_frame_rust_shm(
        panel,
        specs,
        stock_panel=stock_panel,
        bond_stock_map=bond_stock_map,
        compute_backend_params=compute_backend_params,
        workers=workers,
    )


