import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_vwap_deviation_v1")
class T1430VwapDeviationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            last = group["last"]
            amount = group["amount"]
            volume = group["volume"]
            
            vwap = amount / volume.replace(0, 1e-8)
            dev = (last - vwap) / (vwap.replace(0, 1e-8))
            
            return dev.iloc[-1]
            
        return _group_scalar(panel, _calc)