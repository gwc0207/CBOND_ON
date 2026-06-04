from __future__ import annotations

from cbond_on.app.usecases.ai_factor_factory import FactorCandidateDraft, review_candidate


_BASE_CODE = '''
import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("{key}")
class TestFactor(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        def _calc(group: pd.DataFrame) -> float:
            window_df = slice_window(group.sort_values("trade_time"), 30)
            {body}
        return _group_scalar(panel, _calc)
'''


def _candidate(*, key: str, body: str, fields: list[str]) -> FactorCandidateDraft:
    return FactorCandidateDraft(
        factor_key=key,
        factor_name=key,
        formula=body,
        rationale="test candidate",
        python_code=_BASE_CODE.format(key=key, body=body),
        config_spec={"name": key, "factor": key, "params": {"window_minutes": 30}},
        used_panel_fields=fields,
        time_visibility="only uses T 14:30 panel",
    )


def _error_codes(candidate: FactorCandidateDraft) -> set[str]:
    return {finding.code for finding in review_candidate(candidate) if finding.severity == "error"}


def test_review_rejects_existing_factor_key() -> None:
    candidate = _candidate(
        key="t1430_volume_max_v1",
        body='vol = window_df["volume"]; return float(vol.max())',
        fields=["trade_time", "volume"],
    )

    assert "existing_factor_key" in _error_codes(candidate)


def test_review_rejects_duplicate_volume_formula_family_with_np_call() -> None:
    candidate = _candidate(
        key="t1430_new_volume_peak_np_v1",
        body='vol = window_df["volume"].values.astype(float); return float(np.max(vol))',
        fields=["trade_time", "volume"],
    )

    assert "duplicate_formula_family" in _error_codes(candidate)


def test_review_allows_cross_field_interaction_using_volume_sum_component() -> None:
    candidate = _candidate(
        key="t1430_new_price_volume_interaction_v1",
        body=(
            'vol = window_df["volume"]; last = window_df["last"]; denom = last.iloc[-1];\n'
            "            if denom <= 0 or vol.sum() <= 0: return np.nan\n"
            "            return float(((last.iloc[-1] - last.iloc[0]) / denom) * "
            "(vol.iloc[-5:].sum() / vol.sum()))"
        ),
        fields=["trade_time", "volume", "last"],
    )

    assert "duplicate_formula_family" not in _error_codes(candidate)
