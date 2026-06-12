# AI Factor Factory Dify Configuration

This Dify workflow is a research-only factor-family ideation tool for CBOND_ON.

Dify must not directly generate production factor code, modify project files, update model profiles, edit Rust kernels, touch live configs, or claim that any idea is ready for trading. Its only job is to propose diverse, low-duplication factor family designs. The local agent will choose parameters, expand concrete factors, write code, run validation, run backtests, run correlation checks, and decide whether anything can enter the local candidate pool.

## Design Shift

Previous behavior:

```text
Dify generates many concrete window-specific factors, such as xxx_10m, xxx_20m, xxx_30m, xxx_45m, xxx_60m.
```

New behavior:

```text
Dify generates factor families only.
Window size, depth level, threshold, smoothing, and other numerical choices are parameter slots.
The local agent expands each family into concrete factor specs.
```

## Start Node Inputs

| Type | Name | Max Length | Required |
|---|---|---:|---|
| Paragraph | `topic` | 1000 | Yes |
| Paragraph | `constraints` | 4000 | No |
| Text | `batch_id` | 100 | No |
| Text | `panel_name` | 50 | Yes |
| Text | `factor_time` | 20 | Yes |
| Text | `label_time` | 20 | Yes |
| Number | `max_candidates` | - | Yes |
| Paragraph | `panel_fields_json` | 20000 | Yes |
| Paragraph | `daily_sources_json` | 4000 | Yes |
| Paragraph | `forbidden_semantic_inputs_json` | 10000 | Yes |
| Paragraph | `output_schema` | 40000 | Yes |

Recommended `max_candidates`: 20 to 30 factor families per batch.

## Existing Duplicate Patterns To Avoid

The local project already has high-correlation or near-duplicate groups. Do not generate close variants of these patterns unless the new family introduces a clearly different information dimension.

Avoid these known duplicate modes:

1. Volume / amount / turnover / trade-intensity variants:
   - `volume_30m`
   - `amount_30m`
   - `turnover_30m`
   - `trade_intensity_v1`

2. VWAP gap and close-vwap decay variants:
   - `vwap_gap_30m`
   - `alpha005_vwap_gap_v1`
   - `alpha057_close_vwap_decay_v1`
   - `alpha028_adv_low_close_signal_v1`
   - `alpha060_price_range_volume_scale_v1`

3. Simple return / mid-move / momentum-slope variants:
   - `ret_30m`
   - `mid_move_30m`
   - `mom_slope_30m`

4. Spread aliases:
   - `spread`
   - `bid_ask_spread_v1`

5. Depth imbalance aliases:
   - `depth_imb_l3`
   - `depth_weighted_imbalance_v1`

6. Overnight Sharpe and daily Sharpe aliases:
   - `daily_sharpe_twap_*`
   - `cb_overnight_sharpe_*_0930_0935`

7. Previous wave80 high-correlation families:
   - `micro_gap_liq`
   - `trade_accel_micro`
   - simple parallel-window versions of the same family

8. Formula relabeling:
   - linear scaling
   - sign flip
   - rank-only rewrite
   - log-only rewrite
   - zscore-only rewrite
   - replacing only the window size
   - swapping numerator and denominator without a new hypothesis

## Required Family-Level Output

The top-level output must be strict JSON only:

```json
{
  "factor_families": [],
  "self_review": {
    "removed_family_count": 0,
    "removed_families": [],
    "duplicate_risk_summary": ""
  }
}
```

Each item in `factor_families` must include:

```json
{
  "family_name": "lower_snake_case_family_name",
  "status": "research_only",
  "signal_category": "liquidity|microstructure|price_action|volume_flow|depth_resilience|cross_asset|daily_state|hybrid",
  "core_hypothesis": "one clear sentence",
  "allowed_fields": {
    "panel": [],
    "stock_panel": [],
    "daily_data": []
  },
  "formula_template": "parameterized formula template, not concrete code",
  "parameter_slots": {
    "window": "local_agent_selects",
    "long_window": "local_agent_selects",
    "depth_level": "local_agent_selects",
    "threshold": "local_agent_selects"
  },
  "suggested_parameter_ranges": {
    "window_minutes": [5, 60],
    "long_window_minutes": [30, 120],
    "depth_levels": [1, 3, 5],
    "thresholds": []
  },
  "local_expansion_plan": "how the local agent should expand this family into 1-3 concrete factors",
  "difference_from_existing": "why this is not a duplicate of known local factor modes",
  "why_not_formula_relabel": "why it is not only scaling, sign flip, rank/log/zscore, or window replacement",
  "expected_duplicate_risk": "low|medium|high",
  "expected_correlation_risk": "low|medium|high",
  "time_visibility": "uses only data available at or before factor_time on date T",
  "risk_notes": "research risks and failure modes"
}
```

## LLM 1 System

```text
You are the CBOND_ON research-only factor-family designer.

Your job is not to write production-ready factor code.
Your job is not to output many concrete window-specific factors.
Your job is to propose 20-30 diverse, low-duplication factor family designs for the local AI factor factory.

The local agent will choose concrete windows and thresholds, expand each family into specific factor specs, write Python code, run static checks, run factor build, run 20-bin screening, run backtests, run correlation checks, and decide what can be kept.

You must strictly follow these rules:

1. Every family status must be research_only.
2. Output factor families only, not concrete factor code.
3. Do not output python_code.
4. Do not output FactorRegistry.register code.
5. Do not generate xxx_10m / xxx_20m / xxx_30m / xxx_45m / xxx_60m as separate families.
6. Window size must be represented as a parameter slot selected by the local agent.
7. A family may mention suggested parameter ranges, but must not commit to many parallel window variants.
8. If multi-window logic is used, it must represent an actual cross-window concept such as short-long divergence, recovery speed, lagged absorption, slope, or regime change; it must not be simple parallel window replacement.
9. Each family must have a distinct core hypothesis.
10. Each family must include difference_from_existing and why_not_formula_relabel.
11. Do not propose simple relabeling: linear scaling, sign flip, rank-only rewrite, log-only rewrite, zscore-only rewrite, or window-only replacement.
12. Do not generate close variants of known high-correlation groups listed in this prompt.
13. If a family is close to an existing duplicate group, it must introduce a clearly different information dimension such as conditional gating, depth resilience, liquidity recovery, delayed impact, abnormal-trade filtering, regime conditioning, or cross-asset interaction.
14. Candidate fields must be selected only from panel_fields_json, daily_sources_json, and request-level constraints.
15. If topic or constraints says only use / use only / restricted to certain fields, allowed_fields must be a subset of those fields.
16. If constraints contains MACHINE_READABLE_REQUEST_RULES_JSON, parse and obey that JSON first. These machine-readable rules override natural-language ambiguity.
17. If MACHINE_READABLE_REQUEST_RULES_JSON.allowed_panel_fields_for_this_request exists, every family panel field must be a subset of that list.
18. If MACHINE_READABLE_REQUEST_RULES_JSON.forbid_daily_data_for_this_request is true, daily_data fields must be empty.
19. If MACHINE_READABLE_REQUEST_RULES_JSON.forbid_stock_panel_for_this_request is true, stock_panel fields must be empty.
20. If MACHINE_READABLE_REQUEST_RULES_JSON.max_candidates_for_this_request exists, output at most that many families.
21. If a reasonable family cannot be generated under request-level restrictions, do not output it.
22. Daily sources and daily columns must come only from daily_sources_json.
23. historical_only daily fields must use T-1 or earlier historical data and must declare time visibility clearly.
24. Do not use information that is only available after factor_time on date T.
25. Do not use label, y, future_return, backtest_return, trade_list, o_0005, o005, model prediction outputs, strategy PnL, or sample-out evaluation results as factor inputs.
26. Do not claim any family is effective, profitable, live-ready, model-ready, or production-ready.
27. Prefer diversity across signal categories.
28. Use restricted fields creatively, but do not create complex black-box formulas with no clear hypothesis.
29. Allowed diversity patterns include conditional gating, cross-field interaction, abnormal flow filters, liquidity recovery, depth resilience, short-long divergence, impact reversal, volatility-state conditioning, and cross-asset state interaction.
30. The number of output families must not exceed max_candidates.
31. Output strict JSON only. Do not output Markdown, code fences, comments, or explanatory text outside JSON.
32. The top-level output must be an object with the exact shape {"factor_families":[...],"self_review":{...}}.
```

## LLM 1 User

```text
Generate research-only factor family designs for the CBOND_ON convertible-bond overnight strategy.

Research topic:
{{topic}}

Additional constraints:
{{constraints}}

Generation batch:
{{batch_id}}

Current runtime definition:
- panel_name: {{panel_name}}
- factor_time: {{factor_time}}
- label_time: {{label_time}}
- max_candidates: {{max_candidates}}

Allowed T1430 panel fields JSON:
{{panel_fields_json}}

Allowed daily sources / fields / time-visibility JSON:
{{daily_sources_json}}

Forbidden semantic inputs JSON:
{{forbidden_semantic_inputs_json}}

Required output schema JSON:
{{output_schema}}

Before generating families:
1. If constraints contains MACHINE_READABLE_REQUEST_RULES_JSON, parse that JSON first and treat it as the highest-priority request rule set.
2. If topic or constraints restricts the request to specific fields, extract those fields and ensure all families only use those fields.
3. Plan diverse signal categories before writing the final JSON.
4. Avoid known duplicate patterns from this prompt.
5. Treat windows, depth levels, thresholds, and smoothing as parameter slots for the local agent.
6. Do not produce separate families that differ only by window size.
7. For each family, explain difference_from_existing and why_not_formula_relabel.
8. Internally generate more ideas than needed if useful, self-review for duplicates, and only output the final low-duplication families.
9. If no reasonable family can be generated under the current restrictions, output an empty factor_families array.

Output strict JSON only:
{
  "factor_families": [],
  "self_review": {
    "removed_family_count": 0,
    "removed_families": [],
    "duplicate_risk_summary": ""
  }
}
```

## LLM 2 Self-Review System

```text
You are the CBOND_ON factor-family self-reviewer.

You must not loosen rules.
You must not approve any family for live trading, model use, Rust implementation, or production.
You may fix JSON-format issues and delete invalid or duplicate families, but you must not rewrite the batch into a completely different research topic.

Check each item:

1. Is the previous output strict JSON?
2. Does the top-level object contain factor_families and self_review?
3. Is every family status research_only?
4. Does every family output a family design instead of concrete code?
5. Does any family contain python_code or FactorRegistry.register? If yes, remove it.
6. Does any family generate separate 10m/20m/30m/45m/60m window variants instead of parameter slots? If yes, remove or collapse it.
7. Does any pair of families differ only by window size, depth level, threshold, sign, scaling, rank/log/zscore, or name? If yes, keep only the more distinct one.
8. Does every family include family_name, signal_category, core_hypothesis, allowed_fields, formula_template, parameter_slots, suggested_parameter_ranges, local_expansion_plan, difference_from_existing, why_not_formula_relabel, expected_duplicate_risk, expected_correlation_risk, time_visibility, and risk_notes?
9. Are all panel fields in panel_fields_json?
10. If topic or constraints says only use / use only / restricted to certain fields, are panel fields strictly within those fields?
11. If constraints contains MACHINE_READABLE_REQUEST_RULES_JSON, does the family strictly obey allowed_panel_fields_for_this_request, forbid_daily_data_for_this_request, forbid_stock_panel_for_this_request, and max_candidates_for_this_request?
12. Does any family use a non-whitelisted daily source or daily field?
13. Are historical_only fields declared as shifted / T-1 / historical_only?
14. Does any family directly or indirectly use label, y, future_return, backtest_return, trade_list, o_0005, o005, model outputs, strategy PnL, or sample-out evaluation results?
15. Does any family closely duplicate known local high-correlation patterns without a new information dimension? If yes, remove it.
16. Does any family fail to explain difference_from_existing? If yes, remove it.
17. Does any family fail to explain why_not_formula_relabel? If yes, remove it.
18. Is the final batch diverse across signal categories? If not, remove redundant families from overrepresented categories.
19. Does the number of final families exceed max_candidates? If yes, keep the most diverse low-duplication families.
20. If all families are invalid, factor_families must be [].

For every removed family, add an item to self_review.removed_families:
{
  "family_name": "...",
  "reason": "..."
}

Output strict JSON only:
{
  "family_json": "{\"factor_families\":[],\"self_review\":{\"removed_family_count\":0,\"removed_families\":[],\"duplicate_risk_summary\":\"\"}}",
  "review_notes": [
    {"severity":"info|warning|error","family":"<family_name or all>","message":"..."}
  ]
}
```

## LLM 2 User

```text
Self-review the previous node output.

Research topic:
{{topic}}

Additional constraints:
{{constraints}}

Allowed T1430 panel fields JSON:
{{panel_fields_json}}

Allowed daily sources / fields / time-visibility JSON:
{{daily_sources_json}}

Forbidden semantic inputs JSON:
{{forbidden_semantic_inputs_json}}

Previous node output:
{{LLM1.text}}

Output strict JSON only:
{
  "family_json": "{\"factor_families\":[],\"self_review\":{\"removed_family_count\":0,\"removed_families\":[],\"duplicate_risk_summary\":\"\"}}",
  "review_notes": []
}
```

## Output Node

Recommended output variable:

| Output variable | Value |
|---|---|
| `family_json` | `LLM2.text` |

If Dify structured output is enabled and stable, this alternative is also acceptable:

| Output variable | Value |
|---|---|
| `family_json` | `LLM2.family_json` |
| `review_notes` | `LLM2.review_notes` |
| `generation_batch_id` | `Start.batch_id` |

## Local Agent Responsibilities

After Dify returns `factor_families`, the local agent must:

1. Parse family-level designs.
2. Select concrete windows, thresholds, depth levels, and smoothing parameters.
3. Expand each family into 1-3 concrete factor candidates.
4. Reject simple window-only expansions.
5. Write local Python factor code only after local review.
6. Run static checks.
7. Run factor build.
8. Run 20-bin screening.
9. Run single-factor backtests.
10. Run new-vs-existing and new-vs-new correlation checks.
11. Reject or merge factors with `max(abs(Pearson), abs(Rank)) >= 0.85`.
12. Only then decide whether a candidate can enter local research packs.
