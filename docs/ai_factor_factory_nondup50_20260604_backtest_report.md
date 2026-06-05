# AI Factor Factory Nondup50 Backtest 2026-06-04

- Source candidates: 417
- Static accepted: 184
- Synthetic smoke pass: 77
- Exact-key unique candidates: 53
- Landed factors: 50
- Real-data smoke: passed on 2025-01-02, 50/50 factors computed
- Long backtest: 2025-01-01 to 2026-06-02, T1430 14:30 factor, 14:42 label
- Leakage guard: `bin_source=fixed`, `bin_select=null`; end date capped at 2026-06-02 because labels stop there
- Good factors: 36; bad factors: 14; screened shortlist: 5; clean shortlist excluding bad-quality flags: 3

## Screened Shortlist

| factor_name | best_bin | best_alpha_mean | best_alpha_t | best_alpha_hit_rate | best_rolling_score | sharpe | alpha_sharpe | ret_total | alpha_ret_total | ic_mean | rank_ic_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1430_volatility_turnover_v1 | 17 | 0.000952046 | 3.0319 | 0.548673 | 5.89687 | 1.52692 | -0.348333 | 0.180271 | -0.0267977 | 0.00140034 | -0.0263836 |
| t1430_amount_vol_ratio_v1 | 19 | 0.00114546 | 2.74653 | 0.578171 | 5.02812 | 2.56042 | 2.37152 | 0.75775 | 0.459519 | 0.02798 | -0.0310348 |
| t1430_pv_ratio_v1 | 19 | 0.000873403 | 2.53732 | 0.572271 | 4.9125 | 2.94175 | 2.19087 | 0.617678 | 0.335332 | 0.0365906 | -0.0552172 |
| t1430_order_flow_toxicity_v1 | 5 | 0.000333249 | 1.62499 | 0.502994 | 4.48166 | 1.52324 | 0.261576 | 0.161145 | 0.0192911 | 0.00218315 | 0.00152626 |
| t1430_microprice_mid_spread_v1 | 0 | 0.000423481 | 2.44192 | 0.545723 | 3.76563 | 2.46277 | 1.39667 | 0.287998 | 0.0701671 | -0.0256709 | 0.00145656 |

## Shortlist Formulas

- `t1430_volatility_turnover_v1`: ((high - low) / last) * (amount / (volume * last + eps))
- `t1430_amount_vol_ratio_v1`: amount / (volume + eps)
- `t1430_pv_ratio_v1`: (last - low) / (high - low + eps) * (volume / (amount / last + eps))
- `t1430_order_flow_toxicity_v1`: abs(bid_volume1 - ask_volume1) / (bid_volume1 + ask_volume1 + eps)
- `t1430_microprice_mid_spread_v1`: Microprice - MidPrice, normalized by MidPrice. Microprice = (Bid1*AskVol1 + Ask1*BidVol1) / (BidVol1 + AskVol1). MidPrice = (Bid1 + Ask1) / 2.

## Clean Shortlist

- `t1430_volatility_turnover_v1`
- `t1430_amount_vol_ratio_v1`
- `t1430_pv_ratio_v1`

## Bad Factors

- `ob_pressure_spread_recovery_v1`: insufficient_bins_any_day
- `ob_pressure_spread_v1`: insufficient_bins_any_day
- `t1430_microprice_liquidity_v1`: insufficient_bins_any_day
- `ob_pressure_spread_persistence_v1`: insufficient_bins_any_day
- `t1430_close_position_v1`: insufficient_bins_any_day
- `t1430_order_flow_toxicity_v1`: insufficient_bins_any_day
- `t1430_bid_ask_imbalance_v1`: insufficient_bins_any_day
- `t1430_order_imbalance_v1`: insufficient_bins_any_day
- `t1430_orderflow_imbalance_v1`: insufficient_bins_any_day
- `t1430_order_flow_imbalance_v1`: insufficient_bins_any_day
- `t1430_ob_imbalance_flow_v1`: insufficient_bins_any_day
- `t1430_microprice_mid_spread_v1`: insufficient_bins_any_day
- `t1430_pv_interaction_v1`: insufficient_bins_any_day
- `t1430_tail_pressure_v1`: insufficient_bins_any_day

## Paths

- Result root: `D:\cbond_on\results\ai_factor_factory\nondup50_20260604\results\2025-01-01_2026-06-02\Single_Factor\20260604_023247`
- Full config: `cbond_on/config/factor/ai_factory/packs/ai_factor_factory_20260604_nondup50.json5`
- Good config: `cbond_on/config/factor/ai_factory/packs/ai_factor_factory_20260604_nondup50_active.json5`
- Shortlist config: `cbond_on/config/factor/ai_factory/packs/ai_factor_factory_20260604_nondup50_screened_shortlist.json5`
- Clean shortlist config: `cbond_on/config/factor/ai_factory/packs/ai_factor_factory_20260604_nondup50_screened_clean_shortlist.json5`
