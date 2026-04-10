# Factor Time Coverage Report (Registry)

Total factors: 97

| factor | category | time_coverage | time_params | file |
|---|---|---|---|---|
| aacb | point_in_time | point-in-time at T | `{}` | aacb.py |
| alpha001_signed_power_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"stddev_window": 20, "ts_max_window": 5}` | alpha001_signed_power_v1.py |
| alpha002_corr_volume_return_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 6}` | alpha002_corr_volume_return_v1.py |
| alpha003_corr_open_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 10}` | alpha003_corr_open_volume_v1.py |
| alpha004_ts_rank_low_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_rank_window": 9}` | alpha004_ts_rank_low_v1.py |
| alpha005_vwap_gap_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"vwap_window": 10}` | alpha005_vwap_gap_v1.py |
| alpha006_corr_open_volume_neg_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 10}` | alpha006_corr_open_volume_neg_v1.py |
| alpha007_volume_breakout_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 20, "delta_window": 7, "ts_rank_window": 60}` | alpha007_volume_breakout_v1.py |
| alpha008_open_return_momentum_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"sum_window": 5, "delay_window": 10}` | alpha008_open_return_momentum_v1.py |
| alpha009_close_change_filter_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_window": 5}` | alpha009_close_change_filter_v1.py |
| alpha010_close_change_rank_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_window": 4}` | alpha010_close_change_rank_v1.py |
| alpha011_vwap_close_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_window": 3}` | alpha011_vwap_close_volume_v1.py |
| alpha012_volume_close_reversal_v1 | point_in_time | point-in-time at T | `{}` | alpha012_volume_close_reversal_v1.py |
| alpha013_cov_close_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"cov_window": 5}` | alpha013_cov_close_volume_v1.py |
| alpha014_return_open_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delta_window": 3, "corr_window": 10}` | alpha014_return_open_volume_v1.py |
| alpha015_high_volume_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 3, "sum_window": 3}` | alpha015_high_volume_corr_v1.py |
| alpha016_cov_high_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"cov_window": 5}` | alpha016_cov_high_volume_v1.py |
| alpha017_close_rank_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 20, "ts_rank_close_window": 10, "ts_rank_vol_window": 5}` | alpha017_close_rank_volume_v1.py |
| alpha018_close_open_vol_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"stddev_window": 5, "corr_window": 10}` | alpha018_close_open_vol_v1.py |
| alpha019_close_momentum_sign_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delta_window": 7, "sum_window": 250}` | alpha019_close_momentum_sign_v1.py |
| alpha020_open_delay_range_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delay_window": 1}` | alpha020_open_delay_range_v1.py |
| alpha021_close_volatility_breakout_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"sum_window_long": 5, "sum_window_short": 2, "adv_window": 10}` | alpha021_close_volatility_breakout_v1.py |
| alpha022_high_volume_corr_change_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 5, "delta_window": 5, "stddev_window": 10}` | alpha022_high_volume_corr_change_v1.py |
| alpha023_high_momentum_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"sum_window": 10, "delta_window": 2}` | alpha023_high_momentum_v1.py |
| alpha024_close_trend_filter_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"sum_window": 20, "delta_window": 20, "ts_min_window": 20, "short_delta_window": 3}` | alpha024_close_trend_filter_v1.py |
| alpha025_return_volume_vwap_range_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 10}` | alpha025_return_volume_vwap_range_v1.py |
| alpha026_volume_high_rank_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_rank_window": 5, "corr_window": 5, "ts_max_window": 3}` | alpha026_volume_high_rank_corr_v1.py |
| alpha027_volume_vwap_corr_signal_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 6, "sum_window": 2}` | alpha027_volume_vwap_corr_signal_v1.py |
| alpha028_adv_low_close_signal_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 10, "corr_window": 5}` | alpha028_adv_low_close_signal_v1.py |
| alpha029_complex_rank_signal_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_min_window": 2, "ts_rank_window": 5, "delay_window": 3, "min_window": 5}` | alpha029_complex_rank_signal_v1.py |
| alpha030_close_sign_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"sum_window_short": 5, "sum_window_long": 10}` | alpha030_close_sign_volume_v1.py |
| alpha031_close_decay_momentum_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delta_window": 10, "decay_window": 10, "delta_short_window": 3, "corr_window": 12, "adv_window": 10}` | alpha031_close_decay_momentum_v1.py |
| alpha032_vwap_close_mean_reversion_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"sum_window": 7, "corr_window": 60, "delay_window": 5}` | alpha032_vwap_close_mean_reversion_v1.py |
| alpha033_open_close_ratio_v1 | point_in_time | point-in-time at T | `{}` | alpha033_open_close_ratio_v1.py |
| alpha034_return_volatility_rank_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"stddev_window_short": 2, "stddev_window_long": 5, "delta_window": 1}` | alpha034_return_volatility_rank_v1.py |
| alpha035_volume_price_momentum_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_rank_window_long": 20, "ts_rank_window_short": 16}` | alpha035_volume_price_momentum_v1.py |
| alpha036_complex_correlation_signal_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window_1": 15, "corr_window_2": 6, "sum_window": 60, "ts_rank_window": 5, "delay_window": 6, "adv_window": 10}` | alpha036_complex_correlation_signal_v1.py |
| alpha037_open_close_correlation_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 30, "delay_window": 1}` | alpha037_open_close_correlation_v1.py |
| alpha038_close_rank_ratio_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_rank_window": 10}` | alpha038_close_rank_ratio_v1.py |
| alpha039_volume_decay_momentum_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 10, "decay_window": 9, "delta_window": 7, "sum_window": 60}` | alpha039_volume_decay_momentum_v1.py |
| alpha040_high_volatility_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"stddev_window": 10, "corr_window": 10}` | alpha040_high_volatility_corr_v1.py |
| alpha041_geometric_mean_vwap_v1 | point_in_time | point-in-time at T | `{}` | alpha041_geometric_mean_vwap_v1.py |
| alpha042_vwap_close_rank_ratio_v1 | point_in_time | point-in-time at T | `{}` | alpha042_vwap_close_rank_ratio_v1.py |
| alpha043_volume_delay_momentum_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 10, "ts_rank_window_1": 10, "delta_window": 5, "ts_rank_window_2": 5}` | alpha043_volume_delay_momentum_v1.py |
| alpha044_high_volume_rank_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 5}` | alpha044_high_volume_rank_corr_v1.py |
| alpha045_close_sum_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delay_window": 5, "sum_window_long": 20, "corr_window_1": 2, "sum_window_short": 5, "corr_window_2": 2}` | alpha045_close_sum_corr_v1.py |
| alpha046_close_delay_trend_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delay_window_long": 10, "delay_window_short": 5}` | alpha046_close_delay_trend_v1.py |
| alpha047_inverse_close_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 10, "sum_window": 5, "delay_window": 5}` | alpha047_inverse_close_volume_v1.py |
| alpha049_close_delay_threshold_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delay_window_long": 10, "delay_window_short": 5}` | alpha049_close_delay_threshold_v1.py |
| alpha050_volume_vwap_corr_max_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window": 5, "ts_max_window": 5}` | alpha050_volume_vwap_corr_max_v1.py |
| alpha051_close_delay_threshold_v2_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delay_window_long": 10, "delay_window_short": 5}` | alpha051_close_delay_threshold_v2_v1.py |
| alpha052_low_momentum_volume_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_min_window": 5, "delay_window": 5, "sum_window_long": 60, "sum_window_short": 20, "ts_rank_window": 5}` | alpha052_low_momentum_volume_v1.py |
| alpha053_price_position_delta_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delta_window": 9}` | alpha053_price_position_delta_v1.py |
| alpha054_price_power_ratio_v1 | point_in_time | point-in-time at T | `{}` | alpha054_price_power_ratio_v1.py |
| alpha055_close_range_volume_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_window": 12, "corr_window": 6}` | alpha055_close_range_volume_corr_v1.py |
| alpha057_close_vwap_decay_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_argmax_window": 10, "decay_window": 2}` | alpha057_close_vwap_decay_v1.py |
| alpha060_price_range_volume_scale_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"ts_argmax_window": 10}` | alpha060_price_range_volume_scale_v1.py |
| alpha062_vwap_open_rank_compare_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 20, "sum_window": 22, "corr_window": 10}` | alpha062_vwap_open_rank_compare_v1.py |
| alpha065_open_vwap_min_signal_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 30, "sum_window": 9, "corr_window": 6, "ts_min_window": 14}` | alpha065_open_vwap_min_signal_v1.py |
| alpha066_vwap_low_decay_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delta_window": 4, "decay_window_1": 7, "decay_window_2": 11, "ts_rank_window": 7}` | alpha066_vwap_low_decay_v1.py |
| alpha068_high_adv_rank_signal_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 15, "corr_window": 9, "ts_rank_window": 14, "delta_window": 1}` | alpha068_high_adv_rank_signal_v1.py |
| alpha072_vwap_volume_decay_ratio_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 20, "corr_window_1": 9, "decay_window_1": 10, "ts_rank_window_1": 4, "ts_rank_window_2": 19, "corr_window_2": 7, "decay_window_2": 3}` | alpha072_vwap_volume_decay_ratio_v1.py |
| alpha073_vwap_open_decay_max_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"delta_window_1": 5, "decay_window_1": 3, "delta_window_2": 2, "decay_window_2": 3, "ts_rank_window": 17}` | alpha073_vwap_open_decay_max_v1.py |
| alpha074_close_adv_rank_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"adv_window": 20, "sum_window": 37, "corr_window_1": 15, "corr_window_2": 11}` | alpha074_close_adv_rank_corr_v1.py |
| alpha075_vwap_volume_low_adv_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"corr_window_1": 4, "adv_window": 30, "corr_window_2": 12}` | alpha075_vwap_volume_low_adv_corr_v1.py |
| alpha077_mid_price_adv_decay_min_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"decay_window_1": 10, "adv_window": 20, "corr_window": 3, "decay_window_2": 6}` | alpha077_mid_price_adv_decay_min_v1.py |
| alpha078_low_vwap_adv_corr_v1 | rolling_window_no_abs_time | rolling statistical windows only (no explicit absolute time range) | `{"sum_window_1": 20, "adv_window": 20, "sum_window_2": 20, "corr_window_1": 7, "corr_window_2": 6}` | alpha078_low_vwap_adv_corr_v1.py |
| amihud_illiq | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | amihud_illiq.py |
| amount_sum | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | amount_sum.py |
| bid_ask_spread_v1 | point_in_time | point-in-time at T | `{}` | bid_ask_spread_v1.py |
| depth_imbalance | point_in_time | point-in-time at T | `{}` | depth_imbalance.py |
| depth_slope | point_in_time | point-in-time at T | `{}` | depth_slope.py |
| depth_weighted_imbalance_v1 | point_in_time | point-in-time at T | `{}` | depth_weighted_imbalance_v1.py |
| intraday_momentum_v1 | point_in_time | point-in-time at T | `{}` | intraday_momentum_v1.py |
| microprice_bias | point_in_time | point-in-time at T | `{}` | microprice_bias.py |
| midprice_move | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | midprice_move.py |
| mom_slope | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | mom_slope.py |
| order_flow_imbalance_v1 | point_in_time | point-in-time at T | `{}` | order_flow_imbalance_v1.py |
| premium_momentum_proxy_v1 | point_in_time | point-in-time at T | `{}` | premium_momentum_proxy_v1.py |
| price_level_position_v1 | point_in_time | point-in-time at T | `{}` | price_level_position_v1.py |
| price_position | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | price_position.py |
| range_ratio | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | range_ratio.py |
| ret_open_to_time | explicit_time_range | 09:30~14:30 (explicit time range) | `{"start_time": "09:30", "end_time": "14:30"}` | ret_open_to_time.py |
| ret_window | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | ret_window.py |
| return_skew | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | return_skew.py |
| spread | point_in_time | point-in-time at T | `{}` | spread.py |
| stock_bond_momentum_gap_v1 | point_in_time | point-in-time at T | `{}` | stock_bond_momentum_gap_v1.py |
| trade_intensity_v1 | point_in_time | point-in-time at T | `{}` | trade_intensity_v1.py |
| turnover_rate | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | turnover_rate.py |
| volatility | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | volatility.py |
| volatility_scaled_return_v1 | point_in_time | point-in-time at T | `{}` | volatility_scaled_return_v1.py |
| volen | point_in_time | point-in-time at T | `{}` | volen.py |
| volume_imbalance | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | volume_imbalance.py |
| volume_price_trend_v1 | point_in_time | point-in-time at T | `{}` | volume_price_trend_v1.py |
| volume_sum | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | volume_sum.py |
| vwap | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | vwap.py |
| vwap_gap | minute_window | T-30min ~ T (minute window) | `{"window_minutes": 30}` | vwap_gap.py |