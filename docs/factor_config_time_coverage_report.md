# Factor Config Time Coverage Report

Total factor specs: 161

| config_file | factor_name | kernel | time_coverage | time_params |
|---|---|---|---|---|
| factors_20260325.json5 | aacb_l3 | aacb | ???(??????T) | `{}` |
| factors_20260325.json5 | volen_f60_s10_l3 | volen | ???(??????T) | `{}` |
| factors_20260325.json5 | ret_5m | ret_window | T-5min ~ T (???) | `{"window_minutes": 5}` |
| factors_20260325.json5 | ret_10m | ret_window | T-10min ~ T (???) | `{"window_minutes": 10}` |
| factors_20260325.json5 | ret_30m | ret_window | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | ret_60m | ret_window | T-60min ~ T (???) | `{"window_minutes": 60}` |
| factors_20260325.json5 | ret_open_0930_1430 | ret_open_to_time | 09:30~14:30 (????) | `{"start_time": "09:30", "end_time": "14:30"}` |
| factors_20260325.json5 | mom_slope_30m | mom_slope | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | vol_30m | volatility | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | vol_60m | volatility | T-60min ~ T (???) | `{"window_minutes": 60}` |
| factors_20260325.json5 | range_30m | range_ratio | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | pos_30m | price_position | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | volume_30m | volume_sum | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | amount_30m | amount_sum | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | vwap_30m | vwap | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | volimb_30m_l3 | volume_imbalance | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | spread | spread | ???(??????T) | `{}` |
| factors_20260325.json5 | depth_imb_l3 | depth_imbalance | ???(??????T) | `{}` |
| factors_20260325.json5 | mid_move_30m | midprice_move | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | turnover_30m | turnover_rate | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | amihud_30m | amihud_illiq | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | microprice_bias_l1 | microprice_bias | ???(??????T) | `{}` |
| factors_20260325.json5 | depth_slope_l5 | depth_slope | ???(??????T) | `{}` |
| factors_20260325.json5 | ret_skew_30m | return_skew | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | vwap_gap_30m | vwap_gap | T-30min ~ T (???) | `{"window_minutes": 30}` |
| factors_20260325.json5 | order_flow_imbalance_v1 | order_flow_imbalance_v1 | ???(??????T) | `{}` |
| factors_20260325.json5 | depth_weighted_imbalance_v1 | depth_weighted_imbalance_v1 | ???(??????T) | `{}` |
| factors_20260325.json5 | intraday_momentum_v1 | intraday_momentum_v1 | ???(??????T) | `{}` |
| factors_20260325.json5 | volatility_scaled_return_v1 | volatility_scaled_return_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260325.json5 | volume_price_trend_v1 | volume_price_trend_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260325.json5 | trade_intensity_v1 | trade_intensity_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260325.json5 | price_level_position_v1 | price_level_position_v1 | ???(??????T) | `{}` |
| factors_20260325.json5 | stock_bond_momentum_gap_v1 | stock_bond_momentum_gap_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260325.json5 | bid_ask_spread_v1 | bid_ask_spread_v1 | ???(??????T) | `{}` |
| factors_20260325.json5 | premium_momentum_proxy_v1 | premium_momentum_proxy_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260325.json5 | alpha001_signed_power_v1 | alpha001_signed_power_v1 | ??10??????? (seq??) | `{"stddev_window": 20, "ts_max_window": 5, "windowsize": 10}` |
| factors_20260325.json5 | alpha002_corr_volume_return_v1 | alpha002_corr_volume_return_v1 | ??120??????? (seq??) | `{"corr_window": 6, "windowsize": 120}` |
| factors_20260325.json5 | alpha003_corr_open_volume_v1 | alpha003_corr_open_volume_v1 | ??120??????? (seq??) | `{"corr_window": 10, "windowsize": 120}` |
| factors_20260325.json5 | alpha004_ts_rank_low_v1 | alpha004_ts_rank_low_v1 | ??120??????? (seq??) | `{"ts_rank_window": 9, "windowsize": 120}` |
| factors_20260325.json5 | alpha005_vwap_gap_v1 | alpha005_vwap_gap_v1 | ??120??????? (seq??) | `{"vwap_window": 10, "windowsize": 120}` |
| factors_20260325.json5 | alpha006_corr_open_volume_neg_v1 | alpha006_corr_open_volume_neg_v1 | ??120??????? (seq??) | `{"corr_window": 10, "windowsize": 120}` |
| factors_20260325.json5 | alpha007_volume_breakout_v1 | alpha007_volume_breakout_v1 | ????????(???????) | `{"adv_window": 20, "delta_window": 7, "ts_rank_window": 60}` |
| factors_20260325.json5 | alpha008_open_return_momentum_v1 | alpha008_open_return_momentum_v1 | ??120??????? (seq??) | `{"sum_window": 5, "delay_window": 10, "windowsize": 120}` |
| factors_20260325.json5 | alpha009_close_change_filter_v1 | alpha009_close_change_filter_v1 | ????????(???????) | `{"ts_window": 5}` |
| factors_20260325.json5 | alpha010_close_change_rank_v1 | alpha010_close_change_rank_v1 | ????????(???????) | `{"ts_window": 4}` |
| factors_20260325.json5 | alpha011_vwap_close_volume_v1 | alpha011_vwap_close_volume_v1 | ????????(???????) | `{"ts_window": 3}` |
| factors_20260325.json5 | alpha012_volume_close_reversal_v1 | alpha012_volume_close_reversal_v1 | ???(??????T) | `{}` |
| factors_20260325.json5 | alpha013_cov_close_volume_v1 | alpha013_cov_close_volume_v1 | ????????(???????) | `{"cov_window": 5}` |
| factors_20260325.json5 | alpha014_return_open_volume_v1 | alpha014_return_open_volume_v1 | ??10??????? (seq??) | `{"delta_window": 3, "corr_window": 10, "windowsize": 10}` |
| factors_20260325.json5 | alpha015_high_volume_corr_v1 | alpha015_high_volume_corr_v1 | ??120??????? (seq??) | `{"corr_window": 3, "sum_window": 3, "windowsize": 120}` |
| factors_20260325.json5 | alpha016_cov_high_volume_v1 | alpha016_cov_high_volume_v1 | ??120??????? (seq??) | `{"cov_window": 5, "windowsize": 120}` |
| factors_20260325.json5 | alpha017_close_rank_volume_v1 | alpha017_close_rank_volume_v1 | ????????(???????) | `{"adv_window": 20, "ts_rank_close_window": 10, "ts_rank_vol_window": 5}` |
| factors_20260325.json5 | alpha018_close_open_vol_v1 | alpha018_close_open_vol_v1 | ??120??????? (seq??) | `{"stddev_window": 5, "corr_window": 10, "windowsize": 120}` |
| factors_20260325.json5 | alpha019_close_momentum_sign_v1 | alpha019_close_momentum_sign_v1 | ??10??????? (seq??) | `{"delta_window": 7, "sum_window": 250, "windowsize": 10}` |
| factors_20260325.json5 | alpha020_open_delay_range_v1 | alpha020_open_delay_range_v1 | ??120??????? (seq??) | `{"delay_window": 1, "windowsize": 120}` |
| factors_20260326.json5 | alpha021_close_volatility_breakout_v1 | alpha021_close_volatility_breakout_v1 | ????????(???????) | `{"sum_window_long": 5, "sum_window_short": 2, "adv_window": 10}` |
| factors_20260326.json5 | alpha023_high_momentum_v1 | alpha023_high_momentum_v1 | ??10??????? (seq??) | `{"sum_window": 10, "delta_window": 2, "windowsize": 10}` |
| factors_20260326.json5 | alpha024_close_trend_filter_v1 | alpha024_close_trend_filter_v1 | ????????(???????) | `{"sum_window": 20, "delta_window": 20, "ts_min_window": 20, "short_delta_window": 3}` |
| factors_20260326.json5 | alpha025_return_volume_vwap_range_v1 | alpha025_return_volume_vwap_range_v1 | ??10??????? (seq??) | `{"adv_window": 10, "windowsize": 10}` |
| factors_20260326.json5 | alpha026_volume_high_rank_corr_v1 | alpha026_volume_high_rank_corr_v1 | ??10??????? (seq??) | `{"ts_rank_window": 5, "corr_window": 5, "ts_max_window": 3, "windowsize": 10}` |
| factors_20260326.json5 | alpha027_volume_vwap_corr_signal_v1 | alpha027_volume_vwap_corr_signal_v1 | ????????(???????) | `{"corr_window": 6, "sum_window": 2}` |
| factors_20260326.json5 | alpha028_adv_low_close_signal_v1 | alpha028_adv_low_close_signal_v1 | ??10??????? (seq??) | `{"adv_window": 10, "corr_window": 5, "windowsize": 10}` |
| factors_20260326.json5 | alpha029_complex_rank_signal_v1 | alpha029_complex_rank_signal_v1 | ??10??????? (seq??) | `{"ts_min_window": 2, "ts_rank_window": 5, "delay_window": 3, "min_window": 5, "windowsize": 10}` |
| factors_20260326.json5 | alpha030_close_sign_volume_v1 | alpha030_close_sign_volume_v1 | ????????(???????) | `{"sum_window_short": 5, "sum_window_long": 10}` |
| factors_20260326.json5 | alpha031_close_decay_momentum_v1 | alpha031_close_decay_momentum_v1 | ??10??????? (seq??) | `{"delta_window": 10, "decay_window": 10, "delta_short_window": 3, "corr_window": 12, "adv_window": 10, "windowsize": 10}` |
| factors_20260326.json5 | alpha032_vwap_close_mean_reversion_v1 | alpha032_vwap_close_mean_reversion_v1 | ????????(???????) | `{"sum_window": 7, "corr_window": 60, "delay_window": 5}` |
| factors_20260326.json5 | alpha033_open_close_ratio_v1 | alpha033_open_close_ratio_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260326.json5 | alpha034_return_volatility_rank_v1 | alpha034_return_volatility_rank_v1 | ??10??????? (seq??) | `{"stddev_window_short": 2, "stddev_window_long": 5, "delta_window": 1, "windowsize": 10}` |
| factors_20260326.json5 | alpha035_volume_price_momentum_v1 | alpha035_volume_price_momentum_v1 | ??10??????? (seq??) | `{"ts_rank_window_long": 20, "ts_rank_window_short": 16, "windowsize": 10}` |
| factors_20260326.json5 | alpha036_complex_correlation_signal_v1 | alpha036_complex_correlation_signal_v1 | ??10??????? (seq??) | `{"corr_window_1": 15, "corr_window_2": 6, "sum_window": 60, "ts_rank_window": 5, "delay_window": 6, "adv_window": 10, "windowsize": 10}` |
| factors_20260326.json5 | alpha037_open_close_correlation_v1 | alpha037_open_close_correlation_v1 | ??10??????? (seq??) | `{"corr_window": 30, "delay_window": 1, "windowsize": 10}` |
| factors_20260326.json5 | alpha038_close_rank_ratio_v1 | alpha038_close_rank_ratio_v1 | ??10??????? (seq??) | `{"ts_rank_window": 10, "windowsize": 10}` |
| factors_20260326.json5 | alpha039_volume_decay_momentum_v1 | alpha039_volume_decay_momentum_v1 | ??10??????? (seq??) | `{"adv_window": 10, "decay_window": 9, "delta_window": 7, "sum_window": 60, "windowsize": 10}` |
| factors_20260326.json5 | alpha040_high_volatility_corr_v1 | alpha040_high_volatility_corr_v1 | ??10??????? (seq??) | `{"stddev_window": 10, "corr_window": 10, "windowsize": 10}` |
| factors_20260327.json5 | alpha041_geometric_mean_vwap_v1 | alpha041_geometric_mean_vwap_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260327.json5 | alpha042_vwap_close_rank_ratio_v1 | alpha042_vwap_close_rank_ratio_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260327.json5 | alpha043_volume_delay_momentum_v1 | alpha043_volume_delay_momentum_v1 | ??10??????? (seq??) | `{"adv_window": 10, "ts_rank_window_1": 10, "delta_window": 5, "ts_rank_window_2": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha044_high_volume_rank_corr_v1 | alpha044_high_volume_rank_corr_v1 | ??10??????? (seq??) | `{"corr_window": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha045_close_sum_corr_v1 | alpha045_close_sum_corr_v1 | ??10??????? (seq??) | `{"delay_window": 5, "sum_window_long": 20, "corr_window_1": 2, "sum_window_short": 5, "corr_window_2": 2, "windowsize": 10}` |
| factors_20260327.json5 | alpha046_close_delay_trend_v1 | alpha046_close_delay_trend_v1 | ??10??????? (seq??) | `{"delay_window_long": 10, "delay_window_short": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha047_inverse_close_volume_v1 | alpha047_inverse_close_volume_v1 | ??10??????? (seq??) | `{"adv_window": 10, "sum_window": 5, "delay_window": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha049_close_delay_threshold_v1 | alpha049_close_delay_threshold_v1 | ??10??????? (seq??) | `{"delay_window_long": 10, "delay_window_short": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha050_volume_vwap_corr_max_v1 | alpha050_volume_vwap_corr_max_v1 | ??10??????? (seq??) | `{"corr_window": 5, "ts_max_window": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha051_close_delay_threshold_v2_v1 | alpha051_close_delay_threshold_v2_v1 | ??10??????? (seq??) | `{"delay_window_long": 10, "delay_window_short": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha052_low_momentum_volume_v1 | alpha052_low_momentum_volume_v1 | ??10??????? (seq??) | `{"ts_min_window": 5, "delay_window": 5, "sum_window_long": 60, "sum_window_short": 20, "ts_rank_window": 5, "windowsize": 10}` |
| factors_20260327.json5 | alpha053_price_position_delta_v1 | alpha053_price_position_delta_v1 | ??10??????? (seq??) | `{"delta_window": 9, "windowsize": 10}` |
| factors_20260327.json5 | alpha054_price_power_ratio_v1 | alpha054_price_power_ratio_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_20260327.json5 | alpha055_close_range_volume_corr_v1 | alpha055_close_range_volume_corr_v1 | ??10??????? (seq??) | `{"ts_window": 12, "corr_window": 6, "windowsize": 10}` |
| factors_20260327.json5 | alpha057_close_vwap_decay_v1 | alpha057_close_vwap_decay_v1 | ??10??????? (seq??) | `{"ts_argmax_window": 10, "decay_window": 2, "windowsize": 10}` |
| factors_20260327.json5 | alpha060_price_range_volume_scale_v1 | alpha060_price_range_volume_scale_v1 | ??10??????? (seq??) | `{"ts_argmax_window": 10, "windowsize": 10}` |
| factors_20260327.json5 | alpha062_vwap_open_rank_compare_v1 | alpha062_vwap_open_rank_compare_v1 | ??10??????? (seq??) | `{"adv_window": 20, "sum_window": 22, "corr_window": 10, "windowsize": 10}` |
| factors_20260327.json5 | alpha065_open_vwap_min_signal_v1 | alpha065_open_vwap_min_signal_v1 | ??10??????? (seq??) | `{"adv_window": 30, "sum_window": 9, "corr_window": 6, "ts_min_window": 14, "windowsize": 10}` |
| factors_20260327.json5 | alpha066_vwap_low_decay_v1 | alpha066_vwap_low_decay_v1 | ??10??????? (seq??) | `{"delta_window": 4, "decay_window_1": 7, "decay_window_2": 11, "ts_rank_window": 7, "windowsize": 10}` |
| factors_20260327.json5 | alpha068_high_adv_rank_signal_v1 | alpha068_high_adv_rank_signal_v1 | ??10??????? (seq??) | `{"adv_window": 15, "corr_window": 9, "ts_rank_window": 14, "delta_window": 1, "windowsize": 10}` |
| factors_20260327.json5 | alpha072_vwap_volume_decay_ratio_v1 | alpha072_vwap_volume_decay_ratio_v1 | ??10??????? (seq??) | `{"adv_window": 20, "corr_window_1": 9, "decay_window_1": 10, "ts_rank_window_1": 4, "ts_rank_window_2": 19, "corr_window_2": 7, "decay_window_2": 3, "windowsize": 10}` |
| factors_20260327.json5 | alpha073_vwap_open_decay_max_v1 | alpha073_vwap_open_decay_max_v1 | ??10??????? (seq??) | `{"delta_window_1": 5, "decay_window_1": 3, "delta_window_2": 2, "decay_window_2": 3, "ts_rank_window": 17, "windowsize": 10}` |
| factors_20260327.json5 | alpha074_close_adv_rank_corr_v1 | alpha074_close_adv_rank_corr_v1 | ??10??????? (seq??) | `{"adv_window": 20, "sum_window": 37, "corr_window_1": 15, "corr_window_2": 11, "windowsize": 10}` |
| factors_20260327.json5 | alpha075_vwap_volume_low_adv_corr_v1 | alpha075_vwap_volume_low_adv_corr_v1 | ??10??????? (seq??) | `{"corr_window_1": 4, "adv_window": 30, "corr_window_2": 12, "windowsize": 10}` |
| factors_20260327.json5 | alpha077_mid_price_adv_decay_min_v1 | alpha077_mid_price_adv_decay_min_v1 | ??10??????? (seq??) | `{"decay_window_1": 10, "adv_window": 20, "corr_window": 3, "decay_window_2": 6, "windowsize": 10}` |
| factors_20260327.json5 | alpha078_low_vwap_adv_corr_v1 | alpha078_low_vwap_adv_corr_v1 | ??10??????? (seq??) | `{"sum_window_1": 20, "adv_window": 20, "sum_window_2": 20, "corr_window_1": 7, "corr_window_2": 6, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | volatility_scaled_return_v1 | volatility_scaled_return_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | volume_price_trend_v1 | volume_price_trend_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | trade_intensity_v1 | trade_intensity_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | premium_momentum_proxy_v1 | premium_momentum_proxy_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | stock_bond_momentum_gap_v1 | stock_bond_momentum_gap_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha001_signed_power_v1 | alpha001_signed_power_v1 | ??10??????? (seq??) | `{"stddev_window": 20, "ts_max_window": 5, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha014_return_open_volume_v1 | alpha014_return_open_volume_v1 | ??10??????? (seq??) | `{"delta_window": 3, "corr_window": 10, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha019_close_momentum_sign_v1 | alpha019_close_momentum_sign_v1 | ??10??????? (seq??) | `{"delta_window": 7, "sum_window": 250, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha025_return_volume_vwap_range_v1 | alpha025_return_volume_vwap_range_v1 | ??10??????? (seq??) | `{"adv_window": 10, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha029_complex_rank_signal_v1 | alpha029_complex_rank_signal_v1 | ??10??????? (seq??) | `{"ts_min_window": 2, "ts_rank_window": 5, "delay_window": 3, "min_window": 5, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha034_return_volatility_rank_v1 | alpha034_return_volatility_rank_v1 | ??10??????? (seq??) | `{"stddev_window_short": 2, "stddev_window_long": 5, "delta_window": 1, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha035_volume_price_momentum_v1 | alpha035_volume_price_momentum_v1 | ??10??????? (seq??) | `{"ts_rank_window_long": 20, "ts_rank_window_short": 16, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha036_complex_correlation_signal_v1 | alpha036_complex_correlation_signal_v1 | ??10??????? (seq??) | `{"corr_window_1": 15, "corr_window_2": 6, "sum_window": 60, "ts_rank_window": 5, "delay_window": 6, "adv_window": 10, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha039_volume_decay_momentum_v1 | alpha039_volume_decay_momentum_v1 | ??10??????? (seq??) | `{"adv_window": 10, "decay_window": 9, "delta_window": 7, "sum_window": 60, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha052_low_momentum_volume_v1 | alpha052_low_momentum_volume_v1 | ??10??????? (seq??) | `{"ts_min_window": 5, "delay_window": 5, "sum_window_long": 60, "sum_window_short": 20, "ts_rank_window": 5, "windowsize": 10}` |
| factors_patch_20260330_rebuild_semantics.json5 | alpha054_price_power_ratio_v1 | alpha054_price_power_ratio_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha002_corr_volume_return_v1 | alpha002_corr_volume_return_v1 | ??10??????? (seq??) | `{"corr_window": 6, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha003_corr_open_volume_v1 | alpha003_corr_open_volume_v1 | ??10??????? (seq??) | `{"corr_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha004_ts_rank_low_v1 | alpha004_ts_rank_low_v1 | ??10??????? (seq??) | `{"ts_rank_window": 9, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha005_vwap_gap_v1 | alpha005_vwap_gap_v1 | ??10??????? (seq??) | `{"vwap_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha006_corr_open_volume_neg_v1 | alpha006_corr_open_volume_neg_v1 | ??10??????? (seq??) | `{"corr_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha008_open_return_momentum_v1 | alpha008_open_return_momentum_v1 | ??10??????? (seq??) | `{"sum_window": 5, "delay_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha015_high_volume_corr_v1 | alpha015_high_volume_corr_v1 | ??10??????? (seq??) | `{"corr_window": 3, "sum_window": 3, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha016_cov_high_volume_v1 | alpha016_cov_high_volume_v1 | ??10??????? (seq??) | `{"cov_window": 5, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha018_close_open_vol_v1 | alpha018_close_open_vol_v1 | ??10??????? (seq??) | `{"stddev_window": 5, "corr_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha020_open_delay_range_v1 | alpha020_open_delay_range_v1 | ??10??????? (seq??) | `{"delay_window": 1, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha023_high_momentum_v1 | alpha023_high_momentum_v1 | ??10??????? (seq??) | `{"sum_window": 10, "delta_window": 2, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha025_return_volume_vwap_range_v1 | alpha025_return_volume_vwap_range_v1 | ??10??????? (seq??) | `{"adv_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha026_volume_high_rank_corr_v1 | alpha026_volume_high_rank_corr_v1 | ??10??????? (seq??) | `{"ts_rank_window": 5, "corr_window": 5, "ts_max_window": 3, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha028_adv_low_close_signal_v1 | alpha028_adv_low_close_signal_v1 | ??10??????? (seq??) | `{"adv_window": 10, "corr_window": 5, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha031_close_decay_momentum_v1 | alpha031_close_decay_momentum_v1 | ??10??????? (seq??) | `{"delta_window": 10, "decay_window": 10, "delta_short_window": 3, "corr_window": 12, "adv_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha033_open_close_ratio_v1 | alpha033_open_close_ratio_v1 | ??10??????? (seq??) | `{"windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha035_volume_price_momentum_v1 | alpha035_volume_price_momentum_v1 | ??10??????? (seq??) | `{"ts_rank_window_long": 20, "ts_rank_window_short": 16, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha037_open_close_correlation_v1 | alpha037_open_close_correlation_v1 | ??10??????? (seq??) | `{"corr_window": 30, "delay_window": 1, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha038_close_rank_ratio_v1 | alpha038_close_rank_ratio_v1 | ??10??????? (seq??) | `{"ts_rank_window": 10, "windowsize": 10}` |
| factors_patch_ohlc_windowsize_20260327.json5 | alpha040_high_volatility_corr_v1 | alpha040_high_volatility_corr_v1 | ??10??????? (seq??) | `{"stddev_window": 10, "corr_window": 10, "windowsize": 10}` |
| lgbm_factor_MSE.json5 | aacb_l3 | aacb | ???(??????T) | `{}` |
| lgbm_factor_MSE.json5 | volen_f60_s10_l3 | volen | ???(??????T) | `{}` |
| lgbm_factor_MSE.json5 | ret_5m | ret_window | T-5min ~ T (???) | `{"window_minutes": 5}` |
| lgbm_factor_MSE.json5 | ret_10m | ret_window | T-10min ~ T (???) | `{"window_minutes": 10}` |
| lgbm_factor_MSE.json5 | ret_30m | ret_window | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | ret_60m | ret_window | T-60min ~ T (???) | `{"window_minutes": 60}` |
| lgbm_factor_MSE.json5 | ret_open_0930_1430 | ret_open_to_time | 09:30~14:30 (????) | `{"start_time": "09:30", "end_time": "14:30"}` |
| lgbm_factor_MSE.json5 | mom_slope_30m | mom_slope | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | vol_30m | volatility | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | vol_60m | volatility | T-60min ~ T (???) | `{"window_minutes": 60}` |
| lgbm_factor_MSE.json5 | range_30m | range_ratio | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | pos_30m | price_position | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | volume_30m | volume_sum | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | amount_30m | amount_sum | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | vwap_30m | vwap | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | volimb_30m_l3 | volume_imbalance | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | spread | spread | ???(??????T) | `{}` |
| lgbm_factor_MSE.json5 | depth_imb_l3 | depth_imbalance | ???(??????T) | `{}` |
| lgbm_factor_MSE.json5 | mid_move_30m | midprice_move | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | turnover_30m | turnover_rate | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | amihud_30m | amihud_illiq | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | microprice_bias_l1 | microprice_bias | ???(??????T) | `{}` |
| lgbm_factor_MSE.json5 | depth_slope_l5 | depth_slope | ???(??????T) | `{}` |
| lgbm_factor_MSE.json5 | ret_skew_30m | return_skew | T-30min ~ T (???) | `{"window_minutes": 30}` |
| lgbm_factor_MSE.json5 | vwap_gap_30m | vwap_gap | T-30min ~ T (???) | `{"window_minutes": 30}` |