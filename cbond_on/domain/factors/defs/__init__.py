from cbond_on.domain.factors.defs.aacb import AacbFactor
from cbond_on.domain.factors.defs.volen import VolenFactor
from cbond_on.domain.factors.defs.amount_sum import AmountSumFactor
from cbond_on.domain.factors.defs.depth_imbalance import DepthImbalanceFactor
from cbond_on.domain.factors.defs.midprice_move import MidpriceMoveFactor
from cbond_on.domain.factors.defs.mom_slope import MomentumSlopeFactor
from cbond_on.domain.factors.defs.price_position import PricePositionFactor
from cbond_on.domain.factors.defs.range_ratio import RangeRatioFactor
from cbond_on.domain.factors.defs.ret_open_to_time import ReturnOpenToTimeFactor
from cbond_on.domain.factors.defs.ret_window import ReturnWindowFactor
from cbond_on.domain.factors.defs.spread import SpreadFactor
from cbond_on.domain.factors.defs.turnover_rate import TurnoverRateFactor
from cbond_on.domain.factors.defs.volatility import VolatilityFactor
from cbond_on.domain.factors.defs.volume_imbalance import VolumeImbalanceFactor
from cbond_on.domain.factors.defs.volume_sum import VolumeSumFactor
from cbond_on.domain.factors.defs.vwap import VwapFactor
from cbond_on.domain.factors.defs.amihud_illiq import AmihudIlliqFactor
from cbond_on.domain.factors.defs.microprice_bias import MicropriceBiasFactor
from cbond_on.domain.factors.defs.depth_slope import DepthSlopeFactor
from cbond_on.domain.factors.defs.return_skew import ReturnSkewFactor
from cbond_on.domain.factors.defs.vwap_gap import VwapGapFactor
from cbond_on.domain.factors.defs.order_flow_imbalance_v1 import OrderFlowImbalanceV1Factor
from cbond_on.domain.factors.defs.depth_weighted_imbalance_v1 import DepthWeightedImbalanceV1Factor
from cbond_on.domain.factors.defs.intraday_momentum_v1 import IntradayMomentumV1Factor
from cbond_on.domain.factors.defs.volatility_scaled_return_v1 import VolatilityScaledReturnV1Factor
from cbond_on.domain.factors.defs.volume_price_trend_v1 import VolumePriceTrendV1Factor
from cbond_on.domain.factors.defs.trade_intensity_v1 import TradeIntensityV1Factor
from cbond_on.domain.factors.defs.price_level_position_v1 import PriceLevelPositionV1Factor
from cbond_on.domain.factors.defs.bid_ask_spread_v1 import BidAskSpreadV1Factor
from cbond_on.domain.factors.defs.stock_bond_momentum_gap_v1 import StockBondMomentumGapV1Factor
from cbond_on.domain.factors.defs.premium_momentum_proxy_v1 import PremiumMomentumProxyV1Factor
from cbond_on.domain.factors.defs.alpha001_signed_power_v1 import Alpha001SignedPowerV1Factor
from cbond_on.domain.factors.defs.alpha002_corr_volume_return_v1 import Alpha002CorrVolumeReturnV1Factor
from cbond_on.domain.factors.defs.alpha003_corr_open_volume_v1 import Alpha003CorrOpenVolumeV1Factor
from cbond_on.domain.factors.defs.alpha004_ts_rank_low_v1 import Alpha004TsRankLowV1Factor
from cbond_on.domain.factors.defs.alpha005_vwap_gap_v1 import Alpha005VwapGapV1Factor
from cbond_on.domain.factors.defs.alpha006_corr_open_volume_neg_v1 import Alpha006CorrOpenVolumeNegV1Factor
from cbond_on.domain.factors.defs.alpha007_volume_breakout_v1 import Alpha007VolumeBreakoutV1Factor
from cbond_on.domain.factors.defs.alpha008_open_return_momentum_v1 import Alpha008OpenReturnMomentumV1Factor
from cbond_on.domain.factors.defs.alpha009_close_change_filter_v1 import Alpha009CloseChangeFilterV1Factor
from cbond_on.domain.factors.defs.alpha010_close_change_rank_v1 import Alpha010CloseChangeRankV1Factor
from cbond_on.domain.factors.defs.alpha011_vwap_close_volume_v1 import Alpha011VwapCloseVolumeV1Factor
from cbond_on.domain.factors.defs.alpha012_volume_close_reversal_v1 import Alpha012VolumeCloseReversalV1Factor
from cbond_on.domain.factors.defs.alpha013_cov_close_volume_v1 import Alpha013CovCloseVolumeV1Factor
from cbond_on.domain.factors.defs.alpha014_return_open_volume_v1 import Alpha014ReturnOpenVolumeV1Factor
from cbond_on.domain.factors.defs.alpha015_high_volume_corr_v1 import Alpha015HighVolumeCorrV1Factor
from cbond_on.domain.factors.defs.alpha016_cov_high_volume_v1 import Alpha016CovHighVolumeV1Factor
from cbond_on.domain.factors.defs.alpha017_close_rank_volume_v1 import Alpha017CloseRankVolumeV1Factor
from cbond_on.domain.factors.defs.alpha018_close_open_vol_v1 import Alpha018CloseOpenVolV1Factor
from cbond_on.domain.factors.defs.alpha019_close_momentum_sign_v1 import Alpha019CloseMomentumSignV1Factor
from cbond_on.domain.factors.defs.alpha020_open_delay_range_v1 import Alpha020OpenDelayRangeV1Factor
from cbond_on.domain.factors.defs.alpha021_close_volatility_breakout_v1 import Alpha021CloseVolatilityBreakoutV1Factor
from cbond_on.domain.factors.defs.alpha022_high_volume_corr_change_v1 import Alpha022HighVolumeCorrChangeV1Factor
from cbond_on.domain.factors.defs.alpha023_high_momentum_v1 import Alpha023HighMomentumV1Factor
from cbond_on.domain.factors.defs.alpha024_close_trend_filter_v1 import Alpha024CloseTrendFilterV1Factor
from cbond_on.domain.factors.defs.alpha025_return_volume_vwap_range_v1 import Alpha025ReturnVolumeVwapRangeV1Factor
from cbond_on.domain.factors.defs.alpha026_volume_high_rank_corr_v1 import Alpha026VolumeHighRankCorrV1Factor
from cbond_on.domain.factors.defs.alpha027_volume_vwap_corr_signal_v1 import Alpha027VolumeVwapCorrSignalV1Factor
from cbond_on.domain.factors.defs.alpha028_adv_low_close_signal_v1 import Alpha028AdvLowCloseSignalV1Factor
from cbond_on.domain.factors.defs.alpha029_complex_rank_signal_v1 import Alpha029ComplexRankSignalV1Factor
from cbond_on.domain.factors.defs.alpha030_close_sign_volume_v1 import Alpha030CloseSignVolumeV1Factor
from cbond_on.domain.factors.defs.alpha031_close_decay_momentum_v1 import Alpha031CloseDecayMomentumV1Factor
from cbond_on.domain.factors.defs.alpha032_vwap_close_mean_reversion_v1 import Alpha032VwapCloseMeanReversionV1Factor
from cbond_on.domain.factors.defs.alpha033_open_close_ratio_v1 import Alpha033OpenCloseRatioV1Factor
from cbond_on.domain.factors.defs.alpha034_return_volatility_rank_v1 import Alpha034ReturnVolatilityRankV1Factor
from cbond_on.domain.factors.defs.alpha035_volume_price_momentum_v1 import Alpha035VolumePriceMomentumV1Factor
from cbond_on.domain.factors.defs.alpha036_complex_correlation_signal_v1 import Alpha036ComplexCorrelationSignalV1Factor
from cbond_on.domain.factors.defs.alpha037_open_close_correlation_v1 import Alpha037OpenCloseCorrelationV1Factor
from cbond_on.domain.factors.defs.alpha038_close_rank_ratio_v1 import Alpha038CloseRankRatioV1Factor
from cbond_on.domain.factors.defs.alpha039_volume_decay_momentum_v1 import Alpha039VolumeDecayMomentumV1Factor
from cbond_on.domain.factors.defs.alpha040_high_volatility_corr_v1 import Alpha040HighVolatilityCorrV1Factor
from cbond_on.domain.factors.defs.alpha041_geometric_mean_vwap_v1 import Alpha041GeometricMeanVwapV1Factor
from cbond_on.domain.factors.defs.alpha042_vwap_close_rank_ratio_v1 import Alpha042VwapCloseRankRatioV1Factor
from cbond_on.domain.factors.defs.alpha043_volume_delay_momentum_v1 import Alpha043VolumeDelayMomentumV1Factor
from cbond_on.domain.factors.defs.alpha044_high_volume_rank_corr_v1 import Alpha044HighVolumeRankCorrV1Factor
from cbond_on.domain.factors.defs.alpha045_close_sum_corr_v1 import Alpha045CloseSumCorrV1Factor
from cbond_on.domain.factors.defs.alpha046_close_delay_trend_v1 import Alpha046CloseDelayTrendV1Factor
from cbond_on.domain.factors.defs.alpha047_inverse_close_volume_v1 import Alpha047InverseCloseVolumeV1Factor
from cbond_on.domain.factors.defs.alpha049_close_delay_threshold_v1 import Alpha049CloseDelayThresholdV1Factor
from cbond_on.domain.factors.defs.alpha050_volume_vwap_corr_max_v1 import Alpha050VolumeVwapCorrMaxV1Factor
from cbond_on.domain.factors.defs.alpha051_close_delay_threshold_v2_v1 import Alpha051CloseDelayThresholdV2V1Factor
from cbond_on.domain.factors.defs.alpha052_low_momentum_volume_v1 import Alpha052LowMomentumVolumeV1Factor
from cbond_on.domain.factors.defs.alpha053_price_position_delta_v1 import Alpha053PricePositionDeltaV1Factor
from cbond_on.domain.factors.defs.alpha054_price_power_ratio_v1 import Alpha054PricePowerRatioV1Factor
from cbond_on.domain.factors.defs.alpha055_close_range_volume_corr_v1 import Alpha055CloseRangeVolumeCorrV1Factor
from cbond_on.domain.factors.defs.alpha057_close_vwap_decay_v1 import Alpha057CloseVwapDecayV1Factor
from cbond_on.domain.factors.defs.alpha060_price_range_volume_scale_v1 import Alpha060PriceRangeVolumeScaleV1Factor
from cbond_on.domain.factors.defs.alpha062_vwap_open_rank_compare_v1 import Alpha062VwapOpenRankCompareV1Factor
from cbond_on.domain.factors.defs.alpha065_open_vwap_min_signal_v1 import Alpha065OpenVwapMinSignalV1Factor
from cbond_on.domain.factors.defs.alpha066_vwap_low_decay_v1 import Alpha066VwapLowDecayV1Factor
from cbond_on.domain.factors.defs.alpha068_high_adv_rank_signal_v1 import Alpha068HighAdvRankSignalV1Factor
from cbond_on.domain.factors.defs.alpha072_vwap_volume_decay_ratio_v1 import Alpha072VwapVolumeDecayRatioV1Factor
from cbond_on.domain.factors.defs.alpha073_vwap_open_decay_max_v1 import Alpha073VwapOpenDecayMaxV1Factor
from cbond_on.domain.factors.defs.alpha074_close_adv_rank_corr_v1 import Alpha074CloseAdvRankCorrV1Factor
from cbond_on.domain.factors.defs.alpha075_vwap_volume_low_adv_corr_v1 import Alpha075VwapVolumeLowAdvCorrV1Factor
from cbond_on.domain.factors.defs.alpha077_mid_price_adv_decay_min_v1 import Alpha077MidPriceAdvDecayMinV1Factor
from cbond_on.domain.factors.defs.alpha078_low_vwap_adv_corr_v1 import Alpha078LowVwapAdvCorrV1Factor
from cbond_on.domain.factors.defs.daily_overnight_return_mean_v1 import DailyOvernightReturnMeanV1Factor
from cbond_on.domain.factors.defs.daily_sharpe_mean_v1 import DailySharpeMeanV1Factor
from cbond_on.domain.factors.defs.t1430_amount_accel_30m_v1 import T1430AmountAccel30mV1
from cbond_on.domain.factors.defs.t1430_depth_concentration_v1 import T1430DepthConcentrationV1
from cbond_on.domain.factors.defs.t1430_depth_imbalance_change_v1 import T1430DepthImbalanceChangeV1
from cbond_on.domain.factors.defs.t1430_last_return_30m_v1 import T1430LastReturn30mV1
from cbond_on.domain.factors.defs.t1430_microprice_last_gap_v1 import T1430MicropriceLastGapV1
from cbond_on.domain.factors.defs.t1430_mid_return_30m_v1 import T1430MidReturn30mV1
from cbond_on.domain.factors.defs.t1430_price_range_30m_v1 import T1430PriceRange30mV1
from cbond_on.domain.factors.defs.t1430_spread_change_30m_v1 import T1430SpreadChange30mV1
from cbond_on.domain.factors.defs.t1430_spread_mean_guarded_30m_v1 import T1430SpreadMeanGuarded30mV1
from cbond_on.domain.factors.defs.t1430_window_vwap_last_gap_v1 import T1430WindowVwapLastGapV1
from cbond_on.domain.factors.defs.t1430_cumulative_volume_ratio_v1 import T1430CumulativeVolumeRatioV1
from cbond_on.domain.factors.defs.t1430_volume_gini_coefficient_v1 import T1430VolumeGiniCoefficientV1
from cbond_on.domain.factors.defs.t1430_volume_autocorrelation_v1 import T1430VolumeAutocorrelationV1
from cbond_on.domain.factors.defs.t1430_volume_momentum_v1 import T1430VolumeMomentumV1
from cbond_on.domain.factors.defs.t1430_volume_std_dev_v1 import T1430VolumeStdDevV1
from cbond_on.domain.factors.defs.t1430_volume_skewness_v1 import T1430VolumeSkewnessV1
from cbond_on.domain.factors.defs.t1430_volume_kurtosis_v1 import T1430VolumeKurtosisV1
from cbond_on.domain.factors.defs.t1430_volume_count_v1 import T1430VolumeCountV1
from cbond_on.domain.factors.defs.t1430_volume_cumsum_v1 import T1430VolumeCumsumV1
from cbond_on.domain.factors.defs.t1430_volume_mean_v1 import T1430VolumeMeanV1
from cbond_on.domain.factors.defs.t1430_volume_max_v1 import T1430VolumeMaxV1
from cbond_on.domain.factors.defs.t1430_volume_min_v1 import T1430VolumeMinV1
from cbond_on.domain.factors.defs.t1430_volume_count_v2 import T1430VolumeCountV2
from cbond_on.domain.factors.defs.t1430_volume_acceleration_v1 import T1430VolumeAccelerationV1
from cbond_on.domain.factors.defs.t1430_volume_concentration_hhi_v1 import T1430VolumeConcentrationHhiV1
from cbond_on.domain.factors.defs.t1430_volume_cv_v1 import T1430VolumeCvV1
from cbond_on.domain.factors.defs.t1430_volume_entropy_v1 import T1430VolumeEntropyV1
from cbond_on.domain.factors.defs.t1430_volume_gini_simple_v1 import T1430VolumeGiniSimpleV1
from cbond_on.domain.factors.defs.t1430_volume_concentration_gini_v1 import T1430VolumeConcentrationGiniV1
from cbond_on.domain.factors.defs.t1430_volume_weighted_avg_size_v1 import T1430VolumeWeightedAvgSizeV1
from cbond_on.domain.factors.defs.t1430_volume_hhi_v1 import T1430VolumeHhiV1
from cbond_on.domain.factors.defs.t1430_volume_hhi_v2 import T1430VolumeHhiV2
from cbond_on.domain.factors.defs.t1430_volume_entropy_v2 import T1430VolumeEntropyV2
from cbond_on.domain.factors.defs.t1430_volume_cumsum_slope_v1 import T1430VolumeCumsumSlopeV1
from cbond_on.domain.factors.defs.t1430_volume_recent_share_v1 import T1430VolumeRecentShareV1
from cbond_on.domain.factors.defs.t1430_volume_weighted_time_center_v1 import T1430VolumeWeightedTimeCenterV1
from cbond_on.domain.factors.defs.t1430_volume_moment_v1 import T1430VolumeMomentV1
from cbond_on.domain.factors.defs.t1430_volume_sum_v1 import T1430VolumeSumV1
from cbond_on.domain.factors.defs.t1430_volume_max_tick_v1 import T1430VolumeMaxTickV1
from cbond_on.domain.factors.defs.t1430_volume_range_v1 import T1430VolumeRangeV1
from cbond_on.domain.factors.defs.t1430_volume_count_v3 import T1430VolumeCountV3
from cbond_on.domain.factors.defs.t1430_volume_mean_v2 import T1430VolumeMeanV2
from cbond_on.domain.factors.defs.t1430_volume_std_dev_v2 import T1430VolumeStdDevV2
from cbond_on.domain.factors.defs.t1430_volume_max_v2 import T1430VolumeMaxV2
from cbond_on.domain.factors.defs.t1430_volume_min_v2 import T1430VolumeMinV2

from cbond_on.domain.factors.defs.microprice_dev_liquidity_v1 import MicropriceDevLiquidityV1
from cbond_on.domain.factors.defs.ob_pressure_spread_recovery_v1 import ObPressureSpreadRecoveryV1
from cbond_on.domain.factors.defs.ob_pressure_spread_v1 import ObPressureSpreadV1
from cbond_on.domain.factors.defs.microprice_deviation_v1 import MicropriceDeviationV1
from cbond_on.domain.factors.defs.microprice_deviation_liq_cond_v1 import MicropriceDeviationLiqCondV1
from cbond_on.domain.factors.defs.spread_depth_pressure_v1 import SpreadDepthPressureV1
from cbond_on.domain.factors.defs.microprice_deviation_liq_v1 import MicropriceDeviationLiqV1
from cbond_on.domain.factors.defs.vwap_deviation_liquidity_cond_v1 import VwapDeviationLiquidityCondV1
from cbond_on.domain.factors.defs.tail_volume_absorption_v1 import TailVolumeAbsorptionV1
from cbond_on.domain.factors.defs.ob_pressure_spread_persistence_v1 import ObPressureSpreadPersistenceV1
from cbond_on.domain.factors.defs.t1430_pv_interaction_v1 import T1430PvInteractionV1
from cbond_on.domain.factors.defs.t1430_range_vol_norm_v1 import T1430RangeVolNormV1
from cbond_on.domain.factors.defs.t1430_tail_pressure_v1 import T1430TailPressureV1
from cbond_on.domain.factors.defs.t1430_range_volume_v1 import T1430RangeVolumeV1
from cbond_on.domain.factors.defs.t1430_amount_intensity_v1 import T1430AmountIntensityV1
from cbond_on.domain.factors.defs.t1430_close_position_v1 import T1430ClosePositionV1
from cbond_on.domain.factors.defs.t1430_pv_ratio_v1 import T1430PvRatioV1
from cbond_on.domain.factors.defs.t1430_range_volume_norm_v1 import T1430RangeVolumeNormV1
from cbond_on.domain.factors.defs.t1430_range_norm_volume_v1 import T1430RangeNormVolumeV1
from cbond_on.domain.factors.defs.t1430_amount_vol_ratio_v1 import T1430AmountVolRatioV1
from cbond_on.domain.factors.defs.t1430_close_range_position_v1 import T1430CloseRangePositionV1
from cbond_on.domain.factors.defs.t1430_volatility_turnover_v1 import T1430VolatilityTurnoverV1
from cbond_on.domain.factors.defs.t1430_vwap_deviation_v1 import T1430VwapDeviationV1
from cbond_on.domain.factors.defs.t1430_vol_turnover_ratio_v1 import T1430VolTurnoverRatioV1
from cbond_on.domain.factors.defs.t1430_spread_liquidity_ratio_v1 import T1430SpreadLiquidityRatioV1
from cbond_on.domain.factors.defs.t1430_microprice_mid_deviation_v1 import T1430MicropriceMidDeviationV1
from cbond_on.domain.factors.defs.t1430_orderflow_imbalance_v1 import T1430OrderflowImbalanceV1
from cbond_on.domain.factors.defs.t1430_depth_weighted_mid_v1 import T1430DepthWeightedMidV1
from cbond_on.domain.factors.defs.t1430_bid_ask_imbalance_v1 import T1430BidAskImbalanceV1
from cbond_on.domain.factors.defs.t1430_microprice_midpoint_dev_v1 import T1430MicropriceMidpointDevV1
from cbond_on.domain.factors.defs.t1430_order_flow_toxicity_v1 import T1430OrderFlowToxicityV1
from cbond_on.domain.factors.defs.t1430_bid_ask_spread_v1 import T1430BidAskSpreadV1
from cbond_on.domain.factors.defs.t1430_order_imbalance_v1 import T1430OrderImbalanceV1
from cbond_on.domain.factors.defs.t1430_micro_price_deviation_v1 import T1430MicroPriceDeviationV1
from cbond_on.domain.factors.defs.t1430_relative_strength_v1 import T1430RelativeStrengthV1
from cbond_on.domain.factors.defs.t1430_volume_weighted_price_deviation_v1 import T1430VolumeWeightedPriceDeviationV1
from cbond_on.domain.factors.defs.t1430_mid_price_return_v1 import T1430MidPriceReturnV1
from cbond_on.domain.factors.defs.t1430_spread_width_norm_v1 import T1430SpreadWidthNormV1
from cbond_on.domain.factors.defs.t1430_microprice_mid_spread_v1 import T1430MicropriceMidSpreadV1
from cbond_on.domain.factors.defs.t1430_order_flow_imbalance_v1 import T1430OrderFlowImbalanceV1
from cbond_on.domain.factors.defs.t1430_relative_bid_ask_strength_v1 import T1430RelativeBidAskStrengthV1
from cbond_on.domain.factors.defs.t1430_liq_adj_spread_v1 import T1430LiqAdjSpreadV1
from cbond_on.domain.factors.defs.t1430_microprice_imbalance_v1 import T1430MicropriceImbalanceV1
from cbond_on.domain.factors.defs.t1430_volume_weighted_mid_v1 import T1430VolumeWeightedMidV1
from cbond_on.domain.factors.defs.t1430_microprice_liquidity_v1 import T1430MicropriceLiquidityV1
from cbond_on.domain.factors.defs.t1430_ob_imbalance_flow_v1 import T1430ObImbalanceFlowV1
from cbond_on.domain.factors.defs.t1430_spread_volatility_ratio_v1 import T1430SpreadVolatilityRatioV1
from cbond_on.domain.factors.defs.t1430_relative_strength_vwap_v1 import T1430RelativeStrengthVwapV1
from cbond_on.domain.factors.defs.t1430_depth_pressure_return_v1 import T1430DepthPressureReturnV1
from cbond_on.domain.factors.defs.t1430_price_range_depth_ratio_v1 import T1430PriceRangeDepthRatioV1
__all__ = [
    "AacbFactor",
    "VolenFactor",
    "AmountSumFactor",
    "DepthImbalanceFactor",
    "MidpriceMoveFactor",
    "MomentumSlopeFactor",
    "PricePositionFactor",
    "RangeRatioFactor",
    "ReturnOpenToTimeFactor",
    "ReturnWindowFactor",
    "SpreadFactor",
    "TurnoverRateFactor",
    "VolatilityFactor",
    "VolumeImbalanceFactor",
    "VolumeSumFactor",
    "VwapFactor",
    "AmihudIlliqFactor",
    "MicropriceBiasFactor",
    "DepthSlopeFactor",
    "ReturnSkewFactor",
    "VwapGapFactor",
    "OrderFlowImbalanceV1Factor",
    "DepthWeightedImbalanceV1Factor",
    "IntradayMomentumV1Factor",
    "VolatilityScaledReturnV1Factor",
    "VolumePriceTrendV1Factor",
    "TradeIntensityV1Factor",
    "PriceLevelPositionV1Factor",
    "BidAskSpreadV1Factor",
    "StockBondMomentumGapV1Factor",
    "PremiumMomentumProxyV1Factor",
    "Alpha001SignedPowerV1Factor",
    "Alpha002CorrVolumeReturnV1Factor",
    "Alpha003CorrOpenVolumeV1Factor",
    "Alpha004TsRankLowV1Factor",
    "Alpha005VwapGapV1Factor",
    "Alpha006CorrOpenVolumeNegV1Factor",
    "Alpha007VolumeBreakoutV1Factor",
    "Alpha008OpenReturnMomentumV1Factor",
    "Alpha009CloseChangeFilterV1Factor",
    "Alpha010CloseChangeRankV1Factor",
    "Alpha011VwapCloseVolumeV1Factor",
    "Alpha012VolumeCloseReversalV1Factor",
    "Alpha013CovCloseVolumeV1Factor",
    "Alpha014ReturnOpenVolumeV1Factor",
    "Alpha015HighVolumeCorrV1Factor",
    "Alpha016CovHighVolumeV1Factor",
    "Alpha017CloseRankVolumeV1Factor",
    "Alpha018CloseOpenVolV1Factor",
    "Alpha019CloseMomentumSignV1Factor",
    "Alpha020OpenDelayRangeV1Factor",
    "Alpha021CloseVolatilityBreakoutV1Factor",
    "Alpha022HighVolumeCorrChangeV1Factor",
    "Alpha023HighMomentumV1Factor",
    "Alpha024CloseTrendFilterV1Factor",
    "Alpha025ReturnVolumeVwapRangeV1Factor",
    "Alpha026VolumeHighRankCorrV1Factor",
    "Alpha027VolumeVwapCorrSignalV1Factor",
    "Alpha028AdvLowCloseSignalV1Factor",
    "Alpha029ComplexRankSignalV1Factor",
    "Alpha030CloseSignVolumeV1Factor",
    "Alpha031CloseDecayMomentumV1Factor",
    "Alpha032VwapCloseMeanReversionV1Factor",
    "Alpha033OpenCloseRatioV1Factor",
    "Alpha034ReturnVolatilityRankV1Factor",
    "Alpha035VolumePriceMomentumV1Factor",
    "Alpha036ComplexCorrelationSignalV1Factor",
    "Alpha037OpenCloseCorrelationV1Factor",
    "Alpha038CloseRankRatioV1Factor",
    "Alpha039VolumeDecayMomentumV1Factor",
    "Alpha040HighVolatilityCorrV1Factor",
    "Alpha041GeometricMeanVwapV1Factor",
    "Alpha042VwapCloseRankRatioV1Factor",
    "Alpha043VolumeDelayMomentumV1Factor",
    "Alpha044HighVolumeRankCorrV1Factor",
    "Alpha045CloseSumCorrV1Factor",
    "Alpha046CloseDelayTrendV1Factor",
    "Alpha047InverseCloseVolumeV1Factor",
    "Alpha049CloseDelayThresholdV1Factor",
    "Alpha050VolumeVwapCorrMaxV1Factor",
    "Alpha051CloseDelayThresholdV2V1Factor",
    "Alpha052LowMomentumVolumeV1Factor",
    "Alpha053PricePositionDeltaV1Factor",
    "Alpha054PricePowerRatioV1Factor",
    "Alpha055CloseRangeVolumeCorrV1Factor",
    "Alpha057CloseVwapDecayV1Factor",
    "Alpha060PriceRangeVolumeScaleV1Factor",
    "Alpha062VwapOpenRankCompareV1Factor",
    "Alpha065OpenVwapMinSignalV1Factor",
    "Alpha066VwapLowDecayV1Factor",
    "Alpha068HighAdvRankSignalV1Factor",
    "Alpha072VwapVolumeDecayRatioV1Factor",
    "Alpha073VwapOpenDecayMaxV1Factor",
    "Alpha074CloseAdvRankCorrV1Factor",
    "Alpha075VwapVolumeLowAdvCorrV1Factor",
    "Alpha077MidPriceAdvDecayMinV1Factor",
    "Alpha078LowVwapAdvCorrV1Factor",
    "DailyOvernightReturnMeanV1Factor",
    "DailySharpeMeanV1Factor",
    "T1430AmountAccel30mV1",
    "T1430DepthConcentrationV1",
    "T1430DepthImbalanceChangeV1",
    "T1430LastReturn30mV1",
    "T1430MicropriceLastGapV1",
    "T1430MidReturn30mV1",
    "T1430PriceRange30mV1",
    "T1430SpreadChange30mV1",
    "T1430SpreadMeanGuarded30mV1",
    "T1430WindowVwapLastGapV1",
    "MicropriceDevLiquidityV1",
    "ObPressureSpreadRecoveryV1",
    "ObPressureSpreadV1",
    "MicropriceDeviationV1",
    "MicropriceDeviationLiqCondV1",
    "SpreadDepthPressureV1",
    "MicropriceDeviationLiqV1",
    "VwapDeviationLiquidityCondV1",
    "TailVolumeAbsorptionV1",
    "ObPressureSpreadPersistenceV1",
    "T1430PvInteractionV1",
    "T1430RangeVolNormV1",
    "T1430TailPressureV1",
    "T1430RangeVolumeV1",
    "T1430AmountIntensityV1",
    "T1430ClosePositionV1",
    "T1430PvRatioV1",
    "T1430RangeVolumeNormV1",
    "T1430RangeNormVolumeV1",
    "T1430AmountVolRatioV1",
    "T1430CloseRangePositionV1",
    "T1430VolatilityTurnoverV1",
    "T1430VwapDeviationV1",
    "T1430VolTurnoverRatioV1",
    "T1430SpreadLiquidityRatioV1",
    "T1430MicropriceMidDeviationV1",
    "T1430OrderflowImbalanceV1",
    "T1430DepthWeightedMidV1",
    "T1430BidAskImbalanceV1",
    "T1430MicropriceMidpointDevV1",
    "T1430OrderFlowToxicityV1",
    "T1430BidAskSpreadV1",
    "T1430OrderImbalanceV1",
    "T1430MicroPriceDeviationV1",
    "T1430RelativeStrengthV1",
    "T1430VolumeWeightedPriceDeviationV1",
    "T1430MidPriceReturnV1",
    "T1430SpreadWidthNormV1",
    "T1430MicropriceMidSpreadV1",
    "T1430OrderFlowImbalanceV1",
    "T1430RelativeBidAskStrengthV1",
    "T1430LiqAdjSpreadV1",
    "T1430MicropriceImbalanceV1",
    "T1430VolumeWeightedMidV1",
    "T1430MicropriceLiquidityV1",
    "T1430ObImbalanceFlowV1",
    "T1430SpreadVolatilityRatioV1",
    "T1430RelativeStrengthVwapV1",
    "T1430DepthPressureReturnV1",
    "T1430PriceRangeDepthRatioV1",
]

