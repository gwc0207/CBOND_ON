# CBOND_ON 因子公式目录（中文）

- 更新时间：`2026-03-26`
- 源配置：`cbond_on/config/factor/factors_20260325.json5`
- 源代码：`cbond_on/factors/defs/*.py`
- 范围：当前批处理使用的 55 个活跃因子

## 记号约定
- $rank_{cs}(x)$：同一交易日截面分位数排名（百分位）。
- $rank_t(x)$：同一标的、同一交易日内序列分位数排名（百分位）。
- $tsrank_n(x)$：同一标的日内最近 $n$ 个快照内，末值的时序排名。
- $\Delta_n x_t=x_t-x_{t-n}$，$delay_n(x_t)=x_{t-n}$。
- $MA_n(x)$：同一标的日内滚动均值。
- $Corr_n(x,y)$、$Cov_n(x,y)$：同一标的日内最近 $n$ 个快照的相关/协方差。
- $vwap_n=\dfrac{\sum_{k=t-n+1}^{t} amount_k}{\sum_{k=t-n+1}^{t} volume_k+\epsilon}$。
- 变量命名与项目字段一致：如 `last/open/high/low/pre_close/volume/amount/num_trades`，下标 $t$ 表示当前时点。
- `open` 语义统一规则：优先使用 $mid=\dfrac{ask\_price1+bid\_price1}{2}$，若缺失再回退到 `open`。

## 因子公式（55）

1. `aacb_l3`  
公式：$F=\dfrac{1}{T}\sum_{s=1}^{T}\dfrac{\bar A_s-\bar B_s}{(ask_price_{1,s}+bid_price_{1,s})/2}$，其中 $\bar A_s=\dfrac{1}{3}\sum_{i=1}^{3}ask_price_{i,s},\ \bar B_s=\dfrac{1}{3}\sum_{i=1}^{3}bid_price_{i,s}$。  
解释：衡量 L1-L3 盘口的平均价差压力，值越大代表卖盘相对更强。

2. `volen_f60_s10_l3`  
公式：$depth_s=\sum_{i=1}^{3}(ask_volume_{i,s}+bid_volume_{i,s}),\ F=\dfrac{MA_{60}(depth)_t}{MA_{10}(depth)_t}$。  
解释：比较慢速与快速深度均值，刻画盘口深度的趋势变化。

3. `ret_5m`  
公式：$F=\dfrac{last_t-last_{t-5m}}{last_{t-5m}+\epsilon}$。  
解释：近 5 分钟收益率。

4. `ret_10m`  
公式：$F=\dfrac{last_t-last_{t-10m}}{last_{t-10m}+\epsilon}$。  
解释：近 10 分钟收益率。

5. `ret_30m`  
公式：$F=\dfrac{last_t-last_{t-30m}}{last_{t-30m}+\epsilon}$。  
解释：近 30 分钟收益率。

6. `ret_60m`  
公式：$F=\dfrac{last_t-last_{t-60m}}{last_{t-60m}+\epsilon}$。  
解释：近 60 分钟收益率。

7. `ret_open_0930_1430`  
公式：$F=\dfrac{last_{14:30}-last_{09:30}}{last_{09:30}+\epsilon}$。  
解释：从 09:30 到 14:30 的日内主区间收益。

8. `mom_slope_30m`  
公式：在近 30 分钟序列上回归 $last_k=\alpha+\beta k+\varepsilon_k$，取 $F=\hat\beta$。  
解释：价格动量斜率，正值表示上行趋势更陡。

9. `vol_30m`  
公式：$r_k=\ln\dfrac{last_k}{last_{k-1}+\epsilon},\ F=std(r_k,\ 30m)$。  
解释：近 30 分钟对数收益波动率。

10. `vol_60m`  
公式：$r_k=\ln\dfrac{last_k}{last_{k-1}+\epsilon},\ F=std(r_k,\ 60m)$。  
解释：近 60 分钟对数收益波动率。

11. `range_30m`  
公式：$F=\dfrac{\max(last)-\min(last)}{last_t+\epsilon}$（窗口 30 分钟）。  
解释：近 30 分钟振幅占当前价比例。

12. `pos_30m`  
公式：$F=\dfrac{last_t-\min(last)}{\max(last)-\min(last)+\epsilon}$（窗口 30 分钟）。  
解释：当前价格位于区间高低点之间的位置。

13. `volume_30m`  
公式：$F=\sum_{k=t-30m+1}^{t}volume_k$。  
解释：近 30 分钟成交量总和。

14. `amount_30m`  
公式：$F=\sum_{k=t-30m+1}^{t}amount_k$。  
解释：近 30 分钟成交额总和。

15. `vwap_30m`  
公式：$F=\dfrac{\sum amount}{\sum volume+\epsilon}$（窗口 30 分钟）。  
解释：近 30 分钟成交量加权平均价。

16. `volimb_30m_l3`  
公式：$F=\dfrac{\sum_{\tau}\sum_{i=1}^{3}bid_volume_{i,\tau}-\sum_{\tau}\sum_{i=1}^{3}ask_volume_{i,\tau}}{\sum_{\tau}\sum_{i=1}^{3}bid_volume_{i,\tau}+\sum_{\tau}\sum_{i=1}^{3}ask_volume_{i,\tau}+\epsilon}$（窗口 30 分钟）。  
解释：近 30 分钟 L3 累积盘口量不平衡。

17. `spread`  
公式：$F=\dfrac{ask_price1-bid_price1}{(ask_price1+bid_price1)/2+\epsilon}$。  
解释：当前一档相对买卖价差。

18. `depth_imb_l3`  
公式：$F=\dfrac{\sum_{i=1}^{3}bid\_volume_i-\sum_{i=1}^{3}ask\_volume_i}{\sum_{i=1}^{3}bid\_volume_i+\sum_{i=1}^{3}ask\_volume_i+\epsilon}$。  
解释：当前快照 L3 盘口深度不平衡。

19. `mid_move_30m`  
公式：$mid_k=(ask_price_{1,k}+bid_price_{1,k})/2,\ F=\dfrac{mid_t-mid_{t-30m}}{mid_{t-30m}+\epsilon}$。  
解释：近 30 分钟中间价收益。

20. `turnover_30m`  
公式：$F=\dfrac{\sum_{k=t-30m+1}^{t}volume_k}{num\_trades_t+\epsilon}$。  
解释：成交量相对笔数的强度指标（按现有实现口径）。

21. `amihud_30m`  
公式：$F=\dfrac{\left|\dfrac{last_t-last_{t-30m}}{last_{t-30m}+\epsilon}\right|}{\sum_{k=t-30m+1}^{t}amount_k+\epsilon}$。  
解释：经典 Amihud 思路，单位成交额对应的价格冲击。

22. `microprice_bias_l1`  
公式：$micro=\dfrac{ask_price1\cdot bid_volume1+bid_price1\cdot ask_volume1}{bid_volume1+ask_volume1+\epsilon},\ mid=\dfrac{ask_price1+bid_price1}{2},\ F=\dfrac{micro-mid}{mid+\epsilon}$。  
解释：微价格相对中间价的偏离，体现一档订单簿压力。

23. `depth_slope_l5`  
公式：$F=\dfrac{Slope^{ask}_{1\sim5}-Slope^{bid}_{1\sim5}}{mid+\epsilon}$，其中两侧斜率由相邻档位价差按挂单量加权构造。  
解释：衡量五档订单簿“陡峭度”差异。

24. `ret_skew_30m`  
公式：$r_k=\ln\dfrac{last_k}{last_{k-1}+\epsilon},\ F=Skew(r_k,\ 30m)$。  
解释：近 30 分钟收益分布偏度。

25. `vwap_gap_30m`  
公式：$F=\dfrac{last_t-vwap_{30m}}{vwap_{30m}+\epsilon}$。  
解释：当前价相对近 30 分钟 vwap 的偏离。

26. `order_flow_imbalance_v1`  
公式：$F=\dfrac{bid_volume1-ask_volume1}{bid_volume1+ask_volume1+\epsilon}$。  
解释：一档买卖量差驱动的订单流不平衡。

27. `depth_weighted_imbalance_v1`  
公式：$w=[5,4,3,2,1],\ F=\dfrac{\sum_{i=1}^{5}w_i\cdot bid\_volume_i-\sum_{i=1}^{5}w_i\cdot ask\_volume_i}{\sum_{i=1}^{5}w_i\cdot bid\_volume_i+\sum_{i=1}^{5}w_i\cdot ask\_volume_i+\epsilon}$。  
解释：对近档赋更高权重的五档深度不平衡。

28. `intraday_momentum_v1`  
公式：$F=\dfrac{last_t-open_t}{open_t+\epsilon}$。  
解释：开盘到当前的日内动量。

29. `volatility_scaled_return_v1`  
公式：$F=\dfrac{(last_t-open_t)/(open_t+\epsilon)}{(high_t-low_t)/(pre\_close_t+\epsilon)+\epsilon}$。  
解释：用日内波动区间归一化的收益强度。

30. `volume_price_trend_v1`  
公式：$F=\dfrac{last_t-pre\_close_t}{pre\_close_t+\epsilon}\cdot amount_t$。  
解释：价格变动与成交额耦合的趋势强度。

31. `trade_intensity_v1`  
公式：$F=\dfrac{amount_t/(num\_trades_t+\epsilon)}{pre\_close_t+\epsilon}$。  
解释：单位成交笔对应金额的价格尺度归一化。

32. `price_level_position_v1`  
公式：$F=\dfrac{last_t-low_t}{high_t-low_t+\epsilon}$。  
解释：当前价在当日高低区间中的相对位置。

33. `stock_bond_momentum_gap_v1`  
公式：$F=\dfrac{bond_last-bond_open}{bond_open+\epsilon}-\dfrac{stock_last-stock_open}{stock_open+\epsilon}$。  
解释：转债与正股的日内动量差。

34. `bid_ask_spread_v1`  
公式：$F=\dfrac{ask_price1-bid_price1}{(ask_price1+bid_price1)/2+\epsilon}$。  
解释：一档相对价差（与 `spread` 同口径）。

35. `premium_momentum_proxy_v1`  
公式：$F=\dfrac{bond_last-bond_pre\_close}{bond_pre\_close+\epsilon}-\dfrac{stock_last-stock_pre\_close}{stock_pre\_close+\epsilon}$。  
解释：债股当日收益差，作为溢价动量代理。

36. `alpha001_signed_power_v1`  
公式：$ret_t=\dfrac{last_t-pre\_close_t}{pre\_close_t+\epsilon}$，$base_t=\sigma_{20}(ret)_t$（若 $ret_t<0$）否则 $base_t=last_t$，$F=rank_{cs}\!\left(\max_{5}\left(sign(base)\cdot |base|^2\right)\right)-0.5$。  
解释：将负收益波动与价格水平非线性放大后做截面排序。

37. `alpha002_corr_volume_return_v1`  
公式：$F=-Corr_6\!\left(rank_t(\Delta_2\log(volume)),\ rank_t\!\left(\dfrac{last-open}{open+\epsilon}\right)\right)$。  
解释：量变与收益在短窗内的同步性（取负号）。

38. `alpha003_corr_open_volume_v1`  
公式：$F=-Corr_{10}(rank_t(open),\ rank_t(volume))$。  
解释：开盘价序列与成交量序列相关性的反向指标。

39. `alpha004_ts_rank_low_v1`  
公式：$F=-tsrank_9(rank_t(low))$。  
解释：低价序列末值在最近窗口中的相对高低（取反）。

40. `alpha005_vwap_gap_v1`  
公式：$F=rank_{cs}(open-MA_{10}(vwap))\cdot\left(-\left|rank_{cs}(last-vwap)\right|\right)$。  
解释：开盘相对均值偏离与当前相对 vwap 偏离的组合。

41. `alpha006_corr_open_volume_neg_v1`  
公式：$F=-Corr_{10}(open,\ volume)$。  
解释：开盘价与成交量原始相关性的反向指标。

42. `alpha007_volume_breakout_v1`  
公式：$F=\begin{cases}-1,& MA_{20}(amount)\ge volume\\-tsrank_{60}(|\Delta_7 last|)\cdot sign(\Delta_7 last),& \text{otherwise}\end{cases}$。  
解释：先用量能条件判定，再用价格变化幅度与方向给信号。

43. `alpha008_open_return_momentum_v1`  
公式：$S_t=\left(\sum_{k=t-4}^{t}open_k\right)\left(\sum_{k=t-4}^{t}\dfrac{last_k-open_k}{open_k+\epsilon}\right),\ F=-rank_{cs}(S_t-delay_{10}(S_t))$。  
解释：开盘价累计与收益累计乘积的变化强度。

44. `alpha009_close_change_filter_v1`  
公式：$d_t=\Delta_1(last_t),\ F=\begin{cases}d_t,& \min(d,5)>0\ \text{或}\ \max(d,5)<0\\-d_t,& \text{其他}\end{cases}$。  
解释：用短窗单边性过滤的价格变化反转/延续信号。

45. `alpha010_close_change_rank_v1`  
公式：$d'_t=\begin{cases}\Delta_1(last_t),& \min(\Delta_1(last),4)>0\ \text{或}\ \max(\Delta_1(last),4)<0\\-\Delta_1(last_t),& \text{其他}\end{cases},\ F=rank_{cs}(d'_t)$。  
解释：`alpha009` 思路的截面排序版本。

46. `alpha011_vwap_close_volume_v1`  
公式：$F=\left(rank_{cs}(\max_3(vwap-last))+rank_{cs}(\min_3(vwap-last))\right)\cdot rank_{cs}(\Delta_3 volume)$。  
解释：vwap 偏离极值与成交量变化的联合信号。

47. `alpha012_volume_close_reversal_v1`  
公式：$F=sign(\Delta_1 volume)\cdot(-\Delta_1 last)$。  
解释：量增价跌/量减价涨时给出正向反转强度。

48. `alpha013_cov_close_volume_v1`  
公式：$F=-rank_{cs}\!\left(Cov_5(rank_t(last),\ rank_t(volume))\right)$。  
解释：价格与成交量序列协方差的反向截面信号。

49. `alpha014_return_open_volume_v1`  
公式：$r_t=\dfrac{last_t-pre\_close_t}{pre\_close_t+\epsilon},\ F=\left(-rank_{cs}(\Delta_3 r_t)\right)\cdot Corr_{10}(open,\ volume)$。  
解释：收益变化与开盘-成交量相关性叠加。

50. `alpha015_high_volume_corr_v1`  
公式：$c_t=Corr_3(rank_t(high),\ rank_t(volume)),\ F=-\sum_{j=0}^{2}rank_t(c_{t-j})$。  
解释：高价与成交量相关性的短窗累计强度（取反）。

51. `alpha016_cov_high_volume_v1`  
公式：$F=-rank_{cs}\!\left(Cov_5(rank_t(high),\ rank_t(volume))\right)$。  
解释：高价与成交量协方差的反向截面信号。

52. `alpha017_close_rank_volume_v1`  
公式：$F=\left(-rank_{cs}(tsrank_{10}(last))\right)\cdot rank_{cs}(\Delta_1\Delta_1 last)\cdot rank_{cs}\!\left(tsrank_5\!\left(\dfrac{volume}{MA_{20}(amount)+\epsilon}\right)\right)$。  
解释：价格位置、二阶动量与相对量能三者乘积。

53. `alpha018_close_open_vol_v1`  
公式：$F=-rank_{cs}\!\left(std_5(|last-open|)+(last-open)+Corr_{10}(last,open)\right)$。  
解释：价差波动、价差水平与价相关性的综合反向信号。

54. `alpha019_close_momentum_sign_v1`  
公式：$F=-sign\!\left((last-delay_7(last))+\Delta_7 last\right)\cdot\left(1+rank_{cs}\!\left(1+\sum_{250}\dfrac{last-pre\_close}{pre\_close+\epsilon}\right)\right)$。  
解释：长窗累计收益强度由短窗方向符号调制。

55. `alpha020_open_delay_range_v1`  
公式：$F=\left(-rank_{cs}(open-delay_1(high))\right)\cdot rank_{cs}(open-delay_1(last))\cdot rank_{cs}(open-delay_1(low))$。  
解释：开盘价相对昨日高/收/低位置的联合截面信号。

## 维护规则
- `factors_YYYYMMDD.json5` 发生变更时，同提交更新本文件。
- 因子实现发生变更时，必须同步更新本文件中的数学表达与解释。
- 债股联动因子需与 `docs/factor_development.md` 的 `stock_panel`、`bond_stock_map` 假设保持一致。


