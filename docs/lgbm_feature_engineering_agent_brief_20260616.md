# CBOND_ON LGBM 特征工程评估说明报告

生成日期: 2026-06-16

本文面向一个完全不了解 `CBOND_ON` 项目的外部 agent。目标是帮助其快速理解当前项目链路、已有特征工程实验结果、现存问题，以及后续值得评估的特征工程技术方案。

本文不进入代码级实现细节。所有实盘相关改动都必须在执行前向负责人确认最终口径，包括模型 ID、中性化模式、因子集合、交易池、输出路径、数据库写入和调度配置。

## 1. 当前项目现状

### 1.1 项目定位

`CBOND_ON` 是一个可转债日内策略项目，当前重点研究方向是基于 14:30 前后的因子，对后续短窗口收益进行预测，并通过模型分数选出交易标的。

当前主要研究对象是树模型 LightGBM，核心模型为:

`lgbm_screened_no_winsor_neutral_v1`

当前主要输入特征为 27 个已筛选因子，模型以滚动训练方式生成每日分数，再进入回测和实盘链路。

### 1.2 主链路

当前项目的主链路可以理解为:

1. 数据准备
   - 读取原始可转债行情、成交、盘口、日频基础数据等。
   - 形成清洗后的日内和日频数据。

2. Panel 构建
   - 生成指定时间点的截面数据。
   - 当前重点 panel 是 `T1430`。

3. Label 构建
   - 当前模型常用预测目标为从 `14:30` 附近到后续窗口的收益标签。
   - 当前主力 LGBM 配置里，`factor_time = 14:30`，`label_time = 14:42`。

4. Factor Batch
   - 批量计算因子。
   - 因子数据按交易日、panel 维度落地。
   - 研究因子、报告因子、筛选因子主要在这一层完成。

5. Model Score
   - 读取因子和标签。
   - 做训练前处理和滚动训练。
   - 生成每日每只转债的模型分数。

6. Model Eval
   - 用模型分数评估 IC、Rank IC、分箱表现、稳定性等。
   - 也用于参数搜索和模型比较。

7. Backtest
   - 基于模型分数做 topK 选券和组合回测。
   - 输出日收益、月收益、净值曲线、IC 汇总、持仓、换手和图表。

8. Live / LiveLaunch
   - 用当前实盘配置生成交易候选。
   - 实盘链路不能被研究实验自动覆盖。

### 1.3 当前主力 LGBM 配置

当前研究结论里表现最好的口径是:

`zscore 开 + winsor 关 + 全量中性化开`

对应模型:

`lgbm_screened_no_winsor_neutral_v1`

当前 27 个输入因子为:

1. `cb_overnight_return_mean_20d`
2. `cb_overnight_return_mean_5d`
3. `cb_overnight_return_mean_60d`
4. `range_30m`
5. `cb_overnight_return_mean_10d`
6. `amount_30m`
7. `vol_30m`
8. `alpha001_signed_power_v1`
9. `alpha030_close_sign_volume_v1`
10. `volume_30m`
11. `depth_weighted_imbalance_v1`
12. `alpha024_close_trend_filter_v1`
13. `cb_overnight_sharpe_20_0930_0935`
14. `daily_sharpe_twap_5d_mean5`
15. `cb_overnight_sharpe_5_0930_0935`
16. `mid_move_30m`
17. `premium_momentum_proxy_v1`
18. `volen_f60_s10_l3`
19. `daily_sharpe_twap_20d_mean5`
20. `cb_overnight_return_mean_40d`
21. `alpha041_geometric_mean_vwap_v1`
22. `alpha078_low_vwap_adv_corr_v1`
23. `mom_slope_30m`
24. `alpha019_close_momentum_sign_v1`
25. `ret_10m`
26. `alpha025_return_volume_vwap_range_v1`
27. `alpha050_volume_vwap_corr_max_v1`

当前已采用的输入处理:

| 处理项 | 当前状态 | 说明 |
| --- | --- | --- |
| 截面 zscore | 开 | 每日截面对每个因子标准化 |
| winsor | 关 | 历史对比显示关闭 winsor 整体更好 |
| 中性化 | 全量开启 | 对全部 27 个因子统一做风格中性化 |
| 部分中性化 | 不采用 | 已确认效果差，当前策略原则为要么全量中性化，要么不做中性化 |

当前全量中性化使用的暴露变量:

| 暴露变量 | 含义 |
| --- | --- |
| `log_price` | 转债价格取对数 |
| `log_amount` | 成交额取 `log1p` |
| `premium` | 转股溢价率 |
| `remain_size` | 剩余规模取 `log1p` |
| `year_to_mat` | 剩余期限 |

## 2. 已做过的特征工程尝试和结果

### 2.1 winsor 开关

已比较过 winsor 开启和关闭。

在 `2025-01-01` 到 `2026-05-11` 的共同回测区间:

| 配置 | 总收益 | Sharpe | 结论 |
| --- | ---: | ---: | --- |
| zscore 开，winsor 开，中性化关 | 45.47% | 2.50 | 可用，但不是最优 |
| zscore 开，winsor 关，中性化关 | 55.03% | 2.67 | 优于 winsor 开 |
| zscore 开，winsor 关，全量中性化开 | 65.81% | 3.11 | 当前最好 |

结论:

关闭 winsor 更好。当前推断是，尾部信息本身可能包含有效 alpha，简单截尾会削弱模型可用信号。后续不建议直接回到传统 winsor，而应测试更稳健的标准化方式。

相关结果路径:

`D:\cbond_on\results\compare\feature_engineering_monthly_20260616`

### 2.2 全量中性化

已比较过无中性化和全量中性化。

在 `2025-01-01` 到 `2026-05-11` 的共同回测区间:

| 配置 | 总收益 | Sharpe |
| --- | ---: | ---: |
| zscore 开，winsor 关，中性化关 | 55.03% | 2.67 |
| zscore 开，winsor 关，全量中性化开 | 65.81% | 3.11 |

结论:

全量中性化在当前 27 因子组合上有效。它提升了总收益和夏普，是当前最值得保留的特征工程处理。

但需要注意，单月表现并非全胜:

- 2026-03，三个配置都亏，全量中性化亏损更大。
- 2026-04，`zscore 开，winsor 关，中性化关` 的收益高于全量中性化。

因此，中性化提升的是整体风险收益表现，不代表每个市场阶段都占优。

### 2.3 部分中性化

曾尝试过“部分因子中性化”，即部分因子做中性化，部分因子保留原始暴露。

结果较差。此前 2026 年区间的回测表现大致为:

| 配置 | 策略收益 | benchmark 收益 | 超额收益 | Sharpe | 超额 Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| 部分中性化 | 8.82% | 10.28% | -1.14% | 1.63 | -0.31 |

同期对比:

| 配置 | 策略收益 | 超额收益 | Sharpe |
| --- | ---: | ---: | ---: |
| 无中性化 | 19.02% | 8.10% | 3.46 |
| 全量中性化 | 18.37% | 7.50% | 3.43 |

结论:

部分中性化明显不如无中性化和全量中性化。当前项目原则是:

要么全量中性化，要么不做中性化。不要默认恢复部分中性化。

### 2.4 CNN/LSTM 方向的特征工程尝试

虽然当前重点回到 LGBM，但此前也做过 CNN 输入特征工程实验，对后续思路有参考价值。

已验证过的方向包括:

| 方向 | 结果摘要 |
| --- | --- |
| zscore 后 clip | 对 CNN 序列模型帮助最明显 |
| 去重单独使用 | 效果变差 |
| temporal diff / slope | 对去重损失有一定修复，但不是决定性提升 |
| fill missing + missing mask | 让更大因子集合能够稳定运行 |

对 LGBM 的启发:

1. 极端值处理不一定要用 winsor，可以考虑 zscore 后连续压缩或稳健变换。
2. 直接去重可能会损失信息，应该用相关簇和替代特征一起评估。
3. 缺失值不一定要整行删除，缺失本身可能携带信息。

## 3. 当前暴露出来的问题

### 3.1 因子高度重复

对当前 27 因子做抽样诊断后，发现多个因子几乎重复或高度相关。

典型例子:

| 因子 A | 因子 B | 相关性 |
| --- | --- | ---: |
| `daily_sharpe_twap_5d_mean5` | `cb_overnight_sharpe_5_0930_0935` | 0.999995 |
| `cb_overnight_sharpe_20_0930_0935` | `daily_sharpe_twap_20d_mean5` | 0.999982 |
| `amount_30m` | `volume_30m` | 0.939752 |
| `cb_overnight_return_mean_60d` | `cb_overnight_return_mean_40d` | 0.850063 |
| `range_30m` | `vol_30m` | 0.775671 |

问题:

树模型可以处理共线性，但高度重复会带来几个负面影响:

1. 分裂重要性被重复因子分散。
2. 不同滚动窗口里模型选择的重复因子可能切换，导致稳定性下降。
3. 解释性变差。
4. 超参搜索和特征选择容易被重复信号误导。

诊断输出路径:

`D:\cbond_on\results\compare\feature_engineering_diagnostics_20260616\current27_factor_corr_pairs_sample.csv`

### 3.2 当前标准化方式仍然粗糙

当前主力配置是每日截面 zscore。

问题:

1. 普通均值和标准差对极端值敏感。
2. winsor 关闭虽然整体更好，但并不代表极端值完全不需要处理。
3. 对树模型来说，特征单调排序通常比绝对尺度更重要，rank 类标准化可能更稳。

因此后续更合理的方向是测试稳健标准化，而不是简单测试 winsor 开关。

### 3.3 当前缺失值处理可能损失样本

当前训练口径里，如果一行样本任一因子缺失，可能会整行丢弃。

问题:

1. 长窗口因子更容易缺失，样本可能被系统性过滤。
2. 某些缺失不是噪声，而是可交易状态、停牌、流动性或上市时间的隐含信息。
3. LightGBM 原生支持缺失值，直接整行删除可能不是最优。

建议:

让 LightGBM 保留 NaN，同时加 `min_available_ratio` 和缺失指示变量进行对照实验。

### 3.4 全量中性化有效，但暴露变量可能不够

当前全量中性化只剥离 5 类风格暴露:

价格、成交额、转股溢价率、剩余规模、剩余期限。

问题:

1. 可转债还存在信用、评级、正股属性、波动率、流动性状态、行业或主题暴露。
2. 当前中性化可能剥离了部分风格，但仍残留一些结构性偏置。
3. 如果新增 exposure 不当，也可能剥离 alpha，因此必须做分组实验。

### 3.5 IC 和回测收益存在分歧

当前全量中性化版本在 `2025-01-01` 到 `2026-05-11` 的回测结果较好:

- 策略收益: 65.81%
- Sharpe: 3.11
- 超额收益: 37.34%
- 超额 Sharpe: 3.03

但对应回测区间的 Rank IC 均值接近 0:

- IC mean: 0.0177
- Rank IC mean: 0.0008

解释:

当前策略收益可能更多来自 topK 持仓路径、行情阶段、交易规则和少数强行情段，而不是每天稳定的全截面排序能力。

这意味着后续评估不能只看 IC，也不能只看最终收益。必须同时看:

1. 月度收益和夏普。
2. 回撤。
3. topK 选票稳定性。
4. 压力月份表现。
5. 分数分布。
6. 分箱收益。
7. 特征重要性稳定性。

### 3.6 缺少标准化的特征诊断产物

当前训练产物里有:

- `features.json`
- `rolling_metrics.csv`
- `rolling_score_guard.csv`
- backtest 报告

但缺少稳定产出的:

1. 训练后 feature importance。
2. permutation importance。
3. SHAP 或近似解释。
4. 预处理前后因子相关矩阵。
5. 单因子 IC / Rank IC。
6. 因子簇报告。
7. 中性化前后 exposure 残留报告。

没有这些诊断，后续特征工程容易变成只看回测终局，难以判断为什么有效或失效。

## 4. 建议优先评估的特征工程技术方案

### 4.1 P0: 先补诊断体系

目的:

在继续改特征前，先知道当前模型到底依赖什么。

建议产物:

| 诊断项 | 用途 |
| --- | --- |
| feature importance | 看模型主要使用哪些因子 |
| permutation importance | 看因子被扰动后对表现的真实影响 |
| SHAP 或近似 SHAP | 看方向性和非线性关系 |
| 因子相关簇 | 识别重复信号 |
| 中性化前后 exposure 残留 | 确认中性化是否真正生效 |
| 预处理后单因子 IC | 判断预处理是否破坏信号 |
| topK 选票重合度 | 判断改动是否导致持仓剧烈漂移 |

评估标准:

不仅看全周期收益，也看 2025 年以来逐月收益、夏普、最大回撤、压力月份和选票重合。

### 4.2 P1: 高相关因子去重或聚合

目标:

处理几乎重复的因子，降低模型不稳定性。

可测试方案:

1. 相关性大于 0.98 的因子只保留一个。
2. 同一 family 的多窗口因子做聚合。
3. 对多窗口序列构造斜率、短长差、曲率，而不是简单保留所有窗口。

重点候选:

| 因子组 | 建议 |
| --- | --- |
| 两组 5 日 Sharpe 因子 | 二选一或合成一个 |
| 两组 20 日 Sharpe 因子 | 二选一或合成一个 |
| `amount_30m` 和 `volume_30m` | 二选一，或改为成交强度类派生特征 |
| 5/10/20/40/60 日隔夜收益均值 | 保留 level，同时新增 spread/slope，不必全量堆叠 |

风险:

CNN 实验里“单纯去重”曾经变差，所以 LGBM 也不要只做删除。更合理的是“去重 + 派生聚合特征”一起评估。

### 4.3 P1: 稳健标准化替代普通 zscore

当前 winsor 关闭更好，但普通 zscore 仍然对极端值敏感。

建议测试:

| 方法 | 说明 |
| --- | --- |
| rank percentile | 每日截面转成 0 到 1 分位数 |
| rank gaussian | 先做截面 rank，再映射为正态分布 |
| robust zscore | 使用 median 和 MAD 替代 mean/std |
| tanh zscore | 对 zscore 后的值做连续压缩 |
| signed log transform | 对偏态严重因子做符号 log 压缩 |

优先顺序:

1. `rank_pct`
2. `robust_zscore`
3. `tanh_zscore`
4. `rank_gauss`

原因:

树模型更依赖排序和分裂阈值，rank 类处理可能比普通 zscore 更稳。

### 4.4 P1: 缺失值策略重做

当前整行丢弃缺失样本可能不适合 LightGBM。

建议测试:

| 方法 | 说明 |
| --- | --- |
| keep_nan | 保留 NaN，让 LightGBM 自行处理 |
| min_available_ratio | 允许部分因子缺失，例如至少 80% 因子可用 |
| missing indicator | 为关键因子增加是否缺失的 0/1 指示变量 |
| group median fill | 用当日截面中位数填充，同时保留 missing flag |

优先建议:

先测试 `keep_nan + min_available_ratio`，再测试 `median fill + missing flag`。

### 4.5 P2: 扩展全量中性化 exposure

当前全量中性化有效，但 exposure 可能不完整。

建议新增候选 exposure:

| exposure 类型 | 可能字段或构造 |
| --- | --- |
| 信用质量 | rating、评级分层、评级变化 |
| 正股状态 | 正股动量、正股波动、正股成交额 |
| 转债流动性 | 过去 N 日成交额、换手、盘口深度 |
| 波动状态 | 转债历史波动、日内 range、隔夜波动 |
| 估值状态 | 溢价率分位数、价格分位数 |
| 行业或主题 | 正股行业 onehot 或主题分组 |

评估方式:

不要一次性把所有 exposure 加进去。建议逐组加入:

1. 当前 5 exposure baseline。
2. 加信用。
3. 加正股状态。
4. 加流动性和波动。
5. 加行业 onehot。

每次只比较一个新增组，避免无法定位收益变化来源。

### 4.6 P2: 多窗口 family 特征重构

当前多个因子是同一逻辑不同窗口，例如 5/10/20/40/60 日隔夜收益均值。

建议从“堆窗口”改为“描述曲线形态”:

| 派生特征 | 含义 |
| --- | --- |
| short_minus_long | 短周期均值减长周期均值 |
| slope | 多窗口收益随窗口长度变化的斜率 |
| curvature | 短中长周期变化的弯曲程度 |
| acceleration | 近期变化是否加速 |
| stability | 多窗口信号是否一致 |

优点:

1. 减少共线性。
2. 保留趋势结构。
3. 更容易解释市场状态。

### 4.7 P2: 显式交互特征

LightGBM 可以自动学习交互，但金融含义强的交互可以直接构造。

候选交互:

| 交互 | 解释 |
| --- | --- |
| 动量 × 流动性 | 高动量是否需要成交确认 |
| 溢价动量 × 深度不平衡 | 估值变化是否伴随盘口压力 |
| range / volume | 单位成交量推动的价格波动 |
| ret_10m × depth_weighted_imbalance | 短期价格方向是否被盘口支持 |
| overnight_mean_spread × intraday_momentum | 隔夜趋势和日内趋势是否共振 |

风险:

交互特征容易过拟合，必须看滚动稳定性和压力月份表现。

### 4.8 P3: 标签和目标函数联动

当前策略最终是按分数 topK 选券，因此模型目标应和排序质量对齐。

可测试:

| 方向 | 说明 |
| --- | --- |
| day-rank label | 每日截面标签改为排序或分位数 |
| day-zscore label | 每日标签标准化，降低行情日尺度影响 |
| LGBM ranker | 直接优化排序目标 |
| topK-aware objective proxy | 用 top bin 收益和稳定性选择参数 |

注意:

这是模型目标侧改造，不是纯输入特征工程。建议排在输入特征稳定之后。

## 5. 建议实验矩阵

为了让不同 agent 的结果可比较，建议固定:

- 训练区间。
- 回测区间。
- 交易池。
- topK 规则。
- 调仓和交易成本。
- 模型超参。
- 是否全量中性化。

建议第一轮实验:

| 实验名 | 变化点 | 预期验证 |
| --- | --- | --- |
| baseline | 当前最好口径 | 对照组 |
| dedup_high_corr | 去掉相关性极高的重复因子 | 是否提升稳定性 |
| family_spread | 多窗口因子改为 spread/slope | 是否保留信息并降低共线性 |
| rank_pct_norm | 截面 rank percentile 标准化 | 是否优于普通 zscore |
| robust_zscore | median/MAD 标准化 | 是否提升压力月份表现 |
| keep_nan_min_ratio | 保留 NaN，允许部分缺失 | 是否增加有效样本并提升表现 |
| missing_flag | 缺失指示变量 | 缺失本身是否有信息 |
| neutral_plus_credit | 中性化加入信用 exposure | 是否剥离信用暴露后更稳 |
| neutral_plus_stock | 中性化加入正股状态 exposure | 是否降低正股驱动噪声 |
| interaction_core | 少量金融含义明确的交互 | 是否提升 topK 排序 |

## 6. 统一评估指标

每个实验都应输出:

| 类型 | 指标 |
| --- | --- |
| 策略表现 | 总收益、年化收益、Sharpe、最大回撤、Calmar、胜率 |
| 月度稳定性 | 2025 年以来逐月收益和夏普 |
| 超额表现 | 相对 benchmark 的超额收益、超额 Sharpe |
| 压力月份 | 2025-11、2026-03、2026-04 单独表现 |
| 排序质量 | IC、Rank IC、分箱收益、top bin 收益 |
| 持仓变化 | topK 选票重合度、换手率 |
| 特征诊断 | importance、相关簇、缺失率、预处理后分布 |
| 中性化诊断 | exposure 残留、残差分布、被剥离强度 |

重要原则:

不要只用单一指标判断。当前项目已经出现“回测收益好，但 Rank IC 均值接近 0”的情况，因此必须综合评估。

## 7. 目前可直接引用的结果路径

月度特征工程对比表:

`D:\cbond_on\results\compare\feature_engineering_monthly_20260616`

其中:

- `feature_engineering_monthly_return_sharpe.xlsx`
- `feature_engineering_monthly_return_sharpe_display.csv`
- `feature_engineering_common_period_summary.csv`

当前 27 因子诊断:

`D:\cbond_on\results\compare\feature_engineering_diagnostics_20260616`

其中:

- `current27_factor_distribution_sample.csv`
- `current27_factor_corr_pairs_sample.csv`
- `neutral_lgbm_rolling_metrics_monthly_2025_sample.csv`

当前全量中性化 LGBM 回测结果:

`D:\cbond_on\results\2025-01-01_2026-05-11\Backtest_lgbm_screened_no_winsor_neutral_v1_2025\20260615_151958`

当前全量中性化 LGBM 模型训练产物:

`D:\cbond_on\results\models\lgbm_screened_no_winsor_neutral_v1\2024-01-01_2026-05-11\20260615_150042`

## 8. 给外部 agent 的建议结论

当前不要优先做大规模新因子扩张。更应该先处理输入特征工程本身:

1. 建立特征诊断体系。
2. 处理高度重复因子。
3. 测试稳健标准化替代普通 zscore。
4. 允许 LightGBM 使用缺失值信息。
5. 在全量中性化有效的基础上，分组扩展 exposure。
6. 再考虑少量金融含义明确的派生特征和交互特征。

当前最强 baseline 是:

`zscore 开 + winsor 关 + 全量中性化开`

不要把“部分中性化”作为默认候选。除非负责人重新明确要求，否则后续实验只比较“全量中性化”和“无中性化”。

任何实盘链路修改必须先汇报最终口径并等待确认。
