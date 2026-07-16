# 上午 K 线预测下午收益实验记录

更新时间：2026-07-16

## 研究目标

使用当日 `09:30-11:29` 的一分钟 K 线预测同日下午收益，交易口径为：

- T-1 `o_0005` 候选池；
- `13:00-13:05` TWAP 买入；
- `14:50-14:57` TWAP 卖出；
- 买卖成本合计 `2.2bp`；
- 缺失测试收益按 0 计，不按标签可得性过滤推理池；
- 全部实验均为研究链路，不修改 live、调度或数据库。

## 无泄露验证口径

开发阶段使用七个扩展窗口样本外折：`2024Q3` 至 `2026Q1`。每个季度模型只使用该季度之前的数据训练，不使用测试季度早停。

最终留出集为 `2026-04-01` 至 `2026-07-14`。在打开留出集前，已将模型成员、等权 rank 平均方式和 Top5 选择写入：

`D:\cbond_on\results\analysis\intraday_kline_optimization_v1\selection_lock.json`

## 主要迭代

| 阶段 | 尝试 | 主要结果 | 结论 |
| --- | --- | --- | --- |
| 图像表示 | 蜡烛图 CNN、STFT、CWT、蜡烛图+CWT | 原始蜡烛图优于所有时频表示 | 频率表示没有增加有效尾部信息 |
| 架构比较 | Image CNN、1D CNN、Inception、TCN、GRU、Transformer | 普通 1D CNN 最好；GRU/Transformer 为负超额 | 任务信号较弱，复杂架构主要放大噪声 |
| 训练目标 | Huber z-score、Huber rank、ListNet、Hybrid ListNet、Top20 pairwise | Huber rank 有效；ListNet/pairwise 显著失败 | 极端尾部目标会学习不可重复的收益噪声 |
| 输入消融 | 全通道、仅价格水平、仅分钟形态、去跳空 | 仅分钟形态收益 `2.82%`、超额 Sharpe `3.395` | 有效信息来自分钟收益、实体、振幅和影线；绝对价格路径有害 |
| 容量与正则 | 32/40/48/64 通道，2/3/4 轮，学习率、Dropout、卷积核 | 32 通道、3 层、5 分钟核最稳；`lr=5e-4` 和 Dropout 0.2 有效 | 小模型更适合当前信噪比 |
| 多模型稳健化 | 多种子和强候选等权日内 rank 平均 | `robust5_rankavg_v1` 开发期收益 `5.69%`，超额 Sharpe `4.354` | 等权 rank 平均显著降低初始化和 Top20 边界波动 |

完整单模型和组合排行榜：

`D:\cbond_on\results\analysis\intraday_kline_optimization_v1\development\leaderboard.csv`

## 开发期锁定结果

`robust5_rankavg_v1` 使用五个 1D CNN：

1. 48 通道标准版本；
2. 48 通道低学习率版本；
3. 32 通道标准版本；
4. 32 通道低学习率版本；
5. 32 通道高 Dropout 版本。

五个模型的每日截面分数先转为百分位 rank，再等权平均。

### Top20 原始口径

| 指标 | 模型 | 候选池等权 |
| --- | ---: | ---: |
| 交易日 | 422 | 422 |
| 累计收益 | 5.69% | -14.41% |
| 年化收益 | 3.36% | -8.87% |
| Sharpe | 0.580 | -1.254 |
| 最大回撤 | -5.35% | -15.91% |

开发期七个季度全部正超额，21 个月全部正超额。39 个候选/组合的 20 日块 max-t 现实检验得到家族级 `p=0.0006`。

### Top5 开发期选择

Top5 在开发期收益 `18.74%`、Sharpe `1.513`、最大回撤 `-5.21%`，七个季度全部正超额，因此在打开留出集前被锁定为主选择。

## 最终留出集

### 主选择 Top5

| 指标 | 模型 | 候选池等权 |
| --- | ---: | ---: |
| 交易日 | 63 | 63 |
| 累计收益 | -0.71% | -0.41% |
| Sharpe | -0.429 | -0.181 |
| 最大回撤 | -4.92% | -3.75% |

Top5 主选择未通过最终留出集，说明开发期的集中化收益存在过拟合。

### 预先存在的 Top20 参考口径

Top20 是项目原始选股数量，并在选择锁中保留了完整开发指标。同一模型、同一训练 state、同一最终分数按 Top20 评价：

| 指标 | 模型 | 候选池等权 |
| --- | ---: | ---: |
| 累计收益 | 1.01% | -0.41% |
| Sharpe | 0.859 | -0.181 |
| 最大回撤 | -2.75% | -3.75% |
| 超额收益 | 1.36% | - |
| 超额 Sharpe | 1.212 | - |

该结果只能作为预先存在 Top20 参考，不应解释为打开留出集后重新调参得到的新最优。

## 阶段结论

1. 上午纯 K 线包含可重复的宽截面排序信息，但最强的 5 只尾部排序不稳定。
2. 当前最可靠结构是小型 1D CNN，而不是图像 CNN、循环网络、Transformer 或时频网络。
3. 输入应保留局部分钟形态并去掉相对昨收的 OHLC 水平通道。
4. 训练目标应使用平滑截面 rank 回归，避免 ListNet 和极端 TopK pairwise。
5. 当前模型不具备直接切换实盘的证据；后续验证必须使用 `2026-07-15` 之后的新数据，不能继续用现有最终留出集调参。

## 主要产物

- 开发期最优组合：`D:\cbond_on\results\analysis\intraday_kline_optimization_v1\development\ensembles\robust5_rankavg_v1`
- 主选择 Top5 最终结果：`D:\cbond_on\results\analysis\intraday_kline_optimization_v1\final_holdout\robust5_rankavg_top5_final_v1`
- Top20 参考结果：`D:\cbond_on\results\analysis\intraday_kline_optimization_v1\final_holdout\robust5_rankavg_top20_reference_v1`
- 连续样本外 Top20 参考：`D:\cbond_on\results\analysis\intraday_kline_optimization_v1\continuous_oos_top20_reference_v1`
- 多重比较检验：`D:\cbond_on\results\analysis\intraday_kline_optimization_v1\multiple_testing_reality_check.json`
