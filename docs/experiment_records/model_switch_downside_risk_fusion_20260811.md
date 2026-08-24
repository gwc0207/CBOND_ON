# 2026-08-11：三模型 score-level 下行风险平价融合 v1

## 结论

不推广，不改 live。

预先固定的 EWMA 下行风险平价融合，确实比等权 rank 融合多获得了一点
累计收益与 Sharpe，但没有解决本次研究的核心问题——融合后的最大回撤没有下降，
反而从 `-7.53%` 轻微扩大到 `-7.57%`。因此它不能作为“控制等权融合回撤”的
实盘候选，也不应在同一段历史上继续调衰减、预热长度或回撤窗口。

本实验仅写入独立 research scratch；没有修改 live 配置、既有工具、scheduler、
DB、模型 state、生产 score root、名单或 `D:/cbond_on/results/*`。

## 问题与固定口径

此前真实 single-book Top20 回放中，三模型等权 percentile-rank 融合的累计收益
高于 Regsim，但最大回撤从 Regsim 的 `-4.76%` 扩大到 `-7.53%`。本实验只检验：
是否能用**严格因果、无门槛**的模型级下行风险权重改善这条回撤路径。

- 冻结来源：
  `D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/run_20260811_relative_utility_v5_with_ranker/`。
- 主窗口：`2024-05-08` 至 `2026-07-30`；仅 541 个
  `execution_metadata_complete` 的共同执行日。`2026-07-31..2026-08-07`
  的不完整 metadata 后缀不参与。
- 先对冻结 Regsim 原始 score 执行 generic runtime；只有它 541/541 日收益精确
  复现冻结 return 才允许候选融合运行。
- 三条线都保持原来的完整执行合约：三模型同日完整共享 universe 内 percentile rank，
  然后交给未改动的 generic runtime 应用 `o_0005`、市场 mask、
  `strategy01_topk_turnover`、Top20、单券最多 5%、全换手、原费用、benchmark 与
  严格买卖 cycle return。不是 shadow sleeve 或模型收益拼接代理。

## 预注册公式

唯一的新候选为 `ewma_downside_risk_parity_rank_score_fusion`。对 score day `t`
及模型 `i`：

```text
d_i,t = sqrt(EWMA_lambda=0.94(min(r_i,s, 0)^2), s < t)
w_i,t = (1 / d_i,t) / sum_j(1 / d_j,t)
fused_score_t(c) = sum_i w_i,t * percentile_rank(score_i,t(c))
```

- `r_i,s` 是候选模型 i 在 s 日的 standalone 实现收益；公式只读取严格早于
  `t` 的完整主窗口收益，绝不读取当日或未来收益。
- `lambda=0.94`（日频 RiskMetrics 风格衰减）与前 20 个 score day 完全等权的
  warmup，均在运行前固定；没有网格、阈值、clip、数据相关地调节参数或按回撤段选择。
- 对照是相同 rank-score 链路的三模型 `1/3, 1/3, 1/3` 等权融合。
- 没有 Ridge utility tilt、state 相似性、BaseGap/Robust、top1-top2 gap、LCB、
  Champion 偏好、veto 或置信度门控。

## Identity gate

冻结 Regsim 原 score 通过 generic runtime 得到：

```text
expected days = 541
generic days  = 541
aligned days  = 541
max abs(day_return difference) = 0.0
```

之后才生成 541 个等权 score 文件与 541 个风险平价 score 文件，并分别完成
完整 generic 回放。

## 结果（完全对齐的 541 执行日）

| 策略 | 累计收益 | Sharpe | 年化波动 | 最大回撤 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regsim | 172.22% | 3.968 | 11.94% | -4.76% | 60.07% |
| 等权 rank-score | 200.45% | 4.313 | 12.06% | -7.53% | 61.37% |
| EWMA 下行风险平价 rank-score | 203.64% | 4.349 | 12.07% | -7.57% | 61.55% |

相对等权融合，风险平价为 `+3.20pp` 累计收益、`+0.036` Sharpe；但最大回撤
额外加深 `0.04pp`。它是平均收益更好的版本，不是下行风险更受控的版本。

## 回撤归因与路径检查

等权融合的最大回撤发生在 `2025-03-11` 至 `2025-04-08`：

| 同一 peak-to-trough 路径 | 收益 / 回撤 |
| --- | ---: |
| Regsim | -4.58% |
| 等权 rank-score | -7.53% |
| EWMA 下行风险平价 | -7.57% |

风险平价在这 19 个下跌执行日实际平均配置为 Regsim `30.93%`、Ensemble
`34.44%`、HL20 `34.64%`。但事后该段 standalone 表现是 Regsim `-4.58%`、
Ensemble `-7.07%`、HL20 `-5.68%`：滞后的 individual downside 估计恰好低配了
随后更抗跌的 Regsim。这个事实用于解释失败，**不**被用于调本公式。

对 Regsim 的事后相对路径诊断也更差：

| 候选 | 相对 Regsim MDD | 最差 5 日复合相对收益 | 最差 20 日复合相对收益 |
| --- | ---: | ---: | ---: |
| 等权 rank-score | -3.88% | -2.72% | -3.49% |
| EWMA 下行风险平价 | -4.13% | -2.72% | -3.58% |

三模型 standalone 日收益相关性很高：Regsim–Ensemble `0.9012`、
Regsim–HL20 `0.9123`、Ensemble–HL20 `0.9398`。因此按各模型自己的历史
下行波动倒数分配，无法消除它们在共同冲击日同步受损的风险；它也不直接预测
当日融合 Top20 basket 的下行。

## 风险权重行为

541 日内前 20 日为固定等权。后续风险平价权重并不极端：

| 模型 | 全窗平均权重 | 最小 | 最大 |
| --- | ---: | ---: | ---: |
| Regsim | 33.15% | 26.49% | 40.90% |
| Ensemble | 33.83% | 28.08% | 38.33% |
| HL20 | 33.02% | 27.65% | 39.65% |

平均有效模型数为 `2.984`，说明该规则主要是温和地偏离等权，而不是隐性 hard switch。

## 限制与下一步边界

- v5 快照冻结了 score、候选 return、live/strategy config 和 T1430 state，未冻结逐日
  raw execution price 或 `o_0005` pool snapshot。generic replay 使用当前配置的
  DataHub raw/pool；Identity gate 证明当前合约复现，不等于 raw/pool 永久不可变。
- strict generic runtime 从当前 benchmark/fees config 读取执行字段与费用；路径和
  SHA-256 已记录到 run manifest。
- 来源仍是 T1430，不是严格 14:29 PIT 认证；该实验不能支持任何 live 变更。
- 不应在这 541 日上继续调 `lambda`、warmup、回撤触发器或加入事后修正。若继续，
  需先由 owner 确定一个独立的新风险模型家族或 forward shadow 设计。

## 产物与验证

- 结果根：
  `D:/cbond_on/research_scratch/model_switch_downside_fusion_20260811/run_main_20260811/`
- 核心证据：`RESULTS.md`、`run_manifest.json`、`run_status.json`、
  `identity_regsim_parity.csv`、`summary_metrics.csv`、`daily_weights.csv`、
  `weight_diagnostics.csv`、`daily_drawdown.csv`、`drawdown_summary.csv`、
  `relative_path_vs_regsim.csv`、`relative_path_summary.csv`。
- 新 research-only 工具：`harness/tools/model_switch_downside_risk_fusion_replay.py`。
- focused regression 验证：

  ```powershell
  py -3.11 -m pytest -q tests/test_model_switch_downside_risk_fusion_replay.py tests/test_model_switch_score_level_fusion_replay.py tests/test_model_switch_dynamic_weight_replay.py tests/test_model_switch_relative_return_replay.py tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py
  ```

  结果：`54 passed`。
