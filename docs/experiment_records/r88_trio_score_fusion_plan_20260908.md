# R88 P6 + Full88 P3 + Regsim 组合研究计划（2026-09-08）

## 研究问题

在不改变既有 `o_0005`、市场 mask、Top20、执行价格、费用、benchmark 或
实盘链路的前提下，冻结的 P6、Full88 P3 与 Regsim score 是否可通过单一
Top20 的日内 percentile-rank 融合，取得比零参数等权融合更稳定的风险调整
表现？

## 已锁定范围

- 候选只包括 `r88_icir50__p6_lower_bagging`、
  `r88_full88__p3_deep_regularized` 与当前 Regsim score。
- 本轮主线是 score-level fusion；三 sleeve 资金配置是独立的后续研究，
  不得与单一 Top20 结果混称。
- 2025 是设计/描述区间，2026 是报告区间；2026 不得用于重新调权或选择
  新候选。真正的升级证据只能来自冻结后的 forward shadow。
- 不使用 Champion、BaseGap、Robust、LCB、置信度阈值、硬切换或模型重训。

## 分阶段计划

### A0：输入与覆盖预检

1. 冻结 score、return 与 R88 integrity 输入哈希。
2. 审计三模型 score 日期、代码 universe、交集/并集和缺失日期。
3. 审计三条 `day_return` 的共同日期及 benchmark 差异。
4. 只记录缺失-score policy 的决策需求；不生成融合分数、不运行 backtest。

#### A0 完成结果

- 完成时间：2026-09-08；research-only，无 DB、live runtime、scheduler、
  model training、model scoring 或 generic backtest 调用。
- 三模型共同 score 日为 399 日，区间 `2025-01-02` 至 `2026-08-27`；仅
  1 日的三模型 score universe 完全相同。
- 共同日平均代码交集为 345.40、并集为 383.57，最小交集为 202；因此不能
  直接复用要求完全相同 score universe 的旧 rank-fusion 工具。
- P6/P3 各有 401 score 文件，Regsim 有 569；三条 return 历史共同 399 日。
  Regsim 在 `2026-06-11`、`2026-06-12` 缺覆盖，且 18 个共同日 benchmark
  数值不一致，故当前相对比较仍限于原始 `day_return`。
- A0 结论：B 阶段技术上需要一项显式、可审计且不改变既有 `o_0005` universe
  的缺失-score 规则；在规则冻结和 Regsim identity replay 前，不生成任何融合
  score 或回测结果。

### B：静态 rank-fusion

只有在 owner 审核 A0 后才可启动：

1. Regsim raw-score generic-runtime identity replay。
2. 等权 percentile-rank fusion。
3. 预注册的有限 long-only simplex 权重集。
4. 每个权重点均通过同一 generic Top20/cost/strict-cycle backtest。

#### B1 identity 结果（停止）

- owner 已确认使用“自身有效 score universe percentile-rank；原 `o_0005`
  universe 内缺失模型 score 贡献中性 rank 0.5”的主线合同。
- B1 仅执行了 raw Regsim generic-runtime identity。运行在 research scratch
  内，没有 DB、live runtime、scheduler、模型训练或模型打分调用。
- identity 在锁定的 399 日共同窗口失败：`2026-08-12` 的冻结 Regsim
  `day_return=0.0126301509866367`，当前 generic runtime
  `day_return=0.0151291039553838`，绝对差 `0.0024989529687471004`。
- 因此 B1 正确停止为 `blocked_nonparity`，没有写入融合 score，也没有运行
  等权 rank-fusion backtest。该任务不推断差异来自 raw、pool、价格或其他
  历史输入；任何改变冻结基线、当前输入或 identity 容差的动作都需新的合同。

#### B1 差异溯源（只读完成）

- Regsim `2026-08-12` score 文件创建时间为 `2026-08-12 14:31`；
  `2026-08-13` scheduler 日志确认历史 shadow-return 正是用该 score 源将
  history 从 `2026-08-11` 追加至 `2026-08-12`。因此本次 identity 差异没有
  证据指向 warm start 或重新训练后的 score 漂移。
- 当前本地 DataHub 的 `o_0005` 2026-08-11 分区创建于 `2026-08-17`，
  2026-08-12/13 daily TWAP 与 daily price 分区创建于 `2026-08-18/19`，
  均晚于历史 shadow-return 计算时间。
- 当前 shadow-return builder 和 generic B1 对 2026-08-12 均给出
  `0.0151291039553838`，说明两者在当前输入下语义一致；历史值
  `0.0126301509866367` 来自先前执行输入状态。历史 benchmark count 为 149，
  当前为 295；买腿差约 4.26bp、卖腿差约 20.52bp。
- 结论：B1 阻塞于可变的历史 execution-input 重建，而不是模型 warm start。
  由于未保留当时 raw/pool 的不可变字节快照，不能再把当前 DataHub 分区与冻结
  shadow return 混为同一历史执行口径。

### C：低自由度优化器

仅在静态方案有稳健改善后：

- 强收缩 NNLS/Ridge score stacking；
- 因果、强收缩至等权的在线连续权重；
- 或独立 sleeve 对照中的 shrinkage GMV/ERC/robust MVO/CVaR。

## 评价顺序

先检查回撤、CVaR、最差时间块 Sharpe 与权重/换手稳定性；再检查 HAC Alpha
与跨块 Alpha；最后才比较 Sharpe 和收益。相对 Regsim 使用共同日期的原始
`day_return`，直到统一 generic backtest 产生同一 benchmark。

## A0 输出与禁止项

A0 的唯一输出根为：

```text
D:/cbond_on/research_scratch/r88_trio_score_fusion_20260908_r1/
```

禁止写入 live、DB、scheduler、模型状态、FactorStore 或既有结果根。
