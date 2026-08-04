# Tail path efficiency 5m（预注册，2026-07-31）

## 问题与唯一候选

在不改变 27 因子 Regsim 基线、`strategy01_topk_turnover`、所有既有 mask
（包括 T-1 `o_0005`）、交易/执行窗口、费用或 benchmark 的前提下，尾盘报价
路径的“单边效率”是否能为现有分数提供增量 IC？

唯一因子为 `tail_path_efficiency_5m_v1`。对每个 `(T, code)`，仅保留
`[T 14:24:00, T 14:30:00)` 内有效 L1 quote（有限、`0 < bid_price1 <=
ask_price1`）。每个时钟分钟 `14:24..14:29` 取最后一个有效 midpoint，令
`x_m=log((bid_price1+ask_price1)/2)`：

```text
(x_14:29 - x_14:24) / sum_{m=14:25..14:29} abs(x_m - x_{m-1})
```

六个分钟桶必须完整，且总路径长度必须正；否则输出 `NaN`，绝不填 0 或回退到
last/open/daily 数据。14:30 及之后、跨日行、label、pool/mask、数据库和每日数据
不参与计算。它不同于现有 `ret_10m`、`mid_move_30m` 和 `range_30m`：后者只
测端点或幅度，本因子测方向相对路径噪声的效率。

## 固定研究流程

1. 只向隔离 scratch FactorStore 构建此一列，日期锁定
   `2024-05-08..2026-04-30`；先审计 schema/index、14:30 factor dt、时间边界、
   NaN/Inf、与 Regsim score code 的覆盖率及与已有 27 因子的日截面重复度。
2. 健康门槛：每个可评分日不改变 score calendar；用于 Ridge 的当前 Regsim code
   覆盖至少 80%，否则该日/该 code 保留原始 Regsim score。不会因覆盖差删除券或
   调整 mask。
3. 唯一 score arm：`anchored_residual` Ridge、120 个严格先前 Regsim score
   calendar slot、`alpha=20`、至少 96 个可用训练日、逐日/逐 code exact Regsim
   fallback。score 阶段不读 T 日 label；先冻结 scores，再单独打开同日 14:42
   label 评价。
4. 配对评价会对两个 stream 同时应用已有 T-1 `o_0005` allowlist，并冻结
   score/code/value 对；不使用原始历史 Regsim 的未过滤 score rows。

仅在 validation `2025-10-09..2026-04-30` 的 Pearson IC 配对差至少
`+0.003`、t 至少 `1.5`，且 RankIC、Top20 raw-label 不恶化时，才考虑另一个
未查看的前瞻 shadow。否则立即拒绝，不调分钟窗口、价格定义、Ridge alpha、
coverage 门槛或融合方式，不跑策略回测或 live。`2026-05-06+` 已被查看，只能
作探索诊断，绝不用于本候选验收。
