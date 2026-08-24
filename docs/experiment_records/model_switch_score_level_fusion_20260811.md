# 2026-08-11：三模型 score-level 连续融合 v1

## 状态

- 仅限隔离 research，未改 live 配置、scheduler、DB、模型 state、生产 score root、名单或 `D:/cbond_on/results/*`。
- 不作实盘推广；未运行 `2026-07-31..2026-08-07` 的 547 日后缀敏感性。

## 问题与固定口径

检验既有 `state_score_geometry_lags` 的因果 Ridge 相对效用预测，是否更适合用于**连续融合三套当日横截面分数**，而不是每天 hard switch 一个模型。

- 冻结来源：`D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/run_20260811_relative_utility_v5_with_ranker/`；其 1,648 个冻结输入文件均已重验 SHA-256。
- 主评价窗口：`2024-05-08..2026-07-30`，541 个 execution-metadata-complete 且实际有冻结分数的交易日。
- 交易合约：现有 generic `backtest_runtime.run`、`strategy01_topk_turnover`、Top20、单券最大 5%、`turnover_ratio=1.0`、o_0005 唯一 allowlist、既有市场 mask、买卖成本、benchmark 与严格 cycle return。
- 每个模型的原始 score 先在其**完整且三模型完全相同的冻结当日 universe**做 percentile rank；再线性融合；之后才由未修改的 generic runtime 应用 o_0005/mask/Top20。
- 固定变体（结果前锁定）：
  1. 冻结 Regsim 原分数执行基线；
  2. 三模型等权 rank-score 融合；
  3. Ridge-softmax rank-score 融合。
- Ridge-softmax：当天仅使用已冻结的 T 日 `state_score_geometry_lags` utility。温度为最多 360 个、严格早于 T、metadata-complete 历史日的三模型 realised relative utility RMS；预测 warmup/输入不可用时固定等权。无 top1-top2、阈值、LCB、Champion、BaseGap/Robust route、veto、weight clip 或参数扫描。

## 必要的 identity gate

先将冻结 Regsim 分数直接送进 generic runtime。结果为 541/541 日完全对齐，`max_abs(day_return difference)=0`；只有在此 gate 通过后才生成并执行两个融合分数树。

## 结果

| 策略 | 累计收益 | Sharpe | 最大回撤 | 相对 Regsim 日均增量 |
|---|---:|---:|---:|---:|
| Regsim | 172.22% | 3.968 | -4.76% | — |
| 等权 rank-score 融合 | 200.45% | 4.313 | -7.53% | +1.833 bp |
| Ridge-softmax rank-score 融合 | 198.09% | 4.308 | -7.75% | +1.684 bp |

两种单一 Top20 score-level 融合都高于 Regsim；等权版本在这一预先固定的主窗口内略高于 Ridge-softmax。Ridge 权重并未趋于 hard switch：平均有效模型数 2.915，平均权重为 Regsim 34.78%、Ensemble 32.41%、HL20 32.81%。

这不是 sleeve proxy：每个融合分数均实际经过原 Top20 选择与严格 cycle backtest。

## 日历与输入 caveat

- 当前 generic raw calendar 在该区间有 543 日；`2026-06-11`、`2026-06-12` 三个候选都没有冻结 score 文件。三条回测均记录 `missing_score` 并跳过这两日，所以所有上表、identity 与配对统计明确是共同的 541 个执行日，不是 543 个日历日。
- v5 冻结了 score、候选 return、live/strategy 配置和 T1430 state，但没有冻结每日 raw execution price 或 o_0005 pool snapshot。generic replay 读取的是当时的 current DataHub raw/pool；Regsim exact identity 证明该**当前**合约在 541 日可复刻，不证明这些输入是不可变快照。
- generic strict backtest 不消费传入 config 中的 `buy_twap_col` / `sell_twap_col`；执行价格字段来自当前 `benchmark_config`，费用来自当前 `fees` config。运行时记录的 SHA-256：benchmark `2e4440ba904195ac265d89fb083e36741cca99cc9d6f34e095a5e9fa85e60f22`，fees `caaf63fddd5ae94ac2909b5c475fa5748c56e54dc9bf07f51046624a24300ea8`，paths `31587e09bc56d81d7e8e251676adf311c092404e0bcc3b01e576e0de567716d2`。
- 源 state 仍是 T1430，未获严格 14:29 PIT 认证；本记录不能支持 live 修改或推广。

## 产物与验证

- 结果根：`D:/cbond_on/research_scratch/model_switch_dynamic_weight_20260811/score_level_v1/run_main_20260811/`。
- 关键文件：`RESULTS.md`、`run_manifest.json`、`run_status.json`、`identity_regsim_parity.csv`、`daily_weights.csv`、`daily_rank_score_audit.csv`、`summary_metrics.csv`、`paired_vs_regsim.csv`、`generic_backtests/`。
- 新工具：`harness/tools/model_switch_score_level_fusion_replay.py`；新聚焦测试：`tests/test_model_switch_score_level_fusion_replay.py`。
