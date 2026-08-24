# 2026-08-11：50 因子三 candidate 纯 BaseGap 内部调优

## 结论

本轮**不改变实盘**。冻结回放中，`240 日 lookback / Top-60 / 等权
trim20_lcb10 / path44` 比现行 `60 / 40 / trim20_lcb10 / path44` 更能抵抗
2026 年初出现的状态—收益映射漂移；但它仍不是统计显著、宽参数平台上的
稳定优势。新增 score 几何、50 因子结构状态、MAD 标准化、固定距离加权和
相对收益评分都没有形成可推广的提升。

因此：

- 不写 live config、DB、scheduler、model state、score root、factor store 或
  `results/live`；
- 不把任何本轮结果视作实盘模型切换变更授权；
- 如需继续，只能在**新的、未参与调参的 forward shadow 窗口**验证已经固定的
  `240 / 60 / trim20_lcb10 / path44`，而不是继续在本段历史上搜索参数。

## 固定研究合约

- 三 candidate：Regsim、Ensemble、HL20；候选顺序只用于精确数值并列时的稳定
  tie-break。
- 选择器：纯 BaseGap 直接 argmax；无 Champion 偏好、Robust、Fusion、阈值或
  top1-top2 gate。
- 标签：三条现有 standalone full-cycle `day_return`；这是一条
  selector-shadow-return 回放，**不是**重新跑出的单账户真实执行回测。
- 主目标：`selected_return - mean(Regsim, Ensemble, HL20)`；指标还报告命中率、
  regret、相对每一 candidate、Sharpe、MDD、切换次数。
- 评估只用三条 return history 都具有 execution metadata 的共同日期：
  `2024-05-08..2026-07-30` 共 541 日。BaseGap 有 warm-up，active 样本数随
  lookback/状态族变化而变化，均单独披露。
- 切分：设计 `2024-05..2025-06`，验证 `2025-07..2025-12`，最终 OOS
  `2026-01..2026-07`。参数/状态族只按设计段排名；验证与最终 OOS 仅报告。
- 原 T1430 state 的既有 PIT caveat 保留：本研究不是严格 14:29 证明。

所有来源均来自冻结 run：

```text
D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/
  run_20260811_relative_utility_v5_with_ranker/
```

本轮每次运行都重新核验 1,648 个冻结文件的 SHA-256。生产 BaseGap 对
冻结 `daily_direct_basegap.csv` 的 547 日逐字段一致；后续 in-memory
reference 也逐字段一致，才允许变体继续执行。

## A. BaseGap 内部参数网格

预注册 51 点：

```text
lookback = 40/60/90/120/180/240
K        = 20/40/60（K <= lookback）
metric   = mean/lcb10/trim20_lcb10
min_periods = K；margin = 0；邻居等权；原 44 path 状态
```

设计集只按 active-day selection alpha 排名，胜者为：

```text
lb240_k60_trim20_lcb10
```

| 版本 | active 设计 | active 验证 | active 最终 OOS | active 全窗 |
| --- | ---: | ---: | ---: | ---: |
| 现行 60/40/trim | +1.078 bp/日 (239) | +2.005 bp/日 (126) | -5.891 bp/日 (130) | -0.516 bp/日 (495) |
| 设计胜者 240/60/trim | +1.109 bp/日 (219) | +0.156 bp/日 (126) | +1.332 bp/日 (130) | +0.917 bp/日 (475) |

`240/60` 的 active 全窗相对三模型算术等权的累计简单 alpha 为 `+435.664bp`，
selector / 等权复利为 `146.480% / 136.108%`，MDD 为 `-5.252% / -5.778%`。
它有 217 次模型切换（474 个相邻 active 日中的 45.78%），命中当日最佳的比例为
37.90%，平均名次 1.937，平均 regret 16.627bp/日。

它在样本路径中相对 Regsim、Ensemble、HL20 分别为 `+0.217`、`+0.856`、
`+1.678 bp/日`，但均不显著；相对 Regsim 的 MDD 高 0.491 个百分点。

统计检验不支持将该 alpha 视为已证实：

- HAC(5) 配对检验：`t=1.073, p=0.283`；
- IID 配对 t 检验：`p=0.308`；
- 5 日移动块 bootstrap（20,000 次，seed=20260811）95% CI：
  `[-0.865, +2.494] bp/日`；
- 最终 OOS 的 `+1.332 bp/日` 亦不显著：HAC `p=0.490`，bootstrap CI
  `[-3.063, +4.339] bp/日`。

参数稳定性也不足：51 点中只有 2 点在设计、验证、最终 OOS 三段均为正；
设计胜者在验证排名第 33、最终 OOS 排名第 4。其相邻点多为负，不能称为宽
稳健平台。

## B. 预注册状态族：G19 / F12

固定 A 的 `240 / 60 / trim20_lcb10`，只比较整组状态族而不按单列回测筛选：

- `path44`：原 44 列 T1430 path 状态；
- `G19`：冻结的三模型 score Spearman、rank 差、Top20 重叠/边际/连续性等
  19 列几何状态；
- `F12`：冻结 50 因子在三个 score 共同 universe 的覆盖/缺失率、legacy-27 与
  mined-23 的因子水位及离散度、tail asymmetry 与 Spearman 相关结构摘要；
  F12 的 median/MAD 时间标准化严格只使用早于 T 的最多 120 日、至少 60 日。

四组都 ready 的 415 个完整日上：

| 状态族 | alpha vs 等权（bp/日） | 设计 / 验证 / OOS（bp/日） |
| --- | ---: | ---: |
| path44 | +1.091 | +1.635 / +0.156 / +1.332 |
| path44 + G19 | +0.674 | -1.633 / +1.394 / +2.797 |
| path44 + F12 | -0.071 | +0.702 / +1.327 / -2.372 |
| path44 + F12 + G19 | -0.776 | -0.412 / -1.034 / -0.973 |

G19 的 OOS 表面较好却在设计期为负；F12 的设计/验证表面较好却在 OOS 为负。
这不是可推广改善，设计期胜者仍是 `path44`。

## C. 预注册距离与对称收益评分

固定 `240 / 60 / trim20_lcb10 / path44`，比较 12 个变体：

```text
state scale: rolling z-score / rolling median-MAD
neighbour:   equal / median-half Gaussian
target:      absolute / relative-to-three-model-EW / pairwise-mean contrast
```

`median-half Gaussian` 是固定规则：Top-60 的 median-distance 邻居有原始权重
1/2；其有效样本量中位数约 58.8，说明它是温和重排，不是少数邻居的集中押注。

设计期第一仍为原基线：

```text
zscore + equal + absolute: +1.109 bp/日
zscore + median-half + absolute: +1.051 bp/日
```

距离加权在验证期为 `+1.012 bp/日`，但最终 OOS 降至 `+0.125 bp/日`；MAD 与
相对收益评分没有稳定改善。三模型时 `pairwise_mean = 1.5 × relative_equal_weight`，
该对照用于核验对称性，不视作独立的参数自由度。

## 产物与验证

- A：`D:/cbond_on/research_scratch/model_switch_basegap_tuning_20260811/run_stage_a_20260811_r3/`
- B：`D:/cbond_on/research_scratch/model_switch_basegap_state_family_20260811/run_state_families_20260811_r1/`
- C：`D:/cbond_on/research_scratch/model_switch_basegap_scoring_geometry_20260811/run_geometry_20260811_r1/`

新增的研究工具与 focused tests：

```text
harness/tools/model_switch_basegap_tuning_replay.py
harness/tools/model_switch_basegap_state_family_replay.py
harness/tools/model_switch_basegap_scoring_geometry_replay.py
tests/test_model_switch_basegap_tuning_replay.py
tests/test_model_switch_basegap_state_family_replay.py
tests/test_model_switch_basegap_scoring_geometry_replay.py
```

相关 focused tests 共 19 项通过；每个 run 的 `run_manifest.json`、
`input_verification.json`、`leakage_audit.json`、逐日 CSV、指标 CSV、task state 和
图表均已保存。实盘链路未修改。
