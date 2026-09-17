# 23:59 因子补算 V1（研究隔离）

状态：已实现单日 `execute` 能力、dry-run 账本和任务注册准备；尚未注册 Windows Task，尚未运行真实 750 因子补算。

## 目标与范围

```text
canonical Factor Catalog：800（research773 + legacy_live27）
  - immutable live50 release：50
  = 750 个非实盘实例
  + 当日 DataHub clean manifest / .done / run_id / validation gate
  + ResearchCatalogExecutionPermit
  -> 单一 score_date 的 research scratch FactorStore + execution manifest + ledger
```

release manifest 只作为排除集合读取；任务不会读取 live runtime、模型状态、实盘结果或 scheduler state。

## 执行约束

- 普通因子 pipeline 仍然是 Rust-first。研究补算的 Python 例外只能使用 `research_python + research_catalog_python`，且必须持有不可直接构造的 `ResearchCatalogExecutionPermit`。
- Permit 会绑定并重验 canonical catalog SHA、每个因子 definition/contract、固定参数、operator 源码 hash、输出列和 research scratch FactorStore 根。
- `execute` 每次只处理一个 score date。它只能写 `D:/cbond_on/research_scratch/factor_supplement_v1` 下的 FactorStore、manifest、ledger 和 task logs。
- 不会调用 live runtime、live scheduler、DB、model score、trade list 或 live FactorStore。
- 同日已有 scratch 输出时，只有 catalog hash、release hash、完整因子 scope 均一致且存在前序 execution manifest 才可安全恢复；否则 fail-closed。
- 输出 manifest 固化 catalog/release hash、每因子 contract hash、预存/本次执行/完成/缺失 ID、列顺序、列 hash、索引 hash 和 FactorStore 文件 hash。
- `compute_complete` 与覆盖质量分开记录：所有列已物化、无异常且无 Inf 时，任务计算成功；若部分因子全 NaN，则状态为 `completed_with_coverage_gaps`，账本会列出每因子的 finite count/rate 和 all-NaN ID，但不会被当作无限重试的计算失败。

## 命令

默认 CLI 是 dry-run：

```powershell
py harness/tools/run_factor_supplement.py --score-day 2026-08-25
```

只读 plan：

```powershell
py harness/tools/run_factor_supplement.py --mode plan --score-day 2026-08-25
```

单日真实 research 执行（仅在确认后使用；本次未执行）：

```powershell
py harness/tools/run_factor_supplement.py --mode execute --score-day 2026-08-25
```

## 23:59 Windows Task

注册工具默认只展示 action，绝不注册任务。它支持 `-Mode dry-run` 和 `-Mode execute`，采用：

- 每日 23:59；
- `IgnoreNew`，避免同日并发；
- 不设置 `StartWhenAvailable`，避免错过后补跑；
- 8 小时上限；
- stdout/stderr 分别写入 `<scratch_root>/task_logs/YYYY-MM-DD/`。

```powershell
.\harness\tools\register_factor_supplement_task.ps1 -Mode execute
.\harness\tools\register_factor_supplement_task.ps1 -Mode execute -Register
```

第二条命令会改变 Windows Task Scheduler；本次没有执行它。

## 验证

- catalog/release plan：800 / 50 / 750；
- DataHub 通过时可签发 750 实例的只读 permit；
- synthetic 单因子 execute、真实 legacy operator 的合成面板单因子 execute、resume 与失败 manifest 均有测试；
- 不代表这 750 因子已完成真实 PIT/单因子质量验证，也不构成实盘准入。
