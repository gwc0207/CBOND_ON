# CBOND_ON 新负责人完整交接（2026-07-29）

> **用途**：给下一位 CBOND_ON 负责人接手研究、实盘、数据接口、因子、模型、回测和风险模型工作。
>
> **快照时间**：2026-07-29 下午（Asia/Shanghai），已结合当日实盘抢修后的运行产物、当前配置、代码和研究记录核验。
>
> **真实性边界**：这是“可执行的代码地图 + 当前已核验状态”，不是对每一行历史代码的虚假全审阅声明。任何会改动 live、数据库或调度的操作，都必须再读当前配置和运行产物；历史报告中的“current”可能已经过期。

---

## 0. 最重要的结论（先读）

1. **2026-07-29 的 Champion 抢修已经成功入库。**
   - Champion：`lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708`（Regsim）。
   - 本地名单：[D:\\cbond_on\\results\\live\\2026-07-30\\trade_list.csv](D:\cbond_on\results\live\2026-07-30\trade_list.csv)，14:40:03 生成，20 只。
   - 数据库：`quant_factor_dev.researcher_gswzif.o_0001` 的 `trade_date=2026-07-28`，20 行、rank 1--20、code/rank/weight 与 CSV 完全一致；score 按数据库 `numeric(25,8)` 四舍五入后完全一致（CSV 原始 score 最大差 `4.52e-09`）。
   - 该次人工修复显式关闭了模型切换：`model_switch_enabled=false`；不能把它误认为常规 selector 的自然选择。

2. **DataHub 已恢复且当前发布门正常。**
   - `raw`、`clean` 的 `2026-07-29` manifest 都为 `success`。
   - 当前一致的 `run_id`：`20260729_145955_000132`；`clean.validation.passed=true`，profile 为 `cbond_on_live_t1430`，`.done` 存在且 `ready=true`。
   - DataHub scheduler PID `4076` 存活，收盘后 `idle_outside_session`、`last_error=''` 正常。

3. **CBOND_ON 常驻调度器仍是失败状态，不能为了“刷绿”直接启动。**
   - [state.json](D:\cbond_on\results\live\scheduler\state.json) 仍记录 14:29:12 的 gate 失败，`status=failed`、`last_return_code=1`；当前未发现 CBOND_ON live scheduler 进程，Dashboard `pythonw` 仍在。
   - 直接 `POST /api/start` / 重启常规 scheduler 会在已过 cutoff 后重跑正常 `model_switch`，可能把已确认的 Champion 名单覆盖为 Ensemble 或其他候选。
   - 下一位负责人必须先获得“是否允许重跑 / 是否固定 Champion / 如何补审计状态”的明确授权，再恢复 CBOND_ON scheduler。

4. **工作树大量 dirty，且 2026-07-20 之后相关研究没有可依赖的 Git commit。**
   - 不要 `git reset --hard`、`git checkout --`、`git clean`，也不要恢复/删除不明配置。
   - 当前 live config、model-switch 代码、线性研究、similar-day 训练、CB-Risk、因子候选、文档和测试混在同一 worktree；先分组审阅再决定提交策略。

---

## 1. 新会话的建议起手式

在提出任何改动前，下一位 agent 应按以下顺序做只读检查：

```powershell
Set-Location C:\Users\BaiYang\CBOND_ON\cbond_on
git rev-parse --show-toplevel
Get-Content AGENTS.md -Raw -Encoding utf8
Get-Content harness\README.md -Raw -Encoding utf8
py harness\tools\agent_preflight.py --mode incident   # 若是实盘/抢修
Get-Content D:\cbond_on\results\live\scheduler\state.json -Raw -Encoding utf8
Get-Content D:\cbond_on\results\live\2026-07-29\logs\live_scheduler_2026-07-29.log -Tail 100
```

然后核验 DataHub，而不是猜测数据是否已好：

```powershell
Set-Location C:\Users\BaiYang\CBOND_DATA_HUB
py -m cbond_data_hub publish status `
  --manifest-root D:/cbond_data_hub/manifests `
  --trade-day 2026-07-29 `
  --require-datasets raw,clean
```

在今天这个特定状态下，先只读比较下列两者；**禁止先启动常规 scheduler**：

```text
D:\cbond_on\results\live\2026-07-30\trade_list.csv
quant_factor_dev.researcher_gswzif.o_0001 where trade_date=2026-07-28
```

---

## 2. 物理边界、目录和 source of truth

| 项目 | 当前位置 / 说明 |
| --- | --- |
| 真正 Git 根 | `C:\Users\BaiYang\CBOND_ON\cbond_on` |
| 外层目录 | `C:\Users\BaiYang\CBOND_ON`，**不是** Git 根 |
| Python package 根 | `C:\Users\BaiYang\CBOND_ON\cbond_on\cbond_on` |
| DataHub 生产仓库 | `C:\Users\BaiYang\CBOND_DATA_HUB\cbond_data_hub`（外层是 launcher shell） |
| DataHub 数据 | `D:\cbond_data_hub\raw_data`、`D:\cbond_data_hub\clean_data` |
| CBOND_ON panel / label / factor | `D:\cbond_on\panel_data`、`D:\cbond_on\label_data`、`D:\cbond_on\factor_data` |
| CBOND_ON 运行产物 | `D:\cbond_on\results` |
| 主路径配置 | `cbond_on/config/data/paths_config.json5` |

启动时先检查是否被环境变量覆盖：

```text
CBOND_ON_PATHS_CONFIG
CBOND_ON_PATHS_PROFILE
CBOND_ON_RAW_ROOT
CBOND_ON_CLEAN_ROOT
CBOND_ON_RUNTIME_ROOT
CBOND_ON_DATA_ROOT
```

解析实现位于 `cbond_on/core/config.py`。不要把 NFS、Redis snapshot、旧 `window-data` 或历史 raw loader 当成默认生产数据源：当前生产链消费本地 DataHub `raw/clean` 与 `daily_twap`。

### 2.1 必读文件优先级

1. `AGENTS.md`、`harness/README.md`、`harness/context/source_of_truth.md`。
2. 当前实盘：
   - `cbond_on/config/live/live_config.json5`
   - `cbond_on/config/live/live_factors_config.json5`
   - `cbond_on/config/live/live_models_config.json5`
   - `cbond_on/config/live/live_switch_source_regsim_w10_s15_d05_20260708_config.json5`
   - `cbond_on/config/live/live_lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708_config.json5`
3. 当日运行证据：`D:\cbond_on\results\live\<target_day>`、`D:\cbond_on\results\model_state`、DataHub manifests / `.done`。
4. 实验结论：`docs/experiment_records/*.md`；必须以其中的日期、收益口径和 promotion status 为准。
5. 历史交接：[live_model_switch_handoff_20260724.md](live_model_switch_handoff_20260724.md)。它解释 7/24 soft-degrade 变更，但 PID、日期和当时 HL20/Regsim 角色已过期。

---

## 3. 代码架构和入口

当前主架构为：

```text
cbond_on/run/（兼容薄包装）
  -> cbond_on/cli/
  -> cbond_on/bootstrap/
  -> cbond_on/workflows/
  -> cbond_on/app/{pipelines,usecases}/
  -> cbond_on/domain/ + cbond_on/infra/
```

不要把 `run/` 或 `interfaces/cli/` 做成新业务逻辑入口；它们是兼容适配层。`domain` 不能反向 import `app`、`infra`、`cli`、`workflows`；运行时代码也不应 import `harness` 或已废弃的 `cbond_on.services`。

| 环节 | 实际入口 / 关键实现 |
| --- | --- |
| Panel / label | `cbond_on/run/build_panels.py`、`cbond_on/run/build_labels.py`；DataHub 负责 raw/clean，ON 只构造派生 panel/label |
| 因子 batch | `cbond_on/run/factor_batch.py`；因子领域代码在 `cbond_on/domain/factors/`，Rust adapter 在 `cbond_on/infra/factors/` |
| 模型 score | `cbond_on/run/model_score.py`；runner 在 `cbond_on/infra/model/runners/` |
| 模型评估 | `cbond_on/run/model_eval.py` |
| 回测 | `cbond_on/run/backtest.py` -> `cbond_on/app/usecases/backtest_runtime.py` / `cbond_on/infra/backtest/runner.py` |
| 单次 live | `cbond_on/run/live.py` -> `cbond_on/cli/live.py` -> `cbond_on/app/usecases/live_runtime.py:run_once` |
| live pipeline | `cbond_on/app/pipelines/live_pipeline.py` -> `app/usecases/run_live_once.py` |
| 单次调度 runner | `liveLaunch/runner.py` |
| 常驻 scheduler | `liveLaunch/scheduler.py` |
| Dashboard | `liveLaunch/web/app.py`，通常 `127.0.0.1:5002` |

### 3.1 生产实盘链

```text
DataHub publish gate（manifest + .done）
  -> 本地派生数据 / label
  -> T1430 factor build
  -> Champion score
  -> challenger score + shadow return 更新 + T1430 state feature 更新
  -> model switch
  -> clean × score / filter_tradable
  -> o_0005 允许池
  -> strategy01_topk_turnover
  -> trade_list.csv + summaries
  -> PostgreSQL replace_date 写入 o_0001
```

如果 gate 失败，ON 在因子、score、选券和 DB 写入之前停止。不要为了下游绕过 gate；今天的事故正是 gate 正确阻断了不完整快照。

---

## 4. 当前交易、标签、回测和入库合同

### 4.1 时间与价格口径

| 概念 | 当前口径 |
| --- | --- |
| live cutoff | `14:29`；`target_policy=next_trading_day_after_cutoff` |
| T1430 因子配置 | panel=`T1430`、`factor_time=14:30`、label marker=`14:42`；生产可见性仍必须严格到 live cutoff，不能把命名为 T1430 当成可以使用 14:30 后数据 |
| 买入 | 当日 `twap_1442_1457`（14:42--14:57） |
| 卖出 | 下一交易日 `twap_0930_0939`（09:30--09:39） |
| label / strict cycle | 当日买、下一交易日卖；`daily_twap` 是唯一正式成本价源 |
| 当日 `score_day` | 做信号 / score 的交易日 |
| `target_day` | 下一交易日、也出现在 live 输出目录及 CSV `trade_date` 字段 |
| `prev_trade_day` | `score_day` 的前一交易日；DB 使用这个日期分区 |

当前 `label_config.json5` 仅记录 14:42--14:57 close window；卖出正式字段由 benchmark / live config 的 `twap_0930_0939` 控制。历史/legacy 文件中其它 sell TWAP 字段可为因子服务，不要误删仍被因子使用的 `daily_twap` 列。

### 4.2 策略与 universe

- strategy：`strategy01_topk_turnover`。
- 配置：`cbond_on/config/strategies/strategy01/strategy01_config.json5`。
- `top_k=20`，`max_weight=0.05`，`turnover_ratio=1.0`，所以当前为等权 Top20、每日完全换仓；下一日全卖。
- allowlist：`quant_factor_dev.researcher_xuvb.o_0005`，滞后 1 个交易日，`factor_value`（fallback `weight`）大于 0 的可交易池。
- no-filter fallback 被禁止：allowlist 缺失、池为空、过滤后为空均应 hard fail。
- `filter_tradable` 由回测和 live 共用，保证交易约束一致。

### 4.3 DB 合同（务必避免日期误解）

```text
backend      = PostgreSQL
table        = quant_factor_dev.researcher_gswzif.o_0001
db_write     = true
db_mode      = replace_date
DB 日期      = prev_trade_day，不是 target_day，也不是 CSV 的 trade_date
```

在 `live_runtime.py`，写入前会把 picks 的 `trade_date` 改为 `prev_trade_day`，并对该日期执行 replace。字段映射为：

```text
instrument_code, exchange_code, trade_date, factor_value, weight, rank
```

**例子（今日修复）**：`score_day=2026-07-29`、`target_day=2026-07-30`，CSV 中显示 `trade_date=2026-07-30`；实际 DB 分区是 `2026-07-28`。DB 复核或人工修复时，如果按 target day 查，会得到错误结论。

生产 DB 写入、改模型、改 universe、改中性化、改 factor set、重启 scheduler 都是需要 owner 明确确认的 live side effect。

---

## 5. DataHub publish gate 与 2026-07-29 事故

### 5.1 当前 gate

`live_config.json5` 的 `data_hub` 配置要求：

```text
manifest root       D:/cbond_data_hub/manifests
required datasets   clean
allow_partial       false
require_done_marker true
ready_gate_enabled  true
```

DataHub 当前为 CBOND_ON 启用 `cbond_on_live_t1430` profile。它会校验 clean snapshot 的行数 / code 数 / 必填字段非空、schema 指纹、历史覆盖率、`daily_twap` 必填字段（包括 `twap_0930_0939`），以及 cbond/stock 最大快照时间至少达到 `14:29:00`；所有通过后才写 `.done`。

DataHub 负责 manifest 准确性，CBOND_ON 只消费 gate。不要在 ON 侧手工伪造 manifest、`.done` 或将 failed manifest 改为 success。

### 5.2 今日事故和恢复事实

1. DataHub 原 intraday scheduler 在约 14:04:51 退出，日志根因是 `_is_trading_day -> load_calendar_open_days -> pd.read_parquet` 的 pandas `ArrayMemoryError`。
2. 当时 raw manifest 成功，但 clean manifest 的 cbond/stock max snapshot 仅到约 `14:04:40/41`，无法满足 T1430 14:29 gate，clean 被标 `failed`、无 `.done`。
3. CBOND_ON scheduler 于 14:29:12 正确 hard-stop：
   ```text
   data hub publish not ready for 2026-07-29: failed manifests=clean
   ```
   其时未进入因子、score、选券或 DB。
4. 经 DataHub Dashboard control plane 恢复后，scheduler PID `4076` 自 14:37:51 运行，补齐 snapshot 和 prepare-live；最终可用 run 为 `20260729_145955_000132`。
5. 当前证据：
   - `D:\cbond_data_hub\manifests\raw\2026-07-29.json`
   - `D:\cbond_data_hub\manifests\clean\2026-07-29.json`
   - `D:\cbond_data_hub\manifests\publish\2026-07-29.done`
   三者均 success/ready，run_id 一致，clean `validation.errors=[]`。

### 5.3 本次 Champion 抢修的审计缺口

手工修复的输出仅有：

```text
D:\cbond_on\results\live\2026-07-30\trade_list.csv
D:\cbond_on\results\live\2026-07-30\allowlist_summary.json
D:\cbond_on\results\live\2026-07-30\universe_filter_summary.json
```

没有 `model_switch_decision.json` 或独立 repair log。因此结果/DB 已核验正确，但本地审计产物不完整。未来若实现固定 Champion 抢修入口，应增加不可变 repair note（model id、score/target/DB day、DataHub run_id、命令/操作者、DB row count / checksum），而不是依赖口头说明。

---

## 6. 当前 live 模型、特征和模型切换

### 6.1 当前生产角色（以 `live_config.json5` 为准）

| 角色 | 名称 | Model ID |
| --- | --- | --- |
| Champion / base score source | Regsim w10_s15_d05 | `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708` |
| Challenger 1 | Ensemble rankavg + labeltop20 | `ensemble_rankavg_baseline_hl20_labeltop20_20260626` |
| Challenger 2 | HL20 live baseline | `lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625` |

Ensemble 的三个 score source：

```text
baseline   lgbm_screened_no_winsor_neutral_tminus1_refit1_202401_rerun_20260623
hl20       lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625
labeltop20 lgbm_screened_no_winsor_neutral_tminus1_weight_labeltop20_20260625
```

**历史台账提示**：`docs/experiment_records/model_tuning_experiments.md` 中一些“current champion=HL20”的表格是当时的结果标签，已经与 2026-07-29 未提交 live config 漂移。当前实盘角色必须以 live config / 当日产物为准，不能反推历史文档。

### 6.2 Champion LGBM 合同

- shared base：`live_lgbm_screened_no_winsor_config.json5`。
- rolling 60 个交易日、daily refit、warm start / state save、MSE regression；日截面 z-score，winsor 关闭，`min_count=30`，27 个特征。
- HL20 基座有 20 交易日 half-life 的 time-decay sample weight。
- Regsim 在 HL20 基座上增加 regime similarity weight：benchmark window=10，same=1.5、different=0.5、missing=1.0，并按 mean normalize。
- T-1 5 style exposure Ridge neutralization：`min_count=30`、`ridge_alpha=1e-6`、standardize exposures；生产中性化只能全开或全关，禁止恢复部分 banlist/exclude 模式。
- model state：`D:\cbond_on\results\model_state\lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708`。

### 6.3 `scoreopt_t1430_fusion_gate`

代码：`cbond_on/infra/live/model_switch.py`；正常实盘决策输出：

```text
D:\cbond_on\results\live\<target_day>\model_switch_decision.json
```

当前 config：

```text
mode                   scoreopt_t1430_fusion_gate
state feature set      path_full_t1430（44 个 T1430 path 特征）
Base lookback / K      60 / 40
Base min_periods       40
metric / score_mode    trim20_lcb10
Base margin            0.0005（5bp）
Robust lookback        360
Robust min_periods     120
Robust ridge alpha     100
Robust target clip     +/-0.0075（75bp）
Robust margin          0.0005（5bp）
```

当前路由意图：

```text
Base 高置信 winner       -> 选择 Base winner
Base 低置信且 Champion 第一 -> champion_first_override，直接保护 Champion，不受 5bp gap 阻断
Champion 明显落后两 challenger -> champion_third_veto，可选最佳 challenger
其余低置信 Base           -> 执行 Robust pairwise Ridge
Robust 明显领先            -> 可以覆盖 Base
Robust 低置信              -> 保留 Base 当前 winner，记录 soft-degrade
```

Robust 不是绝对收益预测：它拟合三模型两两 `day_return` 差的相对 utility。Robust score / pairwise prediction 只有进入 Robust 阶段才有解释价值；Dashboard 也应只在该阶段展示对应分数，不能把未执行/skip 的空值伪装成 0。

`robust_base_veto` 逻辑代码存在，但当前 live config **没有启用**。不要基于回放中看起来很好的单日结果偷开该开关。

### 6.4 soft-degrade 和 hard fail

7/24 后的实现把模型选择低置信视为 soft-degrade，而不是停实盘：会写 `warnings`、Dashboard note，并继续生成名单/写 DB。低置信包括 feature missing/NA、history insufficient、margin default、rolling-Sharpe fallback 和 `fusion_base_robust_not_confident`。

以下仍然是 hard fail：DataHub / factor / score / shadow return、stale history（当前 `fail`）、clean 为空、clean-score 无匹配、allowlist 不可用、过滤后 universe 为空、策略 picks 为空、DB 写入失败。

`fail_on_fallback=true` 仍在 config 中，但正常运行时不再将模型选择不确定性抛为 hard RuntimeError；不要因字段名误判行为，需读 `live_runtime.py`。

### 6.5 昨日旧产物的陷阱

`D:\cbond_on\results\live\2026-07-29` 是 **2026-07-28 score day / 2026-07-29 target day** 的旧输出，不是今日结果。它自然选择 Ensemble，原因为：

```text
reason                 fusion_base_robust_not_confident
Base gap               0.0003073 < 0.0005
Robust margin          0.0002085 < 0.0005
```

对应 DB 分区是 `trade_date=2026-07-27`。不要把这 20 只、该 decision JSON 或其 Ensemble 选择称为 7/29 的 Champion 抢修。

---

## 7. 当前 27 个 live 因子与因子管理规则

### 7.1 运行配置

source of truth：

```text
cbond_on/config/live/live_factors_config.json5
cbond_on/config/factor/packs/live_screened_no_winsor_27.json5
cbond_on/config/live/live_lgbm_screened_no_winsor_config.json5
cbond_on/factor_contracts/profiles/live.json5
```

当前为 `clean_direct` + Rust CPU，pack 为 `live_screened_no_winsor_27.json5`。factor contract profile 是 phase-1 合同目录，**实际 live 因子来源仍是 live factor config / pack**；不要假定 profile 已替代运行配置。

27 个输出列如下（参数的唯一权威在 pack 文件）：

```text
cb_overnight_return_mean_20d        cb_overnight_return_mean_5d
cb_overnight_return_mean_60d        cb_overnight_return_mean_10d
cb_overnight_return_mean_40d        cb_overnight_sharpe_20_0930_0935
daily_sharpe_twap_5d_mean5          cb_overnight_sharpe_5_0930_0935
daily_sharpe_twap_20d_mean5         range_30m
amount_30m                           vol_30m
volume_30m                           mid_move_30m
mom_slope_30m                        ret_10m
depth_weighted_imbalance_v1          premium_momentum_proxy_v1
volen_f60_s10_l3                     alpha001_signed_power_v1
alpha019_close_momentum_sign_v1      alpha024_close_trend_filter_v1
alpha025_return_volume_vwap_range_v1 alpha030_close_sign_volume_v1
alpha041_geometric_mean_vwap_v1      alpha050_volume_vwap_corr_max_v1
alpha078_low_vwap_adv_corr_v1
```

日频 overnight / Sharpe 因子都通过 `market_cbond.daily_twap` 声明所需列与 lookback；可用于因子服务的历史卖出字段不应为了“实盘只卖 09:30--09:39”而删除。

### 7.2 强制的“一因子一文件”规范

每个新因子必须：

1. 新建 `cbond_on/domain/factors/defs/<snake_case>_vN.py`，一因子一个文件。
2. 以 `@FactorRegistry.register("<factor_key>")` 注册，并同步 `defs/__init__.py` 的 import / `__all__`。
3. 通过 config factor pack 建立实例（`name`=输出列、`factor`=实现、`params`=显式参数）。
4. 更新 `factor_contracts/registry.json5`、对应 profile；进入模型还需更新 model feature list。
5. 如果进 live（当前 Rust engine），新增 Rust kernel、`rust/factor_engine/factor_manifest.json` 和 Rust/Python 对齐验证；没有 kernel 时 Rust 必须 fail-fast，不能静默回退 Python。

禁止的因子行为：

```text
直接读文件 / DB / Redis / raw snapshot
在 compute() 内绕过 daily context 读数据
缺字段静默换列、fillna(0)、吞异常
直接使用当日未来价格、label、回测或交易清单收益
自行重建交易日历或改变 panel 回看边界
```

因子只能消费 `FactorComputeContext` 的 `panel`、声明后的 `stock_panel` / `bond_stock_map`，以及由 `daily_requirements()` 声明的 `daily_data`。缺字段要明确 `KeyError`。详细规则见 `docs/开发规则.md`。

`factor_quality_guard` 默认具有写副作用；只读检查必须同时传：

```powershell
py -m cbond_on.common.factor_quality_guard --config factor `
  --no-apply-disable-bad --no-apply-remove-deprecated
```

`--mode dry-run` 本身不足以保证不写 DB / 不改 guard。

---

## 8. 回测、收益和模型研究的共同合同

- 回测和 live 共用 `filter_tradable` 与 strategy 语义。
- 标准严格收益：`buy twap_1442_1457 -> next trading day sell twap_0930_0939`，完整 cycle return 为 `(1 + buy_leg) * (1 + sell_leg) - 1`。
- 2026-07-09 以后完整 cycle return 才是正式收益口径。更早一些报告若标为 `stale_after_return_fix`，不允许和正式结果混排。
- 任何研究都需对齐：日期、预热/refit、label / sell window、费用、benchmark、T-1 neutralization、winsor/zscore、`o_0005` universe、Top20 和实际 score roots。
- 当前 `turnover_ratio=1.0` 的合同下，模型独立 shadow `day_return` 与 daily selector 的实际收益严格等价；若未来改为部分换手、连续持仓或跨日净再平衡，必须重新构建 switch-aware labels / evaluation，不能复用当前结论。

### 8.1 已验证的历史单模型证据（return-fix 后）

窗口 `2024-05-08` 至 `2026-07-08`，只作为历史模型证据，不等于当前角色定义：

| 模型 | 累计收益 | Sharpe | 说明 |
| --- | ---: | ---: | --- |
| Regsim w10_s15_d05 | 128.67% | 3.282 | 当前 live Champion |
| HL20 | 122.55% | 3.126 | 当前 challenger |
| Labeltop20 | 123.21% | 3.115 | Ensemble source |
| Baseline refit1 | 103.76% | 2.777 | Ensemble source |

来源：`docs/experiment_records/model_tuning_experiments.md`。Ranker、DART、旧 feature-contribution 等许多数字仍是旧口径或需要 return-fix 后重跑，不能直接晋升。

---

## 9. 已做实验：结论、证据和禁止的错误外推

### 9.1 模型本身 / 相似日训练 / 线性模型

| 方向 | 已做设计和结果 | 当前决定 |
| --- | --- | --- |
| Hard Similar60（MT-014） | 176 个共同严格日（2025-10-30--2026-07-23）；44 维 `path_full_t1430`，过去 360 状态取 Top60 相似日，daily cold start。15.99% / Sharpe 1.653，Latest60 为 4.37% / 0.528；日差 +6.01bp，t=1.863、双侧 p=.064。去掉 10 个最佳相对日后优势仅 0.74%。 | research-only；未与当前 Regsim production 重新严格对齐，且历史状态标记 14:30 与 live 14:29 边界未完全一致，不能上线。 |
| Soft360Pool（MT-015） | 用 340 日 Gaussian kernel、ESS=60；8.11% / Sharpe .989，低于 Hard60；完整运行约 23 分钟。 | 不推广，不和其他权重叠加。 |
| Ridge / ElasticNet / Huber 小网格 | 60 日 daily refit、27 因子、Top20 严格收益、开发 140 日 + 固定留出 36 日。开发最优 ElasticNet `alpha=1e-4,l1_ratio=.5` 为 21.95% / 2.930，但留出 -3.98% / -2.289；9 个线性候选留出 Sharpe 均为负。 | 全部 reject；不融合、不接 Champion / Robust / live。Ridge 三个 alpha 排序几乎不变，继续扫 alpha 价值低。 |
| dynamic feature contribution | 5 因子族、IC60、最大调整 15%；535 日：HL20 retrained 120.48% / 3.030，dynfc 124.53% / 3.119，但 MDD -7.22% -> -7.81%，只 15/27 月、265/535 日胜。 | 研究线索，未入 live。 |

主要证据：

```text
docs/experiment_records/similar60_pathfull360_20260727.md
docs/experiment_records/soft360pool_pathfull360_ess60_20260727.md
docs/experiment_records/linear_non_tree_grid_20260728.md
D:\cbond_on\results\analysis\dynfc_rerun_20260723\final_retrained_baseline\summary.json
```

线性研究的未提交实现 / 配置包括 `infra/model/impl/linear/linear_score.py`、`runners/train_linear.py`、`adapters.py`、`config/models/linear/*`、对应 score / backtest configs 和 `tests/test_linear_causal_score.py`。它们实现了因果 `d < target_day` / `label_cutoff` 约束，但没有因为实现存在就获得 promotion。

### 9.2 模型切换 / Robust 研究

共同基线：538 个对齐 score day（2024-05-08--2026-07-27），当前 selector replay 为累计 **+147.40%**、Sharpe **3.415**、MDD **-6.58%**、175 次切换。

- 有 Robust 数值诊断的低置信分支为 239 天，但真正覆盖 Base 的只有 **33 天**：19 胜 / 14 负，平均 +4.2369bp，合计 +139.817bp；180 天低置信保持 Base，26 天高置信但同意 Base。
- 不要说“239 天都在 Ridge 决策下”；这是把 eligibility 和实际 override 混为一谈。
- Robust top-two gap 与实际 override 收益 Spearman 约 `.02`，没有校准证据支持只调 gap / threshold。

| 实验 | 核心结果 | 当前决定 |
| --- | --- | --- |
| Robust strict veto | `robust_base_veto` 的 3/5/7.5bp 结果均低于 147.40%；5bp 将 5 天改为 Regsim，实际少 3.67pp。 | config 默认关闭，不上线。 |
| Base / Robust soft fusion | 全样本 30/70 表面 155.53% / 3.518，但前半段相对当前 -1.27pp / Sharpe -.062；从前半选 90/10，后半仍变差。 | 过拟合风险，且未来若非 full liquidation 需路径化成本，不上线。 |
| 动态 pairwise / reliability weight | 表面最佳 150.21% / 3.447，reliability 149.64% / 3.440；前半段均落后当前（-.82pp / -.54pp），MDD 无实质改善。 | 不接 live。 |
| Long-window KNN Robust | 主 Mahalanobis-kernel K40 相对当前 -4.15pp / Sharpe -.075；K60 表面 +1.7pp 但 multiple-testing block-bootstrap familywise p=.770，收益集中后段。 | 不通过。 |
| 冻结统计 pairwise 弃权门 | 只覆盖 2/239 日，144.49% / 3.385，仍比当前 -2.91pp。 | 不替代 Ridge。 |
| Ridge 输入重建 | disp7 / PCA15 / PCA+score-disagreement 无正 OOS pairwise 预测，分别相对当前 -2.26 / -1.08 / -4.32pp。 | 不继续在同一 538 日样本扫输入参数。 |
| coherent two-contrast Ridge | 两个 Helmert contrast 消除三 pairwise 的代数不一致；最终选择 538/538 与现行完全相同，收益仍 147.4008% / 3.4146。 | 排除“pairwise 代数不一致是主因”的假设。 |
| switch-cost equivalence | 1,614 次真实 score 选券和 538 日 NAV 均完全一致，证明在 `turnover_ratio=1.0` 下 standalone shadow return 与 selector 序列严格等价。 | 维持当前标签；未来改交易合同才重做 switch-aware 评估。 |

可复现工具：

```text
harness/tools/ridge_robust_replay.py
harness/tools/similarity_robust_replay.py
harness/tools/stat_pairwise_gate_replay.py
harness/tools/coherent_switch_ridge_replay.py
harness/tools/switch_cost_equivalence_check.py
```

每项的完整 CSV、月度拆分、`nav_compare.png` 路径在下列记录中：

```text
docs/experiment_records/model_switch_robust_strict_veto_20260728.md
docs/experiment_records/model_switch_soft_fusion_20260728.md
docs/experiment_records/model_switch_dynamic_weight_optimization_20260728.md
docs/experiment_records/model_switch_similarity_robust_20260728.md
docs/experiment_records/model_switch_stat_pairwise_gate_20260728.md
docs/experiment_records/model_switch_ridge_rebuild_20260728.md
docs/experiment_records/model_switch_coherent_two_contrast_ridge_20260728.md
docs/experiment_records/model_switch_switch_cost_equivalence_20260728.md
```

### 9.3 `parity_adjusted_stock_lag_v1` 因子候选

状态：未提交、research-only，尚未进入 live factor pack 或模型。

| 项目 | 事实 |
| --- | --- |
| 代码 | `cbond_on/domain/factors/defs/parity_adjusted_stock_lag_v1.py` |
| 注册 | 仅 `defs/__init__.py`；scratch pack：`cbond_on/config/factor/research/parity_adjusted_stock_lag_v1_scratch.json5` |
| 定义 | T-1 `kappa=clip(conv_value/cb_close_price,0,2)`；T 日 14:00--14:29 `kappa * stock_tail_return - cbond_tail_return`；T-1 `o_0005` pool |
| 预筛窗口 | DataHub clean 严格层 2024-01-03--2026-07-27，615 日 |
| 预筛结果 | RankIC +.02894 / t=8.99，Top20 相对池 +5.27bp/日 / t=3.28；coverage 99.08%，中位截面 467；最后 57 日 RankIC +.02989 / t=2.38、Top20 +12.53bp/日 / t=2.10 |
| 风险 | 2026-03--04 段 RankIC 约 0；尚无正式 FactorBatch 20 分箱 / walk-forward / benchmark / quality guard 结果 |

正式报告 / 图片**尚不存在**：确认 `D:\cbond_on\results` 和 `D:\cbond_on\factor_data` 中没有该候选的 `factor_metrics.csv`、`bin_time_returns.csv`、`diagnostics.csv`、`summary.json` 或 `factor_report.png`，也没有 FactorBatch 进程在跑。此前的预筛报告不是 FactorBatch 图像报告，不能把它交付为正式因子报告。

下一步若用户重新要求推进，应在独立 scratch factor_data / results 下完成：20 分箱、40/30 日 walk-forward、benchmark / Newey-West alpha、bad-factor 筛查、Rust 准入（若拟入 live），再由 owner 决定是否 promotion。

证据：`docs/experiment_records/parity_adjusted_stock_lag_factorbatch_compatible_prescreen_20260729.md`。

---

## 10. Barra 风格风险模型：CB-Risk v1

这不是多因子 alpha，也不是当前 27 个预测因子。它是独立的 convertible-bond Barra-style 风险 / 归因层，用来回答“组合暴露、因子风险、协方差、特异风险、组合风险和收益归因”，当前不选券、不改模型、不写 `o_0001`。

### 10.1 当前实现状态

所有以下文件均为未提交的 shadow-only 研究实现：

```text
cbond_on/config/risk/cb_risk_v1_config.json5
cbond_on/domain/risk/*
cbond_on/infra/risk/*
cbond_on/app/pipelines/risk_pipeline.py
cbond_on/app/usecases/risk_runtime.py
cbond_on/bootstrap/risk.py
cbond_on/cli/risk.py
cbond_on/workflows/research/cb_risk.py
cbond_on/schemas/config/risk.py
tests/test_cb_risk_*.py
```

模式：`offline_shadow`、`write_db=false`、`live_hook=false`。

已实现：CB_MKT + CB_SIZE / LIQ / PREMIUM / DURATION / duration-orthogonal convexity / CREDIT / EQUITY_VOL exposure、robust WLS factor return、EWMA covariance、diagonal specific risk、portfolio risk 和 return attribution。

未实现或未准入：行业、float market cap、fundamental value/quality/growth、risk optimizer、live scheduler hook、trade-list mutation、DB write。

### 10.2 上游数据依赖 / PIT 阻塞

当前本地历史数据没有不可变 `available_at` / revision 证据，因此均标 `PIT-unverified`；不能用来证明一个 14:29 实盘风险判断。DataHub 需要发布：

```text
risk_barra.cbond_exposure_input
primary key = (trade_date, asof_cutoff, cbond_code, revision_id)
asof_cutoff = EOD | 142900
available_at <= D 14:29
```

完整所需合约见 `docs/cb_risk_v1_data_contract.md`。它要求 immutable revision、SCD mapping / industry、风险 benchmark 成分、严格 holding-period returns 和质量门。

已记录 15 个 risk tests 通过、10 日 dry-run；示例报告：

```text
D:\cbond_on\results\risk\cb_risk_v1\yesterday_signal_20260727\run_20260728_145300\risk_report.html
```

该日绝对年化波动约 10.285%，无 benchmark holdings 所以 TE 为空。它只能作为 shadow report。

---

## 11. 当前 dirty worktree：保护、分组和审阅顺序

截至本交接，`git status --short` 包含大量 `M`、`D`、`??`。不要把“未提交”解释成“无用”，也不要把它解释成“已经发布”。

### 11.1 直接影响 live 的未提交改动

```text
cbond_on/config/live/live_config.json5
cbond_on/app/usecases/live_runtime.py
cbond_on/infra/live/model_switch.py
liveLaunch/web/static/app.js
liveLaunch/web/static/style.css
tests/test_live_model_switch.py
tests/test_live_dashboard_model_compare.py
```

它们包含 Regsim 角色切换、`path_full_t1430`、Champion-first / third-veto、soft-degrade warning、Robust / decision 展示等逻辑。当前实盘已依赖当前运行目录中的这些文件；新 agent 若要改动，先看 diff 和对应测试，不能用历史 handoff 覆盖。

### 11.2 研究 / 风险 / 因子未提交改动

```text
cbond_on/infra/model/similar_day_training.py
cbond_on/config/models/lgbm/*similar60* / *soft360pool* / *dynfc*
cbond_on/config/models/linear/*
cbond_on/config/backtest_pipeline/backtest_linear_non_tree_*
cbond_on/config/score/model/model_score_linear_non_tree_grid_20260728_config.json5
cbond_on/infra/model/impl/linear/linear_score.py
cbond_on/infra/model/runners/train_linear.py
cbond_on/infra/model/adapters.py
cbond_on/domain/factors/defs/parity_adjusted_stock_lag_v1.py
cbond_on/config/factor/research/*
cbond_on/{app,bootstrap,cli,config,domain,infra,schemas,workflows}/risk/*
docs/experiment_records/*_20260727.md / *_20260728.md / *_20260729.md
harness/task_state_* / harness/tools/*replay*.py
tests/test_linear_causal_score.py / tests/test_similar_day_training.py / tests/test_cb_risk_*.py
```

### 11.3 其它已修改 / 删除项

还存在 `backtest_runtime.py`、`repo_hygiene_guard.py`、`label_config.json5`、`benchmark/service.py`、`train_lgbm.py`、`test_strict_cycle_returns.py` 等修改，以及外层 `config/models/lgbm/*regsim_grid*`、`config/score/model/*regsim_grid*` 的删除。它们可能来自前序工作，不能在没有 diff 审阅前恢复。

建议先执行：

```powershell
git status --short
git diff --stat
git diff -- cbond_on/config/live/live_config.json5
git diff -- cbond_on/app/usecases/live_runtime.py cbond_on/infra/live/model_switch.py
```

然后按 `live`、`risk`、`linear`、`similar-day`、`factor` 五组分别 review / test / commit；不要混成一次巨大提交。

---

## 12. 代码和操作规范（下一位负责人必须遵守）

1. **先确认 scope**：不静默改 live model / model state / neutralization / factor set / universe / DB target / schedule / output path。用户确认的是 live 变更范围，不是泛泛“改一下”。
2. **实盘抢修顺序**：先 state / log，再 resolved config / path，再 DataHub manifest + `.done`，再 clean、factor、score、decision、allowlist、trade list、DB。既定业务优先级是先保证 Champion Regsim 入库，再排根因；但 DB write 仍须有明确授权。
3. **禁止手工篡改状态**：不要清 state、改 manifest 或写 `.done` 只为让 UI 变绿。
4. **scheduler 不热加载**：修改 live Python/config 后，经授权后通过 Dashboard control plane 重启；核验 PID、started_at、state、日志和实际 feature/config。不要只看 `pid.json`，它可能陈旧。
5. **DataHub 边界**：raw/clean / manifest 的生产和准确性由 DataHub 负责；ON 不要本地重新计算或绕过 publish contract。
6. **因子边界**：一因子一文件、显式依赖、时间可见性、Rust/Python 对齐；没有完整 batch / OOS / factor contract 验证不能进 live。
7. **研究边界**：基线保持当前 live chain；所有 research 用独立 output/config，报告 aligned contract / coverage / promotion boundary；不自动 promotion 到 live。
8. **读取 / 编辑习惯**：使用 inner Git root、UTF-8 PowerShell 读取、`py` 而不是 WindowsApps `python`；编辑文件使用 `apply_patch`；保留 unrelated dirty changes。
9. **测试与 guard**：每次改动先用对应 harness preflight，再跑最小相关 tests。`factor_quality_guard` 默认为可写；只读时显式关闭 apply。历史 `20 passed` / `17+3 passed` 只是此前快照，不替代当前 dirty worktree 的重跑。
10. **实盘输出声明**：不要把历史手动 Regsim repair、自然 selector 结果、CSV target day 和 DB prev_trade_day 混为同一日期 / 同一事件。

---

## 13. 未完成事项和安全的下一步

### P0：当日调度状态（需 owner 决策）

- 已完成：今日 Champion 名单 / DB。
- 未完成：CBOND_ON scheduler state 仍 failed，且没有该次 repair 的完整 audit artifact。
- **不要**直接 normal scheduler restart。
- 需要明确选择之一：
  1. 只保留正确名单和 DB，下一交易日开盘前按规范重新启动 scheduler；
  2. 设计并批准“固定 Champion、幂等、带 audit 的 repair entrypoint”，用于恢复 state；
  3. 明确批准常规 selector 重跑及其可能覆盖当前结果的风险。

### P1：DataHub 稳定性

- 今日 scheduler 的 `ArrayMemoryError` 是触发失败的根因；当时并非持续内存耗尽（恢复时机器仍有约 14.8GB 空闲）。
- 后续应在 DataHub 侧追踪 calendar parquet load 的内存行为 / scheduler resilience；不要降低 CBOND_ON T1430 gate 来掩盖。

### P2：因子候选

- `parity_adjusted_stock_lag_v1` 先补正式 FactorBatch 图 / CSV，严格独立 scratch，再决定是否进入模型。
- 不要继续之前被中断的 FactorBatch / 图片生成，除非用户重新要求；目前没有运行进程、没有正式图片可交付。

### P3：模型与切换研究

- 线性 / Similar60 / Soft360Pool / dynamic fusion / statistical Robust 均没有 live promotion 证据。
- 如果继续研究，首先定义未参与选参的时间外 holdout 或 forward shadow；不要在同一 538 天 replay 上继续扫 alpha、K、margin、PCA 或权重后宣布改进。

### P4：CB-Risk / Barra

- 先推动 DataHub PIT contract，不要把 PIT-unverified 历史风险报告接入 live。
- 风险层的任何上线应与 alpha / live selection 完全隔离，单独 shadow 验证后再授权。

---

## 14. 可直接给下一位 agent 的启动提示

```text
你是 CBOND_ON 新负责人。请先完整阅读
docs/handoffs/owner_takeover_handoff_20260729.md，随后在真实 Git 根
C:\Users\BaiYang\CBOND_ON\cbond_on 做只读健康核验。

当前优先事项：确认 2026-07-29 Champion Regsim repair 的 CSV 和
o_0001 trade_date=2026-07-28 分区仍一致；确认 DataHub 2026-07-29
raw/clean/.done run_id 一致；确认 CBOND_ON scheduler 仍 failed 但不要直接启动它。

不得 reset/clean/revert dirty worktree；不得无授权写 DB、改 live config、
改 model switch、重启 scheduler。报告当前状态和一个不覆盖 Champion 的恢复方案，
等待 owner 决策。
```

---

## 15. 重要证据索引

```text
当前 live config
  cbond_on/config/live/live_config.json5
  cbond_on/config/live/live_factors_config.json5
  cbond_on/config/live/live_switch_source_regsim_w10_s15_d05_20260708_config.json5

今日实盘 / DB 对照
  D:\cbond_on\results\live\2026-07-30\trade_list.csv
  D:\cbond_on\results\live\2026-07-30\allowlist_summary.json
  D:\cbond_on\results\live\scheduler\state.json
  D:\cbond_on\results\live\2026-07-29\logs\live_scheduler_2026-07-29.log

DataHub
  D:\cbond_data_hub\runtime\scheduler_state.json
  D:\cbond_data_hub\manifests\raw\2026-07-29.json
  D:\cbond_data_hub\manifests\clean\2026-07-29.json
  D:\cbond_data_hub\manifests\publish\2026-07-29.done

历史交接 / 研究
  docs/handoffs/live_model_switch_handoff_20260724.md
  docs/experiment_records/model_tuning_experiments.md
  docs/experiment_records/linear_non_tree_grid_20260728.md
  docs/experiment_records/model_switch_*_20260728.md
  docs/experiment_records/similar60_pathfull360_20260727.md
  docs/experiment_records/soft360pool_pathfull360_ess60_20260727.md
  docs/experiment_records/parity_adjusted_stock_lag_factorbatch_compatible_prescreen_20260729.md
  docs/cb_risk_v1_data_contract.md
  docs/开发规则.md
```

