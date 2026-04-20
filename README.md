# CBOND_ON

可转债隔夜策略工程仓库（研究、回测、实盘一体化）。

当前代码已迁移到新分层架构：
`interfaces -> app -> domain / infra`  
旧 `services/*` 架构已移除。

## 1. 全链路总览

1. `build_panels`：从 DataHub `raw/clean` 构建 panel（并联动构建 labels）。
2. `build_labels`：只重建 labels（不重建 panel）。
3. `factor_batch`：批量计算因子、执行单因子回测、输出筛选报告。
4. `model_score`：滚动训练并生成 score。
5. `model_eval`：模型评估与超参搜索（支持 W&B sweep）。
6. `backtest`：把 score 转成仓位并回测净值。
7. `live`：执行一次实盘流程，输出当日交易清单。
8. `liveLaunch`：实盘调度器与 Web 控制台。

## 2. 目录职责

- `cbond_on/config`：业务配置（data、factor、score、live、models、backtest）。
- `cbond_on/core`：通用基础能力（时间、交易日、命名、配置解析等）。
- `cbond_on/domain`：纯领域逻辑（factors/spec、signals、portfolio、strategies）。
- `cbond_on/app`：用例与流水线编排（usecases、pipelines）。
- `cbond_on/infra`：外部实现适配（数据 IO、Rust 因子引擎、模型 runner、回测、实盘）。
- `cbond_on/interfaces/cli`：CLI 入口适配层。
- `cbond_on/run`：兼容入口（薄转发到 `interfaces/cli`）。
- `liveLaunch`：实盘调度与可视化控制台。
- `docs`：架构、部署、因子相关文档。

## 3. 关键口径约束

1. `raw/clean` 由 DataHub 生产，ON 不再本地重算这两层。
2. Label 成本口径固定为 `daily_twap`，不允许回退到项目内自算 TWAP。
3. 默认标签窗口：
   - 买入：`14:42-14:57`
   - 卖出：次日 `09:30-09:45`
4. `filter_tradable` 在 backtest/live 共用，保证交易约束一致。

## 4. 配置文件地图

- 路径配置：`cbond_on/config/data/paths_config.json5`
- Panel 配置：`cbond_on/config/data/panel_config.json5`
- Label 配置：`cbond_on/config/data/label_config.json5`
- 因子总控：`cbond_on/config/factor/factor_config.json5`
- 因子清单：`cbond_on/config/factor/factors_all_in_one.json5`
- 坏因子黑名单：`cbond_on/config/factor/factor_disabled_factors.json`
- 模型打分总控：`cbond_on/config/score/model_score_config.json5`
- 模型评估总控：`cbond_on/config/score/model_eval_config.json5`
- 回测总控：`cbond_on/config/backtest_pipeline/backtest_config.json5`
- 全链路总控：`cbond_on/config/backtest_pipeline/pipeline_all_config.json5`
- 实盘总控：`cbond_on/config/live/live_config.json5`
- 实盘因子：`cbond_on/config/live/live_factors_config.json5`
- 实盘模型注册：`cbond_on/config/live/live_models_config.json5`
- 实盘模型参数：`cbond_on/config/live/live_lgbm_factor_MSE_config.json5`

## 5. 各环节详细说明

### 5.1 Panel 构建（`run/build_panels.py`）

目标：
- 从 `raw/clean` 生成 panel 快照。
- 同一命令内联动重建 labels。

入口命令：

```bash
python cbond_on/run/build_panels.py
```

主要配置：
- `panel_config.json5`
- `label_config.json5`

输入：
- `raw_data_root`
- `clean_data_root`
- `schedule / snapshot / panel_mode / assets`

输出：
- `panel_data_root/panels/{asset}/{panel_name}/{YYYY-MM}/{YYYYMMDD}.parquet`
- `label_data_root/{YYYY-MM}/{YYYYMMDD}.parquet`

说明：
- `build_panels.py` 会先跑 panel，再跑 label。
- 如果只想重算标签，用 `build_labels.py`。

### 5.2 Label 构建（`run/build_labels.py`）

目标：
- 仅重建 labels。

入口命令：

```bash
python cbond_on/run/build_labels.py
```

主要配置：
- `label_config.json5`

输入：
- 标签窗口配置
- `daily_twap` 成本数据

输出：
- `label_data_root/{YYYY-MM}/{YYYYMMDD}.parquet`

关键行为：
- 成本源固定 `daily_twap`。
- TWAP 列名可根据窗口自动解析（如 `twap_1442_1457`）。

### 5.3 因子批处理（`run/factor_batch.py`）

目标：
- 批量计算因子。
- 生成单因子回测结果。
- 输出筛选与报告产物。

入口命令：

```bash
python cbond_on/run/factor_batch.py
```

主要配置：
- `factor_config.json5`
- `factor_files` 指向的因子清单

输入：
- panel 数据
- label 数据
- 因子定义与计算引擎

输出：
- 因子值：`factor_data_root/factors/{panel_name}/{YYYY-MM}/{YYYYMMDD}.parquet`
- 报告目录：`results/{start}_{end}/Single_Factor/{timestamp}/`
- 每因子报告：`.../{factor_name}/factor_report.png`
- 聚合图目录：`.../plot/`
- 筛选结果：
  - `.../screened/factor_screening_all.csv`
  - `.../screened/factor_shortlist.csv`

关键行为：
- 计算引擎支持 `rust` 和 `rust_shm_exp`。
- 支持 `refresh/overwrite`。
- 自动跳过黑名单因子（`factor_disabled_factors.json`）。

### 5.4 因子质量守卫（`cbond_on.common.factor_quality_guard`）

目标：
- 识别废弃因子和坏因子。
- 同步维护黑名单并清理因子库存。

入口命令：

```bash
python -m cbond_on.common.factor_quality_guard --config factor
```

当前默认行为（已开启）：
- 自动 `disable_bad`（坏因子入黑名单）
- 自动 `remove_deprecated`（从因子库存中删除废弃/坏列）

只读扫描模式：

```bash
python -m cbond_on.common.factor_quality_guard --config factor --no-apply-disable-bad --no-apply-remove-deprecated
```

输出与副作用：
- 更新 `cbond_on/config/factor/factor_disabled_factors.json`
- 清理 `factor_data_root/factors/{panel}/...` 历史 parquet 列
- 控制台打印：
  - 新增黑名单因子
  - 既有黑名单因子
  - 各因子列清理命中次数

### 5.5 模型打分（`run/model_score.py`）

目标：
- 执行滚动训练并产出 score。

入口命令：

```bash
python cbond_on/run/model_score.py
```

常用参数：
- `--model-id`
- `--start --end`
- `--label-cutoff`
- `--refit-every-n-days`
- `--train-processes`
- `--parallel-shards --parallel-shard-index`

主要配置：
- `model_score_config.json5`
- 对应模型配置（`config/models/*` 或 `config/live/*`）

输入：
- 因子库
- 标签库

输出：
- 配置中的 `score_output` 路径
- 通常是按交易日拆分的 score CSV

关键行为：
- 支持 rolling/incremental/warm-start（取决于模型配置）。
- 支持 execution 线程参数与分片并行。
- 支持 W&B 日志（`execution.wandb`）。

### 5.6 模型评估与调参（`run/model_eval.py`）

目标：
- 做单次模型评估，或执行超参搜索。

入口命令：

```bash
python cbond_on/run/model_eval.py
```

主要配置：
- `model_eval_config.json5`

输入：
- `model_score` 模型注册信息
- score 与标签数据

输出：
- `results/model_eval/{experiment_name}/{timestamp}/`
- `summary`、`daily`、`trials`、`best_trial`、配置快照与报告

关键行为：
- `run_model_before_eval=true` 时先执行 model_score。
- `tuning.enabled=true` 进入调参模式。
- `search_type=wandb_sweep` 时使用 W&B sweep。

### 5.7 回测（`run/backtest.py`）

目标：
- 将 score 转换为策略持仓并回测表现。

入口命令：

```bash
python cbond_on/run/backtest.py
```

主要配置：
- `backtest_config.json5`
- `strategy01_config.json5`

输入：
- score 数据
- `daily_twap` 买卖价格
- `filter_tradable` 过滤约束

输出：
- `results/{start}_{end}/{batch_id}/{timestamp}/`
- `daily_returns.csv`
- `nav_curve.csv`
- `positions.csv`
- `ic_series.csv`
- `diagnostics.csv`
- 回测图像报告

关键行为：
- 成本模型：`twap_bps + fee_bps`
- 策略语义和 live 共用

### 5.8 实盘单次执行（`run/live.py`）

目标：
- 执行一次完整实盘流程，生成当日 `trade_list`。

入口命令：

```bash
python cbond_on/run/live.py
```

可选参数：
- `--start`
- `--target`
- `--mode`

主要配置：
- `live_config.json5`
- `live_factors_config.json5`
- `live_models_config.json5`

执行步骤（`app/usecases/live_runtime.py`）：
1. 读取并校验 live/factor/model 配置。
2. 解析 `run_day / target_day / lookback_start / prev_trade_day`。
3. 校验 DataHub 发布门控（manifest + done）。
4. 增量构建 panel/label/factor。
5. 执行 model_score 并读取当日 score。
6. 与 clean 合并并执行 `filter_tradable`。
7. 调用策略（默认 `strategy01_topk_turnover`）输出 picks。
8. 写入 `trade_list.csv`，可选写入数据库。

输出：
- `results/live/{YYYY-MM-DD}/trade_list.csv`
- 可选 DB 写入（`output.db_write=true`）

### 5.9 实盘调度与控制台（`liveLaunch`）

目标：
- 定时触发实盘。
- 提供状态监控、启停、重启、急停。

组件：
- 调度器：`liveLaunch/scheduler.py`
- 单次 runner：`liveLaunch/runner.py`
- Web 控制台：`liveLaunch/web/app.py`（Flask，默认 `127.0.0.1:5002`）

常用命令：

```bash
# 单次 runner
python -m liveLaunch.runner

# 常驻调度器
python -m liveLaunch.scheduler

# Web 控制台
python -m liveLaunch.web.app
```

调度状态产物：
- `results/live/scheduler/state.json`
- `results/live/scheduler/pid.json`
- `results/live/{day}/logs/live_scheduler_*.log`

### 5.10 全链路编排（`run/pipeline_all.py`）

目标：
- 一条命令执行 panel -> label -> factor -> model_score -> backtest。

入口命令：

```bash
python cbond_on/run/pipeline_all.py
```

主要配置：
- `pipeline_all_config.json5`

行为：
- 用顶层 `start/end` 统一覆盖各阶段窗口。
- 支持分阶段 `refresh/overwrite`。
- 可统一覆盖 `model_id`、`strategy_id`。

## 6. 命令速查

### Windows

```powershell
& C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\python.exe cbond_on/run/build_panels.py
& C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\python.exe cbond_on/run/factor_batch.py
& C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\python.exe cbond_on/run/model_score.py
& C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\python.exe cbond_on/run/backtest.py
& C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\python.exe cbond_on/run/live.py
```

### Linux

```bash
source ~/venv/cbond/bin/activate
cd ~/cbond_on
python cbond_on/run/build_panels.py
python cbond_on/run/factor_batch.py
python cbond_on/run/model_score.py
python cbond_on/run/backtest.py
python cbond_on/run/live.py
```

## 7. Rust 扩展更新（服务端）

```bash
source ~/venv/cbond/bin/activate
cd ~/cbond_on/rust/factor_engine
python -m pip install -U maturin
python -m maturin develop --release
python -c "import cbond_on_rust; print('cbond_on_rust OK')"
```

## 8. Guard 命令

```bash
# 架构分层守卫
python -m cbond_on.common.architecture_guard

# 仓库卫生守卫
python -m cbond_on.common.repo_hygiene_guard

# 因子质量守卫（默认执行修复动作）
python -m cbond_on.common.factor_quality_guard --config factor
```

## 9. 产物目录总览（基于 paths_config）

- `panel_data_root`：panel 日文件
- `label_data_root`：label 日文件
- `factor_data_root`：因子日文件
- `results_root/scores`：模型 score
- `results_root/model_eval`：模型评估与调参产物
- `results_root/{date_label}/Backtest`：策略回测结果
- `results_root/{date_label}/Single_Factor`：因子批处理结果
- `results_root/live/{day}`：实盘交易清单与日志

## 10. 相关文档

- 架构边界：`docs/architecture_layers.md`
- 服务器更新命令：`docs/server_update_commands.md`
- 因子开发说明：`docs/factor_development.md`
