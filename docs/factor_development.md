# CBOND_ON 因子开发手册（Agent 交接版，2026-03-25）

> 目标：让新 Agent 可以在不反复问人的前提下，独立完成因子开发、回测、筛选、入模对接。

## 1. 项目口径（先统一）

- ON 已经是 DataHub 消费方：
  - `raw/clean` 由 DataHub 生产与维护；
  - ON 不再负责拉 raw / 生成 clean。
- ON 负责的衍生链路：
  - `panel -> label -> factor -> model_score -> backtest/live`。
- 因子开发默认基于 panel（当前是 cbond panel）。

## 2. 当前关键路径（本地配置）

- `raw_data_root`: `D:/cbond_data_hub/raw_data`
- `clean_data_root`: `D:/cbond_data_hub/clean_data`
- `panel_data_root`: `D:/cbond_on/panel_data`
- `label_data_root`: `D:/cbond_on/label_data`
- `factor_data_root`: `D:/cbond_on/factor_data`
- `results_root`: `D:/cbond_on/results`

配置文件：

- 路径配置：`cbond_on/config/data/paths_config.json5`
- panel 配置：`cbond_on/config/data/panel_config.json5`
- label 配置：`cbond_on/config/data/label_config.json5`
- factor 配置：`cbond_on/config/factor/factor_config.json5`

## 3. 代码入口与职责

- 因子定义目录：`cbond_on/factors/defs/`
- 注册入口：`cbond_on/factors/defs/__init__.py`
- 因子基类：`cbond_on/factors/base.py`
- 因子计算流水线：`cbond_on/factors/pipeline.py`
- batch 主流程（构建 + 单因子回测 + 报告 + 过筛）：`cbond_on/factor_batch/runner.py`
- 运行入口：`cbond_on/run/factor_batch.py`

关键事实：

- `run/factor_batch.py` 已显式导入 `cbond_on.factors.defs`，避免未注册错误。
- 因子列名以 `FactorSpec.name`（或 `output_col`）为准，不再自动拼接参数。
- 因子上下文现支持三类输入：
  - `ctx.panel`（cbond panel，主输入）
  - `ctx.stock_panel`（stock panel，可为空）
  - `ctx.bond_stock_map`（债股映射，来自 `market_cbond.daily_base`，可为空）

## 4. 可用数据资源与字段

## 4.1 Panel（因子 compute 直接可用）

当前样本（`panel_data/panels/cbond/T1430/2026-03/20260324.parquet`）：

- 索引：`dt, code, seq`
- 列（34）：

```text
trade_time
pre_close
last
open
high
low
close
volume
amount
num_trades
high_limited
low_limited
ask_price1
ask_volume1
bid_price1
bid_volume1
ask_price2
ask_volume2
bid_price2
bid_volume2
ask_price3
ask_volume3
bid_price3
bid_volume3
ask_price4
ask_volume4
bid_price4
bid_volume4
ask_price5
ask_volume5
bid_price5
bid_volume5
iopv
trading_phase_code
```

## 4.2 Stock 资源（可用，但不会自动进当前因子流水线）

- `clean_data/snapshot/stock/YYYY-MM/YYYYMMDD.parquet`（盘中快照，字段结构与 cbond snapshot 对齐）
- `raw_data/market_cbond__daily_base`（日频，常用字段：`stock_code`, `stock_close_price`, `stock_volatility`, `stk_*`, `conv_value`, `bond_prem_ratio`, `puredebt_prem_ratio`）
- `raw_data/market_cbond__daily_deriv`（日频，常用字段：`stock_code`, `stock_close_price`, `stock_volatility`, `conv_value`, `bond_prem_ratio`, `puredebt_prem_ratio`）

当前 `stock panel` 可直接使用字段（与 cbond panel 对齐，34 列）：

```text
trade_time
pre_close
last
open
high
low
close
volume
amount
num_trades
high_limited
low_limited
ask_price1
ask_volume1
bid_price1
bid_volume1
ask_price2
ask_volume2
bid_price2
bid_volume2
ask_price3
ask_volume3
bid_price3
bid_volume3
ask_price4
ask_volume4
bid_price4
bid_volume4
ask_price5
ask_volume5
bid_price5
bid_volume5
iopv
trading_phase_code
```

当前 `bond_stock_map` 输出字段：

- `code`（转债代码，带交易所后缀）
- `stock_code`（正股代码，带交易所后缀）
- `trade_date`（映射所属交易日）

注意：

- 当前 `run_factor_pipeline` 读取的是 `panel_data/panels/cbond/...`；
- stock 信息已可通过 `ctx.stock_panel` 与 `ctx.bond_stock_map` 进入因子；
- 需要联动时，因子内部显式按映射关系与时间对齐，不要隐式假设 code 一致。

## 4.3 多资产上下文（已落地）

`factor_config.context` 控制是否注入 stock 资源：

```json5
context: {
  stock_panel: { enabled: true, strict: false },
  bond_stock_map: { enabled: true, strict: false, table: "market_cbond.daily_base" }
}
```

说明：

- `stock_panel.enabled=true`：按同一交易日读取 `panel_data/panels/stock/<panel_name>/...`
- `bond_stock_map.enabled=true`：读取 `raw_data/market_cbond__daily_base` 形成 `code -> stock_code` 映射
- `strict=true`：缺失即报错；`strict=false`：缺失则传空，因子自行兜底
- 映射会自动规范为带交易所后缀（例如 `113574.SH -> 603679.SH`）
- 映射读取支持“向前回退”：当天缺失时使用最近可用交易日映射

因子中读取方式示例：

```python
def compute(self, ctx: FactorComputeContext) -> pd.Series:
    bond = ctx.panel
    stock = ctx.stock_panel          # 可能为 None
    mapping = ctx.bond_stock_map     # 可能为 None
    ...
```

## 5. 因果与时间规则（必须遵守）

- `factor_time`（默认 14:30）之前可见信息才允许进入因子。
- `label_time`（默认 14:42）之后的信息严禁泄露到因子。
- 日频正股字段默认按“上一交易日”对齐；不要直接把同日收盘口径字段用于盘中因子。
- 项目交易日口径以交易日服务为准，不按自然日推断。

## 6. 因子开发规范（硬约束）

1. 一因子一文件：`cbond_on/factors/defs/<factor>.py`
2. 类继承 `Factor`
3. 必须注册：`@FactorRegistry.register("<factor_key>")`
4. `compute(ctx)` 返回 `pd.Series`，索引对齐 `(dt, code)`（最终落盘带 `seq` 结构）
5. 输出列名用 `self.output_name(self.name)`
6. 显式处理空值和异常（不得 silent fail）
7. 新增因子后必须在 `cbond_on/factors/defs/__init__.py` 导入

## 7. 命名规范（配置层）

- `factor`：注册键（例如 `ret_window`）
- `name`：该实例落盘列名/报告目录名（必须唯一）
- `output_col`：可选，覆盖最终列名

建议：

- 用小写 snake_case
- 结构建议：`<factor_short>_<key_param>`
- 不要用空格、中文、路径符

## 8. 当前 batch 行为（非常重要）

运行命令：

```bash
python cbond_on/run/factor_batch.py
```

当前 `factor_batch` 包含以下环节：

1. 因子构建（支持多线程，`factor_config.workers`）
2. 单因子回测（支持多线程，`factor_config.backtest.workers`）
3. 每个因子输出报告目录
4. 按筛选规则生成入围表
5. 把入围因子的图片复制到 `screened/selected_reports`

每次 batch 输出目录：

`D:/cbond_on/results/<start>_<end>/Single_Factor/<batch_ts>/`

其中关键文件：

- `<factor_name>/factor_report.png`
- `<factor_name>/factor_metrics.csv`
- `<factor_name>/summary.json`
- `screened/factor_screening_all.csv`
- `screened/factor_shortlist.csv`
- `screened/selected_reports/<factor_name>.png`（仅图片）
- `screened/screening_config.json`

## 9. 筛选逻辑（当前实现）

筛选配置在 `factor_config.screening`，核心字段：

- `enabled`
- `ic_metric`（例如 `rank_ic_mean`）
- `ir_metric`（例如 `rank_ic_ir`）
- `ic_abs_min`（比较 `abs(ic_metric_value)`）
- `ir_abs_min`（比较 `abs(ir_metric_value)`）
- `sharpe_min`（比较 `sharpe >= sharpe_min`）
- `copy_reports`（是否输出 `selected_reports`）

说明：

- 阈值不应硬编码在代码里，统一从配置读取；
- 新 Agent 改阈值只改配置，不改逻辑代码。

## 10. 入模对接要求

新因子进入模型训练前，必须同步修改模型配置中的 `factors` 列表：

- `cbond_on/config/models/lgbm/lgbm_factor_MSE_config.json5`
- `cbond_on/config/models/lgbm_ranker/lgbm_factor_ranker_config.json5`
- `cbond_on/config/models/linear/linear_factor_default_config.json5`

不改这里会出现“因子已落盘但模型未读取”的假象。

## 11. 推荐开发流程（给新 Agent）

1. 选一个模板因子复制开发（推荐 `ret_window.py` 或 `depth_imbalance.py`）
2. 实现 `compute()`，先只用 panel 字段，确保无泄露
3. 在 `defs/__init__.py` 注册导入
4. 在 `factor_config.factors` 增加配置项（唯一 `name`）
5. 先小区间跑 `factor_batch` 验证
6. 看 `factor_metrics.csv`、`factor_report.png`、`screened/factor_shortlist.csv`
7. 通过后再加入模型配置 `factors`

## 12. 常见错误与排查

- `RegistryError: 未找到注册项`
  - 没注册或没在 `defs/__init__.py` 导入，或入口没加载 `defs`
- `KeyError: panel missing column`
  - 因子请求了 panel 不存在字段，先打印 `ctx.panel.columns`
- 因子结果全 0 / 全 NaN
  - 窗口参数过大、过滤过严、字段口径不对
- IC 异常高
  - 高概率发生信息泄露（尤其误用收盘口径字段）

## 13. 新 Agent 交付标准

- 代码：
  - 因子实现文件
  - `defs/__init__.py` 导入
  - `factor_config` 新增项
- 结果：
  - 至少一轮 `factor_batch` 成功
  - 报告图片可读
  - 过筛表可读（含 pass/fail）
- 文档：
  - 在本文件补充“新因子说明 + 字段依赖 + 泄露风险检查”

---

## 14. 多 Agent 协作约定（ON 维护 Agent 与因子研究 Agent）

为减少来回沟通成本，因子研究 Agent 需要提供以下内容：

1. 因子规格卡（必须）
   - 因子名（最终 `name`，snake_case，唯一）
   - 因子公式（字段级表达，不要只给概念）
   - 依赖字段清单（标注来自 panel / raw 日频 / stock 快照）
   - 时间因果声明（每个字段是否在 `factor_time` 前可见）
   - 参数清单（默认值、可调范围）

2. 变更说明（每次迭代必须）
   - 本次变更内容（公式改动 / 参数改动 / 字段改动）
   - 兼容性影响（是否需要重跑 factor / model）
   - 风险点（可能泄露、可能稀疏、可能不稳定）

3. 结果回报模板（建议）
   - 因子版本号（如 `xxx_v2`）
   - 关键指标摘要（IC/IR/Sharpe）
   - 是否建议入模（是/否 + 理由）

职责边界（必须）：

- 因子研究 Agent：
  - 只负责产出公式与规格卡（字段、参数、因果声明）
  - 不负责跑回测、不负责改筛选阈值、不负责入模配置改动
- ON 维护 Agent：
  - 负责代码落地、batch 回测、筛选配置执行、报告产物与入模对接

注意：

- “可执行验证包”（最小回测区间、目标阈值、失败判据、对照基线）由项目 Owner 决定。
- 该部分由 Owner 直接配置到 `factor_config.screening`，或由 Owner 明确告知 ON 维护 Agent 后代为配置。
- 因子研究 Agent 不负责擅自修改筛选阈值。

---

参考模板：

- `cbond_on/factors/defs/ret_window.py`
- `cbond_on/factors/defs/depth_imbalance.py`
- `cbond_on/factors/defs/vwap_gap.py`

## Factor Formula Catalog
- See: docs/factor_expression_catalog.md (active signal formulas, params, required fields).

## 16. Factor Rebuild Switch Semantics
- `refresh = true`: full rebuild in selected date range. The factor day file is rebuilt from current `specs` only.
- `overwrite = true` and `refresh = false`: recompute selected factor columns and overwrite those columns only; keep all other existing columns unchanged.
- `overwrite = false` and `refresh = false`: incremental append mode. Only missing factor columns are computed; existing columns are kept untouched.

## 15. Per-Factor `windowsize` OHLC Rebuild Rule (2026-03-27)
- This is now the **active** intraday OHLC convention for ON factor development.
- No global switch is used. OHLC rebuild is controlled **per factor** via params only.

### 15.1 Trigger condition
- Rebuild is enabled only when a factor params contains:
  - `windowsize: <int>`
- Alias `window_size` is also accepted for compatibility.
- If no `windowsize` is provided, the factor keeps legacy field behavior.

### 15.2 Scope
- Rebuild only affects factors that use OHLC semantic fields:
  - `open`, `high`, `low`, `close`
- Factors that only use non-OHLC fields (e.g. `last`, `volume`, `amount`, depth fields) are not affected.

### 15.3 Rebuild semantics
- Rebuild is rolling by `(dt, code)` sequence, based on `last` as base price:
  - `open`: first value in the trailing `windowsize` points
  - `high`: max value in the trailing `windowsize` points
  - `low`: min value in the trailing `windowsize` points
  - `close`: last value in the trailing `windowsize` points
- When rebuild is enabled, open-like semantics use rebuilt `open` directly (no mid/open fallback path).

### 15.4 Config example
```json5
{
  name: "alpha026_volume_high_rank_corr_v1",
  factor: "alpha026_volume_high_rank_corr_v1",
  params: {
    ts_rank_window: 5,
    corr_window: 5,
    ts_max_window: 3,
    windowsize: 120
  }
}
```

### 15.5 Factors already switched to `windowsize`
- `alpha002_corr_volume_return_v1`
- `alpha003_corr_open_volume_v1`
- `alpha004_ts_rank_low_v1`
- `alpha005_vwap_gap_v1`
- `alpha006_corr_open_volume_neg_v1`
- `alpha008_open_return_momentum_v1`
- `alpha015_high_volume_corr_v1`
- `alpha016_cov_high_volume_v1`
- `alpha018_close_open_vol_v1`
- `alpha020_open_delay_range_v1`
- `alpha023_high_momentum_v1`
- `alpha025_return_volume_vwap_range_v1`
- `alpha026_volume_high_rank_corr_v1`
- `alpha028_adv_low_close_signal_v1`
- `alpha031_close_decay_momentum_v1`
- `alpha033_open_close_ratio_v1`
- `alpha035_volume_price_momentum_v1`
- `alpha037_open_close_correlation_v1`
- `alpha038_close_rank_ratio_v1`
- `alpha040_high_volatility_corr_v1`

### 15.6 Notes for new factor agents
- If your formula depends on intraday OHLC dynamics, set `windowsize` explicitly per factor.
- Different factors are expected to use different `windowsize`; there is no required shared default.
- If formula does not use OHLC semantics, do not add `windowsize` unnecessarily.

