# CBOND_ON 因子开发说明（2026-03 版）

> 目标：让新 Agent 或新同学可以按统一规范开发因子，并安全接入回测与实盘。

## 1. 当前架构（先对齐口径）

- ON 已是 DataHub 消费方：`raw/clean` 由 DataHub 维护，ON 不再自建原始数据。
- ON 负责的衍生产物：`panel -> label -> factor -> score -> backtest/live`。
- 典型路径（以当前本地配置为准）：
  - `raw_data_root`: `D:/cbond_data_hub/raw_data`
  - `clean_data_root`: `D:/cbond_data_hub/clean_data`
  - `panel_data_root`: `D:/cbond_on/panel_data`
  - `factor_data_root`: `D:/cbond_on/factor_data`

---

## 2. 因子开发入口与配置

### 2.1 代码入口

- 因子实现目录：`cbond_on/factors/defs/`
- 注册入口：`cbond_on/factors/defs/__init__.py`
- 基类与上下文：`cbond_on/factors/base.py`

### 2.2 配置入口（已更新）

- 因子构建配置：`cbond_on/config/factor/factor_config.json5`
- 运行命令：`python cbond_on/run/factor_batch.py`

说明：
- 旧文档里提到的 `cbond_on/config/factor_batch_config.json5` 不是当前主配置入口。
- 新因子如果要入模，还需要把因子名加入模型配置中的 `factors`：
  - `cbond_on/config/models/lgbm/lgbm_factor_MSE_config.json5`
  - `cbond_on/config/models/lgbm_ranker/lgbm_factor_ranker_config.json5`
  - `cbond_on/config/models/linear/linear_factor_default_config.json5`

---

## 3. 可用字段（重点）

## 3.1 Panel 直接可用字段（因子 compute 可直接使用）

扫描基准：
- 样本文件：`D:/cbond_on/panel_data/panels/T1430/2026-03/20260324.parquet`
- 扫描日期：2026-03-24

索引字段（MultiIndex）：
- `dt`, `code`, `seq`

列字段（38）：

```text
__index_level_0__
amount
ask_price1
ask_price2
ask_price3
ask_price4
ask_price5
ask_volume1
ask_volume2
ask_volume3
ask_volume4
ask_volume5
bid_price1
bid_price2
bid_price3
bid_price4
bid_price5
bid_volume1
bid_volume2
bid_volume3
bid_volume4
bid_volume5
close
high
high_limited
iopv
last
low
low_limited
num_trades
open
pre_close
source
symbol
trade_date
trade_time
trading_phase_code
volume
```

说明：
- `__index_level_0__` 是 parquet 索引落盘副产物，通常忽略。
- 因子开发时请以 `dt/code/seq` 为主索引语义，不要把 `symbol` 误当“正股代码”。

## 3.2 正股相关字段（新增）

当前可用正股字段主要不在 panel 默认列中，而在 DataHub 的日频原始表：

数据集 A：`raw_data/market_cbond__daily_base`
- 显式正股相关：`stock_code`, `stock_close_price`, `stock_volatility`
- 额外常用（正股/转债联动）：`stk_prev_close_price`, `stk_act_prev_close_price`, `stk_close_price`, `stk_volume`, `stk_amount`, `stk_deal`, `conv_value`, `bond_prem_ratio`, `puredebt_prem_ratio`

数据集 B：`raw_data/market_cbond__daily_deriv`
- 显式正股相关：`stock_code`, `stock_close_price`, `stock_volatility`
- 额外常用：`conv_value`, `bond_prem_ratio`, `puredebt_prem_ratio`

数据集 C：`clean_data/snapshot/stock/YYYY-MM/YYYYMMDD.parquet`
- 为正股快照（字段结构与 cbond snapshot 类似），可做盘中联动因子。

关键事实：
- ON 当前的 panel 构建默认只消费 `clean_data/snapshot/cbond/...`，不会自动并入上述正股字段。
- 需要正股字段时，必须在因子流程中显式 merge（推荐）或扩展 panel 构建逻辑。

## 3.3 正股字段可用性矩阵

| 数据源 | 粒度 | 是否默认进 panel | 典型用途 |
|---|---|---|---|
| `clean_data/snapshot/cbond` | 盘中快照 | 是 | 转债盘口/成交/微观结构因子 |
| `raw_data/market_cbond__daily_base` | 日频 | 否 | 正股映射、溢价率、正股波动率 |
| `raw_data/market_cbond__daily_deriv` | 日频 | 否 | 正股映射、衍生估值字段 |
| `clean_data/snapshot/stock` | 盘中快照 | 否 | 转债-正股盘中联动因子 |

---

## 4. 时间因果规则（含正股字段）

- `factor_time`：因子可见时点（默认 14:30）
- `label_time`：标签起点（默认 14:45）

必须遵守：
- 因子只能使用 `factor_time` 及之前可观测信息。
- 同日 `*_close_price`（例如 `stock_close_price`, `stk_close_price`）是收盘口径，若在 14:30 因子里直接使用，属于未来函数。

安全做法：
1. 日频正股字段一律先做“上一交易日”对齐，再入因子。
2. 若要做同日盘中联动，请使用 `snapshot/stock` 且截断到 `factor_time` 之前。
3. 严禁使用 T+1 或标签窗后信息。

---

## 5. 开发规范

- 一因子一文件：`cbond_on/factors/defs/<factor_name>.py`
- 类必须继承 `Factor`
- 必须 `@FactorRegistry.register("<factor_name>")`
- `compute()` 返回 `pd.Series`，索引是 `(dt, code)`
- 输出名统一 `self.output_name(self.name)`
- 必须显式处理空值/异常（`fillna(0.0)` 或清晰约定）

---

## 6. 接入流程（从开发到实盘）

1. 新建因子文件到 `cbond_on/factors/defs/`
2. 在类上注册 `@FactorRegistry.register("xxx")`
3. 在 `cbond_on/factors/defs/__init__.py` 导入该类
4. 在 `cbond_on/config/factor/factor_config.json5` 增加配置：

```json5
{ name: "ret_30m", factor: "ret_window", params: { window_minutes: 30, price_col: "last" } }
```

5. 运行因子构建：`python cbond_on/run/factor_batch.py`
6. 需要入模时，把因子名加入目标模型配置 `factors` 列表后再跑 `model_score/pipeline_all`

---

## 7. 常见错误与排查

- `RegistryError`：未注册或重复注册
- `KeyError: panel missing column`：panel 没有该列（先打印 `ctx.panel.columns`）
- 因子全 0 或无分箱：样本不足/列离散化过重/窗口参数不合理
- 用正股字段后 IC 异常飙高：优先排查是否误用了同日收盘字段导致泄露

---

## 8. 上线前 Checklist

- [ ] 因子文件独立、命名规范
- [ ] 已注册并在 `defs/__init__.py` 导入
- [ ] `config/factor/factor_config.json5` 已配置
- [ ] 无未来函数（尤其正股 `close` 类字段）
- [ ] 空值/异常处理明确
- [ ] 单因子回测统计稳定
- [ ] 入模验证通过后再接实盘

---

参考模板：
- `cbond_on/factors/defs/ret_window.py`
- `cbond_on/factors/defs/depth_imbalance.py`
