# CBOND_ON 因子开发说明（简明版）

> 目标：让完全不了解项目的人，能按统一规范开发因子，并安全接入回测/实盘。

## 1. 因子开发流程

### 1.1 总体链路
- 数据链路：`raw -> clean -> panel -> factor -> score -> backtest/live`
- 因子开发只处理 `panel -> factor`，但必须理解时间因果与标签对齐规则。

### 1.2 开发规范
- 一因子一文件：`cbond_on/factors/defs/<factor_name>.py`
- 因子类必须继承 `Factor`
- 必须用 `@FactorRegistry.register("<factor_name>")` 注册
- 返回值必须是 `pd.Series`，索引为 `("dt", "code")`
- 输出名统一用 `self.output_name(self.name)`
- 必须处理空值/异常：不可计算时返回 `0.0`（或统一约定值）
- 禁止未来函数：只能使用当前 `dt` 可见信息

### 1.3 可用数据（当前项目完整字段）
- 以下字段来自你当前 `D:/cbond_on/panel_data` 全部 parquet 的并集扫描（共 278 个文件，扫描日期：2026-03-09）。
- 因子开发可直接使用的核心字段如下（38 个）：

```text
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
code
dt
high
high_limited
iopv
last
low
low_limited
num_trades
open
pre_close
seq
source
trade_time
trading_phase_code
volume
```

- 说明：
  - 索引语义是 `("dt", "code", "seq")`，部分 parquet 里会看到 `__index_level_0__`，这是 parquet 索引落盘副产物，开发因子时忽略。
  - 若你后续改了 panel 构建配置（如 `snapshot_columns`），字段集合会变化，需重新扫描并更新文档。

### 1.4 时间与因果（必须遵守）
- `factor_time`：因子截面时点（如 14:30）
- `label_time`：标签起点（如 14:45）
- 因子必须由 `factor_time` 前可观测数据构成，不可引用 `label_time` 后数据。

### 1.5 注册规范
1. 新建因子文件到 `cbond_on/factors/defs/`
2. 类上加 `@FactorRegistry.register("因子名")`
3. 在 `cbond_on/factors/defs/__init__.py` 导入该类
4. 在 `cbond_on/config/factor_batch_config.json5` 增加一行配置

### 1.6 配置规范（factor_batch）
- `name`：输出列名（建议含参数后缀）
- `factor`：注册名，必须与 `register` 名一致
- `params`：可调参数字典

示例：

```json5
{ name: "ret_30m", factor: "ret_window", params: { window_minutes: 30, price_col: "last" } }
```

---

## 2. 因子开发示例

### 2.1 示例 A：窗口收益率因子

```python
from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar, slice_window, first_last_price

@FactorRegistry.register("ret_window")
class ReturnWindowFactor(Factor):
    name = "ret_window"

    def compute(self, ctx: FactorComputeContext):
        panel = ensure_trade_time(ctx.panel)
        win = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))

        def _calc(df):
            df = df.sort_values("trade_time")
            df = slice_window(df, win)
            p0, p1 = first_last_price(df, price_col)
            if p0 is None or p0 == 0:
                return 0.0
            return float((p1 - p0) / p0)

        out = group_apply_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out
```

### 2.2 示例 B：盘口不平衡因子

```python
from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar

@FactorRegistry.register("depth_imbalance")
class DepthImbalanceFactor(Factor):
    name = "depth_imbalance"

    def compute(self, ctx: FactorComputeContext):
        panel = ensure_trade_time(ctx.panel)
        levels = int(ctx.params.get("levels", 3))
        bid_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        ask_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]

        def _calc(df):
            x = df.sort_values("trade_time")
            bid = x[bid_cols].iloc[-1].sum()
            ask = x[ask_cols].iloc[-1].sum()
            d = bid + ask
            return 0.0 if d <= 0 else float((bid - ask) / d)

        out = group_apply_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out
```

---

## 3. 回测与实盘使用规范

### 3.1 因子到回测
- 因子写入后，由 `factor_batch` 统一产出单因子评估与报告。
- 核心检查指标：`IC`, `RankIC`, `ICIR/RankICIR`, 分箱可分性、分箱收益曲线。
- 若分箱不足（大量重复值/样本太少），应先查：
  - 因子是否离散化过重
  - 当日可交易样本是否不足
  - `bin_count/min_count` 是否过严

### 3.2 因子到实盘
- 实盘模型读取因子后生成 `score`，策略只消费 `score`。
- 因子上线前至少满足：
  - 无未来数据泄露
  - 空值行为可解释
  - 在回测区间统计稳定（非偶然单日驱动）

### 3.3 增量/覆盖约定
- 日常：优先增量更新
- 历史修复：按日期区间覆盖重算
- 回测与实盘配置分离，避免互相污染

### 3.4 变更流程（建议）
1. 新因子开发并本地自检
2. 加入 `factor_batch_config` 小范围试跑
3. 看单因子报告（IC/分箱/NAV）
4. 入模验证（线性/LGBM）
5. 回测通过后再入实盘

---

## 4. 常见错误与排查

- `RegistryError`：未注册或重复注册
- `KeyError`：panel 缺列（先打印 `panel.columns`）
- `ValueError`：空样本/参数非法
- 分箱异常：检查重复值占比、样本量、`bin_count/min_count`

---

## 5. 上线前 Checklist

- [ ] 因子文件独立、命名规范
- [ ] 已注册并在 `__init__.py` 导入
- [ ] `factor_batch_config` 已配置且参数清晰
- [ ] 无未来数据泄露
- [ ] 空值/异常处理明确
- [ ] 单因子报告通过基本阈值
- [ ] 回测通过后再接入实盘

---

如需快速复制模板，直接参考：
- `cbond_on/factors/defs/ret_window.py`
- `cbond_on/factors/defs/depth_imbalance.py`
