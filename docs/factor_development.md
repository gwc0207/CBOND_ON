# 因子开发说明（CBOND_ON）

> 本文说明如何在 CBOND_ON 中开发新因子，包括开发规范、可用字段、注册方式、配置方法与示例。

## 1. 目录结构

- 因子定义目录：`cbond_on/factors/defs/`
- 每个因子**必须**独立一个 `.py` 文件
- 公共工具函数请放在 `_intraday_utils.py` 这类私有模块中

## 2. 开发规范

- 一个文件 = 一个因子类
- 因子类必须继承 `cbond_on.factors.base.Factor`
- 必须使用 `FactorRegistry.register("因子名")` 注册
- 输入 panel 必须是 MultiIndex：`("dt", "code", "seq")`
- 返回 `pd.Series`，索引为 `("dt", "code")`
- 计算失败或不可用时，**返回 0.0**
- 输出列名必须通过 `self.output_name(self.name)` 设置

## 3. 必要接口

### Base Class

```python
from cbond_on.factors.base import Factor, FactorComputeContext

class MyFactor(Factor):
    name = "my_factor"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        # ctx.panel: panel dataframe
        # ctx.params: 配置参数
        raise NotImplementedError
```

### Registry

```python
from cbond_on.core.registry import FactorRegistry

@FactorRegistry.register("my_factor")
class MyFactor(Factor):
    ...
```

## 4. panel 可用字段

具体字段取决于 panel 生成配置，但常见 snapshot 面板包含：

- `trade_time`
- `last`, `open`, `high`, `low`, `close`, `pre_close`
- `volume`, `amount`, `num_trades`
- `ask_price1`..`ask_price5`
- `bid_price1`..`bid_price5`
- `ask_volume1`..`ask_volume5`
- `bid_volume1`..`bid_volume5`
- `trading_phase_code`

**检查实际字段**：

```python
import pandas as pd
from pathlib import Path

sample = Path("<panel_root>").rglob("*.parquet").__next__()
df = pd.read_parquet(sample)
print(df.columns.tolist())
```

## 5. 因子示例

### 示例 1：最近 N 分钟收益率

```python
from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, group_apply_scalar, slice_window, first_last_price

@FactorRegistry.register("ret_window")
class ReturnWindowFactor(Factor):
    name = "ret_window"

    def compute(self, ctx: FactorComputeContext):
        panel = ensure_trade_time(ctx.panel)
        window_minutes = int(ctx.params.get("window_minutes", 30))
        price_col = str(ctx.params.get("price_col", "last"))

        def _calc(df):
            df = df.sort_values("trade_time")
            df = slice_window(df, window_minutes)
            first, last = first_last_price(df, price_col)
            if first is None or first == 0:
                return 0.0
            return (last - first) / first

        out = group_apply_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
```

### 示例 2：盘口深度不平衡

```python
@FactorRegistry.register("depth_imbalance")
class DepthImbalanceFactor(Factor):
    name = "depth_imbalance"

    def compute(self, ctx: FactorComputeContext):
        panel = ensure_trade_time(ctx.panel)
        levels = int(ctx.params.get("levels", 3))
        bid_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        ask_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]

        def _calc(df):
            df = df.sort_values("trade_time")
            bid = df[bid_cols].iloc[-1].sum()
            ask = df[ask_cols].iloc[-1].sum()
            denom = bid + ask
            if denom <= 0:
                return 0.0
            return float((bid - ask) / denom)

        out = group_apply_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
```

## 6. 注册检查清单

1. 新建文件：`cbond_on/factors/defs/<name>.py`
2. 实现因子类，并用 `@FactorRegistry.register("name")`
3. 在 `cbond_on/factors/defs/__init__.py` 里导入
4. 在 `cbond_on/config/factor_batch_config.json5` 配置因子

## 7. 配置示例（JSON5）

```json5
{ name: "ret_30m", factor: "ret_window", params: { window_minutes: 30, price_col: "last" } }
```

## 8. 命名规则

- 因子注册名使用 lower_snake_case
- 配置里的 `name` 是输出列名
- 配置里的 `factor` 必须等于注册名

## 9. 排查建议

- 确保 `trade_time` 存在且是 `datetime64`
- 确保索引是 `("dt", "code", "seq")`
- 缺列时直接报 `KeyError`
- 无法计算时统一返回 `0.0`

## 10. 常见错误

- `RegistryError`：因子未注册或重复注册
- `KeyError`：panel 缺列
- `ValueError`：panel 为空或参数非法

---

如需模板，可以复制 `cbond_on/factors/defs` 下的现有因子文件改写。
