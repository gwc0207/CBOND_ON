# DataHub Schema 修复请求（给 DataHub Agent）

更新时间：2026-03-31  
提出方：CBOND_ON（GPU panel consumer）

## 1. 问题背景

CBOND_ON 已切到 `cudf` 读取 clean snapshot 并拼接多交易日（lookback）构建 panel。  
当前在 `cudf.concat(frames)` 阶段报错：

`ValueError: All columns must be the same type`

这不是 ON 逻辑问题，而是 DataHub 发布的同一数据集跨天 schema 不一致。

## 2. 已定位到的 schema 漂移

检查目录：

- `clean_data/snapshot/cbond/2026-03/*.parquet`
- `clean_data/snapshot/stock/2026-03/*.parquet`
- `clean_data/snapshot/etf/2026-03/*.parquet`

三类资产结论一致：

1. `trade_time` 类型漂移：
- `20260302~20260325`: `timestamp[ns]`
- `20260326/27/30/31`: `timestamp[ms]`

2. `__index_level_0__` 类型漂移（且仅在 4 天出现）：
- `20260326/20260331`: `double`
- `20260327/20260330`: `int64`

3. 非标准列仅在 4 天出现（20260326/27/30/31）：
- `source`
- `symbol`
- `trade_date`
- `__index_level_0__`

## 3. 目标契约（必须统一）

对 `clean_data/snapshot/{asset}/YYYY-MM/YYYYMMDD.parquet`（asset in `cbond/stock/etf`）统一以下约束：

1. 列集合固定（同一资产跨天完全一致，不多不少）。
2. 列类型固定（同一资产跨天完全一致）。
3. `trade_time` 固定为 `timestamp[ns]`。
4. 不写入 pandas index（禁止产出 `__index_level_0__`）。
5. `source/symbol/trade_date` 这类发布元信息不要混入 snapshot 主表（如需保留请放 manifest 或单独 sidecar）。

## 4. 需回补修复的历史日期

请至少回补以下日期（3 类资产都要）：

- `2026-03-26`
- `2026-03-27`
- `2026-03-30`
- `2026-03-31`

说明：这 4 天是当前已确认的 schema 漂移窗口，会导致 GPU 消费链路失败。

## 5. 参考基准 schema（cbond）

建议以 `cbond/20260325.parquet` 为基准（35 列），例如：

- `code: large_string`
- `trade_time: timestamp[ns]`
- `pre_close,last,open,high,low,close,volume,amount,...: double`
- `num_trades: int64`
- `trading_phase_code: large_string`

不应包含：

- `__index_level_0__`
- `source`
- `symbol`
- `trade_date`

## 6. DataHub 侧建议修改点

1. 统一 snapshot writer：
- 显式 `index=False`
- 发布前做 schema cast 到固定 arrow schema

2. 发布门禁增加 schema 校验：
- 当日文件 schema 必须与基准 schema 完全一致，否则发布失败

3. 历史回补后重发 manifest（确保 consumer 可按 manifest 重拉/重算）。

## 7. 验收标准（DataHub 自检）

### 7.1 Schema 一致性检查

```python
from pathlib import Path
import pyarrow.parquet as pq

def check(asset, month):
    root = Path(f"/mnt/cbond_data_hub_ro/clean_data/snapshot/{asset}/{month}")
    files = sorted(root.glob("*.parquet"))
    base = pq.read_schema(files[0])
    base_map = {base.names[i]: str(base.types[i]) for i in range(len(base))}
    for p in files[1:]:
        s = pq.read_schema(p)
        m = {s.names[i]: str(s.types[i]) for i in range(len(s))}
        assert m == base_map, f"schema drift: {asset} {p.name}"
    print(asset, month, "OK", len(files))

for a in ["cbond", "stock", "etf"]:
    check(a, "2026-03")
```

### 7.2 ON 侧 GPU 消费验收

`build_panels` 在 `compute.dataframe_backend=cudf` 下可跑通，且无 `All columns must be the same type`。

