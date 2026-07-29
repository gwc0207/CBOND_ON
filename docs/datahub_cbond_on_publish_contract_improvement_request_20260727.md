# CBOND_DATA_HUB 面向 CBOND_ON 的发布就绪校验增强需求

日期：2026-07-27  
提出方：CBOND_ON

## 1. 背景

CBOND_ON live 当前依赖 DataHub 的 `clean` manifest 与 `.done` 文件判断数据是否可用。

当前 DataHub 已具备以下能力：

- manifest / `.done` 的 `run_id` 一致性校验；
- clean snapshot 的固定 schema；
- parquet 原子写入。

但“发布成功”仍不能充分证明 CBOND_ON 所需内容可用：

- `daily_twap` 的必需列未包含正式卖出列 `twap_0930_0939`；
- clean snapshot 缺字段时可能被补成空列，schema 通过但内容不可用；
- 发布成功不要求行数、code 数或关键字段有效值；
- manifest 未向下游输出足够的字段质量、行数、时间覆盖和 schema 指纹信息。

目标是为 CBOND_ON 提供可验证、可追溯的发布合同，使下游能在启动 live 前明确确认数据是否满足 T1430 因子、14:42 标签和 `09:30-09:39` 卖出收益口径。

## 2. 范围与边界

本需求只增强 DataHub 发布合同和数据校验，不修改：

- CBOND_ON 模型、策略、数据库或调度逻辑；
- 历史原始数据和历史运行产物；
- 非 CBOND_ON 下游的业务口径。

建议涉及的 DataHub 模块：

```text
publish_contract.py
raw_history.py
clean_build.py
intraday.py
data_scheduler.py
prepare_live.py
```

## 3. 当前已确认能力

以下能力已存在，应保留：

1. `publish_contract.py` 已要求所需 manifest 成功、manifest `run_id` 一致、`.done` 存在且 `.done.run_id` 与 manifest 一致。
2. `clean_build.py` 已对 clean snapshot 执行固定 schema 规整，并使用原子 parquet 写入。
3. `raw_history.py` 已对 `market_cbond.daily_twap` 校验部分必需字段。

本需求是补齐“内容有效性”和“CBOND_ON 所需字段”的发布校验，不应回退已有 run_id 与原子写能力。

## 4. 功能需求

### 4.1 `daily_twap` 必需字段

同步 `market_cbond.daily_twap` 时，必须校验以下字段：

```text
twap_0930_0935
twap_0930_1000
twap_0930_0939
twap_1442_1457
```

其中：

```text
twap_1442_1457：CBOND_ON 买入价格
twap_0930_0939：CBOND_ON 次日正式卖出价格
```

任一字段缺失时：

```text
daily_twap 校验失败
-> clean manifest.status = failed
-> 不生成有效 publish .done
-> reason 明确包含 missing_required_columns
```

### 4.2 clean snapshot 内容校验

对 CBOND_ON 使用的 `clean_data/snapshot/cbond` 与 `clean_data/snapshot/stock`，发布前至少校验：

```text
code 非空
trade_time 非空
文件行数 > 0
code 数 > 0
```

并校验下列关键字段存在且不是全空：

```text
pre_close, last, volume, amount
ask_price1..5, bid_price1..5
ask_volume1..5, bid_volume1..5
```

仅“补齐空列后 schema 一致”不能视为通过；发布校验必须检查关键字段至少存在有效值。

### 4.3 T1430 freshness 与覆盖校验

manifest 中需要按 asset 记录：

```text
row_count
code_count
min_trade_time
max_trade_time
non_null_count_by_required_field
schema_fingerprint
file_path
file_mtime
```

新增可配置 profile：

```text
required_profile = cbond_on_live_t1430
```

该 profile 默认检查：

```text
cbond / stock snapshot 的整体 max_trade_time 覆盖到 14:29:00；
不要求每个 code 在 14:29 后均有成交。
```

### 4.4 行数与覆盖异常校验

除 `row_count > 0` 外，支持与最近 N 个交易日中位数比较：

```text
当前 code_count / 最近 N 日 code_count 中位数
当前 row_count / 最近 N 日 row_count 中位数
当前关键字段非空比例
```

阈值必须配置化：

```text
min_code_coverage_ratio
min_row_coverage_ratio
min_required_field_non_null_ratio
```

## 5. 发布合同扩展

`clean/{trade_day}.json` manifest 建议新增以下结构：

```json
{
  "status": "success",
  "run_id": "...",
  "schema_version": "cbond_on_t1430_v1",
  "required_profile": "cbond_on_live_t1430",
  "validation": {
    "passed": true,
    "errors": [],
    "warnings": []
  },
  "assets": {
    "cbond": {
      "row_count": 0,
      "code_count": 0,
      "min_trade_time": "...",
      "max_trade_time": "...",
      "schema_fingerprint": "...",
      "required_field_non_null_ratio": {}
    },
    "stock": {}
  },
  "daily_twap": {
    "path": "...",
    "row_count": 0,
    "required_columns": [
      "twap_0930_0935",
      "twap_0930_1000",
      "twap_0930_0939",
      "twap_1442_1457"
    ],
    "missing_columns": []
  }
}
```

`.done` 只能在以下条件全部成立时写出：

```text
manifest status = success
全部 required assets 校验通过
daily_twap 校验通过
manifest run_id 一致
.done run_id 与 manifest run_id 一致
```

## 6. 实现建议

1. 在 `raw_history.py` 的 `REQUIRED_TABLE_COLUMNS["market_cbond.daily_twap"]` 中加入 `twap_0930_0939`。
2. 在 `clean_build.py` / `intraday.py` 的 schema 规整后增加内容质量校验；缺字段补为 `NA` 后不得自动判定发布成功。
3. 在 `publish_contract.py` 增加 `cbond_on_live_t1430` profile 与结构化 `validation` 结果。
4. 在 `prepare_live.py` 和 `data_scheduler.py` 写 manifest 前调用该 profile 校验。
5. 将校验指标、失败原因和告警写入 manifest，而不是仅写日志。
6. 保持当前 run_id 一致性校验与原子写能力。

## 7. 验收标准

| 场景 | 预期结果 |
| --- | --- |
| `daily_twap` 缺 `twap_0930_0939` | 发布失败，无有效 `.done` |
| clean snapshot 缺盘口字段 | 发布失败，无有效 `.done` |
| 字段存在但全为 `NA` | 发布失败，无有效 `.done` |
| clean 文件行数为 0 | 发布失败，无有效 `.done` |
| manifest / done run_id 不一致 | 发布失败，无有效 `.done` |
| 正常 cbond、stock、daily_twap 数据 | manifest 带完整质量指标，`.done` 正常生成 |
| code / row 覆盖显著低于历史 | 第一阶段告警；启用强校验后阻断发布 |

## 8. 分阶段上线建议

```text
第一阶段：新增指标与 manifest 字段，report-only，不阻断 live
第二阶段：连续 5 个交易日观察阈值与误报
第三阶段：对缺关键列、空文件、run_id 不一致启用 hard fail
第四阶段：CBOND_ON 消费 DataHub 的增强发布合同，而非只检查 success + done 存在
```

## 9. 下游衔接说明

本需求使 DataHub 输出更完整、可信的发布合同。

CBOND_ON 仍需单独改造 `publish_gate.py`，消费 DataHub 的增强校验结果；否则即使 DataHub 写出更严格的 `validation` 字段，CBOND_ON 仍可能沿用旧的 `manifest success + done exists` 判断。
