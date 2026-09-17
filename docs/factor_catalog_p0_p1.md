# 全量因子目录 P0/P1

状态：P0/P1 因子身份层已完成；后续已接入 Catalog-driven live50 admission 与独立
research supplement。本文保留 P0/P1 的资产边界和生成验证说明。

## 本阶段边界

本阶段仅新增：

```text
factor_engine/
cbond_on/domain/factor_catalog/
harness/tools/build_factor_catalog.py
```

P0/P1 生成器本身不会直接启动 scheduler、写数据库或修改 Rust binary。Catalog 产物已被
后续 live admission 只读引用，live50 仍保留原有 Rust capability、profile hash、Feature
order 与 FactorStore permit 校验。

## 口径

- 冻结 2026-08-04 research catalog 中的 773 个实例全部注册为因子；
- 与其不重名的 legacy live27 也全部注册为因子；canonical full catalog 共 800 个实例；
- 原共享计算入口登记为通用算子，而不是因子；
- 每个因子有独立 `definition.py` 与 `contract.json`；
- 筛选、模型引用、Rust 能力、live 准入都不是注册前提；
- 实盘准入仍需后续独立的 immutable release manifest，且只能引用
  `factor_id + factor_version + contract_hash`。

## 现阶段契约状态

生成器可机械冻结：因子 ID、来源集、主因子族、来源位置、共享算子、固定参数、
可声明的日频 source/columns/lookback、stock/map 依赖、实现文件 hash 与
contract hash。

旧算子未声明到单因子粒度的 panel 字段与 PIT 证据时，contract 会明确标记
为 `pending`，而不会把它误写成已经完成的个体 PIT 认证。

每个 `definition.py` 都是该实例的可调用薄实现入口：它提供
`definition_payload()` 和懒加载的 `build_factor_spec()`。导入文件本身不 import
legacy `defs`，也不注册算子；调用后者只构造精确 `FactorSpec`，实际算子准入仍由后续
research/live admission 层控制。

## 同名历史冲突

`drrc_return_amount_rank_spearman60` 同时存在旧 v1 与 v2 实现。冻结合并
source ledger 表明本次 773 catalog 的该列来自 source position 3，因此
`catalog/manual_overrides.json` 显式绑定 v2。生成器遇到任何未声明的类似
歧义都会 fail-closed。

## 静态算子与 live50 元数据

`operator_catalog.json` 记录当前全部 266 个静态 `FactorRegistry` 算子；800 个
因子仅引用其中一部分。空的 `factor_ids` 表示“当前未被 canonical 800 引用”，不是
未注册。

`releases/live/live50_rust50_20260806.json` 固定了既有 live50 的 50 个有序
`factor_id + factor_version + contract_hash + operator_id + rust_contract_id`。它是当前
实盘准入的身份绑定；它不复制公式，也不替换原有的 Rust capability、profile hash、
Feature order 或 FactorStore permit。

这 50 个实例在全量 Catalog 及其独立 contract 中均显式标记为：

```text
live_admission_status = live_released
rust_status = live50_rust_capability_required
```

其余 750 个因子仍是已注册因子，只是没有因此被推定为可进入实盘。

## 验证

```powershell
py -B harness/tools/build_factor_catalog.py --check
py -m pytest -q tests/test_factor_catalog.py
```

这两项仅校验源码目录与 metadata；不运行 factor batch，不写入数据，也不改变
实盘。

## 对后续 permit / admission 的只读 API

```python
from cbond_on.domain.factor_catalog import (
    catalog_path,
    load_factor_catalog,
    resolve_factor_instance,
    resolve_operator_modules,
    load_live_release,
)
```

这些 API 只读取 `factor_engine` 下的 JSON，返回只读 metadata mapping，不会 import
`cbond_on.domain.factors.operators`。后续 permit 应传入并固定
`factor_id + factor_version + contract_hash`；能否实际 import/执行算子，仍应由对应研究
或 live admission 层独立决定。
