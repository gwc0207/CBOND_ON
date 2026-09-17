# 因子开发兼容页

本文件不再定义可执行的因子开发或因子结果读写路径。

唯一有效的规范是 [因子工程治理规则](因子工程治理规则.md)，并以
`harness/skills/cbond-factor-governance/SKILL.md` 为执行入口。

当前约束：

- 所有因子实例先进入 `factor_engine/catalog/factor_catalog.json`；通用计算能力单独进入 Operator Catalog。
- 正常结果只能使用 `D:/cbond_on/factor_store/live`、`experiment`、`factor_library/<family>` 三张正式表。
- 正常读者只能声明 `factor_table` 并验证表 manifest、日 manifest 与 `.done`；不得直接指定 `factor_data_root`。
- 正常写者只有 admitted live runtime、显式 experiment publisher、23:59 factor supplement。
- 历史 `FactorStore`、旧 `factor_data` 与临时 parquet 仅可用于显式 migration/audit/no-DB staging，不能作为模型、回测、Dashboard、scheduler 或实盘输入。

新增、修改、回测、筛选、准入和退役的完整步骤见治理规则与其 lifecycle reference。
