# WC/SP Consumer-Only 迁移作业手册（以 ON 为模板）

更新时间：`2026-03-18`

适用对象：后续维护 `CBOND_WC`、`CBOND_SP` 的 agent。

目标：将 WC/SP 迁移到与 ON 一致的 **consumer-only** 模式。

---

## 1. 迁移目标（必须满足）

1. DataHub 是唯一 `raw/clean` 生产方。
2. WC/SP 只消费共享目录，不再本地拉取 raw、不再本地构建 clean。
3. 读取数据前必须做发布门禁检查（`publish status`）。
4. WC/SP 不调用 DataHub 控制接口（`prepare-live`、`schedule`、`dashboard start/stop` 等）。
5. 保留项目自有产物（panel/label/factor/signal/model/trade_list）。

说明：允许读取类调用（`context resolve`、`publish status`）。

---

## 2. ON 模板参考点

按 ON 当前实现对齐：

1. 共享路径配置：  
`[paths_config.json5](/c:/Users/BaiYang/CBOND_ON/cbond_on/cbond_on/config/data/paths_config.json5)`
2. 门禁配置：  
`[live_config.json5](/c:/Users/BaiYang/CBOND_ON/cbond_on/cbond_on/config/live/live_config.json5)`
3. live 门禁 + 消费主流程：  
`[live_service.py](/c:/Users/BaiYang/CBOND_ON/cbond_on/cbond_on/services/live/live_service.py)`
4. ON 已删除的生产残留（对 WC/SP 同样应删除）：  
`raw_service/clean_service/raw_sync_ops/run/sync_data/run/build_cleaned_data`

---

## 3. 通用改造步骤（WC/SP 都执行）

## 3.1 配置层

1. `paths_config` 改为共享目录：
   - `raw_data_root = D:/cbond_data_hub/raw_data`
   - `clean_data_root / cleaned_data_root = D:/cbond_data_hub/clean_data`
2. `live_config` 增加 `data_hub` 块（参考 ON）：
   - `manifest_root = D:/cbond_data_hub/manifests`
   - `require_datasets = ["raw","clean"]`
   - `allow_partial_manifest = false`
   - `require_done_marker = true`
   - `ready_gate_enabled = true`
3. 若项目有 `snapshot_source`，保留 `redis` 语义，但不要再把 redis 结果写入项目 raw 目录。

## 3.2 代码层（live 主链）

1. 在 `run_once` 开头增加 DataHub 只读门禁：
   - 计算 `run_day`（redis 模式通常为当天）
   - `publish status` 检查 `gate_day` 是否 ready
   - 不 ready 直接失败并重试
2. 删除/移除 `run_raw`、`run_clean`、`sync_nfs`、`sync_db`、`sync_redis` 在 live 主链中的调用。
3. 保留下游项目内逻辑：
   - `panel/label/factor/signal/model/strategy` 按各项目自己的链路运行。

## 3.3 清理层

1. 删除 raw/clean 生产入口脚本（如 `run/sync_*`、`run/build_clean*`）。
2. 删除 `services/data/__init__.py` 里对 `run_raw/run_clean` 的导出。
3. 更新 README：明确“raw/clean 由 DataHub 统一生产”。

---

## 4. WC 专项改造清单

当前关键差异（待改）：

1. `liveLaunch/runner.py` 仍在做本地数据生产（NFS 同步、clean 构建、panel 增量前置依赖本地写 raw/clean）。
2. `cbond/config/paths_config.json5` 仍指向 `D:/cbond_data/...`，非 DataHub 共享目录。
3. `live_config.json5` 仍有 `data_sync_mode/nfs_root` 等生产参数。

建议改造顺序：

1. 先改路径到 DataHub 共享目录。
2. 在 `liveLaunch/runner.py` 增加门禁检查函数（对接 `publish status`），并在 `run_once` 入口先检查。
3. 去掉 `sync_raw_data_from_nfs`、`build_cleaned_snapshot`、`_append_raw_snapshot` 等生产写入链路。
4. 保留 panel/label/factor/model/交易执行逻辑。
5. 将 `data_sync_*` 参数标记为废弃并逐步清理。

建议重点文件：

1. `C:/Users/BaiYang/CBOND_WC/cbond_wc/liveLaunch/runner.py`
2. `C:/Users/BaiYang/CBOND_WC/cbond_wc/cbond/config/paths_config.json5`
3. `C:/Users/BaiYang/CBOND_WC/cbond_wc/cbond/config/live_config.json5`
4. `C:/Users/BaiYang/CBOND_WC/cbond_wc/cbond/data/sync.py`
5. `C:/Users/BaiYang/CBOND_WC/cbond_wc/cbond/data/clean.py`

---

## 5. SP 专项改造清单

当前关键差异（待改）：

1. `cbond_sp/services/live/live_service.py` 直接调用 `run_raw` 与 `run_clean`。
2. `cbond_sp/config/data/paths_config.json5` 仍是项目本地 runtime 目录。
3. `live_config` 还没有 `data_hub` 门禁块。

建议改造顺序：

1. 改 `paths_config` 到 DataHub 共享 raw/clean。
2. 在 `live_service.run_once` 引入 ON 同款门禁流程：
   - `context resolve`（可选）
   - `publish status`（必须）
3. 删除 `run_raw/run_clean` 调用和相关模块导出。
4. 保留 `signal/model/strategy/output` 链路。

建议重点文件：

1. `C:/Users/BaiYang/CBOND_SP/cbond_sp/services/live/live_service.py`
2. `C:/Users/BaiYang/CBOND_SP/cbond_sp/config/data/paths_config.json5`
3. `C:/Users/BaiYang/CBOND_SP/cbond_sp/config/live/live_config.json5`
4. `C:/Users/BaiYang/CBOND_SP/cbond_sp/services/data/raw_service.py`
5. `C:/Users/BaiYang/CBOND_SP/cbond_sp/services/data/clean_service.py`

---

## 6. 验收标准（必须全部通过）

## 6.1 功能验收

1. 运行 live 前，日志出现 `publish status` 检查记录。
2. `ready=false` 时，live 失败且不产出交易文件。
3. `ready=true` 时，live 成功并产出当日/目标日交易文件。

## 6.2 解耦验收

1. 仓库内不存在 live 主链路上的 `run_raw/run_clean` 调用。
2. 运行 live 不会修改 `D:/cbond_data_hub/raw_data` 与 `D:/cbond_data_hub/clean_data`（由 DataHub 独占写入）。
3. 项目不调用 DataHub 控制接口（`prepare-live`、`schedule`、`dashboard`）。

## 6.3 命令验收（示例）

```powershell
# 1) 门禁检查（读取）
python -m cbond_data_hub publish status `
  --manifest-root D:/cbond_data_hub/manifests `
  --trade-day 2026-03-18 `
  --require-datasets raw,clean

# 2) 项目单次 live（WC/SP 各自入口）
# 期望：日志中有 publish status；无 raw/clean 本地构建日志
```

---

## 7. 迁移后边界（给所有 agent）

1. 项目内可以做：
   - 读取 shared raw/clean
   - 读取 publish status
   - 本地衍生计算（panel/label/factor/signal/model）
2. 项目内禁止做：
   - 写 shared raw/clean
   - 启停 DataHub 调度
   - 触发 DataHub prepare/生产流程

若要做运维控制，请在 DataHub 仓库内实现，不要下沉到消费端项目。

