# AI 因子工厂有数据机器使用手册

本文档给接手 agent 使用。目标是在有数据机器上完成以下工作：

1. 验证 Dify / Qwen 候选因子生成通路。
2. 生成 research-only 候选因子包。
3. 做本地静态审查。
4. 人工/agent 审阅后把候选因子接入 research 因子链路。
5. 跑单日、多日 batch、bad factor report、stable_bin_alpha。
6. 再进入模型筛选和回测对比。

重要边界：Dify 只生成候选方案和 Python 草稿，不是可信执行器。可信部分必须在本地完成：字段白名单、时间可见性、静态审查、batch 检验、样本外筛选、人工准入。

## 1. 迁移后先检查

进入项目根目录：

```powershell
cd C:\Users\BaiYang\CBOND_ON\cbond_on
```

检查 Python：

```powershell
python --version
```

如果机器上没有 Python，安装 Python 3.11 或 3.12。然后安装至少这些依赖：

```powershell
python -m pip install json5 pandas
```

如果后续要跑完整数据链路和模型链路，按项目原环境补齐依赖，例如 numpy、pyarrow、scikit-learn、lightgbm 等。以有数据机器现有项目环境为准。

检查代码能否导入/编译：

```powershell
python -m py_compile cbond_on\app\usecases\ai_factor_factory.py cbond_on\cli\ai_factor_factory.py cbond_on\infra\ai\dify.py cbond_on\run\ai_factor_factory.py
```

## 2. Dify 密钥配置

不要把真实 key 写进 repo。

在有数据机器当前用户目录创建：

```text
C:\Users\<USER>\.cbond_on\dify.json
```

内容示例：

```json
{
  "endpoint": "http://byai.boyuandigital.com/v1/workflows/run",
  "api_key": "app-REPLACE_WITH_REAL_DIFY_WORKFLOW_KEY",
  "response_mode": "streaming",
  "user": "cbond_on_ai_factor_factory",
  "timeout_seconds": 180
}
```

说明：

- `endpoint` 必须是 workflow run endpoint，不是页面地址。
- `api_key` 是 Dify Workflow App API key。
- `response_mode` 推荐 `streaming`，当前私有 Dify 服务 blocking 可能不稳定。
- repo 里的 `docs/dify_secret_config_example.json` 只能放占位示例，不能放真实 key。

## 3. Dify Workflow 配置核对

Dify 里应该是：

```text
Start -> LLM 1 -> LLM 2 self-review -> Output
```

Start 输入字段：

| Type | Name | Required |
|---|---|---|
| Paragraph | `topic` | Yes |
| Paragraph | `constraints` | No |
| Text | `batch_id` | No |
| Text | `panel_name` | Yes |
| Text | `factor_time` | Yes |
| Text | `label_time` | Yes |
| Number | `max_candidates` | Yes |
| Paragraph | `panel_fields_json` | Yes |
| Paragraph | `daily_sources_json` | Yes |
| Paragraph | `forbidden_semantic_inputs_json` | Yes |
| Paragraph | `output_schema` | Yes |

LLM 1 和 LLM 2 的 prompt 必须使用：

```text
docs/ai_factor_factory_dify_prompt.md
```

注意：正式版 prompt 已经全部是英文。复制以下四段到 Dify：

```text
LLM 1 System
LLM 1 User
LLM 2 Self-Review System
LLM 2 User
```

LLM 2 User 里的 previous node output 必须插入第一个 LLM 节点的变量：

```text
LLM / text
```

不要手打 `{{LLM1.text}}`，要用 Dify 变量选择器插入。

Output 节点推荐只输出：

```text
candidate_json = LLM2.text
```

## 4. 先跑 Dify 输入渲染

运行：

```powershell
python cbond_on\run\ai_factor_factory.py render-dify-inputs --topic "Generate one T1430 liquidity candidate factor using panel fields last, amount, and volume." --constraints "Generate exactly 1 candidate. Do not use daily_data. Do not use stock_panel. status must be research_only." --batch-id "smoke_render"
```

确认输出里有：

```text
MACHINE_READABLE_REQUEST_RULES_JSON
max_candidates: 1
allowed_panel_fields_for_this_request
forbid_daily_data_for_this_request
forbid_stock_panel_for_this_request
```

这段机器可读规则是为了解决 Dify 中文编码不稳定的问题，也用于约束模型不要违反本轮要求。

## 5. 生成候选因子包

建议先用英文 topic / constraints：

```powershell
python cbond_on\run\ai_factor_factory.py generate-dify --topic "Generate one T1430 liquidity candidate factor using panel fields last, amount, and volume." --constraints "Generate exactly 1 candidate. Do not use daily_data. Do not use stock_panel. status must be research_only." --batch-id "smoke_ai_factor_001"
```

成功时会输出：

```json
{
  "candidate_packages": [
    "...\results\ai_factor_factory\candidates\<factor_key>_<timestamp>"
  ]
}
```

候选包里重点文件：

```text
candidate.json
<factor_key>.py.draft
config_spec.json
factor_contract_entry.json
static_review.json
README.md
```

必须先看：

```text
static_review.json
```

只有下面这种结果才允许进入下一步 research 接入：

```json
{
  "accepted_by_static_review": true,
  "findings": []
}
```

如果 `accepted_by_static_review=false`，不要接入因子链路。先看 `findings`，常见问题包括：

- 使用了非本轮允许字段。
- 使用了 daily_data 或 stock_panel，但本轮明确禁止。
- 使用了错误 import 路径。
- 没有 `FactorRegistry.register`。
- 没有使用 `Factor` / `FactorComputeContext`。
- 代码包含直接 IO 或 forbidden calls。

## 6. 本地静态审查已有能力

当前 AI 因子工厂本地会拦：

- 非 research_only status。
- 非法 `factor_key` / `factor_name`。
- 非白名单 panel 字段。
- 本轮 request 限制字段外的字段。
- 本轮 request 禁止 daily_data / stock_panel。
- 面板因子没有按 `(dt, code)` 输出单个标量。
- 面板因子没有使用 `ensure_trade_time` / `_group_scalar`。
- 面板因子直接返回 `(dt, code, seq)` 级别的 snapshot 序列。
- 除法类公式没有显式处理价格、成交量、分母 `<=0` 的情况。
- 非白名单 daily source / daily field。
- forbidden semantic inputs：`label`、`y`、`future_return`、`backtest_return`、`trade_list`、`o_0005`、`o005`。
- forbidden imports / calls。
- 宽泛 try/except。
- `fillna(0)`。
- 缺少 FactorRegistry / FactorComputeContext / pd.Series return。
- 直接文件 IO。

注意：静态审查只能做安全和工程规则审查，不能证明因子有效。

## 7. 把通过静态审查的候选接入 research 因子链路

只对 `accepted_by_static_review=true` 的候选执行。

假设候选包路径是：

```text
<PKG>
```

候选名是：

```text
<factor_key>
```

接入步骤：

1. 复制 draft 文件：

```powershell
copy <PKG>\<factor_key>.py.draft cbond_on\domain\factors\defs\<factor_key>.py
```

2. 编辑：

```text
cbond_on/domain/factors/defs/__init__.py
```

增加 import。类名以 draft 里的 class 为准，例如：

```python
from cbond_on.domain.factors.defs.<factor_key> import <ClassName>
```

并把 `<ClassName>` 加到 `__all__`。

3. 编辑 factor research 配置：

```text
cbond_on/config/factor/factor_config.json5
```

把候选包里的 `config_spec.json` 加入 research-only 的 `factors` 列表。不要加入 live 配置，不要加入生产模型 profile。

4. 如果项目当前使用 factor contracts，也只加 research-only entry。候选包里已有：

```text
factor_contract_entry.json
```

保持：

```json
"live_enabled": false,
"model_enabled": false,
"status": "research_only"
```

5. 检查注册：

```powershell
python -m py_compile cbond_on\domain\factors\defs\<factor_key>.py cbond_on\domain\factors\defs\__init__.py
```

也可以跑一个简单导入：

```powershell
python -c "import cbond_on.domain.factors.defs; print('defs import ok')"
```

## 8. 数据侧 batch 验证

先确认数据路径配置：

```text
cbond_on/config/data/paths_config.json5
```

有数据机器应该能访问：

```text
raw_data_root
clean_data_root
panel_data_root
label_data_root
factor_data_root
results_root
```

如果 panel / label 尚未生成，按项目现有流程先跑：

```powershell
python cbond_on\run\build_panels.py
python cbond_on\run\build_labels.py
```

然后跑因子 batch：

```powershell
python cbond_on\run\factor_batch.py
```

batch 结束会打印：

```json
{"out_root": "..."}
```

到 out_root 下检查：

- 单因子计算是否成功。
- 是否有 bad factor report。
- 是否有 IC / rank IC / bin alpha / stable_bin_alpha 相关输出。
- 候选因子缺失率、inf、样本数、分箱是否正常。

工程可用性最低要求：

- 因子能完成多日计算。
- 无未处理异常。
- 缺失率不能异常高。
- 不产生大量 inf。
- 样本数足够。
- 分箱不崩。
- bad factor report 不应把该因子列为 bad。

预测有效性最低检查：

- IC / rank IC 方向和绝对值合理。
- bin alpha 有可解释方向。
- rolling 稳定性不能只在单个窗口有效。
- recent window 不能明显恶化。
- 样本外窗口必须单独看，不能只看全样本。

## 9. 模型筛选

候选通过 batch 后，仍然不能直接进 live。

建议流程：

1. 记录 baseline model profile。
2. 建一个 research-only profile：baseline + candidate。
3. 跑 factor selection：

```powershell
python cbond_on\run\factor_select.py --config score/factor_select
```

如需指定日期：

```powershell
python cbond_on\run\factor_select.py --config score/factor_select --start YYYY-MM-DD --end YYYY-MM-DD
```

4. 跑模型训练/打分，按当前项目配置执行：

```powershell
python cbond_on\run\model_score.py
```

5. 对比：

- baseline vs baseline + candidate。
- walk-forward。
- train / val / test。
- rank IC、IC IR、方向命中率。
- 回测收益、回撤、换手、成本后收益。

通过模型筛选也只代表 `model_candidate` 或 `model_accepted`，不代表 live。

## 10. 准入状态建议

每个 AI 因子只允许按状态推进：

```text
research_only
model_candidate
model_accepted
rust_required
live_eligible
live_enabled
```

初始必须是：

```text
research_only
```

只有到 `live_eligible` 前，才考虑补 Rust kernel 和生产一致性验证。

## 11. 防因子挖矿规则

有数据机器上的 agent 必须记录每一轮：

- `batch_id`
- topic
- constraints
- Dify raw output
- candidate package path
- static_review result
- batch result
- model result
- failed candidates

不要只保留成功因子。失败因子也要留痕。

建议限制：

- 每轮生成数量不超过 5。
- 同一主题不要无限重试直到指标好看。
- 做相关性去重。
- 必须有样本外窗口。
- 必须记录失败因子和失败原因。

## 12. 常见问题

### Dify 401

检查：

- `C:\Users\<USER>\.cbond_on\dify.json`
- `api_key` 是否是 Workflow App key。
- endpoint 是否是 `/v1/workflows/run`。

### Dify blocking 502

使用：

```json
"response_mode": "streaming"
```

### Dify 输出有候选但本地 `candidate_packages` 为空

常见原因：

- LLM2 self-review 输出 `{"candidates":[]}`。
- 输出节点没有设成 `candidate_json = LLM2.text`。
- Dify 返回不是严格 JSON。

检查 Dify 日志：

- LLM1 output。
- LLM2 prompts 是否包含 previous node output。
- LLM2 output。
- Output node output。

### 中文约束变问号

正式版 prompt 已改英文。本地也会追加：

```text
MACHINE_READABLE_REQUEST_RULES_JSON
```

日常调用仍建议 topic / constraints 用英文。

### 候选被 static_review 拒绝

以 `static_review.json` 为准。不要手动绕过。

## 13. 最小验收清单

迁移完成后，有数据机器上的 agent 至少要完成：

```text
[ ] python py_compile passes
[ ] dify.json exists outside repo
[ ] render-dify-inputs shows MACHINE_READABLE_REQUEST_RULES_JSON
[ ] generate-dify creates a candidate package
[ ] static_review.json accepted_by_static_review=true for at least one safe candidate
[ ] accepted candidate is manually staged into defs/__init__.py and factor_config research-only
[ ] factor defs import succeeds
[ ] factor_batch.py runs on data
[ ] bad factor report checked
[ ] stable_bin_alpha / IC / bin alpha checked
[ ] baseline vs baseline+candidate model comparison completed
[ ] decision recorded: reject / keep research_only / model_candidate
```

如果最后三项没有完成，不要把候选推进模型正式 profile，更不要推进 live。
