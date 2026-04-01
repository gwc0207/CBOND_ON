# CBOND_ON 远端更新与运行命令

更新时间：2026-04-01

## 1. 覆盖更新远端代码（丢弃服务器本地改动）

```bash
cd ~/cbond_on
git fetch origin
git checkout main
git reset --hard origin/main
git clean -fd
git rev-parse --short HEAD
git status
```

说明：
- `git reset --hard origin/main`：强制覆盖到远端最新代码。
- `git clean -fd`：删除未跟踪文件/目录。

## 2. 如果服务器还没有项目目录

```bash
git clone http://10.1.30.16/JebGong/cbond_on.git ~/cbond_on
cd ~/cbond_on
git branch --show-current
git rev-parse --short HEAD
```

## 3. 激活环境

```bash
source ~/venv/cbond/bin/activate
cd ~/cbond_on
```

## 4. 运行 model_score（按配置默认参数）

```bash
python cbond_on/run/model_score.py
```

当前默认读取：
- `cbond_on/config/score/model_score_config.json5`
- `model_id = lob_st_default`
- `start = 2025-12-01`
- `end = 2026-03-30`

## 5. 指定参数运行（可选）

```bash
python cbond_on/run/model_score.py \
  --model-id lob_st_default \
  --start 2025-12-01 \
  --end 2026-03-30
```

## 6. 日志落盘运行（推荐）

```bash
mkdir -p ~/cbond_on_runtime/logs
python cbond_on/run/model_score.py 2>&1 | tee ~/cbond_on_runtime/logs/model_score_$(date +%Y%m%d_%H%M%S).log
```

## 7. 进程检查与停止（可选）

检查：
```bash
pgrep -af "cbond_on/run/model_score.py"
```

停止：
```bash
pkill -f "cbond_on/run/model_score.py"
```

## 8. 本阶段约束

- 当前服务器用途：训练模型、测试因子。
- 不启动 live scheduler（除非明确要求）。

