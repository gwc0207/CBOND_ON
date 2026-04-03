# CBOND_ON Server Update Commands
更新时间：2026-04-03

## 1. 覆盖远端代码
```bash
cd ~/cbond_on && git fetch origin && git checkout main && git reset --hard origin/main && git clean -fd && git rev-parse --short HEAD && git status
```

## 2. Rust 扩展更新
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on/rust/factor_engine && python -m pip install -U maturin && python -m maturin develop --release && python -c "import cbond_on_rust; print('cbond_on_rust OK')"
```

## 3. 激活环境
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on
```

## 4. 流程运行代码
### 4.1 Panel（含 Label）
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on && python cbond_on/run/build_panels.py
```

### 4.2 因子
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on && python cbond_on/run/factor_batch.py
```

### 4.3 模型打分
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on && python cbond_on/run/model_score.py
```

可选：并发分片（吞吐模式，开启 2 个进程）
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on && python cbond_on/run/model_score.py --parallel-shards 2 --parallel-shard-index 0
source ~/venv/cbond/bin/activate && cd ~/cbond_on && python cbond_on/run/model_score.py --parallel-shards 2 --parallel-shard-index 1
```

### 4.4 回测
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on && python cbond_on/run/backtest.py
```
