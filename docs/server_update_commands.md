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

### 2.1 Windows 本地更新 Rust 扩展（推荐）
说明：
- 项目在本地运行时优先加载仓库内 `cbond_on/cbond_on_rust/*.pyd`。
- 仅执行 `pip install wheel` 可能不会覆盖仓库内本地 `.pyd`，从而出现“Rust 扩展过旧”报错。

```powershell
cd C:\Users\BaiYang\CBOND_ON\cbond_on\rust\factor_engine
$py = 'C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\python.exe'
$env:PYO3_PYTHON = $py

# 1) 编译 wheel（显式指定解释器，避免 Windows Store python 干扰）
& $py -m maturin build --release -i $py

# 2) 安装 wheel（用于 site-packages）
$whl = Get-ChildItem .\target\wheels\cbond_on_rust-*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $py -m pip install --force-reinstall $whl.FullName

# 3) 用新 wheel 覆盖仓库内本地 .pyd（关键步骤）
& $py -c "import zipfile,glob,os,shutil,tempfile;w=sorted(glob.glob(r'c:/Users/BaiYang/CBOND_ON/cbond_on/rust/factor_engine/target/wheels/cbond_on_rust-*.whl'))[-1];target=r'c:/Users/BaiYang/CBOND_ON/cbond_on/cbond_on_rust/cbond_on_rust.cp311-win_amd64.pyd';z=zipfile.ZipFile(w);name='cbond_on_rust/cbond_on_rust.cp311-win_amd64.pyd';tmp=tempfile.mkdtemp();z.extract(name,tmp);src=os.path.join(tmp,name);shutil.copy2(src,target);print('COPIED',src,'->',target)"

# 4) 验证签名（需包含 daily_data 参数）
& $py -c "import cbond_on_rust,inspect;print(cbond_on_rust.__file__);print(inspect.signature(cbond_on_rust.compute_factor_frame))"
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
说明：并行与预取在 `cbond_on/config/score/model_score_config.json5` 的 `execution` 中配置（`train_processes / prep_workers / prefetch_windows`）。

### 4.4 回测
```bash
source ~/venv/cbond/bin/activate && cd ~/cbond_on && python cbond_on/run/backtest.py
```
