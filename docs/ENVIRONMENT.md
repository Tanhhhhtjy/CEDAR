# CEDAR 环境管理

本项目统一使用名为 `cedar` 的 Conda 环境。仓库内不创建 `.venv`，模型权重也不写入 Conda 环境。

## 创建与同步

```bash
conda env create -f environment.yml
conda activate cedar
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --frozen --group dev
```

已有环境只需执行后两行。必须显式设置 `UV_PROJECT_ENVIRONMENT`；Conda 只设置
`CONDA_PREFIX`，直接运行 `uv sync` 会错误地创建仓库内 `.venv`。

## 模型缓存

```bash
export HF_HOME=/data1/WuMingrui/HF_HOME
```

模型与数据缓存放在 `/data1`，实验小型输出放在仓库的 `artifacts/`。不要将权重、token、
原始有害生成文本提交到 Git。

## 验证

```bash
python -c "import sys, torch; print(sys.prefix, torch.__version__, torch.cuda.is_available())"
pytest -q
ruff check src tests scripts
```

预期环境前缀为 `/home/WuMingrui/miniconda3/envs/cedar`，Python 3.11，PyTorch
2.6.0+cu124。
