# RG1-P Stage A · 新服务器配置与实验执行 Runbook

> 用途：交给实验执行者（人或 Codex）逐条执行。
> 背景：原 4090 服务器被占用，实验在 272/864 处安全暂停；改在新租用服务器上按 **路线 A（从 0 重跑全部 864 cells）** 执行。
> 本文档基于我在上一台 AutoDL 机器（4×RTX 5090）上实际踩过的坑写成，每一条“注意”都是实测结果，不是推测。

---

## 0. 先决判断：为什么必须从 0 重跑，而不是续跑 272

| 项 | 原服务器 | 新服务器 | 后果 |
|---|---|---|---|
| GPU | RTX 4090（Ada, sm_89） | 请填写 `[TBD]` | 若不是 4090，BF16 kernel 路径不同 |
| PyTorch | 2.6.0+cu124（冻结栈） | Blackwell(sm_120) 必须 ≥2.7/cu128 | 冻结软件栈无法复制 |
| transformers | 4.56.1 | 4.56.1（可对齐） | 可对齐 |

**结论**：只要 GPU 架构或 torch 版本与冻结栈不一致，`272/864` 与新跑的 `592` 就不能拼进同一个冻结矩阵。旧的 272 条**归档保留、不删除、不参与分析**，新机器上从 0 跑满 864。

**这对论文的影响是可控的**，理由要写进方法/附录：

> RG1-P Stage A 的主对比是**同一台机器、同一次运行内**的 `uncapped` vs `capped`（以及 norm-matched 随机方向对照）。跨机器数值差异不会污染这个组内对比。RG1-P **不**声称与 E2A/E2B 的 token 级数值等价，也**不**参与任何需要与 `0c8517bb` / `6a7e2862` 逐条等价的闸门。

因此 RG1-P 的定位固定为：**self-contained within-run propagation diagnostic**。不要试图让它去过 E2A anchor 等价闸——在新硬件上它必然过不了，硬过反而是造假。

---

## 1. 机器验收（跑任何东西之前，5 分钟）

```bash
# 1.1 GPU 型号、数量、显存
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

# 1.2 驱动与 CUDA
nvidia-smi | head -4

# 1.3 磁盘：这是上一台机器翻车的地方
df -h | grep -vE "tmpfs|overlay"
```

**硬性要求**

- 可见 GPU 数：与 RG1-P 协议冻结值一致（原为 4 张）。若数量不同，必须写进 environment amendment。
- 总显存 ≥ 80 GB（Qwen3-32B bf16 权重约 62 GB + KV cache + hidden-state 缓冲）。4×32G=128G 足够。
- **可用磁盘 ≥ 90 GB 且在同一个挂载点**：权重 62–66 GB + 下载临时文件 + hidden states 输出。

> ⚠️ **上一台机器的真实教训**：`/root/autodl-tmp` 只有 **50 GB 配额**，而 Qwen3-32B bf16 是 **17 个分片、约 62–66 GB**，根本放不下。必须先 `df -h` 找到真正够大的挂载点。AutoDL 上通常是：
> - `/root/autodl-tmp`：本地盘，快，但受套餐配额限制（可能只有 50 G）；
> - `/root/autodl-fs/data`：网络盘，200 G 起，慢一些但够大；
> - 若两者都不够，先在控制台扩容数据盘，**不要**边下边删。
>
> 建议：权重放能放下的那个盘，输出的 hidden states 放本地盘（读写频繁）。

---

## 2. Python 环境

```bash
# 2.1 确认解释器（AutoDL 镜像通常是 /root/miniconda3）
/root/miniconda3/bin/python -V
/root/miniconda3/bin/pip list | grep -iE "^torch|transformers"
```

### 2.2 torch 选择（按 GPU 架构）

| GPU | compute cap | torch |
|---|---|---|
| RTX 4090 / A100 | sm_89 / sm_80 | **2.6.0+cu124**（与冻结栈一致，优先） |
| RTX 5090 | sm_120 | ≥ **2.8.0+cu128**（sm_120 在 2.7 之前不受支持） |

若镜像自带 torch 且版本符合，**不要动它**。需要装时：

```bash
# 仅在 GPU 为 Ada/Ampere 且需要对齐冻结栈时
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 2.3 其余依赖（版本必须钉死）

```bash
export PIP="https://pypi.tuna.tsinghua.edu.cn/simple"   # 清华源可用
pip install -i $PIP "transformers==4.56.1" accelerate safetensors huggingface_hub numpy scipy pandas matplotlib pytest ruff
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.device_count(),torch.cuda.get_device_capability(0))"
```

> ⚠️ **实测坑**：AutoDL 默认 `pip.conf` 指向 `mirrors.aliyun.com`，但一旦 `source /etc/network_turbo` 开了学术代理，阿里源反而 **`No matching distribution found`**。解决：装 pip 包时**关掉代理**或显式 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。学术加速只对 GitHub/HuggingFace 有用，对 pip 源是负作用。

---

## 3. 下载 Qwen3-32B 权重（最耗时的一步，先启动）

**目标 revision（必须钉死）**：`9216db5781bf21249d130ec9da846c4624c16137`

### 3.1 首选：HuggingFace 镜像

```bash
source /etc/network_turbo          # AutoDL 学术加速
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1        # 关键
export HF_HUB_ENABLE_HF_TRANSFER=0 # 关键

cat > /root/dl.py <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "Qwen/Qwen3-32B",
    revision="9216db5781bf21249d130ec9da846c4624c16137",
    local_dir="<够大的盘>/qwen3-32b",
    max_workers=8, max_retries=20)
print("DONE", p)
PY
screen -dmS dl bash -lc "/root/miniconda3/bin/python /root/dl.py > /root/dl.log 2>&1"
```

> ⚠️ **两个实测坑**
> 1. **不要开 `HF_HUB_ENABLE_HF_TRANSFER=1`**：会直接 `RuntimeError: An error occurred while downloading using 'hf_transfer'`。
> 2. **必须 `HF_HUB_DISABLE_XET=1`**：否则 hf-mirror 会把请求转给 `cas-bridge.xethub.hf.co`，在国内代理下持续 `ReadTimeoutError` 重试，**字节数完全不增长**（我实测卡在 17 G 一动不动）。

### 3.2 备选（更快，推荐）：ModelScope

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com modelscope

cat > /root/run_dl.sh <<'EOS'
#!/bin/bash
unset http_proxy https_proxy all_proxy    # modelscope 是国内源，走代理反而慢
/root/miniconda3/bin/python - <<PY
from modelscope import snapshot_download
p = snapshot_download("Qwen/Qwen3-32B", local_dir="<够大的盘>/qwen3-32b", max_workers=12)
print("DONE", p)
PY
EOS
chmod +x /root/run_dl.sh
setsid bash /root/run_dl.sh > /root/ms.log 2>&1 < /dev/null &
```

实测速率约 **14 MB/s**，62 GB 约需 **70–80 分钟**。

> ⚠️ **走 ModelScope 的代价**：它不按 HF commit hash 组织版本。若用它，必须在下载后做一次 **文件级校验**，并在附录写明来源：
> ```bash
> # 与 HF 官方 revision 的分片 sha256 逐个比对
> python - <<'PY'
> from huggingface_hub import HfApi
> api = HfApi(endpoint="https://hf-mirror.com")
> info = api.model_info("Qwen/Qwen3-32B",
>                       revision="9216db5781bf21249d130ec9da846c4624c16137", files_metadata=True)
> for s in info.siblings:
>     if s.rfilename.endswith(".safetensors"):
>         print(s.rfilename, s.lfs.sha256 if s.lfs else None)
> PY
> # 本地：sha256sum <够大的盘>/qwen3-32b/*.safetensors
> ```
> 全部一致才可以在论文里写 “Qwen3-32B, revision 9216db57…”。不一致就必须改回 HF 下载。

### 3.3 后台任务不要用裸 `&`

> ⚠️ **实测坑**：通过 SSH 单次 `exec` 起的后台进程，会在 channel 关闭时被杀。
> 必须用 `screen -dmS <name> bash -lc "..."` 或 `setsid <cmd> > log 2>&1 < /dev/null &`，
> 并在下一次连接时用 `pgrep -af` + 目录字节增长**双重确认**它真的在跑。
> 只看 `pgrep` 返回 1 是不够的——我遇到过进程在、但因为参数错误一个字节都没下的情况。

---

## 4. 取代码（当前的真正阻塞点）

**问题**：RG1-P 的分支/脚本目前**没有推到 GitHub**。`git branch -r` 上没有任何 `rg1` 相关分支，最新的只到 `agent/aaai27-paper-writing`（`d183700c`）。

**必须先做**（在原服务器或本地）：

```bash
git push origin agent/rg1-prompt-end-representation-geometry   # 或该分支的实际名字
```

推上去之后，在新机器上：

```bash
cd /root && git clone https://<PAT>@github.com/Tanhhhhtjy/CEDAR.git cedar
cd cedar && git fetch --all
git checkout agent/rg1-...            # RG1-P 分支
git log -1 --format="%H %s"           # 记录 HEAD，写进 manifest
git status --short --branch           # 必须干净
```

> 论文与协议要求：**正式 GPU 运行前，代码必须已提交并推送，worktree 必须 clean**。

---

## 5. 零 GPU CPU Gate（必须全过才允许开 GPU）

按 RG1-P 协议执行，逐项留证据文件：

1. 96 条 prompts 与 E2A manifest 的内容、顺序、SHA-256 完全一致；
2. 模型 revision、axis 资产、threshold、层范围、chat template hash 一致；
3. token 边界可唯一重建（`prompt_end` / `followup_end` / `followup_mean`）；
4. 层号映射与 hook module 与协议约定一致（**不一致立即停，不许静默平移层号**）；
5. `nohook` 与 `monitor_only` 的 next-token logits 一致，monitor hook 不改 residual；
6. 张量 shape 正确、finite、无 NaN/Inf；
7. 每条记录后 hook 计数清零，无泄漏；
8. 两次确定性重建 hash 一致；
9. 单元测试与 lint 全绿。

产出：`CPU_GATE.json`（`status: passed`, `gpu_authorized: true`）、`ANALYSIS_PLAN.json`、`manifest.json`。
**在 `CPU_GATE.json` 写出并提交之前，不得占用 GPU。**

---

## 6. Smoke（正式跑之前的小样）

- 覆盖 N/R/V/F 四臂 × 两个条件 × 三个 token position × 全部捕获层；
- 检查 prompt/token hash、hook 安装与清理、层 46–53 实际触发率、phase telemetry 完整；
- **科学效应大小不作为 smoke 的通过条件**——只看工程正确性；
- 失败就保留 artifact 并停止，不放宽容差、不删 row、不换 seed。

---

## 7. 正式运行：RG1-P Stage A

### 7.1 冻结矩阵

按 RG1-P 协议冻结的 **864 cells**（`[请在协议里核对这个数字的分解方式：prompts × conditions × 随机方向 battery]`）。执行者必须在 `manifest.json` 里写出分解式，不能只写总数。

### 7.2 运行纪律

- `CUDA_VISIBLE_DEVICES` 固定为协议冻结的那组，数量必须与协议一致；
- batch size 1，`model.eval()` + inference mode，不生成回答（纯 forward）；
- **每条记录写原子 checkpoint**，中断后同配置续跑自动跳过已完成 cell；
- 只有**基础设施中断**允许同配置续跑；**科学结果不理想不是重跑理由**；
- 任何中断、恢复、GPU 集合变化都要写进 `INFRASTRUCTURE_EVENTS.json`（原服务器已有此文件的写法，照抄格式）。

### 7.3 采集内容

- block output：layer 45（cap 前负控制）、46–53（全部 capped 层）、54、以及 `L_final`；
- capped 层额外存 `cap_input` / `cap_output` / `immediate_correction`；
- 逐层 trigger rate、correction sum、mean correction，分 prefill/decode 记录；
- 三个 token position 全部记录，**主位置固定为 `prompt_end`**，不许看图之后再挑。

### 7.4 存储估算

`96 × 2 × 11 层 × 3 position × 5120 dim × 4 B ≈ 130 MB`（fp32）。若把 864 cells 全量存则按比例放大，仍在 GB 量级。**存本地盘，不要存网络盘。**

---

## 8. 分析与出图

RG1-P Stage A 要出的两张图（你之前定的）：

### 图一：逐层扫描（layer-wise propagation）

- 横轴 layer 45 → `L_final`；
- 纵轴分三条：`parallel energy`、`orthogonal energy`、`orthogonal fraction`；
- **layers 46–53** 用逐层官方 axis 投影；**layers 54 以上没有官方 axis**，统一用冻结的 **layer-53 exit direction**，并在图注写明；
- **必须画 norm-matched + schedule-matched 随机方向 battery 作为灰色地板**。没有这条地板，正交分量随深度增长是几乎必然的，图没有证据价值；
- family-level CI（12 families 为统计单位，不是 96 prompts）。

### 图二：点云位移（paired representation displacement）

- 在 `L_final` + `prompt_end` 上，把 V/F 的 uncapped 与 capped residual 一起拟合 **一次** PCA，所有条件投到同一 basis；
- **每个 item 一个点**，不是每个 token 一个点；
- 同一 item 的 uncapped → capped 用细箭头连接；
- N/R 作为浅色 overlay；
- 68% covariance ellipse 只能标为 *descriptive*，不是置信区间；
- 报告 PC1/PC2 解释方差；二维图不能替代高维定量检验。

### 定量（图之外必须报的）

- V/F **family-blocked** 线性可分性（leave-one-family-out 或 GroupKFold，标准化/降维/拟合全部在 training fold 内）；
- family-level label-flip permutation（12 families 可枚举全部 4096 种）；
- `cos(mean Δh_V, mean Δh_F)`、各自 norm、`||mean Δh_V − mean Δh_F||`、`D_rel`；
- 10,000 次 family-cluster bootstrap，seed 写进 manifest。

---

## 9. 必须新增的一份文档：Environment Amendment

因为换了机器，必须在仓库里留一份 `docs/RG1P_ENVIRONMENT_AMENDMENT.md`，写清楚：

- 原服务器 GPU/torch/CUDA 与新服务器的差异；
- 旧 272 条的归档路径与“作废但保留”的声明；
- RG1-P 的定位是 within-run diagnostic，不与 E2A/E2B 做 token 级等价；
- 权重来源（HF 还是 ModelScope）与校验结果；
- 若 GPU 数量与协议冻结值不同，明确记录。

论文里对应一句（放方法或附录，不要藏）：

> RG1-P was executed on a separate machine from the E2A/E2B runs. All RG1-P contrasts are within-run comparisons on identical hardware and software; we therefore do not claim token-level numerical equivalence between RG1-P and the frozen E2A traces.

---

## 10. 停止规则（任一触发立即停，保留失败产物）

- 输入/prompt/token/asset hash 不一致；
- 层或 token 边界无法唯一定位；
- `nohook` 与 `monitor_only` 不等价；
- layer 45 负控制异常；
- hook 泄漏；
- 张量非 finite、row 缺失或顺序漂移；
- worktree dirty 或运行的代码未提交；
- 可见 GPU 数量、驱动、torch 版本发生未记录的变化；
- 磁盘写满（**上一台机器就是差点在这里翻车**）。

**不许**通过放宽容差、删 family、换 seed、换层、重画 PCA 来“修复”科学结果。

---

## 11. 交付清单

```
docs/RG1_PROMPT_END_REPRESENTATION_PROTOCOL.md
docs/RG1P_ENVIRONMENT_AMENDMENT.md
datasets/m2_2b_formal/rg1_prompt_end_geometry/{manifest.json, CPU_GATE.json, ANALYSIS_PLAN.json}
scripts/{run_,audit_,analyze_}rg1_prompt_end_geometry.py
tests/test_rg1_prompt_end_geometry.py
artifacts/m2/rg1_prompt_end_geometry/
    smoke-audit.json, records.jsonl, hidden_states.safetensors, telemetry.jsonl,
    analysis.json, analysis-report.md, plot_points.csv, plot_stats.json,
    figures/{layerwise-propagation, representation-displacement}.{pdf,svg,png},
    RESULT_SUMMARY_ZH.md, DELIVERY_MANIFEST.json, INFRASTRUCTURE_EVENTS.json
```

---

## 12. 时间预算（按新机器从零开始）

| 阶段 | 预计 | 是否阻塞下一步 |
|---|---|---|
| 机器验收 + 磁盘规划 | 10 min | 是 |
| Python 环境 | 15 min | 是 |
| 权重下载（并行启动） | 70–90 min | 是 |
| 拉代码（需先 push RG1 分支） | 10 min | 是 |
| CPU Gate + smoke | 30–60 min | 是 |
| Formal 864 cells | 视 forward 长度，**按原机器 272 条的耗时外推** | 是 |
| 审计 + 分析 + 两张图 | 60–90 min | — |

**关键路径是“推 RG1 分支”和“下权重”**，这两件事现在就可以并行开始。