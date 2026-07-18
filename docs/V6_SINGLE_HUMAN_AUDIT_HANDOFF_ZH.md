# v6 单人盲态审计：CPU 交接说明

## 当前完成状态

- 1152 条最终表面输出已完成确定性解析；
- 1151 条为严格 schema 合法 JSON；
- 1 条为空输出，固定为 `invalid`，不重试、不修复；
- 174 条盲审包已由数据负责人全部查看；人工发现与解析器一致；
- `adjudicated_labels.csv` 与 `label_provenance.jsonl` 已生成并冻结哈希；
- 未读取 private key；
- 未调用 CUDA、模型加载、生成、capping、J-Lens、probe、MUSE、p1 或新 seed。

## 是否需要 GPU

不需要。标签生成、揭盲映射、bootstrap、Monte Carlo、遥测聚合和 Gate A/B/C 都是 CPU 工作。分析脚本已改为直接加载纯标准库 helper，不再因顶层包导入而要求安装 PyTorch。

## 数据负责人可执行的命令

在仓库根目录运行：

```powershell
python scripts/finalize_v6_single_human_audit.py
```

预期输出必须包含：

```text
records = 1152
strict_exact_json = 1151
empty_final_surface = 1
private_key_read = false
gpu_or_model_execution = false
```

重复运行必须得到同一个 `adjudicated_sha256`。

## private key 持有人执行的命令

将冻结的 key 放在：

```text
artifacts/m2/m2_2b_v6_zero_gpu_reaudit/private/blind_key.jsonl
```

key 的 SHA-256 必须与 `input_freeze_manifest.json` 中的 `blind_key_sha256` 完全一致。不要通过聊天、公开 Git 提交或标注工作簿传输 key。

然后在仓库根目录运行：

```powershell
python scripts/analyze_v6_zero_gpu_reaudit.py
```

该命令只读取冻结结果和遥测，不重新运行模型。它会生成：

- strict/lenient 记录级四分类；
- behavior report；
- telemetry report；
- family robustness；
- family power simulation；
- delivery audit；
- 唯一的 Gate A、Gate B 或 Gate C 路由。

## 硬停止条件

出现任一情况立即停止：

- key 哈希不符；
- packet、labels 或原始响应哈希不符；
- 非唯一空输出；
- 1152 条配对不完整；
- 遥测交叉校验失败；
- 脚本试图加载模型、CUDA 或生成新文本。

不得临时修复输出、重试截断记录、改 Gate 阈值、删题或先查看某个实验格再决定是否继续。
