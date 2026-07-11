# M2.1 HarmBench 运行手册

M2.1 使用固定 Qwen、Assistant Axis、HarmBench 与 HarmBench classifier 版本。原始生成只写入
Git 忽略的 `artifacts/`。

## 前置资源

- Conda 环境：`cedar`；
- Hugging Face 缓存：`/data1/WuMingrui/HF_HOME`；
- HarmBench checkout：`/data1/WuMingrui/HF_HOME/external/HarmBench`，必须位于 commit
  `8e1604d1171fe8a48d8febecd22f600e462bdcdd`；
- Qwen 与 classifier 必须已下载，正式运行使用离线模式。

## Validation

先验证清单，不加载模型：

```bash
HF_HOME=/data1/WuMingrui/HF_HOME \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_harmbench_m2_1.py \
  --stage prepare --split validation --run-name m2_1_validation_v2
```

确认四张连续 GPU 空闲后运行完整 validation：

```bash
HF_HOME=/data1/WuMingrui/HF_HOME \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/run_harmbench_m2_1.py \
  --stage all --split validation --run-name m2_1_validation_v2
```

运行以固定 batch 为断点单元；重复相同命令会跳过已完成记录。运行配置或代码版本不同会拒绝复用
同一目录。

## Test 防火墙

test 默认拒绝加载。只有 persona YAML 已改为 `status: frozen` 后，命令同时提供
`--allow-test` 和与文件完全一致的 `--test-template-sha256` 才能解封。解封前必须先汇报
validation 聚合结果并取得明确确认。

## 输出

每个运行目录包含：

- `manifest.json`：数据、模型、模板、代码、环境及配置指纹；
- `generations/`：逐记录原始生成，敏感且不提交；
- `classifications/`：逐记录 yes/no 裁判结果；
- `summary.json`：ASR、类别 macro、paired bootstrap、McNemar 与过闸状态；
- `runtime.json`：实际模型设备映射。
