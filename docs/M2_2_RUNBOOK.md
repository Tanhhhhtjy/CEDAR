# M2.2 Belief-R 运行手册

M2.2 用固定的 Qwen3-32B 和官方 Assistant Axis capping 配置，测量 activation capping
对形式域合理信念更新的影响。所有运行均使用 `cedar` Conda 环境、离线模型缓存和 BF16，结果写入
Git 忽略的 `artifacts/`。

## 冻结数据与运行规模

- 来源：Belief-R revision `3719f5804c63318037465fecf298a7fd78d99121`；
- 项目题库：`datasets/formal_domain/formal_full.jsonl`；
- 冻结清单：`data/splits/m2_belief_r_ids.json`，抽样种子 `20260711`；
- development：25 条 BU + 25 条 BM，仅运行 uncapped，共 100 条 step-level score；
- Pilot：100 条 BU + 100 条 BM，运行 uncapped/p1/p25/p50/p75，共 2,000 条 score；
- extension：另 100 条 BU，仅在 Pilot 要求扩样时运行 uncapped/p25，共 400 条 score。

启动时会把 350 条冻结样本与固定 Belief-R 源重新配对，并逐题校验题干、选项、标签、元数据、
清单顺序和组间不重叠。`formal_balanced_200.jsonl` 不参与 M2.2。

## 通用前置检查

```bash
conda activate cedar
export HF_HOME=/data1/WuMingrui/HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

ruff check src tests scripts
pytest -q
git status --short
```

正式运行要求工作区干净。四张卡的选择必须在 `prepare` 前固定；manifest 会记录可见 GPU，后续阶段
若改变 GPU 可见性、batch size、代码 commit 或任一资源哈希，将拒绝复用原运行目录。

## Development：只验证答题范式

先生成并人工核对运行清单，不加载 Qwen：

```bash
python scripts/run_belief_m2_2.py \
  --stage prepare \
  --split development \
  --run-name m2_2_development_v1
```

预期为 50 个 item、1 个条件和 100 条 score。确认后执行；相同命令可安全断点续跑：

```bash
python scripts/run_belief_m2_2.py \
  --stage all \
  --split development \
  --run-name m2_2_development_v1
```

只读取 `summary.json` 中的 uncapped development 指标。执行期修订后的判读规则为：

- overall trajectory accuracy `>= 50%` 且 time_t1 BU-Acc `>= 10%`：提示有效，冻结当前
  prompt SHA256；
- 任一条件不满足：不查看 Pilot，最多修订一次答题模板，改用新 run name；
- 第二次仍未同时满足：Belief-R/Qwen 范式判为基线无效，停止 M2.2。

## Pilot 防火墙

Pilot 不会因普通命令被误触发。只有 development 过闸并取得明确运行确认后，才从其
`summary.json` 复制 `prompt_sha256`，同时提供通过的 summary、显式 `--allow-pilot` 和完全一致的
prompt hash：

```bash
python scripts/run_belief_m2_2.py \
  --stage prepare \
  --split pilot \
  --run-name m2_2_pilot_v1 \
  --allow-pilot \
  --prompt-sha256 '<development summary 中的值>' \
  --authorization-summary artifacts/m2/belief_r/development/m2_2_development_v1/summary.json
```

人工核对 200 个 item、5 个条件和 2,000 条 score 后，使用完全相同的参数将 `--stage prepare`
改为 `--stage all`。未取得确认前不得执行该步骤。

主判据为 p25 的 `Delta_BU` 及按 item 配对的 10,000 次 bootstrap 单侧 95% CI 下界。summary
同时报告各条件 BU/BM accuracy、BREU、trajectory accuracy、保留率及 logit-margin 变化。

## 第二阶段 BU extension

仅当 Pilot summary 的 `pilot_gate.next_step` 为 `expand_bu` 时，extension 才能解封。它还要求
显式 `--allow-extension`、冻结 prompt hash 和 Pilot summary：

```bash
python scripts/run_belief_m2_2.py \
  --stage prepare \
  --split extension \
  --run-name m2_2_extension_v1 \
  --allow-extension \
  --prompt-sha256 '<冻结值>' \
  --authorization-summary artifacts/m2/belief_r/pilot/m2_2_pilot_v1/summary.json
```

确认后再以相同参数运行 `--stage all`。第二阶段最终判读必须把 Pilot 的 100 条 BU 与新增 100 条
BU 按 item 合并后重算，不能把新增批次单独当作最终结论。

## 输出与恢复

每个运行目录包含：

- `manifest.json`：固定来源、模型、干预、提示、统计、环境、Git 和硬件指纹；
- `scores/`：每个 item × condition × step 的原子 JSON checkpoint；
- `runtime.json`：实际模型 device map；
- `summary.json`：聚合统计与阶段闸门；
- `m22_belief_results.parquet`：可复算的逐记录表格。

同一配置中断后原命令可直接重跑，已完成记录会跳过。配置发生变化时必须使用新的 run name，
不得删除 manifest 后混用旧 score。
