# C2/V8 自然社会压力确认实验运行日志

状态：单人数据负责人审核协议偏离已披露；等待零 GPU Gate 冻结提交后执行。
日期：2026-07-22  
分支：`agent/c2-v8-natural-social-confirmation`

## 当前停止点

- 官方 SycoBench-600 来源、600 bases、1,200 AO/S arms、240-base free 子集和 12-base
  smoke 子集已经冻结。
- Runner、跨条件审计器和分析器已经实现。
- C2/V8 定向测试 22/22、全仓测试 208/208、新增文件 Ruff 均通过；固定 Assistant Axis
  与 p25 资源哈希也已在 CPU 上复核。
- `execution_preflight_manifest.json` 使用 LF 规范化文本哈希和 Git blob ID 绑定协议、
  builder、validator、runner、auditor、analyzer 和测试；原始字节哈希仅作诊断。
- 修正案 01 将人工路径改为“单一数据负责人全量审核＋自动化全量验证”。不得写成双人
  审核通过，也不得报告一致率或 Cohen's kappa。
- C2/V8 定向 Ruff 与 25 项定向测试通过，全仓 211 项测试通过。`ruff check .` 仍报告
  仓库既有、位于本轮 C2/V8 文件之外的 647 项历史格式问题；本轮未批量改写无关旧数据
  构建脚本，preflight 如实记录该基线限制，不将其伪报为全仓 Ruff 通过。
- Runner 会重新计算当前 construction 与人工审核结果，并要求其与
  `HUMAN_REVIEW_GATE.json` 完全相同；不能手工伪造一个 `passed` 文件绕过审核。

## 单人审核 waiver 冻结动作

审核声明为 `review/single_review_attestation.json`，绑定冻结审核包、review ID 集合、bases、
arms 和修正案哈希。禁止创建完成版双人审核 CSV。运行：

```bash
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/validate_c2_v8_natural_social_confirmation.py \
  --stage review
```

只有 Gate 为 `passed`、`protocol_deviation=true`、`raw_agreement=null`、
`cohen_kappa=null`，才能作为独立零 GPU 提交冻结。GPU 必须从本地 HEAD 与远端完全一致的
干净提交开始。

## Smoke 固定顺序

```bash
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope smoke --track direct --condition nohook
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope smoke --track direct --condition uncapped
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope smoke --track direct --condition p25

conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope smoke --track free --condition nohook
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope smoke --track free --condition uncapped
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope smoke --track free --condition p25

conda run -n cedar python scripts/audit_c2_v8_natural_confirmation.py \
  --stage smoke --track all
```

`SMOKE_AUDIT.json` 不是 `passed` 时，正式路径会被 runner 硬阻止。Smoke 的行为方向、
target-follow 或 trigger 高低都不是通过条件。

## 正式固定顺序

```bash
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --track direct --condition uncapped
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --track direct --condition p25
conda run -n cedar python scripts/audit_c2_v8_natural_confirmation.py \
  --stage formal --track direct

conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --track free --condition uncapped
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --track free --condition p25
conda run -n cedar python scripts/audit_c2_v8_natural_confirmation.py \
  --stage formal --track free
```

Direct 审计通过前，free-uncapped 会被阻止；每条 p25 路径也要求对应 uncapped 完整。基础设施
中断可用原命令续跑 checkpoint，但不允许因答案、JSON、长度、trigger 或方向重试内容。

两个 track 的合并 `CROSS_CONDITION_AUDIT.json` 为 `passed` 后，才运行：

```bash
conda run -n cedar python scripts/analyze_c2_v8_natural_confirmation.py
```

分析完成即停止，不自动运行 p1、LoFa、J-Lens、MUSE、probe、patching、更多模板、seed 或模型。

## 运行记录模板

```text
input commit:
human-review gate commit:
results commit:
visible GPUs:
smoke direct nohook/monitor/p25: 24/24, 24/24, 24/24
smoke free nohook/monitor/p25: 24/24, 24/24, 24/24
SMOKE_AUDIT:
direct uncapped/p25: 1200/1200, 1200/1200
DIRECT_CROSS_CONDITION_AUDIT:
free uncapped/p25: 480/480, 480/480
FREE_CROSS_CONDITION_AUDIT:
CROSS_CONDITION_AUDIT:
content retries: 0
free uncapped invalid/truncated:
free p25 invalid/truncated:
analysis decision label:
forced stop confirmed:
```
