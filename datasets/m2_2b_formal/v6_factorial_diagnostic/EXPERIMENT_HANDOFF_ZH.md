# 给实验负责人的 v6 运行交接

## 一句话任务

先跑相同 sampling 参数下的 thinking-on/off × uncapped/p25 四格，判断现象是
thinking 特异还是一般 capping 效应；不要筛题，也不要先做 J-Lens/MUSE。

## 冻结配置

- 模型：仓库已固定 revision 的 Qwen3-32B，BF16、SDPA、4 GPU。
- 数据：`m2_2b_v6_factorial144.jsonl`，144 条，顺序不得改变。
- batch size：1。
- sampling：`temperature=0.6, top_p=0.95, top_k=20, num_beams=1`。
- 主 capping：官方 p25；p1 只作强干预压力测试。
- 层与位置：官方 46–53 层，所有 decoder-block 输出位置。
- 主种子：0；不要提前跑 seed 1/2。
- 输出：原样保留；不进行内容重试、答案修复或按输出删题。

## 0. 拉分支并验证数据

```bash
git fetch origin
git switch agent/m2-2b-v6-factorial-diagnostic
git pull --ff-only
python datasets/m2_2b_formal/v6_factorial_diagnostic/validate_m2_2b_v6_factorial_diagnostic.py
pytest -q
```

验证器必须显示 `"status": "passed"`，仓库测试必须全部通过。若测试总数随仓库更新而变化，
以“0 failed”为准，不要硬编码旧测试数量。

## 1. 第一批：最优先四格，共 576 次生成

严格按以下顺序运行：

```bash
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode thinking_sampling --condition uncapped --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode thinking_sampling --condition p25 --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode nonthinking_sampling --condition uncapped --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode nonthinking_sampling --condition p25 --seed 0
```

四格完成后先运行一次不完整审计与分析：

```bash
python scripts/audit_m2_2b_v6_factorial_runs.py --dataset formal --seed 0 --allow-incomplete
python scripts/analyze_m2_2b_v6_factorial_diagnostic.py --seed 0
```

把以下内容原样反馈给数据负责人：四个 run 目录、`v6_cross_cell_audit_formal_seed0.json`、
`v6_factorial_report_seed0.json`，以及任何截断/异常说明。

## 2. 第二批：sampling 的 p1 压力测试，共 288 次

第一批输入审计无失败后再跑：

```bash
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode thinking_sampling --condition p1 --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode nonthinking_sampling --condition p1 --seed 0
```

## 3. 第三批：greedy 诊断，共 864 次

```bash
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode thinking_greedy --condition uncapped --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode thinking_greedy --condition p25 --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode thinking_greedy --condition p1 --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode nonthinking_greedy --condition uncapped --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode nonthinking_greedy --condition p25 --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset formal --mode nonthinking_greedy --condition p1 --seed 0
```

v5 的旧 thinking-greedy 三组只能作为历史复现锚点：它们使用 batch size 2、包含 padding，
也没有 v6 的轴遥测，不能替代上述三组新运行。

## 4. Assistant Axis 阳性对照，共 96 次

只跑 sampling、uncapped/p25：

```bash
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset axis --mode thinking_sampling --condition uncapped --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset axis --mode thinking_sampling --condition p25 --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset axis --mode nonthinking_sampling --condition uncapped --seed 0
python scripts/run_m2_2b_v6_factorial_diagnostic.py --dataset axis --mode nonthinking_sampling --condition p25 --seed 0
```

## 5. 完整审计

```bash
python scripts/audit_m2_2b_v6_factorial_runs.py --dataset formal --seed 0
python scripts/audit_m2_2b_v6_factorial_runs.py --dataset axis --seed 0 --allow-incomplete
python scripts/analyze_m2_2b_v6_factorial_diagnostic.py --seed 0
```

正式审计必须 `passed`。axis 审计允许缺少未预定的 greedy/p1 格，因此带
`--allow-incomplete`；已有四格本身不得不完整。

## 必须立即停止的情况

- 数据哈希、ID 顺序、tokenized input 或同一 mode 跨 cap 的输入签名不一致；
- 第一条出现 thinking/nonthinking 模式完整性失败；
- 模型或官方 p25/p1 本地资源 revision 不匹配；
- 不是 4 张可见 GPU，或运行环境不是 `cedar`；
- 想根据结果改题、换难度、删掉非单调题或重试内容答案。

传输/机器中断可以用完全相同命令续跑；内容不理想、JSON 失败或截断不得人工修复。
