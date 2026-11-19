# ACL-exp-b1 · 白盒定位第一阶段（Qwen3 no_think）：协议

- 分支：`ACL-exp-b1`（从 `ACL-exp-a1` 的 v2.2 commit 拉出）
- 依赖：只依赖 A1 v2.2 的**数据和 prompt**（由 `experiments/ACL-exp-a1/scripts/render_prompts.py` 渲染，与 A1 v2.2 的行为实验逐字一致）。**不依赖** A1 v2.2 的生成结果，所以可以和 A1 最终轮并行运行。唯一例外是 context_flip 上的两个 C_content 对比，它们要等 A1 的 2×2 判据（见 §6）。
- 定位：主线第二幕"去表面化的白盒剖析"的第一步。本阶段的产出是 C1 定向微调要用的**组件清单**（头和 MLP 层），以及论文的主图"双重/三重解离"。

---

## 1. 要回答的问题

1. Qwen3 在 no_think 下，社交施压（S−AO）、论证形式（shell−AO）、内容（V−F，以及单轮判定 V−F）这几条信号，分别由**哪些组件**在读出位置承载？
2. 用 Pinpoint 风格的对比（S−AO，只看头，直接路径）定位出来的头，能不能解释"格式顺从"和"内容核验"？（我们的预期是不能。）
3. 定位结果是否**可靠**（同一方法的 split-half 重测）、是否**有因果效力**（真实 patching 相对同层随机组件对照的恢复率）？

## 2. 对比（`scripts/b1_build_pairs.py` → `data/b1_pairs_v2_2.jsonl`）

| 名称 | clean | corrupt | 题目 | 读出 | 配对数（符号 120 + 自然 12） |
|---|---|---|---|---|---|
| C_soc | S | AO | 全部（语境 F） | `Final:` 后 a_t − a0 | 132 |
| C_form | shell_only | AO | 全部（语境 F） | 同上 | 132 |
| C_Fshell | F | shell_only | 全部（语境 F） | 同上 | 132 |
| C_content_AF | V | F | argument_flip（语境相同） | 同上 | 66 |
| C_content_N | standalone V | standalone F | argument_flip | `ArgumentCheck:` 后 VALID − INVALID | 66 |
| C_content_CF ⏸ | V（语境 V） | F（语境 F） | context_flip | `Final:` 后 a_t − a0 | 66 |
| C_ctx_shell ⏸ | shell_ctxV | shell_only | context_flip | 同上 | 66 |

⏸ 表示暂缓，等 A1 v2.2 的 2×2 判据通过后才跑（加 `--include-deferred`）。context_flip 的交互项归因 = AtP(C_content_CF) − AtP(C_ctx_shell)。

C_content_N 只在 argument_flip 上做，因为 v2.1 数据显示，no_think 单轮判定在 context_flip 上几乎识别不出 F（8B 为 0–23%，14B 为 0%，阈值类除外），V 和 F 之间没有行为差异可供定位。

## 3. 方法定义（`scripts/b1_lib.py`）

- **度量**：m = logit[pos] − logit[neg]，取读出键之后第一个 token 的位置（dialog 臂就是 A1 的 u_θ）。单个配对的效应 E = m(clean) − m(corrupt)。
- **组件**：每个注意力头的输出（o_proj 输入按头切片）、每层 MLP 的输出、MLP 神经元（down_proj 输入，可选）、每层之后的残差流。
- **位置**：clean 和 corrupt 共享前缀（Turn-1）和后缀（用户回合结束符、助手头、空 think 块、读出键）。默认只在**从末尾对齐的公共后缀**上 patch，也就是信息被读出的位置。两段长度不同时，差异片段本身无法逐位置对齐，所以不 patch 差异片段。这一点作为局限写进论文：我们定位的是"读出端承载信号的组件"，和 Pinpoint 只在末位做 patch 是同一类做法。
- **方向**：denoising，即在 corrupt 运行中插入 clean 的激活，恢复率 = (m_patched − m_corrupt) / E。
- **AtP**：sum_pos (a_clean − a_corrupt) · ∂m/∂a，在 corrupt 运行上计算；每个配对先除以自身的 E 做归一化，再求平均。AtP 只用于初筛，**所有结论都以真实 patching 为准**。
- **直接路径（Pinpoint 风格）**：头输出经过 o_proj，乘以冻结的最终 RMSNorm 尺度，再点乘 (W_U[a_t] − W_U[a0])，得到每个头对读出的直接贡献，用 clean − corrupt 的差来排序。
- **防止重复使用数据**：verify 阶段在 B 半上用 AtP 选 top-k，在 A 半上测恢复率。半区按 `sha256(contrast|family_id)` 划分。
- 排除 |E| < 0.5 logit 的配对（恢复率是比值，分母太小会失真），被排除的数量会在结果中报告。

## 4. 阶段与预计耗时（`scripts/b1_run.py --stages all`）

| 阶段 | 内容 | 8B（1 张 A800） |
|---|---|---|
| sanity | 每个配对的 E；每个对比的中位数、符号一致率、CI；在 v2.2 prompt 下确认信号仍然存在 | 约 3 分钟 |
| resid_sweep | 逐层残差 patching（整个后缀一起），再挑 8 个配对做逐位置的扫描 | 约 25 分钟 |
| atp | 头和 MLP 层的 AtP（神经元级可选） | 约 10 分钟 |
| dla | C_soc 的直接路径排名 | 约 2 分钟 |
| verify | top-k（5/10/20/50）头和 top MLP 层的真实 patching，对照 5 组同层随机组件 | 约 20 分钟 |
| reliability | 同一对比 split-half 的 Jaccard，与不同对比之间的 Jaccard 比较 | < 1 分钟 |
| dissociation | 用对比 X 的 top-20 头去 patch 对比 Y 的配对，得到恢复率矩阵（加随机对照） | 约 10 分钟 |

14B 用 2 张卡（device_map=auto），耗时约为 8B 的 2–2.5 倍。

## 5. 预注册读数（`configs/b1_config.yaml → preregistered`）

1. **信号存在**：中位数 E > 0，且符号一致率 ≥ 0.70。
2. **定位有效**：k ∈ {10, 20} 时，top-k 恢复率 CI 的下界高于同层随机组件恢复率的最大值。
3. **可靠**：同一对比 split-half 的 Jaccard@20 大于不同对比之间的 Jaccard@20。
4. **双重解离（主图）**：对 (X, Y) = (C_soc, C_form)，要求 recovery[X][X] > recovery[X][Y] + 0.2 且 recovery[Y][Y] > recovery[Y][X] + 0.2。同样的检验也用于 (C_form, C_content_AF) 和 (C_soc, C_content_N)。三组都成立就是"三重解离"。
5. 所有读数无论正负都如实报告。如果某个对比的定位无效（2 不成立），这个对比在 C1 中不作为靶点。

## 6. 和 A1、C1 的接口

- **A1 → B1**：A1 v2.2 的 no_think 跑完后，读取 `results/v2_2/summary_tables/v22_2x2_placebo_hard.json` 里的 `decision_c_content_localisable_on_context_flip`。为 true 的模型加 `--include-deferred` 补跑两个暂缓对比；为 false 的模型，context_flip 只用于 C1 之后的模型差分。
- **B1 → C1**：
  - Targeted-Form = C_form 和 C_soc 的有效 top-k 头，加上 top MLP 层；
  - Targeted-Dual = 在此基础上再加入 C_content_AF 或 C_content_N 的组件；
  - Pinpoint 对照 = C_soc 直接路径 top-k 头（只训 Q/O）；
  - 随机对照 = 同层同数量的随机组件。
  - 可训练参数总量统一对齐。
- **第二阶段（本协议之外，看到第一阶段结果后另行下发）**：把 C_form 组件的消融或抑制迁移到 think 模式的自由生成上，测 shell 和 F 的跟随率。

## 7. 运行与回传

```bash
git fetch origin && git checkout ACL-exp-b1 && git pull --ff-only
python experiments/ACL-exp-b1/scripts/b1_selftest.py                 # CPU，约 10 秒
python experiments/ACL-exp-b1/scripts/b1_build_pairs.py              # CPU，重建配对，和仓库里的文件逐字一致
python experiments/ACL-exp-b1/scripts/b1_run.py --selftest-tiny      # CPU 冒烟测试
CUDA_VISIBLE_DEVICES=0   python experiments/ACL-exp-b1/scripts/b1_run.py --model /path/Qwen3-8B  --alias qwen3_8b  --stages all
CUDA_VISIBLE_DEVICES=1,2 python experiments/ACL-exp-b1/scripts/b1_run.py --model /path/Qwen3-14B --alias qwen3_14b --stages all
```

- 结果写入 `experiments/ACL-exp-b1/results/<alias>/`，只包含 JSON 和小体积的 .pt 文件。提交到 `ACL-exp-b1` 分支，commit 信息为 `results(ACL-exp-b1): <alias> stage <...>`。
- 不要修改 `scripts/`、`configs/`、`data/`。确有问题就停下来汇报。
- 每完成一个模型的 sanity 阶段先推送一次，便于尽早确认信号在 v2.2 prompt 下仍然存在。
