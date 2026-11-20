# ACL-exp-b1 · 白盒定位第一阶段（Qwen3 no_think）：协议

- 分支：`ACL-exp-b1`（从 `ACL-exp-a1` 的 v2.2 commit 拉出）
- 依赖：只依赖 A1 v2.2 的**数据和 prompt**（由 `experiments/ACL-exp-a1/scripts/render_prompts.py` 渲染，与 A1 v2.2 的行为实验逐字一致）。**不依赖** A1 v2.2 的生成结果，所以可以和 A1 最终轮并行运行。唯一例外是 context_flip 上的两个 C_content 对比，它们要等 A1 的 2×2 判据（见 §6）。
- 定位：主线第二幕"去表面化的白盒剖析"的第一步。本阶段的产出是 C1 定向微调要用的**组件清单**（头和 MLP 层），以及论文的主图"双重/三重解离"。

---

## 1. 要回答的问题

1. Qwen3 在 no_think 下，Pinpoint 式社交建议（`C_pinpoint`: S−N）、纯施压语气增量（`C_soc`: S−AO）、论证形式（`C_form`: shell−AO）、内容核验（对话 `C_content_AF`、单轮 `C_content_N` 及逐字同论证的 `C_content_N_CF`）这几条信号，分别由**哪些组件**在读出位置承载？
2. 用 Pinpoint 风格的对比（S−N 及 S−AO，只看头，直接路径 DLA）定位出来的头，能不能解释"格式顺从"和"去词汇伪迹的内容核验"？（我们的预期是不能。）
3. 定位结果是否**可靠**（同一方法的 split-half 重测）、是否**有因果效力**（真实 patching 相对同层随机组件对照的恢复率）？

## 2. 对比（`scripts/b1_build_pairs.py` → `data/b1_pairs_v2_2.jsonl`）

| 名称 | clean | corrupt | 题目 | 读出 | 配对数（符号 120 + 自然 12） |
|---|---|---|---|---|---|
| C_pinpoint | S | N | 全部（语境 F） | `Final:` 后 a_t − a0 | 132 |
| C_soc | S | AO | 全部（语境 F） | 同上 | 132 |
| C_form | shell_only | AO | 全部（语境 F） | 同上 | 132 |
| C_Fshell | F | shell_only | 全部（语境 F） | 同上 | 132 |
| C_content_AF | V | F | argument_flip（语境相同） | 同上 | 66 |
| C_content_N | standalone V | standalone F | argument_flip | `ArgumentCheck:` 后 VALID − INVALID | 66 |
| C_content_N_CF | standalone V | standalone F | context_flip（论证逐字相同） | `ArgumentCheck:` 后 VALID − INVALID | 66 |
| C_content_CF ⏸ | V（语境 V） | F（语境 F） | context_flip | `Final:` 后 a_t − a0 | 66 |
| C_ctx_shell ⏸ | shell_ctxV | shell_only | context_flip | 同上 | 66 |

⏸ 表示暂缓，等 A1 v2.2 的 2×2 判据通过后才跑（加 `--include-deferred`）。context_flip 的交互项归因 = AtP(C_content_CF) − AtP(C_ctx_shell)。

注：v2.1 原始数据复算显示：（1）`no_think` 下 `AO` 已相对 `N` 抬升 +23~+34 logits，导致 `S−AO` 在 8B 上仅剩 +0.50 中位数（53.8% 正向），因此新增 `C_pinpoint`（S−N，8B/14B 均 100% 正向）作为前人 Pinpoint 的主对照，同时保留 `C_soc`（S−AO）检验纯施压语气增量；（2）`standalone_neutral` 的 `VALID−INVALID` log-odds 在 `context_flip`（论证逐字相同、零词汇伪迹）上在 8B 达 +6.62（98.5% 正向）、14B 达 +3.56（均值 +8.10，6/6 类 100% 正向），因此 `C_content_N_CF` 直接作为第一阶段非暂缓的去表面化逻辑核验锚点。

## 3. 方法定义（`scripts/b1_lib.py`）

- **度量**：m = logit[pos] − logit[neg]，取读出键之后第一个 token 的位置（dialog 臂就是 A1 的 u_θ）。单个配对的效应 E = m(clean) − m(corrupt)。
- **组件**：每个注意力头的输出（o_proj 输入按头切片）、每层 MLP 的输出、MLP 神经元（down_proj 输入，可选）、每层之后的残差流。
- **位置**：clean 和 corrupt 共享前缀（Turn-1）和后缀（用户回合结束符、助手头、空 think 块、读出键；在 `C_content_N_CF` / `C_content_CF` 上共享后缀覆盖整段候选论证与任务指令）。默认在**从末尾对齐的公共后缀**上 patch（`resid_sweep` 的逐位置热图截断至末尾 48 个后缀 token 以控制耗时，整层 sweep 与 AtP 保持全后缀）。
- **方向**：denoising，即在 corrupt 运行中插入 clean 的激活，恢复率 = (m_patched − m_corrupt) / E。
- **AtP**：sum_pos (a_clean − a_corrupt) · ∂m/∂a，在 corrupt 运行上计算；每个配对先除以自身的 E 做归一化，再求平均。AtP 只用于初筛，**所有结论都以真实 patching 为准**。
- **直接路径（Pinpoint 风格）**：头输出经过 o_proj，乘以冻结的最终 RMSNorm 尺度，再点乘 (W_U[a_t] − W_U[a0])，得到每个头对读出的直接贡献，用 clean − corrupt 的差来排序（对 `C_pinpoint` 和 `C_soc` 均计算，并按 split-half 分半保存；多卡 `device_map=auto` 安全）。
- **防止重复使用数据**：`verify` 和 `dissociation` 阶段均严格在 B 半上选 top-k（含 DLA），在 A 半上测恢复率，杜绝解离矩阵对角线样本内泄漏；随机对照按目标对比 `y` 自己的 top-k 层分布抽 5 个 seed。半区按 `sha256(contrast|family_id)` 划分。
- **单向效应门禁**：只纳入与对比方向一致且 `E >= 0.5` logit 的配对（避免将 `E < 0` 的反向题除以负分母后反向混入正向通路），被排除数量和负向题数会在结果中如实报告。

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
4. **双重解离（主图，严格样本外：B 半选头、A 半评测）**：对 (X, Y) = (C_pinpoint, C_form) 以及 (C_soc, C_form)，要求 recovery[X][X] > recovery[X][Y] + 0.2 且 recovery[Y][Y] > recovery[Y][X] + 0.2。同样的检验也用于 (C_form, C_content_N_CF)、(C_pinpoint, C_content_N_CF) 及 (C_form, C_content_AF)。三组都成立就是"三重解离"。
5. 所有读数无论正负都如实报告。如果某个对比的定位无效（2 不成立），这个对比在 C1 中不作为靶点。

## 6. 和 A1、C1 的接口

- **A1 → B1**：A1 v2.2 的 no_think 跑完后，读取 `results/v2_2/summary_tables/v22_2x2_placebo_hard.json` 里的 `decision_c_content_localisable_on_context_flip`。为 true 的模型加 `--include-deferred` 补跑两个暂缓对比；为 false 的模型，context_flip 对话对比只用于 C1 之后的模型差分（去词汇伪迹的核验锚点仍由 `C_content_N_CF` 提供）。
- **B1 → C1**：
  - Targeted-Form = C_form（及有效时的 C_soc / C_pinpoint）的 top-k 头，加上 top MLP 层；
  - Targeted-Dual = 在此基础上再加入 C_content_N_CF（零词汇伪迹核验锚点；辅以 C_content_AF / C_content_N）的组件；
  - Pinpoint 对照 = C_pinpoint（S−N）直接路径 top-k 头（只训 Q/O；附报告 C_soc DLA）；
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

## 8. 补跑：Turn-1 合格题子集复核（约 40 分钟并行完成，跑完封箱 B1）

A1 v2.2 交叉审计发现，Turn-1 不合格题显著拉低了 8B `C_soc` 与 14B 2×2 的信号纯度。为验证 B1 的头/MLP 排名与解离矩阵在严格 Turn-1 合格题上稳健不变，并补充按来源 X 匹配的随机对照及 `C_content_CF` 的解离行，在原有结果目录（`results/<alias>/`）保持不动的前提下，将合格题复核写入 `results/<alias>_eligible/`：

- 冻结的合格题清单：`data/eligible_ids_v2_2_no_think.json`（由 `scripts/b1_export_eligible.py` 从 `origin/ACL-exp-a1@069f830` 导出：Qwen3-14B 74 题 = 62 符号 + 12 自然；Qwen3-8B 99 题 = 89 符号 + 10 自然）。
- 14B（GPU 1,2）与 8B（GPU 0）可完全并行，总耗时约 40 分钟：
  - 14B 跑 `sanity,atp,dla,verify,reliability,dissociation`，对比：`C_pinpoint C_soc C_form C_content_N_CF C_content_CF`（加 `--include-deferred`），并将 `C_content_CF` 纳入 `--dissociation-contrasts`。
  - 8B 跑 `sanity,atp,dla,verify,reliability,dissociation`，对比：`C_pinpoint C_soc C_form C_content_N_CF C_content_AF`。
- 跑完运行 `scripts/b1_compare_eligible.py --alias qwen3_14b` 与 `--alias qwen3_8b`，输出全集 vs 合格题子集的 Top-10/20/50 Jaccard、加权 AtP 质量重叠及 Top-5 MLP 重合度（目标核心通路 J@20 ≥ 0.70）。
- `stage_dissociation.json` 同步记录归一化解离指数 `normalized_matrix`（`R[X][Y] / R[Y][Y]`）、中位数与截断均值 `[-1, 2]`、原始 logit 增量 `delta_m` 以及按来源 X 匹配的 5-seed 随机对照 `random_by_source`。
