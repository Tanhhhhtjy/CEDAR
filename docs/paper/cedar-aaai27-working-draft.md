# CEDAR AAAI-27 Working Draft / CEDAR AAAI-27 论文工作稿 v0.4

> This bilingual working draft mirrors the manuscript structure in Overleaf. The English submission source is `main.tex`; the Chinese review companion is `main_zh.tex`.
>
> **中文：** 本双语工作稿与 Overleaf 论文结构保持一致。英文投稿源为 `main.tex`，中文审阅稿为 `main_zh.tex`。
>
> Fixed drafting rule: write in completed-study prose. Missing configurations, values, analyses, or result-dependent conclusions occupy narrow `TBD` slots; the paper does not tell the reader that an experiment is unfinished.
>
> **中文：** 固定写作规则：正文始终采用研究已经完整执行的成稿口吻。尚缺的配置、数值、分析或依赖结果的结论只放入窄范围 `TBD` 槽位；正文不向读者说明某项实验尚未完成。

## Provisional Title / 暂定标题

**An Axis Is Not a Selective Controller: Auditing Epistemic Selectivity under Assistant-Axis Capping**

**表征轴不等同于选择性控制器：Assistant-Axis 钳制的认识论选择性审计**

## Abstract / 摘要

Activation-space interventions can suppress unwanted language-model behavior, but standard evaluations reveal little about their effect on sensitivity to the grounds for revising an answer. We introduce CEDAR, a reason-sensitive audit framework comprising a solver-certified paired Evidence dataset that varies warrant validity and a separately analyzed Social protocol that isolates social framing from target exposure. On Qwen3-32B, the released Assistant-Axis p25 cap increased the pooled non-initial-versus-initial margin by $+3.130$ logits (95% family-cluster CI $[1.834,4.486]$), with positive shifts even in controls that concealed the target. In the development assay, target following rose by $20.8$ and $16.7$ percentage points under valid and single-fault arguments, leaving no stable improvement in valid--fallacious selectivity; a new-salt instantiation retained a $+1.538$-logit aggregate shift but showed different arm-level answer changes. The Social audit likewise separated continuous from discrete effects: capping changed the confident-endorsement-minus-target-exposure direct-margin contrast by $-0.786$ logits, while frozen free-generation SocialExcess was zero. Fixed-trace replay produced positive continuous shifts under both trace-source conditions; none of the 192 paired replays changed the four-option argmax. *[TBD-XM-ABSTRACT]* Across these assays, Assistant-Axis capping repeatedly altered answer geometry without a stable gain in reason selectivity, motivating direct reason-sensitive evaluation of activation-space control.

**中文：** 激活空间干预可以抑制不期望的语言模型行为，但标准评测很少说明此类干预如何改变模型对改答依据的敏感性。我们提出 CEDAR，一套理由敏感的审计框架；其中经求解器认证的配对 Evidence 数据集改变论证是否成立，单独分析的 Social 协议则隔离社会措辞相对于纯目标暴露的增量。在 Qwen3-32B 上，公开的 Assistant-Axis p25 钳制使汇总的非初始选项相对初始答案 margin 增加 $+3.130$ logits（95% family-cluster CI $[1.834,4.486]$），且隐藏目标的对照条件也呈现正向位移。在 development assay 中，有效论证与 single-fault 谬误下的目标跟随分别增加 $20.8$ 和 $16.7$ 个百分点，valid--fallacious 选择性未出现稳定改善；new-salt 实例保留了 $+1.538$ logits 的总体位移，但各条件的答案改变有所不同。Social 审计同样显示连续与离散读出分离：钳制使“确信背书减目标暴露”的直接 margin 对比变化 $-0.786$ logits，而冻结自由生成读出中的 SocialExcess 为零。固定轨迹重放在两种轨迹来源下均产生正向连续位移；192 组配对重放均未改变四选项 argmax。*[TBD-XM-ABSTRACT]* 综合这些评测，Assistant-Axis 钳制反复改变答案几何，却没有带来稳定的理由选择性增益；激活空间控制因此需要直接接受理由敏感评测。

## 1. Introduction / 引言

Runtime activation interventions can change language-model behavior without retraining, providing lightweight control over refusal, truthfulness, and persona at deployment time (Turner et al., 2023; Arditi et al., 2024; Lu et al., 2026). They are commonly evaluated with a target-behavior metric and broad capability tests or checks on benign inputs. These tests provide limited evidence about how an intervention changes the model's response to the grounds offered for revising an answer. Static accuracy can remain stable even when the intervention changes which arguments or social signals move the model away from its current answer.

**中文：** 推理时激活干预无需重新训练即可改变模型行为，为部署阶段调节拒答、真实性与 persona 提供了轻量手段（Turner 等，2023；Arditi 等，2024；Lu 等，2026）。现有工作通常同时报告目标行为指标与一般能力测试或良性输入检查。这些评测对干预如何改变模型对改答理由的响应说明有限。即使静态准确率保持不变，干预仍可能改变哪些论证或社会信号会推动模型放弃当前答案。

The same proposed answer may be accompanied by a valid derivation, a plausible fallacy, social endorsement, or simple repetition. These cases call for different responses: sufficient evidence should support revision, whereas an invalid inference, target exposure, or confidence alone should carry less weight. We call this differential response *epistemic selectivity*. It is a dynamic property of revision, so neither a static capability score nor an overall flip rate identifies it (Figure 1).

**中文：** 同一候选答案可能伴随有效推导、看似合理的谬误、社会性背书或简单重复。这些情形要求不同的响应：充分证据应支持修正答案，无效推理、纯目标暴露或单纯的自信措辞应当具有较小权重。本文将这种差异化响应称为*认识论选择性*。它是改答过程中的动态属性，单纯的静态能力分数或总体翻转率都无法识别它（图 1）。

We introduce CEDAR, a reason-sensitive audit framework with two paired modules that are constructed and analyzed separately. CEDAR-Evidence is a solver-certified paired dataset that varies warrant validity; CEDAR-Social is a natural-task protocol that isolates social framing from target exposure. Within each module, the base task, initial answer, proposed target, and output contract remain fixed. Four-way outcomes are paired with continuous margins that separate general movement away from the initial answer from target-specific attraction. We apply CEDAR to the released Assistant-Axis p25 cap on Qwen3-32B and run a matched forced-choice CEDAR-Evidence comparison of the released Qwen and Llama model--configuration pairs; CEDAR-Social and the diagnostic analyses remain Qwen-only.

**中文：** 我们提出 CEDAR，一套由两个分别构造和分析的配对模块组成的理由敏感审计框架。CEDAR-Evidence 是改变理由有效性的经求解器认证配对数据集；CEDAR-Social 是在自然任务中隔离社会措辞与纯目标暴露的评测协议。每个模块内部均固定基础任务、初始答案、候选目标和输出约定。四态行为结果与连续 margin 联合使用，以区分离开初始答案的整体位移和目标特异吸引。我们用 CEDAR 审计 Qwen3-32B 的公开 Assistant-Axis p25 钳制，并在匹配的 forced-choice 制度下，对公开的 Qwen 与 Llama 模型—配置组合进行 CEDAR-Evidence 比较；CEDAR-Social 与诊断分析仍仅在 Qwen 上开展。

In the Qwen3-32B development CEDAR-Evidence assay, full-p25 capping increased the pooled non-initial-versus-initial margin, $m_{\mathrm{general}}$, by $3.130$ logits (95% family-cluster CI $[1.834,4.486]$); all four arm estimates were positive, including neutral and evidence-free recheck ($N/R$) controls that concealed the target. Target following increased by $20.8$ and $16.7$ percentage points under valid ($V$) and single-fault fallacious ($F$) arguments, while the change in $V$-minus-$F$ selectivity remained imprecise. A new-salt re-instantiation showed a $+1.538$-logit complete-case aggregate shift, while its arm-level answer changes differed. In CEDAR-Social, capping changed the confident-endorsement-minus-target-exposure ($S-AO$) direct-margin contrast by $-0.786$ logits; both component shifts were positive, and the corresponding SocialExcess was zero in frozen free-generation readouts. The two Evidence banks shared the positive aggregate $m_{\mathrm{general}}$ direction; across Evidence and Social, continuous effects were accompanied by different selected-answer outcomes.

**中文：** 在 Qwen3-32B 的 development CEDAR-Evidence assay 中，full-p25 钳制使汇总 $m_{\mathrm{general}}$（非初始选项相对初始答案的 margin）增加 $3.130$ logits（95% family-cluster CI $[1.834,4.486]$）；包括隐藏目标的中性延续与无证据复核（$N/R$）对照在内，四个条件的估计均为正。有效论证（$V$）与 single-fault 谬误论证（$F$）下的目标跟随分别增加 $20.8$ 和 $16.7$ 个百分点，$V$ 减 $F$ 的选择性变化仍不精确。new-salt 重新实例化的 complete-case 总体位移为 $+1.538$ logits，但各条件的离散改答不同。在 CEDAR-Social 中，钳制使“确信背书减目标暴露”（$S-AO$）的直接 margin 对比变化 $-0.786$ logits；两个分量位移均为正，冻结自由生成读出中相应的 SocialExcess 为零。两个 Evidence bank 的汇总 $m_{\mathrm{general}}$ 均为正向；在 Evidence 与 Social 评测中，连续效应对应的最终答案结果并不相同。

Fixed-trace replay produced positive continuous shifts for traces generated without and with capping ($+2.078$ and $+0.700$ logits, respectively); none of 192 paired replays changed the four-option argmax. *[TBD-XM-INTRO-ROUTE]* This work makes three contributions. (i) We operationalize epistemic selectivity as an audit target for activation interventions and introduce CEDAR, which combines a solver-certified Evidence dataset with a separately analyzed Social protocol and pairs continuous with discrete readouts. (ii) On the released Qwen Assistant-Axis p25 cap, we document a broad compression of initial-answer advantage, heterogeneous discrete consequences, and no stable improvement in valid-versus-fallacious selectivity. (iii) We diagnose the shift with phase-restricted interventions, an independent validity assay, trajectory comparisons, and fixed-trace replay; the replay result shows that the shift persists with identical visible reasoning.

**中文：** 固定轨迹重放在由未钳制与 p25 生成的轨迹上均产生正向连续位移（分别为 $+2.078$ 与 $+0.700$ logits）；192 组配对重放均未改变四选项 argmax。*[TBD-XM-INTRO-ROUTE]* 本文有三项贡献。（i）我们将认识论选择性操作化为激活干预的审计目标，并提出 CEDAR，将经求解器认证的 Evidence 数据集与单独分析的 Social 协议结合，同时报告连续与离散读出。（ii）在公开的 Qwen Assistant-Axis p25 钳制中，我们记录了初始答案优势的广泛压缩、随评测变化的离散后果，以及有效与谬误论证之间的选择性未出现稳定改善。（iii）我们通过分阶段干预、独立有效性评测、轨迹比较和固定轨迹重放诊断该位移；固定轨迹结果表明，该位移在可见推理完全一致时仍然存在。

## 2. Related Work / 相关工作

**Inference-Time Activation Control. / 推理时激活控制。**

Post-training pipelines such as supervised fine-tuning and preference optimization remain a common route to aligning model behavior (Ouyang et al. 2022; Rafailov et al. 2023). Activation-level methods provide a complementary control surface at inference time, modifying internal states without updating base-model weights. Representation engineering treats population-level activation patterns as objects that can be read and manipulated (Zou et al. 2023). Existing methods span additive steering along contrastive directions (Turner et al. 2023; Rimsky et al. 2024), localized interventions on attention heads (Li et al. 2023), directional ablation (Arditi et al. 2024), and context-conditioned control (Lee et al. 2025).<br>
**中文：** 监督微调和偏好优化等训练后方法，是使模型行为符合预期的常用途径（Ouyang et al. 2022；Rafailov et al. 2023）。与此同时，激活层方法提供了互补的推理期控制途径，可在不更新基础模型权重的情况下直接调整内部状态。表征工程将样本总体上的激活模式作为可读取和操控的对象（Zou et al. 2023）。现有方法包括沿对比方向进行加性 steering（Turner et al. 2023；Rimsky et al. 2024）、对注意力头实施局部干预（Li et al. 2023）、方向消融（Arditi et al. 2024），以及依赖上下文的条件控制（Lee et al. 2025）。

Threshold capping defines a state-dependent operator: the controlled projection remains unchanged below a calibrated boundary, and the cap removes only the excess above it. The Assistant Axis work extracts model-specific directions associated with the default assistant persona and pairs them with one-sided, percentile-calibrated caps (Lu et al. 2026). CEDAR audits the released model-specific caps as instances of this intervention, including their effects on responses to social pressure and corrections supported by sufficient evidence.<br>
**中文：** 阈值钳制定义了一种状态依赖算子：受控投影低于校准边界时保持不变，超过边界时仅移除超出的部分。Assistant Axis 工作为不同模型提取与默认助手角色相关的方向，并为其配置按分位数校准的单侧 cap（Lu et al. 2026）。CEDAR 将公开的模型专属 cap 作为这一干预的实例，审计其对社会压力响应和有充分依据的修正所产生的影响。

**Sycophancy and Answer Revision. / 谄媚与答案修正。**

Social influence makes the grounds for answer revision difficult to infer from behavior alone. Language models often shift answers toward a user's stated view, including when that view conflicts with the correct answer (Sharma et al. 2024). SycoBench-600 measures susceptibility to doubt, authority, and explicit wrong suggestions alongside correction selectivity, showing that willingness to update does not by itself establish selective updating (Sinha 2026). Mechanistic work traces how incorrect user opinions alter answer representations across layers before affecting the final prediction (Wang et al. 2026). Intervention evaluation therefore requires explicit controls for the grounds of revision. This produces two comparison problems: isolating social framing beyond target exposure and comparing reasons that warrant revision with matched reasons that do not.<br>
**中文：** 社会影响使得仅凭外显行为很难判断模型改答的依据。语言模型常会把答案移向用户表达的观点，包括该观点与正确答案冲突时（Sharma et al. 2024）。SycoBench-600 同时测量模型对怀疑、权威和明确错误建议的易感性及其修正选择性，并表明愿意改答本身不足以证明更新具有选择性（Sinha 2026）。机制研究进一步追踪错误用户意见如何在最终预测改变前逐层改变答案表征（Wang et al. 2026）。因此，干预评测需要明确控制促使模型改答的依据，包括两类分别分析的比较：隔离超出 target exposure 的社会性措辞影响，以及比较足以支持修正的理由与匹配的无效理由。

**Reason-Sensitive Evaluation and Intervention Side Effects. / 理由敏感评测与干预副作用。**

Belief-revision benchmarks provide criteria for judging such answer changes. Belief-R treats answer revision as a two-sided problem: models should incorporate premises that change the conclusion and preserve conclusions that remain supported (Wilie et al. 2024). ProverGen combines language-model generation with symbolic proving to construct problems with checkable first-order structure (Qi et al. 2025), while LoFa evaluates robustness to fallacious persuasion after accounting for limitations in base knowledge (Shen et al. 2026). These resources support controlled evaluation of reason quality. A parallel line of work evaluates intervention side effects by pairing target behavior with retention criteria. KL-then-steer limits distributional change on benign inputs while maintaining additive-steering efficacy (Stickland et al. 2024); dual-stance evaluation shows that centroid-difference anti-sycophancy steering can also reduce agreement with factually correct statements (Buchan 2026). An audit of state-dependent capping therefore needs estimates of the intervention's effect on warranted-versus-unwarranted revision and on the additional target following attributable to social framing beyond target exposure.<br>
**中文：** 信念修正 benchmark 为判断这类答案变化提供了判据。Belief-R 将答案修正视为一个双向问题：模型应当吸收足以改变结论的新前提，同时维持仍有充分支持的原结论（Wilie et al. 2024）。ProverGen 将语言模型生成与符号证明结合，构造具有可核验一阶逻辑结构的题目（Qi et al. 2025）；LoFa 则在控制基础知识不足后评估模型抵抗谬误说服的能力（Shen et al. 2026）。这些资源支持对理由质量进行受控评测。与之并行，另一类工作借助保留指标评估干预副作用。KL-then-steer 在维持加性 steering 效果的同时限制良性输入上的分布变化（Stickland et al. 2024）；dual-stance evaluation 表明，基于 centroid difference 的反谄媚 steering 也可能降低模型对事实正确陈述的赞同（Buchan 2026）。因此，对状态依赖 capping 的审计需要估计干预对有据与无据修正差异的影响，以及它对社会性措辞在 target exposure 之上所引起的额外 target following 的影响。

## 3. Designing a Reason-Selectivity Audit / 设计理由选择性审计

Capping may alter warrant discrimination, the general criterion for leaving an answer, or sensitivity to social framing. CEDAR separates these accounts with two independently analyzed paired contrasts that vary warrant validity, target exposure, and social framing while holding the task, initial answer, and candidate target fixed. Both categorical transitions and continuous answer geometry are recorded.

**中文：** 钳制可能改变理由有效性判别、离开既有答案的一般倾向，或对社会措辞的敏感程度。CEDAR 用两组独立分析的配对对照区分这些作用；对照在固定任务、初始答案和候选目标的同时，分别改变理由有效性、目标暴露和社会措辞，并同时记录类别迁移与连续答案几何。

### 3.1 Paired Revision Setting / 配对改答设定

Each dialogue supplies a base question, an initial answer $a_0$, and a paired alternative $a_t$. A parseable final option different from $a_0$ is an *answer change*; choosing $a_t$ is *target following*, and is warranted only when the follow-up sufficiently supports it. In CEDAR-Evidence, $N/R$ retain $a_t$ for analysis but hide it from the model, whereas $V/F$ present the same candidate edit with valid or fallacious justification. CEDAR-Social presents $a_t$ in both $AO/S$ and varies social framing. Thus $V-F$ isolates warrant validity and $S-AO$ isolates social framing, each at constant target exposure (Figure 1).

**中文：** 每段对话包含基础问题、初始答案 $a_0$ 和配对替代项 $a_t$。可解析的最终选项偏离 $a_0$ 时记为*答案改变*；选择 $a_t$ 时记为*目标跟随*，仅当后续信息充分支持 $a_t$ 时，这一改变才具有充分依据。CEDAR-Evidence 在分析中为 $N/R$ 保留 $a_t$，但不向模型展示；$V/F$ 则呈现同一候选修改，并分别附加有效论证或谬误论证。CEDAR-Social 的 $AO/S$ 均展示 $a_t$，只改变社会措辞。因此，$V-F$ 在固定目标暴露时隔离理由有效性，$S-AO$ 在同样条件下隔离社会措辞（图 1）。

**Figure 1 / 图 1：** CEDAR paired, reason-sensitive audit design / CEDAR 成对理由敏感审计设计。

**Caption / 图注：** **CEDAR's paired reason-selectivity audit.** CEDAR-Evidence fixes the task, $a_0$, and $a_t$: $N/R$ hide the target, while $V/F$ present the same edit with a valid or single-fault warrant. CEDAR-Social fixes a natural-task answer pair: $AO$ exposes $a_t$, and $S$ adds confident social framing. Capped--uncapped comparisons distinguish target movement from warranted movement; $S-AO$ isolates social framing beyond target exposure. Templates and construction details appear in the supplementary material.

**中文图注：** **CEDAR 的配对理由选择性审计。** CEDAR-Evidence 固定任务、$a_0$ 与 $a_t$：$N/R$ 隐藏目标，$V/F$ 为同一候选修改提供有效或 single-fault 理由。CEDAR-Social 固定自然任务中的答案对；$AO$ 展示 $a_t$，$S$ 再加入确信式社会措辞。钳制前后对比区分一般目标位移与有据位移，$S-AO$ 则隔离目标暴露之外的社会措辞效应。模板与构造细节见补充材料。

### 3.2 Certified Valid--Fallacious Contrast / 经认证的有效—谬误对照

Each CEDAR-Evidence item specifies a finite, function-free Horn theory with explicit negation and a four-way query. The base has unique answer $a_0$; a deterministic atomic edit yields $a_t$. $V$ gives a sufficient justification for this edit, while its paired $F$ argument contains one fatal inference. A Python fixed-point engine and Prover9 must agree on the labels and support relation; every $F$ item also has a Mace4 countermodel, with all remaining steps supported. Certification makes warrant validity checkable before evaluation and independent of the evaluated model and argument author.

**中文：** 每个 CEDAR-Evidence 样本给出一个带显式否定、不含函数符号的有限 Horn 理论和一道四分类查询。基础理论的唯一答案为 $a_0$；一次确定性的原子修改产生 $a_t$。$V$ 提供足以支持该修改的论证，其配对的 $F$ 论证包含一处致命推理错误。Python 不动点推理器与 Prover9 必须就标签和支持关系达成一致；每个 $F$ 样本还带有 Mace4 反模型，其余推理步骤均有支持。由此，理由有效性在评测前即可核验，并独立于被测模型和论证撰写者。

Each bank contains 24 bases: easy and medium variants of 12 structural families spanning six error classes. A new-salt bank re-instantiates the same grammar and family structures and is analyzed separately. Difficulty varies through proof structure and distractors while wording remains simple. Conditions are paired within family, the independent sampling unit. Certificates, examples, and review records appear in the supplementary material.

**中文：** 每个 bank 包含 24 个基础样本，由 12 个结构族的简单与中等变体组成，覆盖六类错误。new-salt bank 使用同一语法和结构族重新实例化，并单独分析。难度由证明结构和干扰项调节，措辞保持简明。条件在结构族内配对，结构族作为独立统计单位。证书、示例和审核记录见补充材料。

### 3.3 Exposure-Controlled Social Contrast / 控制目标暴露的社会对照

CEDAR-Social covers the 600 multiple-choice questions in SycoBench-600. The gold option is $a_0$ and another option is $a_t$. Both follow-ups expose $a_t$: $AO$ adds no speaker, evidence, authority, reward, or consensus cue, while $S$ adds first-person confidence and a request to reconsider without task evidence. The paired $S-AO$ contrast therefore measures social framing net of target exposure.

**中文：** CEDAR-Social 覆盖 SycoBench-600 的 600 道多选题。标准答案为 $a_0$，另一个选项为 $a_t$。两种后续输入都展示 $a_t$：$AO$ 不加入说话者、证据、权威、奖励或共识线索；$S$ 加入第一人称确信和重新考虑的请求，但不提供任务证据。配对的 $S-AO$ 对比测量扣除目标暴露后的社会措辞效应。

Four minimally edited template pairs are balanced across domain and difficulty. Automated checks verify pairing and frozen-tokenizer length bounds. Two researchers review all pairs independently, without model outputs; agreement and adjudication are *[TBD-SOCIAL-AUDIT]*. Templates and review records appear in the supplementary material.

**中文：** 四组最小改写模板在领域和难度层内均衡分配。自动检查核验配对一致性及冻结分词器下的长度界。两名研究者在不接触模型输出的情况下独立审核全部配对；一致率与裁决记录为 *[TBD-SOCIAL-AUDIT]*。模板和审核材料见补充材料。

### 3.4 Intervention and Model Scope / 干预与模型范围

At layer $\ell$, let $\mathbf{v}_{\ell}$ be the unit direction stored in the released model-specific configuration and $\tau_{\ell,c}$ its threshold. The stored direction is oriented opposite to the human-readable Assistant Axis, so the released minimum cap is an upper cap in this coordinate:

$$
\mathbf{h}'_{\ell,t}=\mathbf{h}_{\ell,t}-\mathbf{v}_{\ell}
\left[\langle\mathbf{h}_{\ell,t},\mathbf{v}_{\ell}\rangle-\tau_{\ell,c}\right]_{+},
$$

where $[z]_{+}=\max(z,0)$; only projection above the threshold is removed.

**中文：** 令 $\mathbf{v}_{\ell}$ 表示公开模型专属配置在第 $\ell$ 层存储的单位方向，$\tau_{\ell,c}$ 表示对应阈值。存储方向与可读的 Assistant Axis 方向相反，因此公开的 minimum cap 在该坐标中为 upper cap。上式中 $[z]_{+}=\max(z,0)$；算子只移除超过阈值的投影。

Qwen3-32B is primary and uses `layers_46:54-p0.25` with reasoning-enabled greedy decoding. The matched transport audit uses Qwen3-32B and Llama-3.3-70B-Instruct on the same 24-base Evidence bank under deterministic reasoning-disabled forced choice; Llama uses `layers_56:72-p0.25`. Each model retains its released direction and configuration, so model identity and configuration are not separately identified. CEDAR-Social and all diagnostics remain Qwen-only. Revisions, decoding settings, and cross-regime sensitivities appear in the supplementary material.

**中文：** 主要模型 Qwen3-32B 使用 `layers_46:54-p0.25` 和开启推理的 greedy 解码。匹配迁移评测在同一个 24 题 Evidence bank 上比较 Qwen3-32B 与 Llama-3.3-70B-Instruct，统一采用关闭推理的确定性 forced choice；Llama 使用 `layers_56:72-p0.25`。每个模型保留其公开方向与配置，模型身份和配置作用因而无法分离。CEDAR-Social 与全部诊断仅在 Qwen 上运行。模型 revision、解码设置与跨制度敏感性见补充材料。

### 3.5 Behavioral Outcomes and Answer Geometry / 行为结果与答案几何

We record both continuous option geometry and four final outcomes: initial, paired target, another option, or invalid/abstaining. Let $z(a)$ be the next-token logit of option label $a$ at the fixed answer position:

$$
\begin{aligned}
m_{\mathrm{general}} &= \tfrac{1}{3}\!\sum_{a\neq a_0}z(a)-z(a_0),\\
m_{\mathrm{target}} &= z(a_t)-z(a_0),\\
m_{\mathrm{background}} &= \tfrac{1}{2}\!\sum_{a\notin\{a_0,a_t\}}z(a)-z(a_0),\\
m_{\mathrm{specific}} &= z(a_t)-\tfrac{1}{2}\!\sum_{a\notin\{a_0,a_t\}}z(a).
\end{aligned}
$$

$m_{\mathrm{general}}$ measures movement away from $a_0$ across all alternatives; $m_{\mathrm{specific}}$ compares the paired target with the two backgrounds. Broad compression of initial-answer advantage raises the former without requiring the latter, whereas target-specific attraction raises $m_{\mathrm{specific}}$. Item-level transitions show when either continuous shift crosses the choice boundary.

**中文：** 我们记录连续选项几何，以及初始答案、配对目标、其他选项和无效输出/弃答四类最终结果。令 $z(a)$ 表示固定答案位置上选项标签 $a$ 的 next-token logit，并按上式定义四项 margin。$m_{\mathrm{general}}$ 衡量相对于 $a_0$、共同朝所有替代项的位移；$m_{\mathrm{specific}}$ 比较配对目标与两个背景选项。初始答案优势的广泛压缩会提高前者，却不要求后者提高；目标特异吸引则会提高 $m_{\mathrm{specific}}$。逐题迁移进一步显示连续位移何时跨过选择边界。

For outcome $Y$, follow-up condition $x$, and cap configuration $c$, let $\delta_x^Y(c)=\mathbb{E}[Y\mid x,c]-\mathbb{E}[Y\mid x,0]$, where $0$ denotes uncapped inference. The Evidence estimands are

$$
\begin{aligned}
\Delta_{\mathrm{broad}}(c)
&=\tfrac{1}{4}\!\sum_{x\in\{N,R,V,F\}}\delta_x^{m_{\mathrm{general}}}(c),\\
\Delta_{\mathrm{sel}}^{(m)}(c)
&=\delta_V^{m_{\mathrm{target}}}(c)-\delta_F^{m_{\mathrm{target}}}(c),\\
\Delta_{\mathrm{sel}}^{(p)}(c)
&=[p_V(c)-p_V(0)]-[p_F(c)-p_F(0)],
\end{aligned}
$$

where $p_x(c)$ is target-following probability. The pooled change in $m_{\mathrm{specific}}$ is a prespecified target-specificity check; arm decompositions are post hoc.

**中文：** 对于结果 $Y$、后续条件 $x$ 和钳制配置 $c$，$\delta_x^Y(c)$ 表示该条件相对于未钳制推理的效应。$\Delta_{\mathrm{broad}}$ 汇总四个 Evidence 条件中的 $m_{\mathrm{general}}$ 变化；$\Delta_{\mathrm{sel}}^{(m)}$ 和 $\Delta_{\mathrm{sel}}^{(p)}$ 分别从连续 margin 与目标跟随概率衡量模型对有效理由和谬误理由的差异化响应。汇总的 $m_{\mathrm{specific}}$ 变化是预设的目标特异性检查；各条件分解属于 post hoc 分析。

CEDAR-Social uses

$$
\Delta_{\mathrm{social}}(c)
=\delta_S^{m_{\mathrm{target}}}(c)-\delta_{AO}^{m_{\mathrm{target}}}(c)
=[S-AO]_c-[S-AO]_0.
$$

A negative value means that capping reduces target support associated with social framing beyond target exposure; $S$ and $AO$ component effects accompany the interaction. Direct margins are primary. On a frozen 240-question subset, $\mathrm{SocialExcess}_q=[q_S(p25)-q_{AO}(p25)]-[q_S(0)-q_{AO}(0)]$ is the discrete analogue under strict-option or completion parsing.

**中文：** CEDAR-Social 使用 $\Delta_{\mathrm{social}}$ 衡量钳制对 $S-AO$ 差异的影响。负值表示钳制削弱社会措辞在目标暴露之外带来的目标支持；交互项与 $S$、$AO$ 分量效应一并报告。直接 margin 是主要读数。在冻结的 240 题子集上，$\mathrm{SocialExcess}_q=[q_S(p25)-q_{AO}(p25)]-[q_S(0)-q_{AO}(0)]$ 是严格选项或完整回答解析规则 $q$ 下的离散对应量。

### 3.6 Statistical Inference and Diagnostic Checks / 统计推断与诊断检查

CEDAR-Evidence averages paired effects within each of 12 structural families and weights families equally; primary intervals use the frozen family-cluster bootstrap. CEDAR-Social resamples questions with domain/difficulty stratification. All scheduled outputs enter categorical analysis, with unparsable responses coded invalid/abstaining and no content-based reruns. Continuous contrasts disclose paired-logit coverage. Data banks, protocols, and models are estimated separately.

**中文：** CEDAR-Evidence 先在 12 个结构族内平均配对效应，再对结构族等权汇总；主要区间采用冻结的结构族聚类 bootstrap。CEDAR-Social 按领域和难度分层重采样题目。全部计划内输出均进入类别分析，无法解析的回答记为无效输出或弃答，且不按内容选择性重跑。连续对比披露配对 logit 覆盖率；数据 bank、协议和模型分别估计。

The development bank is exploratory. $\Delta_{\mathrm{broad}}$ is primary for new-salt and matched transport, and $\Delta_{\mathrm{social}}$ for CEDAR-Social; matched Qwen anchors Llama. Other analyses are labeled secondary, sensitivity, or post hoc. The four-margin decomposition and decision rules were frozen before capped-model execution. Leave-one-family-out, small-cluster, missing-data, output-integrity, and protocol-amendment details appear in the supplementary material.

**中文：** development bank 用于探索。$\Delta_{\mathrm{broad}}$ 是 new-salt 和匹配迁移评测的主要估计量，$\Delta_{\mathrm{social}}$ 是 CEDAR-Social 的主要估计量；匹配 Qwen 作为 Llama 的参照。其他分析标为次要、敏感性或 post hoc。四项 margin 分解和决策规则在 capped 模型执行前冻结。leave-one-family-out、小聚类、缺失数据、输出完整性和协议调整见补充材料。

## 4. Auditing Assistant-Axis Capping / 审计 Assistant-Axis 钳制

Across Qwen assays, p25 capping shifted continuous margins while selected-answer effects varied by reason condition and readout. We calibrate the released procedures, compare margins with final choices in CEDAR-Evidence and CEDAR-Social, and then test matched Qwen--Llama transport in CEDAR-Evidence.

**中文：** 在各项 Qwen 评测中，p25 钳制改变了连续 margin，而最终答案效应随理由条件和读出方式而异。我们先校准公开流程，再比较 CEDAR-Evidence 与 CEDAR-Social 中的 margin 和最终选择，最后在 CEDAR-Evidence 中检验匹配的 Qwen--Llama 迁移。

### 4.1 Calibrating Steering Direction and Cap Exposure / 校准 Steering 方向与钳制暴露

Signed steering and threshold capping behaved differently even along the released direction. At $\alpha=+200$, steering in the PhilPapers setup used by the source study reproduced its endpoint ($-2.4312$ logits, 95% CI $[-2.6248,-2.2407]$, versus approximately $-2.41$ reported by Kelkar et al. 2026). In this source-style calibration, every evaluation triggered the released Qwen p25 cap at least once; capping nevertheless decreased the continuous sycophancy margin by $0.094$ logits ($[-0.133,-0.054]$) while increasing the discrete sycophancy rate by $1.44$ percentage points ($[0.11,2.78]$; post hoc). We therefore keep continuous and discrete readouts separate; complete dose, exposure, and distributional diagnostics appear in the supplementary material.

**中文：** 即使沿同一公开方向，signed steering 与阈值钳制也呈现不同后果。在 $\alpha=+200$ 时，源工作采用的 PhilPapers 设定复现了其 steering 端点（$-2.4312$ logits，95% CI $[-2.6248,-2.2407]$；Kelkar 等 2026 原报告约为 $-2.41$）。在这项源工作风格的校准中，每道评测都至少触发一次公开 Qwen p25 钳制；连续 sycophancy margin 仍下降 $0.094$ logits（$[-0.133,-0.054]$），离散 sycophancy rate 却增加 $1.44$ 个百分点（$[0.11,2.78]$；post hoc）。因此下文始终分开报告连续与离散读数；完整剂量、暴露和分布诊断见补充材料。

**Figure 2 asset:** `figures/drafts/cedar-figure2-results-bars-v2-paper/result-bars-paper-vector.pdf` (editable SVG and Python source are in the same directory).

**Caption / 图注：** **Answer geometry under model-specific p25 capping.** **(A)** Qwen development-bank changes in $m_{\mathrm{general}}$ across $N/R/V/F$. **(B)** Uncapped and p25 target following for $V/F$. **(C)** Pooled changes in target, background, and target-specific margins. **(D)** Within-model $\Delta m_{\mathrm{general}}$ and $V-F$ contrasts for matched Qwen and Llama CEDAR-Evidence forced-choice audits. Whiskers, where shown, are 95% family-cluster bootstrap intervals over 12 families. Each model uses its released direction and p25 configuration, so panel D compares model--configuration pairs. Complete estimates appear in the supplement.

**中文图注：** **模型专属 p25 钳制下的答案几何。** **(A)** Qwen development bank 中 $N/R/V/F$ 的 $m_{\mathrm{general}}$ 变化。**(B)** $V/F$ 在 uncapped 与 p25 下的目标跟随率。**(C)** 目标、背景及目标特异 margin 的汇总变化。**(D)** 匹配 CEDAR-Evidence forced-choice 评测中 Qwen 与 Llama 各自的 $\Delta m_{\mathrm{general}}$ 和 $V-F$ 对比。图中所示误差线为 12 个结构族上的 95% 聚类 bootstrap 区间。各模型使用自身公开方向与 p25 配置，因此 D 比较的是两个模型—配置组合。完整估计见补充材料。

### 4.2 Broad Geometry and Categorical Consequences / 广泛几何位移及其类别后果

The margins obey

$$
m_{\mathrm{target}}
=m_{\mathrm{general}}+\frac{2}{3}m_{\mathrm{specific}},
\qquad
m_{\mathrm{background}}
=m_{\mathrm{general}}-\frac{1}{3}m_{\mathrm{specific}}.
$$

Thus $m_{\mathrm{general}}$ summarizes average movement across non-initial options, whereas $m_{\mathrm{specific}}$ separates the paired target from the backgrounds. This decomposition distinguishes broad compression of initial-answer advantage from target-specific attraction.

**中文：** 四项 margin 满足上式。因此，$m_{\mathrm{general}}$ 概括各非初始选项的平均位移，$m_{\mathrm{specific}}$ 则分离配对目标与背景选项；二者可区分初始答案优势的广泛压缩和目标特异吸引。

In the exploratory Qwen development bank, p25 increased pooled $m_{\mathrm{general}}$ by $3.130$ logits (95% family-cluster CI $[1.834,4.486]$; Figure 2A); all four arm estimates were positive, including $N/R$, which concealed the target. The pooled target-specificity check was $\Delta m_{\mathrm{specific}}=-0.084$ logits ($[-1.110,0.953]$; Figure 2C), too imprecise for equivalence. All 48 $N/R$ pairs retained $a_0$ under both conditions despite the continuous shift. With the target presented, following increased by $20.83$ and $16.67$ percentage points under $V$ and $F$ (Figure 2B); their capped--uncapped difference was $+4.17$ points ($[-20.83,29.17]$). The displacement was therefore latent in controls and accompanied both warranted and unwarranted updating in the argument arms.

**中文：** 在探索性的 Qwen development bank 中，p25 使汇总 $m_{\mathrm{general}}$ 增加 $3.130$ logits（95% 结构族聚类 CI $[1.834,4.486]$；图 2A）；包括隐藏目标的 $N/R$ 在内，四个条件均为正。汇总目标特异性检查为 $\Delta m_{\mathrm{specific}}=-0.084$ logits（$[-1.110,0.953]$；图 2C），精度不足以支持等效。尽管 48 个 $N/R$ 配对在两种条件下都保持 $a_0$，呈现目标后，$V$ 与 $F$ 的目标跟随分别增加 $20.83$ 和 $16.67$ 个百分点（图 2B），二者的 capped--uncapped 差为 $+4.17$ 个百分点（$[-20.83,29.17]$）。该连续位移在对照条件中未跨越选择边界，在论证条件中则同时伴随有据与无据改答。

### 4.3 New-Salt Re-instantiation and Natural-Task Boundaries / New-Salt 重新实例化与自然任务边界

Paired logits were complete for 95 of 96 new-salt comparisons. Their complete-case aggregate broad shift was $+1.538$ logits (95% family-cluster CI $[0.455,2.741]$), and the adverse value required to reverse its sign exceeded the prespecified tipping-point boundary by *[TBD-TIPPING-POINT-MARGIN]*. Its categorical pattern differed: $V$ target following decreased by $12.5$ percentage points while $F$ increased by $16.7$, yielding a $V-F$ change of $-29.17$ points ($[-58.33,4.17]$). The two banks agreed on broad continuous movement, while their arm-level answer changes diverged.

**中文：** new-salt 的 96 个比较中，95 个具有完整配对 logits；complete-case 总体 broad shift 为 $+1.538$ logits（95% 结构族聚类 CI $[0.455,2.741]$），而推翻其正向符号所需的不利值比预设 tipping-point 边界高出 *[TBD-TIPPING-POINT-MARGIN]*。类别模式则有所不同：$V$ 目标跟随下降 $12.5$ 个百分点，$F$ 增加 $16.7$ 个百分点，$V-F$ 变化为 $-29.17$ 个百分点（$[-58.33,4.17]$）。两个 bank 的广泛连续位移方向一致，各条件的答案变化存在差异。

In the separate 600-item Social audit, the $AO$ and $S$ target margins increased by $+1.268$ and $+0.482$ logits, giving $\Delta_{\mathrm{social}}(p25)=-0.786$ (95% CI $[-0.860,-0.711]$). Both strict-option and completion-based SocialExcess at p25 were zero, and no paired $S$-arm answer migrated. The observed attenuation is confined to the direct-margin effect of social framing beyond target exposure; the free-generation assay showed no corresponding answer migration. The complementary bidirectional SycoBench estimates are reported separately and remain inconclusive under the prespecified mirrored-argmax gate; complete results appear in the supplement.

**中文：** 在独立的 600 题 Social 审计中，$AO$ 与 $S$ 的目标 margin 分别增加 $+1.268$ 和 $+0.482$ logits，得到 $\Delta_{\mathrm{social}}(p25)=-0.786$（95% CI $[-0.860,-0.711]$）。p25 下严格选项解析和完整回答解析的 SocialExcess 均为零，且没有 $S$ 臂配对答案迁移。衰减仅出现在社会措辞相对目标暴露的直接 margin 增量中；自由生成评测未出现对应的答案迁移。互补的双向 SycoBench 估计单独报告，并在预设 mirrored-argmax gate 下仍为 inconclusive；完整结果见补充材料。

### 4.4 Matched CEDAR-Evidence Transport Across Model-Specific Configurations / 跨模型专属配置的匹配 CEDAR-Evidence 迁移

In the matched reasoning-disabled CEDAR-Evidence forced-choice audit, within-model $\Delta m_{\mathrm{general}}$ was *[TBD-XM-Q-BROAD]* for Qwen and *[TBD-XM-L-BROAD]* for Llama (Figure 2D). *[TBD-XM-ROUTE]* Each model used its released direction and p25 configuration; the estimates therefore apply to two model--configuration pairs and do not isolate model identity. Target-specificity, continuous $V-F$, categorical, exposure, and integrity results appear in the supplement.

**中文：** 在关闭推理的匹配 CEDAR-Evidence forced-choice 评测中，Qwen 与 Llama 的模型内 $\Delta m_{\mathrm{general}}$ 分别为 *[TBD-XM-Q-BROAD]* 和 *[TBD-XM-L-BROAD]*（图 2D）。*[TBD-XM-ROUTE]* 每个模型使用自身公开方向与 p25 配置；这些估计适用于两个模型—配置组合，无法单独识别模型身份。目标特异性、连续 $V-F$、类别结果、暴露与完整性分析见补充材料。

## 5. Diagnosing the Shift / 诊断该位移

We combine phase restriction, independent validity judgments, trajectory comparisons, and fixed-trace replay to narrow the source of the observed shift. Downstream representation geometry and full diagnostic tables appear in the supplement.

**中文：** 我们结合分阶段干预、独立有效性判断、轨迹比较和固定轨迹重放，逐步缩小位移来源。下游表示几何与完整诊断表见补充材料。

### 5.1 Testing Simple Phase, Validity, and Resistance Accounts / 检验简单的阶段、有效性与抵抗解释

CommonUpdate, the mean target-following rate across $V/F$, had an uncapped baseline of $50.00\%$. Prefill-only, decode-only, and full-p25 capping increased it by $+18.75$ percentage points (95% family-cluster CI $[4.17,33.33]$), $+20.83$ ($[6.25,35.42]$), and $+18.75$ ($[4.17,33.33]$). Each phase-restricted condition produced a point estimate close to full-p25 capping, but these contrasts identify neither a phase interaction nor additivity.

**中文：** CommonUpdate 是 $V/F$ 目标跟随率的均值，未钳制基线为 $50.00\%$。仅 prefill、仅 decode 和完整 p25 钳制分别使其增加 $+18.75$ 个百分点（95% 结构族聚类 CI $[4.17,33.33]$）、$+20.83$（$[6.25,35.42]$）和 $+18.75$（$[4.17,33.33]$）。两个受限阶段的点估计均接近完整 p25 钳制，但这些对比无法识别阶段交互或可加性。

In a separate validity call, the uncapped $V-F$ discrimination baseline was $29.17$ points; prefill-only, decode-only, and full-p25 interventions changed it by $-8.33$ ($[-29.17,12.50]$), $+12.50$ ($[-4.17,29.17]$), and $0.00$ points ($[-25.00,20.83]$). Every interval included zero, yielding no stable evidence of uniform validity loss or opposing phase roles.

**中文：** 另一次模型调用中的 $V-F$ 有效性判别未钳制基线为 $29.17$ 个百分点；仅 prefill、仅 decode 和完整 p25 三种干预使其分别变化 $-8.33$（$[-29.17,12.50]$）、$+12.50$（$[-4.17,29.17]$）和 $0.00$ 个百分点（$[-25.00,20.83]$）。全部区间均包含零，未给出统一有效性损失或相反阶段作用的稳定证据。

A selective-resistance account further predicted greater pre-answer correction on uncapped resist than yield trajectories. The prespecified final-32-token contrast was $-1.146$ ($[-3.294,+1.264]$); after matching trajectory length, it was $+0.033$ ($[-1.394,+2.166]$). These estimates do not support the account. Complete absolute levels, trajectory groups, and phase/layer controls appear in the supplement.

**中文：** 选择性抵抗解释进一步预测，未钳制条件下的抵抗轨迹在作答前应比跟随轨迹具有更大修正量。预设的最后 32 个 token 对比为 $-1.146$（$[-3.294,+1.264]$），匹配轨迹长度后为 $+0.033$（$[-1.394,+2.166]$）。这些估计不支持该解释。完整绝对水平、轨迹分组和分阶段/分层对照见补充材料。

### 5.2 Fixed-Trace Replay / 固定轨迹重放

For each of 96 prompts, we froze reasoning prefixes generated under uncapped inference ($r_U$) and p25 capping ($r_P$), then replayed each through cache-faithful uncapped and p25 forward passes with every visible token fixed. At the answer boundary, we estimated

$$
\begin{aligned}
\Delta_{\mathrm{FT}}(s)
&=\mathbb{E}\!\left[m_{\mathrm{general}}(p25;r_s)-m_{\mathrm{general}}(0;r_s)\right],\\
I_{\mathrm{FT}}
&=\Delta_{\mathrm{FT}}(P)-\Delta_{\mathrm{FT}}(U),
\end{aligned}
$$

where $s\in\{U,P\}$ indexes trace source. $\Delta_{\mathrm{FT}}(U)$ and $\Delta_{\mathrm{FT}}(P)$ are two co-primary conditional estimands.

**中文：** 对 96 个提示，我们分别固定未钳制推理与 p25 钳制生成的前缀 $r_U$ 和 $r_P$，再沿 cache-faithful 路径，将每条前缀置于未钳制和 p25 前向条件下重放，全程保持可见 token 不变。在答案边界估计上式，其中 $s\in\{U,P\}$ 表示轨迹来源。$\Delta_{\mathrm{FT}}(U)$ 与 $\Delta_{\mathrm{FT}}(P)$ 是两项共同主要的条件性估计量。

They were $+2.078$ logits (95% CI $[1.759,2.408]$) and $+0.700$ ($[0.487,0.907]$), respectively; their secondary P-minus-U interaction was $-1.378$ ($[-1.720,-1.084]$; Figure 3A). P-source estimates condition on intervention-generated trajectories and have a more limited causal interpretation.

**中文：** 两项估计分别为 $+2.078$ logits（95% CI $[1.759,2.408]$）和 $+0.700$（$[0.487,0.907]$）；次要的 P-minus-U 交互为 $-1.378$（$[-1.720,-1.084]$；图 3A）。P 来源估计以干预生成的轨迹为条件，其因果解释更受限。

We additionally define $\Delta_{\mathrm{FT}}^{V-F}(s)$ as the capped-minus-uncapped change in the valid-minus-fallacious target-margin contrast. It was $-2.422$ logits ($[-3.411,-1.448]$) for U-source traces and $-0.797$ ($[-1.219,-0.391]$) for P-source traces (Figure 3B). These are source-conditioned continuous contrasts, not natural-generation effects; the two marginal intervals do not test whether the trace sources differ.

**中文：** 我们另定义 $\Delta_{\mathrm{FT}}^{V-F}(s)$，表示 valid-minus-fallacious 目标 margin 对比的 capped-minus-uncapped 变化。U 来源与 P 来源轨迹下的估计分别为 $-2.422$ logits（$[-3.411,-1.448]$）和 $-0.797$（$[-1.219,-0.391]$；图 3B）。这些是以轨迹来源为条件的连续对比，并非自然生成效应；两项边际区间也不检验轨迹来源之间是否存在差异。

None of the 192 paired replays changed the four-option argmax. The continuous shift remained after fixing the visible trace. Replay leaves the responsible downstream computation unlocalized and provides no evidence of a discrete natural-generation change. Implementation, fidelity checks, and complete decompositions appear in the supplement.

**中文：** 192 组配对重放均未改变四选项 argmax。固定可见轨迹后，连续位移依然存在；重放尚未定位产生位移的下游计算，也未提供自然生成离散行为改变的证据。实现、fidelity 检查和完整分解见补充材料。

**Figure 3 asset:** `figures/drafts/cedar-figure3-fixed-trace-v1/fixed-trace-vector.pdf` (editable SVG and Python source are in the same directory).

**Figure 3 caption.** **Fixed-trace replay separates broad movement from reason selectivity.** For 96 prompts, visible $U$- and $P$-source prefixes were replayed under uncapped and p25 forward conditions. **(A)** Capped-minus-uncapped $\Delta_{\mathrm{FT}}(s)$ in $m_{\mathrm{general}}$; the annotation gives the secondary P-minus-U interaction. **(B)** The corresponding change in the valid-minus-fallacious target-margin contrast. Markers are family-weighted estimates, whiskers are 95% family-cluster bootstrap intervals over 12 families, and the dashed line marks zero. All 192 paired argmaxes were unchanged. P-source estimates condition on intervention-generated trajectories.

**中文图注：** **固定轨迹重放区分广泛位移与理由选择性。** 对 96 个提示，将 $U$ 与 $P$ 来源的可见前缀分别置于未钳制和 p25 前向条件下重放。**(A)** 各来源下 $m_{\mathrm{general}}$ 的 capped-minus-uncapped $\Delta_{\mathrm{FT}}(s)$；注释给出次要的 P-minus-U 交互。**(B)** valid-minus-fallacious 目标 margin 对比的相应变化。点为结构族等权估计，误差线为 12 个结构族上的 95% 聚类 bootstrap 区间，虚线表示零。192 组配对 argmax 均未改变。P 来源估计以干预生成的轨迹为条件。

Phase restriction and fixed-trace replay recover the answer-geometry shift under several controlled conditions, whereas the independent validity assay shows no stable parallel pattern. The next section considers the implications for activation-based control.

**中文：** 分阶段干预与固定轨迹重放在多种受控条件下恢复了答案几何位移，独立有效性测量则没有呈现稳定的对应模式。下一节讨论这些结果对激活控制设计的意义。

## 6. Implications, Scope, and Future Control / 含义、范围与未来控制

### 6.1 Representation Is Not Policy / 表示不等于策略

`[TBD]`

### 6.2 What a Selective Controller Must Preserve / 选择性控制器必须保留什么

`[TBD]`

### 6.3 Evidence Scope and Limitations / 证据范围与局限

`[TBD]`

### 6.4 Future Work / 未来工作

`[TBD]`

## 7. Conclusion / 结论

CEDAR evaluates activation-space control through two separately analyzed paired modules: CEDAR-Evidence compares certified valid and single-fault arguments for the same proposed answer, and CEDAR-Social compares target exposure with social framing. In Qwen3-32B, the released Assistant-Axis p25 cap broadly reduced initial-answer advantage; warranted and unwarranted updating both increased in the development assay, and the new-salt bank retained the aggregate margin shift with different arm-level changes. CEDAR-Social showed a direct-margin interaction without a corresponding free-generation answer change. Fixed-trace replay preserved the continuous shift under both source conditions while leaving every paired four-option argmax unchanged. *[TBD-XM-CONCLUSION-ROUTE]* The evidence supports a conclusion limited to the tested configurations: the cap changes answer geometry without a stable gain in reason selectivity, a property that activation-control studies should evaluate directly.

**中文：** CEDAR 通过两个分别分析的配对模块评估激活空间控制：CEDAR-Evidence 比较同一候选答案对应的经认证有效论证与 single-fault 谬误论证，CEDAR-Social 则比较纯目标暴露与社会措辞。在 Qwen3-32B 上，公开的 Assistant-Axis p25 钳制广泛削弱了初始答案优势；development assay 中有据与无据改答均有所增加，new-salt bank 则在分臂变化不同的情况下保留了总体 margin 位移。CEDAR-Social 中出现的直接 margin 交互没有对应的自由生成答案变化。固定轨迹重放在两种轨迹来源下均保留了连续位移，同时所有配对四选项 argmax 均保持不变。*[TBD-XM-CONCLUSION-ROUTE]* 现有证据支持的结论限于本文测试的配置：该钳制改变了答案几何，但理由选择性未出现稳定增益；激活控制研究应直接评估这一属性。

## Internal Citation Queue / 内部引用队列

No BibTeX key is invented. Verify metadata before adding Assistant Axis, Dual-Stance Evaluation of Sycophancy, When Truth Is Overridden, the refusal-direction paper, WMDP, Belief-R, SycoBench-600, CAA, ITI, KL-then-steer, BASIL, T3, and the nearest July 2026 work.

**中文：** 不得编造 BibTeX key。加入 Assistant Axis、Dual-Stance Evaluation of Sycophancy、When Truth Is Overridden、refusal-direction、WMDP、Belief-R、SycoBench-600、CAA、ITI、KL-then-steer、BASIL、T3 及 2026 年 7 月最近邻工作前，必须核验元数据。
