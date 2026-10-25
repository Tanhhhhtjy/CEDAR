# 2 Related Work

## 2.1 Activation-Space Control and Persona Stabilization / 激活空间控制与人格稳定

Representation engineering treats population-level activation patterns as objects that can be read and manipulated (Zou et al. 2023). Activation Addition and Contrastive Activation Addition steer residual-stream states along directions obtained from prompt or response contrasts (Turner et al. 2023; Rimsky et al. 2024), while Inference-Time Intervention targets attention heads associated with truthful answers (Li et al. 2023). Work on refusal likewise identified a direction whose addition or removal can induce or suppress refusal behavior (Arditi et al. 2024). These studies made activation steering a practical control mechanism. Selective control, however, requires more than showing that a direction predicts and changes a behavior.

> 中文对照：表征工程把群体层面的激活模式视为可读取、可操纵的研究对象（Zou et al. 2023）。Activation Addition 与 Contrastive Activation Addition 根据提示或回答之间的对比得到方向，并沿这些方向移动残差流状态（Turner et al. 2023; Rimsky et al. 2024）；Inference-Time Intervention 则干预与真实回答相关的注意力头（Li et al. 2023）。关于拒答的研究也发现，加入或移除一个特定方向即可诱发或抑制拒答行为（Arditi et al. 2024）。这些工作使激活 steering 成为一种可实际使用的控制手段。不过，能够预测并改变某种行为，并不等于能够只控制这种行为而不牵动其他行为。

Several methods address selectivity directly. CAST conditions steering on the input context (Lee et al. 2024), whereas KL-then-steer reduces distributional changes on benign inputs before steering is applied (Stickland et al. 2024). SCANS uses safety-conscious activation steering to reduce over-refusal while evaluating whether harmful-query defense and general capabilities are retained (Cao et al. 2025). Persona Vectors extend activation-space control to traits such as sycophancy and hallucination (Chen et al. 2025). The Assistant Axis, our direct point of departure, models the default assistant persona and introduces one-sided threshold capping to keep activations within a fixed region of that axis (Lu et al. 2026). Unlike continuous addition or ablation, this operator acts only after a state crosses a percentile-calibrated threshold. Existing selectivity tests focus on context gating or broad off-target retention. CEDAR asks a complementary question: whether the official cap preserves sensitivity to the quality of the reason offered for revising an answer.

> 中文对照：另一些方法直接处理选择性问题。CAST 根据输入上下文决定是否启用 steering（Lee et al. 2024）；KL-then-steer 在应用 steering 前，先减少模型在良性输入上相对原模型的分布偏移（Stickland et al. 2024）；SCANS 则以安全敏感的激活 steering 缓解过度拒答，同时检查有害请求防御和一般能力能否保留（Cao et al. 2025）。Persona Vectors 将激活空间控制扩展到谄媚、幻觉等人格特质（Chen et al. 2025）。作为本文直接出发点，Assistant Axis 刻画模型默认的助手人格，并提出单侧阈值钳制，使激活停留在该轴的固定区域内（Lu et al. 2026）。它不同于连续加法或消融：只有当状态越过按分位数校准的阈值时才会介入。现有选择性评测主要关注上下文门控或宽泛的非目标能力保留；CEDAR 补充追问：官方 capping 是否仍能按照所给理由的质量，选择性地决定何时修改答案。

## 2.2 Sycophancy, User Influence, and Selective Correction / 谄媚、用户影响与选择性纠正

Sycophancy research has documented that assistants may adapt their judgments to a user's stated beliefs or preferences even when this conflicts with truthfulness (Sharma et al. 2024). MUSE relates yielding to uncertainty before the challenge, distinguishing uncertainty-linked conformity from conformity that persists despite high initial confidence (Guo et al. 2026). SycoBench-600 evaluates both resistance to misleading pressure and acceptance of correct suggestions, making correction selectivity---not resistance alone---the central quantity (Sinha 2026). This distinction is essential for intervention studies: suppressing every answer change can appear robust while also blocking warranted correction.

> 中文对照：谄媚研究表明，助手可能会迎合用户陈述的信念或偏好，即使这种迎合与真实性发生冲突（Sharma et al. 2024）。MUSE 将模型受质疑后的让步行为与其此前的不确定性联系起来，从而区分“不确定性相关的趋同”与“在初始高置信度下仍然发生的趋同”（Guo et al. 2026）。SycoBench-600 同时测量模型抵抗误导性压力和接受正确建议的能力，因此把修正选择性——而非单纯抵抗——作为核心指标（Sinha 2026）。这一点对干预研究至关重要：一味压低所有改答，表面上可能更稳健，却也可能阻断本应发生的纠正。

Other work studies the internal basis of user influence. *When Truth Is Overridden* uses layerwise logit-lens analysis and activation patching to localize changes induced by incorrect user opinions (Wang et al. 2026). *Playing Devil's Advocate* finds that persona directions associated with doubt or scrutiny can reduce agreement with incorrect users, yet do not behave as a simple inverse of agreeable personas (Kelkar et al. 2026). Closest to our objective, dual-stance evaluation shows that anti-sycophancy steering can also suppress agreement with factually correct statements, despite separable activation geometry for the two behaviors (Buchan 2026). Thus, neither lower agreement nor a decodable direction establishes selective control. CEDAR tests the remaining gap: how a state-dependent cap affects matched warranted and unwarranted revisions, with target exposure controlled and continuous and discrete outcomes reported together.

> 中文对照：另一组工作考察用户影响的内部基础。《When Truth Is Overridden》借助逐层 logit lens 与激活 patching，定位错误用户意见所引起的内部变化（Wang et al. 2026）。《Playing Devil’s Advocate》发现，沿怀疑或审视型人格方向进行 steering 可以减少模型对错误用户意见的附和，但这种作用并不是“宜人型人格”的简单反向镜像（Kelkar et al. 2026）。与本文目标最接近的是 dual-stance evaluation：它发现，反谄媚 steering 也可能压低模型对正确事实的赞同，尽管两类行为在激活几何上可以分离（Buchan 2026）。因此，赞同率下降或某个方向可被解码，都不足以证明控制具有选择性。CEDAR 检验尚未解决的问题：当有效与无效的修正理由严格配对时，状态依赖的 capping 如何影响二者；同时控制 target exposure，并联合报告连续与离散结果。

## 2.3 Belief Revision and Reason-Sensitive Evaluation / 信念修订与理由敏感评测

Belief-revision benchmarks ask whether a model maintains or withdraws an earlier conclusion after receiving new information. Belief-R formalizes this setting through delta reasoning and reports a tension between updating when revision is required and maintaining a conclusion when it is not (Wilie et al. 2024). Logical evaluation offers complementary controls. ProverGen combines language-model generation with symbolic proving to create natural-language problems with checkable first-order-logic structure (Qi et al. 2025). LoFa studies resistance to synthetic, natural-language fallacious persuasion while screening for failures of base knowledge (Shen et al. 2026). On controlled syllogisms, Valentino et al. (2026) disentangle formal validity from content plausibility and introduce conditional activation steering (K-CAST) to mitigate content-driven reasoning errors. Together, these approaches offer logical certification, adversarial fallacy coverage, and a direct precedent for conditioning activation control on a reasoning-relevant distinction.

> 中文对照：信念修订 benchmark 关注模型在得到新信息后，能否正确维持或撤回先前结论。Belief-R 通过 delta reasoning 将这一问题形式化，并发现模型在“该更新时更新”与“不该更新时维持”之间存在张力（Wilie et al. 2024）。逻辑评测提供了互补的控制工具。ProverGen 将语言模型生成与符号证明结合，构造具有可核验一阶逻辑结构的自然语言问题（Qi et al. 2025）。LoFa 研究模型抵抗合成的自然语言谬误说服的能力，并筛查基础知识缺陷造成的混淆（Shen et al. 2026）。Valentino et al.（2026）则在受控三段论任务中分离形式有效性与内容可信度，并提出条件激活 steering（K-CAST），以缓解内容驱动的推理错误。这些方法共同提供了逻辑认证、谬误攻击覆盖，以及根据推理相关区分来调节激活控制的直接先例。

CEDAR repurposes these ideas for intervention auditing. Within each contrast cluster, the initial problem and proposed target are shared: a solver-certified valid reason changes the correct answer, whereas a matched invalid argument contains one localized failure and leaves the initial answer correct. Neutral and recheck conditions measure unsupported movement; the social assay separates target exposure from social pressure. The resulting question is not merely whether the model revises, but whether activation capping preserves the distinction between reasons that warrant revision and inputs that do not.

> 中文对照：CEDAR 将这些思想用于干预审计，而不是一般逻辑能力排名。在每个对照簇内，初始问题和诱导 target 完全相同：经证明器认证的有效理由会改变正确答案；与之匹配的无效论证只含一个局部错误，因而初始答案仍然正确。中性和复核条件测量无依据的移动，社会域则把 target exposure 与社会压力分开。由此，问题不再只是模型会不会改答，而是 activation capping 能否保留“哪些理由足以支持改答、哪些输入不足以支持改答”这一关键区分。

# References

Arditi, A.; Obeso, O.; Syed, A.; Paleka, D.; Panickssery, N.; Gurnee, W.; and Nanda, N. 2024. Refusal in Language Models Is Mediated by a Single Direction. In *Advances in Neural Information Processing Systems 37*.

Buchan, M. J. 2026. Dual-Stance Evaluation of Sycophancy: The Structure of Agreement and the Limits of Intervention. In *Technical AI Safety Conference (TAIS 2026)*. arXiv:2606.11205.

Cao, Z.; Yang, Y.; and Zhao, H. 2025. SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(22): 23523--23531.

Chen, R.; Arditi, A.; Sleight, H.; Evans, O.; and Lindsey, J. 2025. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.

Guo, K. H.; Yan, C.; Baidya, A.; Brown, K.; Gao, X.; Xiong, J.; Yin, Z.; and Malin, B. A. 2026. It's Not Always Sycophancy: Measuring LLM Conformity as a Function of Epistemic Uncertainty. arXiv:2605.27288.

Kelkar, I.; Alam, N.; Kakaria, V.; Panwar, M.; Sharma, V.; and Chaudhary, M. 2026. Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy. In *ICML 2026 Workshop on Pluralistic Alignment*.

Lee, B. W.; Padhi, I.; Ramamurthy, K. N.; Miehling, E.; Dognin, P.; Nagireddy, M.; and Dhurandhar, A. 2024. Programming Refusal with Conditional Activation Steering. arXiv:2409.05907.

Li, K.; Patel, O.; Viégas, F.; Pfister, H.; and Wattenberg, M. 2023. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. In *Advances in Neural Information Processing Systems 36*.

Lu, C.; Gallagher, J.; Michala, J.; Fish, K.; and Lindsey, J. 2026. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. arXiv:2601.10387.

Qi, C.; Ma, R.; Li, B.; Du, H.; Hui, B.; Wu, J.; Laili, Y.; and He, C. 2025. Large Language Models Meet Symbolic Provers for Logical Reasoning Evaluation. arXiv:2502.06563.

Rimsky, N.; Gabrieli, N.; Schulz, J.; Tong, M.; Hubinger, E.; and Turner, A. 2024. Steering Llama 2 via Contrastive Activation Addition. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*, 15504--15522.

Sharma, M.; Tong, M.; Korbak, T.; Duvenaud, D.; Askell, A.; Bowman, S. R.; et al. 2024. Towards Understanding Sycophancy in Language Models. In *International Conference on Learning Representations*.

Shen, X.; Yuan, L.; Chen, Y.; Wu, X.; Cai, Y.; and Wu, Z. 2026. Truth or Sophistry? LoFa: A Benchmark for LLM Robustness Against Logical Fallacies. arXiv:2606.31039.

Sinha, D. 2026. SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants. In *Findings of the Association for Computational Linguistics: ACL 2026*, 35278--35284.

Stickland, A. C.; Lyzhov, A.; Pfau, J.; Mahdi, S.; and Bowman, S. R. 2024. Steering Without Side Effects: Improving Post-Deployment Control of Language Models. In *NeurIPS 2024 Workshop on Safe Generative AI*.

Turner, A. M.; Thiergart, L.; Leech, G.; Udell, D.; Vazquez, J. J.; Mini, U.; and MacDiarmid, M. 2023. Steering Language Models with Activation Engineering. arXiv:2308.10248.

Valentino, M.; Kim, G.; Dalal, D.; Zhao, Z.; and Freitas, A. 2026. Mitigating Content Effects on Reasoning in Language Models Through Fine-Grained Activation Steering. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(39): 33314--33322.

Wang, K.; Li, J.; Yang, S.; Zhang, Z.; and Wang, D. 2026. When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(39): 33566--33574.

Wilie, B.; Cahyawijaya, S.; Ishii, E.; He, J.; and Fung, P. 2024. Belief Revision: The Adaptability of Large Language Models Reasoning. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 10480--10496.

Zou, A.; Phan, L.; Chen, S.; Campbell, J.; Guo, P.; Ren, R.; et al. 2023. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
