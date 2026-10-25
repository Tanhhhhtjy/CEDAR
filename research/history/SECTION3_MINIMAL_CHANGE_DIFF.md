# Section 3 Minimal-Change Revision

## Scope

- Baseline: `CEDAR_overleaf_AAAI27/main.tex`, lines 70–168.
- The previous full rewrite preserved the six-subsection order and all five equation blocks, but increased the prose from approximately 1,290 to 1,532 English words (+18.8%) and rewrote most explanatory paragraphs.
- This patch keeps the original order, equations, labels, sample counts, model configurations, estimands, and statistical procedures.
- Only the scientific title, identification framing, several transition sentences, and one inaccurate review claim are changed.

## English unified diff

```diff
@@ Section title and opening
-\section{CEDAR: Auditing Epistemic Selectivity}
+\section{Identifying Reason Selectivity under Activation Capping}
 \label{sec:cedar}
 
-CEDAR combines CEDAR-Evidence, a new solver-certified paired dataset, with CEDAR-Social, a separate protocol over natural tasks. The modules estimate responses to reason validity and social framing with distinct estimands and sampling units. We next define their shared answer-change setting, interventions, outcomes, and inference.
+Distinguishing whether capping changes evidence discrimination, the general criterion for revision, target-specific attraction, or social accommodation requires varying warrant validity, target exposure, and social framing independently while holding the task and candidate answer fixed. We implement this with two paired contrasts and read both discrete transitions and continuous answer geometry. We next define the shared revision setting, released intervention, estimands, and inference.
 
@@ Common setting
-\subsection{Shared Answer-Change Setting and Controlled Contrasts}
+\subsection{Paired Revision Setting and Identification Strategy}
 \label{sec:cedar-setting}
 
 Each evaluation begins with a dialogue containing a base question and a supplied initial answer $a_0$. We associate the base with a different, paired target $a_t$ used to classify the subsequent response. A parseable final option different from $a_0$ is an \emph{answer change}; a final answer equal to $a_t$ is \emph{target following}. When the follow-up supplies sufficient support for $a_t$, target following constitutes a \emph{warranted revision}.
 
-CEDAR-Evidence fixes the paired target across four conditions: $N/R$ retain it for analysis without revealing it, whereas $V/F$ present the same update with valid or fallacious justification. CEDAR-Social reveals $a_t$ in both $AO/S$ and varies the accompanying social framing. Each comparison fixes the base, $a_0$, target identity, and response format (Figure~\ref{fig:cedar-overview}).
+The six conditions form two identification contrasts rather than one undifferentiated task. In the evidence contrast, $N/R$ retain the paired target for analysis without revealing it, whereas $V/F$ present the same candidate update and vary only whether its justification is valid. In the social contrast, both $AO/S$ reveal $a_t$, and only the accompanying social framing changes. Thus $V-F$ isolates warrant validity, while $S-AO$ isolates social framing beyond target exposure; both comparisons fix the base, $a_0$, target identity, and response format (Figure~\ref{fig:cedar-overview}).
 
@@ Figure caption title
-\caption{\textbf{CEDAR's paired, reason-sensitive audit design.}
+\caption{\textbf{Paired identification design for reason-selective auditing.}
 
@@ Evidence subsection title and scientific role of certification
-\subsection{CEDAR-Evidence: A Solver-Certified Paired Dataset}
+\subsection{Solver-Certified Valid--Fallacious Contrast}
 \label{sec:cedar-evidence}
 
-CEDAR-Evidence pairs formally valid and invalid justifications whose status is determined independently of model responses.
+The evidence contrast pairs formally valid and invalid justifications whose status is determined independently of model responses.
 Each item specifies a finite, function-free Horn theory with explicit negation and a four-way query: supported, undetermined, contradicted, or both. The base theory has the unique answer $a_0$; a deterministic atomic edit yields the paired target $a_t$. $N$ continues neutrally, and $R$ requests rechecking without new information. $V$ and $F$ present the same candidate edit: $V$ provides a solver-certified justification that warrants $a_t$, whereas $F$ contains one certified invalid inference and does not warrant changing $a_0$.
 
 A Python fixed-point engine and Prover9 independently verify the labels and whether each justification warrants the edit; retained instances require agreement. Every $F$ argument contains one certified invalid step and a Mace4 countermodel, with all remaining steps supported. The prompt scaffold and response format are matched within $V/F$ pairs.
+This certification is not only data cleaning: it makes warrant validity, the independent variable of the audit, checkable before the evaluated model is run.
 Certificates and review records appear in \emph{[TBD-APPENDIX]}.
 
 Each evaluation bank contains 24 bases: easy and medium variants of 12 structural families spanning six error classes.
+Difficulty is varied through proof structure and distractors rather than deliberately obscure wording.
 A second bank uses a fresh generation salt to instantiate the same task grammar and family structures, and is analyzed separately. Structural families are the independent sampling units; conditions and model runs are paired within family.
 
@@ Social subsection and review-protocol correction
-\subsection{CEDAR-Social: Controlling for Target Exposure}
+\subsection{Target-Exposure-Controlled Social Contrast}
 \label{sec:cedar-social}
 
-CEDAR-Social applies the answer-change setting to the 600 natural multiple-choice questions in SycoBench-600.
+The social contrast applies the answer-change setting to the 600 natural multiple-choice questions in SycoBench-600.
 Each dialogue supplies the gold option as $a_0$ and a different option as $a_t$. Both follow-ups present $a_t$. The answer-only condition ($AO$) contains no speaker, correctness, evidence, authority, reward, or consensus cues. The social condition ($S$) adds a first-person expression of confidence and a request for reconsideration, without adding task-relevant facts or reasons. The paired $S-AO$ contrast estimates the incremental contribution of social framing.
+It is therefore an estimate of social framing beyond target exposure, not a generic sycophancy rate.
 
 Four minimally edited template pairs are balanced across domain and difficulty strata. Automated checks verify matching and token-length bounds under the frozen tokenizer.
-Two researchers independently review all pairs for template fidelity and linguistic naturalness without access to model outputs; agreement and adjudication are \emph{[TBD-SOCIAL-AUDIT]}.
+A pre-output human review checked template fidelity and linguistic naturalness without access to model outputs; the recorded deviation from the originally intended independent-review procedure is disclosed with the audit materials.
 Templates and review materials appear in \emph{[TBD-APPENDIX]}.
 
@@ Model-scope wording only
-CEDAR-Social and diagnostic analyses remain Qwen-only.
+The social and diagnostic analyses remain Qwen-only.
 
@@ Margin interpretation
-\subsection{Behavioral Outcomes and Logit Margins}
+\subsection{Behavioral Outcomes and Answer Geometry}
 \label{sec:cedar-readouts}
 
 [Equations (2)–(4) remain unchanged.]
 
 Here $m_{\mathrm{general}}$ measures broad movement away from $a_0$; a positive capped--uncapped change indicates a smaller initial-answer advantage. The remaining margins locate the shift across the target and background options, with $m_{\mathrm{specific}}$ comparing the target directly with the two background alternatives. We also report changes in the initial, target, and mean background logits to identify which answer roles contribute to each margin shift.
+Accordingly, a controller that merely lowers the initial-answer advantage should move $m_{\mathrm{general}}$ without a corresponding positive change in $m_{\mathrm{specific}}$; genuine target-specific attraction should appear in $m_{\mathrm{specific}}$. Pairing these margins with discrete transitions operationalizes the margin--choice dissociation introduced in Section~2.
 
@@ Remove branded names from estimand and statistics prose
-CEDAR-Social uses
+The social contrast uses
 
 [Equation (5) remains unchanged.]
 
-CEDAR-Evidence treats its 12 structural families as the independent units.
+The evidence contrast treats its 12 structural families as the independent units.
 Paired capped--uncapped differences are averaged within family, and families receive equal weight. Primary intervals use the frozen family-cluster bootstrap; family effects and equal-weight leave-one-family-out analyses assess influence. Small-cluster sensitivities appear in \emph{[TBD-APPENDIX]}.
-CEDAR-Social resamples questions with domain/difficulty stratification.
+The social contrast resamples questions with domain/difficulty stratification.
 
-The development bank is exploratory. Primary contrasts are $\Delta_{\mathrm{broad}}$ for the new-salt and matched Llama studies and $\Delta_{\mathrm{social}}$ for CEDAR-Social;
+The development bank is exploratory. Primary contrasts are $\Delta_{\mathrm{broad}}$ for the new-salt and matched Llama studies and $\Delta_{\mathrm{social}}$ for the social contrast;
```

## 中文对应修改

```diff
@@ 章节标题与开头
-\section{CEDAR：审计认识论选择性}
+\section{激活钳制下理由选择性的可识别评测}
 
-CEDAR 包含两个模块：CEDAR-Evidence 是一套经求解器认证的新配对数据集，CEDAR-Social 是基于自然任务的独立评测协议。二者分别估计模型对理由有效性和社会性措辞的响应，并采用不同的估计量与统计单位。下文依次定义共同的答案改变设定、干预、行为指标和统计推断。
+要区分 capping 改变的是证据分辨、一般改答准则、目标特异吸引还是社会性迎合，必须在固定任务与候选答案的同时，独立操控理由有效性、目标暴露和社会框架。我们通过两组配对对照实现这一识别，并同时读取离散迁移与连续答案几何。下文依次定义共同的改答设定、官方干预、估计量和统计推断。
 
@@ 共同设定
-\subsection{答案改变的共同设定与受控对照}
+\subsection{配对改答设定与识别策略}
 
-CEDAR-Evidence 的四种条件使用同一个配对目标：$N/R$ 在分析中保留该目标，但不向模型展示；$V/F$ 呈现同一候选修改，论证分别有效或包含谬误。CEDAR-Social 的 $AO/S$ 都展示 $a_t$，仅改变伴随的社会性措辞。各项对比均固定基础题、$a_0$、目标身份和回答格式（图~\ref{fig:cedar-overview-zh}）。
+六个条件构成两组识别对照，而不是一个未加区分的统一任务。在证据对照中，$N/R$ 在分析中保留配对目标但不向模型展示，$V/F$ 呈现相同候选修改，只改变理由是否有效。在社会对照中，$AO/S$ 都展示 $a_t$，只改变伴随的社会框架。因此，$V-F$ 隔离理由有效性，$S-AO$ 隔离超出目标暴露的社会框架；两组比较均固定基础题、$a_0$、目标身份和回答格式。
 
@@ 小节标题
-\subsection{CEDAR-Evidence：经求解器认证的配对数据集}
+\subsection{经求解器认证的有效--谬误对照}
 
-\subsection{CEDAR-Social：控制目标暴露效应}
+\subsection{控制目标暴露的社会对照}
 
-\subsection{行为结果与 Logit 边际}
+\subsection{行为结果与答案几何}
 
@@ 求解器的科学角色
+这种认证并不只是数据清理：它使本审计的自变量——理由有效性——能够在被评估模型运行之前得到机器检验。
 
@@ 社会评测含义及审核表述
+因此，$S-AO$ 估计的是超出目标暴露的社会框架作用，而不是泛化的谄媚率。
 
-两名研究人员在不接触模型输出的情况下，分别检查全部配对的模板一致性和语言自然度；一致率和分歧裁决为 \emph{[TBD-SOCIAL-AUDIT]}。
+一次不接触模型输出的预输出人工审核检查了模板一致性和语言自然度；相对于原定独立复核流程的偏离随审计材料如实披露。
 
@@ Margin 判读句
+因此，如果控制器只是降低初始答案优势，它应改变 $m_{\mathrm{general}}$，却不必带来正向的 $m_{\mathrm{specific}}$；真正的目标特异吸引才应体现在 $m_{\mathrm{specific}}$ 中。将这些 margin 与离散迁移配对，正是对第二节 margin--choice 分离的操作化。
```

## What remains verbatim

- All five displayed equation blocks.
- The definition of \(a_0\), \(a_t\), answer change, and warranted revision.
- Horn-theory and four-way-query specification.
- Python/Prover9/Mace4 certification procedure.
- 24 bases, 12 families, two difficulty levels, and six error classes.
- SycoBench-600 item count and AO/S prompt constraints.
- Qwen and Llama model/configuration details.
- Family-cluster bootstrap, leave-one-family-out analysis, intention-to-treat handling, and missing-logit policy.
