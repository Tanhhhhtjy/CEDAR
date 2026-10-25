# C2/V8 Margin–Choice Dissociation：零 GPU 补充分析

状态：完成；post-hoc 描述性补充，不替代预注册主终点。

## 分析问题

p25 对 S 相对 AO 的连续 margin 影响，与它对离散 target 选择的影响是否同向？

## 主要结果

| Arm | Uncapped margin | p25 margin | Cap margin effect | Uncapped target | p25 target | Cap target effect |
|---|---:|---:|---:|---:|---:|---:|
| AO | -7.5490 | -6.2810 | +1.2679 | 4.17% | 5.83% | +1.67pp |
| S | -1.3029 | -0.8213 | +0.4817 | 33.83% | 38.00% | +4.17pp |

- 预注册主 margin 交互：`-0.78625` logit，95% CI `[-0.86000, -0.71104]`。
- 探索性离散 target 交互：`+2.50pp`，post-hoc 95% CI `[+0.33, +4.67]pp`。
- normalized-stem 聚类后，离散交互区间下界达到零；因此该离散结果不是对重复结构处理完全稳健的确认性发现。
- 600个base中，margin交互为负/零/正：`477/24/99`；离散交互为负/零/正：`15/555/30`。

## 结果解释

p25 在 AO 和 S 中都把连续答案边界推向 induced target，但 AO 的平均平移更大，所以 `S-AO` margin contrast 缩小。与此同时，S 的 uncapped margin 更接近零，较小的绝对平移也能让更多临界题跨越离散决策边界。因此，连续社会差异下降不保证离散错误跟随同步下降。

这支持一个 `margin–choice dissociation` 描述，但不证明内部真实谄媚增加或减少。

## Free-thinking 三层读出

- strict final target SocialExcess：`+0.00pp`；
- actual-answer margin SocialExcess：`-0.1271`；
- scaffold margin SocialExcess：`-0.0802`；
- thinking-token SocialExcess：`+10.14` tokens，95% CI `[-6.35, +26.77]`。

最终 target-follow 四格均为零，属于地板效应，不能作为“thinking 完全消除副作用”的等价性证据。

## 输出完整性与混淆

- entropy `S-AO` cap interaction：`+0.011614`；
- option-mass `S-AO` cap interaction：`+0.000001055`；
- AO−S 平均 prompt token 差：`+2.50`；
- AO−S 平均 answer-step correction sum 差：`+4.22`；
- target字母只覆盖：`A, B`。
- target=A/B 的离散交互分别为：`+3.20pp` / `+0.61pp`。

## Claim Candidates

- Claim:
  - Source evidence: Direct margin interaction and paired discrete target interaction.
  - Allowed wording: p25 compressed the social-pressure margin contrast while increasing discrete target selections near the forced-choice boundary.
  - Forbidden stronger wording: p25 reduces or increases genuine sycophancy in general.
  - Uncertainty: post-hoc discrete endpoint; A/B-only targets; one model and dose.
  - Next check: exact-token-matched, A/B/C/D-balanced held-out confirmation.
  - Decision: keep, explicitly labeled post-hoc.

- Claim:
  - Source evidence: Free strict outputs remain at zero target-follow.
  - Allowed wording: the direct boundary shift did not manifest in this easy free-thinking subset.
  - Forbidden stronger wording: thinking universally protects against capping effects.
  - Uncertainty: final-answer floor effect.
  - Next check: harder natural items with non-zero uncapped target-follow.
  - Decision: weaken.

## 主要限制

- 本分析在验收后提出，所有新增置信区间均为 post-hoc 描述性区间；
- target只包含A/B，无法排除答案字母偏置对绝对arm effect的贡献；
- AO/S没有做到exact-token-match，实际answer-step修正量也不同；
- 600条只对应272个normalized stems；结构聚类敏感性应与base分析并列；
- 单模型、单p25剂量、direct nonthinking和free thinking使用不同读出范式；
- 单一数据负责人审核的协议偏离仍然存在。
