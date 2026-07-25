# CEDAR 相关论文阅读包

本目录用于 CEDAR 的论文结构与相关工作研究，不属于实验证据目录，也不会改变 C1/C2/C3/E2 的结论状态。

## 已下载并核验的原文

原始 PDF 仅保存在本地阅读目录，并通过 `.gitignore` 排除，不随论文写作分支发布；下表链接指向公开论文页面。

| 论文 | 正式版本 | 页数 | SHA-256 |
|---|---|---:|---|
| [When Truth Is Overridden](https://arxiv.org/abs/2508.02087) | AAAI-26 | 9 | `7EA1955329478DA7C6DE027B5EB357F6FB26ABF12CFF5C69029B3308C5058BBB` |
| [Refusal Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) | NeurIPS 2024 | 47 | `00B84FCE7160C8CEC3AD5C02DD434292EC386E4B97A5D0E53C03FA40A0854CC6` |
| [The WMDP Benchmark](https://arxiv.org/abs/2403.03218) | ICML 2024 | 26 | `1E27CE419CAD0CE79D93D53A204391E7DBB0383FBCA3F25962A3AC9A4A3976E3` |

页数包含参考文献和附录。校验值对应上表中精确文件名；目录内其他衍生文件不属于这组三个官方原文的校验对象。

## 中文主文译读版

- [AAAI-26：当真相被覆盖](zh/2026-aaai-when-truth-is-overridden.zh.md)
- [NeurIPS 2024：拒答由单一方向介导](zh/2024-neurips-refusal-single-direction.zh.md)
- [ICML 2024：WMDP Benchmark](zh/2024-icml-wmdp.zh.md)

中文文件是项目内部阅读材料，不是会议或作者发布的官方译本。它们保留主文结构、公式、关键数字、图表阅读顺序和结论边界；引用时必须回到英文原文。

## 其余论文快速导读

- [HarmBench、Belief-R、Sycophancy、CAA、ITI 逐章节导读](notes/remaining-five-papers-guide.zh.md)

## 建议阅读顺序

1. 先读 AAAI-26 中文版，观察 7 页正文如何组织“行为现象—机制—因果验证”。
2. 再读 Refusal Direction，重点看双向干预与机制证据层级。
3. 读 WMDP，重点看 benchmark 如何同时承载安全收益与能力保留。
4. 最后用五篇导读按需进入 Belief-R、HarmBench 或 intervention 文献。

## 使用边界

- 这些论文只能作为相关工作和行文参考，不能作为 CEDAR 实验结论。
- 翻译和导读不得替代对原文方法、数值和引用的核验。
- 原始 PDF 及其版权归论文作者和相应出版方所有，本目录仅用于项目内部科研阅读。
