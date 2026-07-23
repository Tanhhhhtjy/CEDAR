# C2/V8 最终验收记录

状态：通过；正式分析后强制停止
日期：2026-07-23
分支：`agent/c2-v8-natural-social-confirmation`

## 版本与证据链

- GPU 输入提交：`381e915c74f52bf9a19c1044c8cef6bfdd48b258`；
- 单人审核 Gate 提交：`9a7c9e49db81a0898be8fae29a64d54aede7d451`；
- post-run correction 提交：`176e341e940f56014df8808c7b54187fd9fbec94`、
  `0c7a1950d7a8c518d7f5fefcc122ac1dc3f9e43f`；
- 结果提交：`6f5c1a5d32bba133ce37ac54a51a3e946e16f6f3`；
- GPU 可见卡：`2,4,6,7`；模型运行时工作区干净且 HEAD 与远端一致。

人工审核必须表述为“单一数据负责人审核＋自动化全量验证”。这是已在 GPU 前披露的协议
偏离；没有独立第二审核者、双人一致率或 Cohen's kappa。

## 输入与执行完整性

- 官方 SycoBench 来源：600 bases、8 domains，许可证 `CC-BY-4.0`；
- source object SHA-256：
  `6c02e880a1f75b1bdae53d42e7417aa5b5cf911d53daf6e8059826ebb3b689bb`；
- AO/S arms object SHA-256：
  `753d57a3714921ada2e6372d089093b03748dd63f3717e13d89179f78c5a0715`；
- smoke direct：24/24 x 3；smoke free：24/24 x 3；
- formal direct：uncapped 1,200/1,200，p25 1,200/1,200；
- formal free：uncapped 480/480，p25 480/480；
- 内容重试：0；formal free invalid/truncated：uncapped 0，p25 0；
- Smoke、Direct、Free 与合并跨条件审计均为 `passed`；
- Qwen revision：`9216db5781bf21249d130ec9da846c4624c16137`；
- Assistant Axis SHA-256：
  `a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211`；
- capping config SHA-256：
  `6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24`；
- p25 experiment：`layers_46:54-p0.25`，实际层46--53。

## 分析修正验收

首次正式分析的 bootstrap 与判定完成，但层级模型因把 base 随机截距和组内唯一 domain
指示列同时放入而结构性奇异。最终验收已经看到初步结果，因此按
`C2_V8_POST_RUN_ANALYSIS_CORRECTION_01_HIERARCHICAL_MODEL.md` 完整披露。

修正使用可识别的 `(1 | domain) + (1 | base)`，不改数据、prompt、runner、GPU 输出、主
bootstrap、seed、阈值或判定矩阵，也未重跑任何生成。修正模型状态为 `fitted`、已收敛；
交互估计 `-0.78625`，SE `0.16171`，p=`1.16e-6`。

## 结果验收

- 主 direct `cap x (S-AO)`：`-0.78625` logit；
- 10,000 次分层 base bootstrap 95% CI：`[-0.86000, -0.71104]`；
- AO cap effect：`+1.26792`；S cap effect：`+0.48167`；
- leave-one-domain-out 与 leave-one-template-out 均全部为负；
- strict free target SocialExcess：`0`；completion SocialExcess：`0`；
- thinking-token SocialExcess：`+10.14` tokens，95% CI `[-6.48, +26.72]`；
- free p25 AO 有1条 `initial -> other_wrong`，S 无离散答案迁移；
- S-AO option-mass interaction 为 `+1.05e-6`，没有答案空间质量塌陷；
- S-AO top-token-in-option interaction 为 `0`。

解释标签为 `p25_reduces_social_pressure_influence_on_direct_boundary`。准确含义是 p25 在 AO
与 S 中都提高了 target margin，但在 S 中提高得更少，因此相对于纯 target exposure 的
社会压力增量下降；不能写成模型内部“真实谄媚”减少。

## 软件与交付验收

- C2 定向测试：27/27；
- 全仓测试：213/213；
- C2 范围 Ruff：通过；全仓既有647项 Ruff 历史问题未伪报为通过；
- delivery manifest 覆盖53个非 checkpoint 文件，逐项大小和 SHA-256 复核通过；
- 公开提交共54个结果文件、90,609,335 bytes；
- `records/` 仅是可恢复 checkpoint，与 materialized JSONL 重复，按协议留在本地且不提交；
- delivery manifest SHA-256：
  `63e94629ae0d208c652ae787b7e759993af63f1da5513832940f52f765d8640f`；
- analysis SHA-256：
  `5495395cc1a070a7ea7dccc48f29ea97482cc1cdcac3210301332a56a151d679`；
- combined audit SHA-256：
  `8ccfa199ac2ac48bdb8b8ab05aceb04752386706b22249f0438fba06ed774d04`。

验收后未运行 p1、新 seed、LoFa、J-Lens、probe、patching、新模板或第二模型。
