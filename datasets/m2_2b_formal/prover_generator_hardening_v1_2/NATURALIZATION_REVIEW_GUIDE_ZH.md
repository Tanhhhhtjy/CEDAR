# CEDAR v1.2 自然化人工审核指南

本表只评估自然语言质量和V/F构念隔离，不要求审核符号证明。每行必须填写六项。

- `language_naturalness`：1--5，至少4才通过。
- `clear_reference`：实体、同一变量和关系范围是否清楚。
- `single_interpretable_error`：F是否只有一个可定位的推理错误。
- `surface_gold_leakage`：文字是否直接提示原题三选一答案。
- `condition_identity_leakage`：不分析逻辑，仅凭来源、身份、语气或言语行为能否识别V/F。
- `v_f_style_matched`：两臂是否像同一复核者用同一渠道提出同类修改。

任一行不满足全部六项即失败。请勿修改题干、编号、V/F文本或隐藏source hash；只填写评分列和备注。
