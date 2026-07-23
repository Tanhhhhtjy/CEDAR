# CEDAR Phase 1.1 自然化审核指南 v2

## 审核目的

本次审核只判断 12 条英文自然化能否作为清晰、自然、无表面捷径的模型输入。请勿复算隐藏
gold，也不要推测 F 的内部错误类别。中文是英文的完整辅助翻译；语言自然度仍以英文为准。

请使用 `naturalization_review12_bilingual_v2.xlsx`。生成者没有填写任何人工评分。

## 操作步骤

1. 每次独立阅读一行的英文完整题目、V 英文和 F 英文。
2. 必要时参考相邻中文，但不要修改任何题干或 V/F 文本。
3. 填写自然度分数和五个 `yes/no` 字段；备注可留空。
4. 12 行全部完成后另存文件，保留原编号、工作表和隐藏列。
5. 将完成版交给 `finalize_naturalization_review_v2.py` 只读导入。

## 审核字段

- `language_naturalness`：英文是否自然、完整、易读；填写 1–5。
- `clear_reference`：实体、规则及复核段落中的指代是否清晰；填写 `yes/no`。
- `single_interpretable_error`：F 是否至多包含一个可定位的问题点；填写 `yes/no`。
- `surface_gold_leakage`：措辞是否直接暗示三选一答案；填写 `yes/no`。
- `condition_identity_leakage`：不分析具体逻辑，仅凭固定写法是否能轻易判断该段是 V 还是
  F；填写 `yes/no`。
- `v_f_style_matched`：V 与 F 的复核场景、语气、信息密度和长度是否大致匹配；填写
  `yes/no`。
- `notes`：可选；请记录具体位置和修改建议。

`condition_identity_leakage` 关注的是表面条件身份捷径。例如，若某个固定词组只在所有 V
出现、从不在 F 出现，即使逻辑内容本身不同，也应考虑标为 `yes`。V 与 F 必然在证据来源上
不同，但这种差异不应被一个无需阅读逻辑的固定标签完全替代。

## 冻结通过标准

每一行都必须同时满足：

```text
language_naturalness >= 4
clear_reference = yes
single_interpretable_error = yes
surface_gold_leakage = no
condition_identity_leakage = no
v_f_style_matched = yes
```

12 行中任何一行不满足，整体即为 Gate C。不得使用平均分、总分、通过率或多数意见覆盖单题
失败。Gate C 只允许修订 renderer，不允许修改符号题、gold、V edit、F certificate 或候选
枚举结果。

## 文件完整性

只允许修改评分列和备注列。以下内容均已冻结：

- 编号；
- 英文和中文题干；
- V/F 英文和中文；
- 隐藏 source hash；
- 工作表结构和冻结元数据。

finalizer 会逐格比较完成版与空白模板。缺失评分、非法取值、增删工作表或任何冻结文本变化
都会被判为无效审核包，并且不会更新 FINAL_GATE。
