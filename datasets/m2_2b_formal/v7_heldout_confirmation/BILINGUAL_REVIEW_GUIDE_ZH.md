# C1/V7 双语盲审操作说明

状态：`awaiting_two_independent_human_reviewers`

## 分发

- Reviewer A 只接收 `blind_review_a_bilingual.xlsx`。
- Reviewer B 只接收 `blind_review_b_bilingual.xlsx`。
- 两人不得互看表格、讨论标签，或接触答案键、family、difficulty、作者构建证书和模型输出。

英文原文是唯一规范文本；中文列是逐字段的完整辅助译文。`language_natural` 必须依据英文原文判断。

## 填写

打开“逐题审核”工作表，只填写浅黄色列。答案字段使用 `A/B/C/unclear`；十个判断字段使用
`yes/no/unclear`；置信度使用 `1–5`。备注可选，但遇到歧义、第二答案或语言问题时必须说明。

第一列会自动显示“完成/待填写”。可以筛选“待填写”定位遗漏项。不要修改 blind ID、英文原文、
中文译文或行顺序。

## 返回与导入

两份 Excel 必须分别返回，不能先查看 A 后提示 B。收到后由协调者运行：

```powershell
python scripts/build_c1_v7_bilingual_reviews.py --import-a --workbook <Reviewer-A-returned.xlsx>
python scripts/build_c1_v7_bilingual_reviews.py --import-b --workbook <Reviewer-B-returned.xlsx>
```

导入器会拒绝不完整标签、修改过的英文原文、改变的 blind ID/顺序、错误的数据集绑定，以及覆盖已经
导入的 raw review。之后按 `docs/C1_V7_PHASE2_TO_CONFIRMATION_RUNBOOK.md` 执行 `--score-only`，
不得自动裁决或修改审核者原始标签。
