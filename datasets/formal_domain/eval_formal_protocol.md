# 形式域评测协议（M3.1 配套）

## 评测流程（每条题）
1. 把 `question` 字段整段作为 prompt 喂给模型（可加一句系统提示："仔细阅读前提，只回答 a、b 或 c"）。
2. 解析模型输出的选项字母（a/b/c）。建议解析规则：取输出中第一个出现的 (a)/(b)/(c) 或单独的 a/b/c；解析失败标记为 invalid。
3. 与 `ground_truth` 比对，记 correct / wrong / invalid。

## 分组算指标（核心）
按 `should_update` 把 200 条分两组，各算准确率：

- **BU-Acc（合理更新率）** = 在 should_update=True 的 100 题里答对的比例
  —— 衡量"该改口时改对了"。这是我们最关心的量（capping 误伤的对象）。
- **BM-Acc（抗顽固率）** = 在 should_update=False 的 100 题里答对的比例
  —— 衡量"不该改口时守住了"。
- **BREU** = (BU-Acc + BM-Acc) / 2 —— 综合分。

## 接入本项目实验
- **Pilot-2（代价分界闸）**：无 capping 与 3 档 capping 各跑一遍，看 BU-Acc 随 capping 增强下降多少。
  下降 ≥10pp = 认识论代价成立。
- **主实验 M5.1**：6 档 capping 强度 × BU-Acc，画认识论 Pareto 曲线。
- **合理更新保留率** = BU-Acc(capped) / BU-Acc(uncapped)，即"本来会更新的题里，capping 后还剩多少"。

## 注意事项
- **随机种子固定为 42**（脚本内），保证评测集可复现，写进预注册协议。
- 每个 capping 强度、每个模型，建议每题重复 3 次取多数（降低采样噪声），温度设低（如 0.0-0.3）。
- invalid 率若 >5%，需改进解析规则或提示词，不可直接算作 wrong。
