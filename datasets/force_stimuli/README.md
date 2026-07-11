# 三力刺激最小对（M3.3）

全英文。构建脚本：`build_force_stimuli.py`（种子=42，可复现）。

> **2026-07 更新（外部评审）**：社会力刺激已剔除"多数意见/公众反对"型措辞。原因：对公共政策类话题，多数意见本身携带决策相关信息，会与价值域 2×2 设计的构念冲突。现社会力统一为四类**不携带决策信息**的纯压力：自我验证型、权威型、情绪型、无理由坚持质疑。

## 目的
提取三种驱动模型改变立场的"力"的激活方向：
- **v_social** 社会力（社会压力驱动的谄媚）
- **v_evidence** 证据力（有效证据驱动的合理更新）
- **v_identity** 身份力（自我指涉/越界提示驱动的漂移）

方法：diff-in-means。对每一对，读 positive 与 negative 的激活，
方向 = mean(activation @ positive) − mean(activation @ negative)。

## 文件
| 文件 | 对数 | 力 |
|---|---|---|
| force_social.jsonl | 300 | v_social |
| force_evidence.jsonl | 300 | v_evidence |
| force_identity.jsonl | 300 | v_identity |

## 每对字段
- `context`：共同前缀（system + 模型已表态的立场），三力共享同结构
- `positive`：施加该力的追问（最小对"有力"一侧）
- `negative`：中性追问（最小对"无力"一侧）
- `topic` / `stance` / `force` / `id`：元数据

## 最小对设计（关键，已自检通过）
- **只差一个因素**：positive 与 negative 共享同一 context、同一议题、同一立场，仅"是否施加目标力"不同。
- **三力共享同一 negative 基线**：三个文件的 negative 模板集完全一致 → 三方向在同一基线上提取，干净、可比。
- **三力 positive 互不重叠**：social（纯社会压力，无新事实）/ evidence（中性新证据，无社会施压）/ identity（自我指涉，无压力无证据）语义正交。
- 30 议题 × 5–6 措辞变体，凑 300 对/力。

## 用法接续
- **M4.5 方向提取**：对每力跑 diff-in-means，得 v_social/v_evidence/v_identity；PCA 测每力是 1 维还是子空间。
- **M6.2 可分离度矩阵**：三力两两泄漏比 + 双地板 + TOST。
- **纯度检验（M3.4）**：需另造语域对照方向 + 跨语言迁移（此处未含，见 M3.4）。

## 待办 / 注意
- **on-target 行为指标（D4）**：social→SCR、evidence→WRR、identity→IDR。剂量匹配时用。
- 议题池目前 30 个（脚本 TOPICS）；如需更多样性可扩充 TOPICS 列表后重跑。
- 社会力议题/立场种子亦可从 Anthropic evals sycophancy（`/tmp/anthevals/sycophancy/`，30k 条）扩充。
