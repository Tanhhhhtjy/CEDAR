"""
形式域题库构建脚本（M3.1）—— 输出完整集 + 高 agreement_lv 干净子集
输入：Belief-R queries_time_t1.csv
输出（写到 datasets/formal_domain/）：
  - formal_full.jsonl        完整集（全部 1744 条，保跨论文可比性，作主结果）
  - formal_clean.jsonl       干净子集（agreement_lv>=5，可废止推理强度高，作稳健性检验）
  - formal_balanced_200.jsonl 平衡采样 200 条（该更新/不该更新各100，供 Pilot 快速试跑）
字段：id, question, options{a,b,c}, ground_truth, should_update, modus, agreement_lv
  should_update = (ground_truth=='c')  # 结论退化为不确定=该更新
"""
import csv, json, random

SRC = "/tmp/beliefr/dataset/belief_r/queries_time_t1.csv"
OUTDIR = "/Data/tanh/ASUKA/paper_proposal/datasets/formal_domain"

rows = list(csv.DictReader(open(SRC)))

def to_item(r):
    gt = r["ground_truth"].strip()
    return {
        "id": r["dataset_id"].strip(),
        "question": r["questions"].strip(),
        "options": {"a": r["a"].strip(), "b": r["b"].strip(), "c": r["c"].strip()},
        "ground_truth": gt,
        "should_update": gt == "c",
        "modus": r["modus"].strip(),
        "agreement_lv": int(r["agreement_lv"].strip()) if r["agreement_lv"].strip().isdigit() else None,
    }

items = [to_item(r) for r in rows]

def dump(path, data):
    with open(path, "w") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    n_upd = sum(x["should_update"] for x in data)
    print(f"  {path.split('/')[-1]}: {len(data)} 条 (该更新 {n_upd} / 不该更新 {len(data)-n_upd})")

# 1) 完整集
dump(f"{OUTDIR}/formal_full.jsonl", items)

# 2) 干净子集：agreement_lv >= 5（可废止推理冲击最强）
clean = [x for x in items if x["agreement_lv"] is not None and x["agreement_lv"] >= 5]
dump(f"{OUTDIR}/formal_clean.jsonl", clean)

# 3) 平衡 200（Pilot 用，固定种子可复现）
random.seed(42)
upd  = [x for x in items if x["should_update"]]
keep = [x for x in items if not x["should_update"]]
bal = random.sample(upd, 100) + random.sample(keep, 100)
random.shuffle(bal)
dump(f"{OUTDIR}/formal_balanced_200.jsonl", bal)

print("完成。随机种子=42（预注册可复现）。干净子集判据：agreement_lv>=5。")
