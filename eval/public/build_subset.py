# -*- coding: utf-8 -*-
"""从 T2Retrieval 构造评测子集：
- 采样 N 条 query，取其全部相关文档作为「正样本」
- 再从语料池采样其余文档作为「难负样本」（它们对其他 query 是正样本）
- 清理 HTML 标签、去重，输出 corpus.jsonl / queries.jsonl / qrels.tsv
"""
import json, random, re, sys, statistics
sys.stdout.reconfigure(encoding="utf-8")
import pyarrow.parquet as pq

N_QUERIES = 200
N_CORPUS = 10000
SEED = 42

rng = random.Random(SEED)
TAG = re.compile(r"<[^>]+>")

def clean(t: str) -> str:
    t = TAG.sub(" ", t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

print("读取数据 ...")
qrels = pq.read_table("qrels.parquet").to_pylist()
corpus = pq.read_table("corpus.parquet").to_pylist()
queries = pq.read_table("queries.parquet").to_pylist()

qmap = {r["_id"]: clean(r["text"]) for r in queries}
cmap = {r["_id"]: clean(r["text"]) for r in corpus}
print(f"  corpus {len(cmap)} / queries {len(qmap)} / qrels {len(qrels)}")

# 按 query 聚合相关文档
by_q = {}
for r in qrels:
    by_q.setdefault(r["query-id"], []).append(r["corpus-id"])
by_q = {k: v for k, v in by_q.items() if k in qmap and all(c in cmap for c in v)}
print(f"  有效 query: {len(by_q)}")

# 采样 query
sample_q = rng.sample(sorted(by_q.keys()), min(N_QUERIES, len(by_q)))
pos_ids = set()
for qid in sample_q:
    pos_ids.update(by_q[qid])
print(f"  采样 {len(sample_q)} 条 query，涉及正样本 {len(pos_ids)} 篇")

# 补齐负样本（从其余语料中采）
others = [cid for cid in cmap if cid not in pos_ids]
rng.shuffle(others)
need = max(0, N_CORPUS - len(pos_ids))
neg_ids = set(others[:need])
keep = pos_ids | neg_ids
print(f"  负样本 {len(neg_ids)} 篇 → 子集语料共 {len(keep)} 篇")

# 输出
with open("corpus.jsonl", "w", encoding="utf-8") as f:
    for cid in keep:
        if cmap[cid]:
            f.write(json.dumps({"_id": cid, "text": cmap[cid]}, ensure_ascii=False) + "\n")

with open("queries.jsonl", "w", encoding="utf-8") as f:
    for qid in sample_q:
        f.write(json.dumps({"_id": qid, "text": qmap[qid]}, ensure_ascii=False) + "\n")

with open("qrels.tsv", "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for qid in sample_q:
        for cid in by_q[qid]:
            f.write(f"{qid}\t{cid}\t1\n")

# 成本估算
tot = sum(len(cmap[c]) for c in keep if cmap[c])
print()
print("=== 输出完成 ===")
print(f"  corpus.jsonl  : {len(keep)} 篇，共 {tot:,} 字符，平均 {tot//max(len(keep),1)} 字符/篇")
print(f"  粗估 token    : 约 {int(tot*0.9):,} tokens（中文约 0.9 token/字，保守）")
print(f"  queries.jsonl : {len(sample_q)} 条")
print(f"  qrels.tsv     : {sum(len(by_q[q]) for q in sample_q)} 条标注")
