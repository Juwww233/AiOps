# -*- coding: utf-8 -*-
"""在 T2Retrieval 子集上评测「LLM 重排」：同一批 hybrid 候选，对比重排前/后的排序质量。

公平口径：hybrid 召回 RECALL 个候选 →
  基线 = 该候选列表的前 10（hybrid 自身排序）
  重排 = 对同一批候选做 LLM 重排，取前 10
两者用同一候选池，只差「是否重排」一个变量。
"""
import json, math, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from loguru import logger
logger.remove()

from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from langchain_core.documents import Document
from app.config import config
from app.services.rerank_service import rerank_service
from app.services.vector_embedding_service import vector_embedding_service

COLL = "bench_t2retrieval"
RECALL = 20          # 重排候选窗口
TOP = 10             # 评估截断
SEED = 42

config.rag_rerank_max_chars = 300   # 控制每次重排的 token


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_qrels(p):
    rel = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, s = line.rstrip("\n").split("\t")
            rel.setdefault(a, {})[b] = int(s)
    return rel


def dcg(g):
    return sum(x / math.log2(i + 2) for i, x in enumerate(g))


def ndcg(ranked, rel, k):
    ideal = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(d, 0) for d in ranked[:k]]) / ideal if ideal else 0.0


def rr(ranked, rel, k):
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def rec(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel) if rel else 0.0


def hybrid_candidates(col, qv, q, k):
    reqs = [AnnSearchRequest([qv], "dense", {"metric_type": "L2", "params": {"nprobe": 16}}, limit=k),
            AnnSearchRequest([q], "sparse", {"metric_type": "BM25", "params": {}}, limit=k)]
    r = col.hybrid_search(reqs, RRFRanker(k=60), limit=k,
                          output_fields=["id", "text"], consistency_level="Strong")
    return [(h.entity.get("id"), h.entity.get("text") or "") for h in r[0]]


def main():
    print("=== LLM 重排评测（T2Retrieval 子集）===")
    queries = load_jsonl("queries.jsonl")
    qrels = load_qrels("qrels.tsv")
    print(f"query {len(queries)} 条 | 候选窗口 {RECALL} | 评估截断 {TOP}\n")

    connections.connect(alias="default", host=config.milvus_host, port=str(config.milvus_port))
    col = Collection(COLL)
    col.load()

    agg = {"hybrid": {"ndcg": 0.0, "rr": 0.0, "recall": 0.0},
           "rerank": {"ndcg": 0.0, "rr": 0.0, "recall": 0.0}}
    n = 0
    t0 = time.time()
    for q in queries:
        rel = qrels.get(q["_id"])
        if not rel:
            continue
        qv = vector_embedding_service.embed_query(q["text"])
        cands = hybrid_candidates(col, qv, q["text"], RECALL)
        if not cands:
            continue

        base_ids = [c[0] for c in cands][:TOP]
        docs = [Document(page_content=t, metadata={"_id": i}) for i, t in cands]
        try:
            reranked = rerank_service.rerank(q["text"], docs, TOP)
            rr_ids = [d.metadata["_id"] for d in reranked]
        except Exception as e:
            print(f"  重排失败({str(e)[:50]})，该条按基线算")
            rr_ids = base_ids

        for name, ids in (("hybrid", base_ids), ("rerank", rr_ids)):
            agg[name]["ndcg"] += ndcg(ids, rel, TOP)
            agg[name]["rr"] += rr(ids, rel, TOP)
            agg[name]["recall"] += rec(ids, rel, TOP)
        n += 1
        if n % 25 == 0:
            print(f"    已评测 {n} 条 ({time.time()-t0:.0f}s)")

    print(f"\n=== 结果（{n} 条 query，同一候选池，只差是否重排）===")
    print(f"{'模式':<10}{'nDCG@10':>10}{'MRR@10':>10}{'Recall@10':>12}")
    print("-" * 42)
    out = {}
    for m in ("hybrid", "rerank"):
        v = {k: agg[m][k] / n for k in agg[m]}
        out[m] = {k: round(x, 4) for k, x in v.items()}
        print(f"{m:<10}{v['ndcg']:>10.4f}{v['rr']:>10.4f}{v['recall']:>12.4f}")

    d = out["rerank"]["ndcg"] - out["hybrid"]["ndcg"]
    print(f"\nnDCG@10 变化: {d:+.4f} ({'重排更优 ✅' if d > 0 else '重排未提升'})")

    with open("rerank_report.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "T2Retrieval(subset)", "queries": n,
                   "recall_window": RECALL, "top": TOP, "results": out},
                  f, ensure_ascii=False, indent=2)
    print("已保存 rerank_report.json")


if __name__ == "__main__":
    main()
