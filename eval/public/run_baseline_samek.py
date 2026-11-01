# -*- coding: utf-8 -*-
"""与重排评测同口径（候选窗口 20、截断 10）补跑 dense/sparse/hybrid 基线，
使四者的 nDCG@10 / MRR@10 / Recall@10 可以严格横向对比。
纯检索，无 LLM 调用，成本为零。
"""
import json, math, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from loguru import logger
logger.remove()

from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from app.config import config
from app.services.vector_embedding_service import vector_embedding_service

COLL = "bench_t2retrieval"
K = 20      # 与重排评测一致的候选窗口
TOP = 10


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


def main():
    print("=== 同口径基线（候选窗口 20 / 截断 10）===")
    queries = load_jsonl("queries.jsonl")
    qrels = load_qrels("qrels.tsv")
    connections.connect(alias="default", host=config.milvus_host, port=str(config.milvus_port))
    col = Collection(COLL); col.load()

    modes = ["dense", "sparse", "hybrid"]
    agg = {m: {"ndcg": 0.0, "rr": 0.0, "recall": 0.0} for m in modes}
    n = 0
    t0 = time.time()
    for q in queries:
        rel = qrels.get(q["_id"])
        if not rel:
            continue
        qv = vector_embedding_service.embed_query(q["text"])
        rd = col.search([qv], "dense", {"metric_type": "L2", "params": {"nprobe": 16}},
                        limit=K, output_fields=["id"], consistency_level="Strong")
        rs = col.search([q["text"]], "sparse", {"metric_type": "BM25", "params": {}},
                        limit=K, output_fields=["id"], consistency_level="Strong")
        reqs = [AnnSearchRequest([qv], "dense", {"metric_type": "L2", "params": {"nprobe": 16}}, limit=K),
                AnnSearchRequest([q["text"]], "sparse", {"metric_type": "BM25", "params": {}}, limit=K)]
        rh = col.hybrid_search(reqs, RRFRanker(k=60), limit=K,
                               output_fields=["id"], consistency_level="Strong")
        res = {"dense": [h.entity.get("id") for h in rd[0]],
               "sparse": [h.entity.get("id") for h in rs[0]],
               "hybrid": [h.entity.get("id") for h in rh[0]]}
        for m in modes:
            agg[m]["ndcg"] += ndcg(res[m], rel, TOP)
            agg[m]["rr"] += rr(res[m], rel, TOP)
            agg[m]["recall"] += rec(res[m], rel, TOP)
        n += 1
        if n % 50 == 0:
            print(f"    {n} 条 ({time.time()-t0:.0f}s)")

    print(f"\n=== 基线结果（{n} 条 query，窗口 {K}，截断 {TOP}）===")
    print(f"{'模式':<10}{'nDCG@10':>10}{'MRR@10':>10}{'Recall@10':>12}")
    print("-" * 42)
    out = {}
    for m in modes:
        v = {k: agg[m][k] / n for k in agg[m]}
        out[m] = {k: round(x, 4) for k, x in v.items()}
        print(f"{m:<10}{v['ndcg']:>10.4f}{v['rr']:>10.4f}{v['recall']:>12.4f}")

    # 合并重排结果，输出最终对比表
    try:
        rr_rep = json.load(open("rerank_report.json", encoding="utf-8"))
        out["rerank"] = rr_rep["results"]["rerank"]
        print(f"{'rerank':<10}{out['rerank']['ndcg']:>10.4f}{out['rerank']['rr']:>10.4f}{out['rerank']['recall']:>12.4f}")
    except Exception as e:
        print("  (未能合并 rerank 结果:", e, ")")

    with open("baseline_samek.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "T2Retrieval(subset)", "queries": n,
                   "window": K, "top": TOP, "results": out}, f,
                  ensure_ascii=False, indent=2)
    print("\n已保存 baseline_samek.json")


if __name__ == "__main__":
    main()
