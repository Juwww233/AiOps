# -*- coding: utf-8 -*-
"""T2Retrieval 公开基准评测：dense / sparse(BM25) / hybrid(RRF) 三路对比。

- 使用独立 collection（bench_t2retrieval），不影响项目原有 biz 数据
- 指标：nDCG@10、Recall@100、MRR@10（对齐 MTEB 惯例）
- embedding 结果增量落盘（emb_cache.jsonl），中断可续跑
"""
import json, math, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from loguru import logger
logger.remove()

from pymilvus import (connections, utility, Collection, CollectionSchema,
                      FieldSchema, DataType, Function, FunctionType,
                      AnnSearchRequest, RRFRanker)
from app.config import config
from app.services.vector_embedding_service import vector_embedding_service

COLL = "bench_t2retrieval"
DIM = 1024
TOP_K = 100          # 召回宽度（用于 Recall@100）
NDCG_K = 10
BATCH = 10           # DashScope text-embedding-v4 单次批量上限为 10
MAX_CHARS = 4000
CACHE = "emb_cache.jsonl"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cut_bytes(s: str, limit: int = 60000) -> str:
    """按 UTF-8 字节数截断（Milvus VARCHAR max_length 以字节计，中文 1 字 3 字节）"""
    b = s.encode("utf-8")
    if len(b) <= limit:
        return s
    return b[:limit].decode("utf-8", errors="ignore")


def load_qrels(p):
    rel = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b, s = line.rstrip("\n").split("\t")
            rel.setdefault(a, {})[b] = int(s)
    return rel


# ---------------- ① 向量化（带缓存） ----------------
def embed_all(corpus):
    done = {}
    if os.path.exists(CACHE):
        for r in load_jsonl(CACHE):
            done[r["id"]] = r["vec"]
        print(f"  从缓存恢复 {len(done)} 篇")
    todo = [c for c in corpus if c["_id"] not in done]
    print(f"  待向量化 {len(todo)} / {len(corpus)} 篇")
    if not todo:
        return [done[c["_id"]] for c in corpus]

    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as fout:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            texts = [c["text"][:MAX_CHARS] for c in chunk]
            for attempt in range(4):
                try:
                    vecs = vector_embedding_service.embed_documents(texts)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"    重试 {attempt+1}/4 ({str(e)[:60]}) 等待{wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError("批量向量化连续失败")
            for c, v in zip(chunk, vecs):
                done[c["_id"]] = v
                fout.write(json.dumps({"id": c["_id"], "vec": v}) + "\n")
            fout.flush()
            if (i // BATCH + 1) % 25 == 0:
                el = time.time() - t0
                dn = i + len(chunk)
                print(f"    进度 {dn}/{len(todo)} ({dn/el:.1f} 篇/秒)")
    print(f"  向量化完成，耗时 {time.time()-t0:.0f}s")
    return [done[c["_id"]] for c in corpus]


# ---------------- ② 建库 ----------------
def build_collection(corpus, vectors):
    if utility.has_collection(COLL):
        print(f"  删除旧 collection {COLL}")
        utility.drop_collection(COLL)
    fields = [
        FieldSchema("id", DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema("text", DataType.VARCHAR, max_length=65535,
                    enable_analyzer=True, analyzer_params={"tokenizer": "jieba"}),
        FieldSchema("dense", DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
    ]
    fn = Function(name="bm25", function_type=FunctionType.BM25,
                  input_field_names=["text"], output_field_names=["sparse"])
    col = Collection(COLL, CollectionSchema(fields, functions=[fn]))
    col.create_index("dense", {"index_type": "IVF_FLAT", "metric_type": "L2",
                               "params": {"nlist": 128}})
    col.create_index("sparse", {"index_type": "SPARSE_INVERTED_INDEX",
                                "metric_type": "BM25", "params": {}})
    col.load()
    for i in range(0, len(corpus), 500):
        ch = corpus[i:i + 500]
        rows = [{"id": c["_id"], "text": cut_bytes(c["text"]), "dense": vectors[i + j]}
                for j, c in enumerate(ch)]
        col.insert(rows)
    col.flush()
    print(f"  已插入 {col.num_entities} 条")
    return col


# ---------------- ③ 检索 ----------------
def dense_search(col, qv, k=TOP_K):
    r = col.search([qv], "dense", {"metric_type": "L2", "params": {"nprobe": 16}},
                   limit=k, output_fields=["id"], consistency_level="Strong")
    return [h.entity.get("id") for h in r[0]]


def sparse_search(col, q, k=TOP_K):
    r = col.search([q], "sparse", {"metric_type": "BM25", "params": {}},
                   limit=k, output_fields=["id"], consistency_level="Strong")
    return [h.entity.get("id") for h in r[0]]


def hybrid_search(col, qv, q, k=TOP_K):
    reqs = [AnnSearchRequest([qv], "dense", {"metric_type": "L2", "params": {"nprobe": 16}}, limit=k),
            AnnSearchRequest([q], "sparse", {"metric_type": "BM25", "params": {}}, limit=k)]
    r = col.hybrid_search(reqs, RRFRanker(k=60), limit=k,
                          output_fields=["id"], consistency_level="Strong")
    return [h.entity.get("id") for h in r[0]]


# ---------------- ④ 指标 ----------------
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(ranked, rel, k):
    gains = [rel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at(ranked, rel, k):
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def rr_at(ranked, rel, k):
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            return 1.0 / i
    return 0.0


# ---------------- main ----------------
def main():
    print("=== T2Retrieval 公开基准评测 ===")
    corpus = load_jsonl("corpus.jsonl")
    queries = load_jsonl("queries.jsonl")
    qrels = load_qrels("qrels.tsv")
    print(f"语料 {len(corpus)} 篇 | query {len(queries)} 条 | 标注 {sum(len(v) for v in qrels.values())} 条\n")

    connections.connect(alias="default", host=config.milvus_host, port=str(config.milvus_port))

    print("[1/3] 向量化语料")
    vectors = embed_all(corpus)

    print("\n[2/3] 建库索引")
    col = build_collection(corpus, vectors)

    print("\n[3/3] 检索评测")
    modes = ["dense", "sparse", "hybrid"]
    agg = {m: {"ndcg10": 0.0, "recall100": 0.0, "mrr10": 0.0} for m in modes}
    n = 0
    t0 = time.time()
    for qi, q in enumerate(queries):
        rel = qrels.get(q["_id"])
        if not rel:
            continue
        qv = vector_embedding_service.embed_query(q["text"])
        res = {"dense": dense_search(col, qv),
               "sparse": sparse_search(col, q["text"]),
               "hybrid": hybrid_search(col, qv, q["text"])}
        for m in modes:
            r = res[m]
            agg[m]["ndcg10"] += ndcg_at(r, rel, NDCG_K)
            agg[m]["recall100"] += recall_at(r, rel, TOP_K)
            agg[m]["mrr10"] += rr_at(r, rel, NDCG_K)
        n += 1
        if n % 25 == 0:
            print(f"    已评测 {n} 条 query ({time.time()-t0:.0f}s)")

    print(f"\n=== 结果（{n} 条 query，语料 {len(corpus)} 篇）===")
    print(f"{'模式':<10}{'nDCG@10':>10}{'Recall@100':>12}{'MRR@10':>10}")
    print("-" * 44)
    out = {}
    for m in modes:
        v = {k: agg[m][k] / n for k in agg[m]}
        out[m] = {k: round(x, 4) for k, x in v.items()}
        print(f"{m:<10}{v['ndcg10']:>10.4f}{v['recall100']:>12.4f}{v['mrr10']:>10.4f}")

    with open("public_report.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "T2Retrieval(subset)", "queries": n,
                   "corpus_size": len(corpus), "results": out}, f,
                  ensure_ascii=False, indent=2)
    print("\n已保存 public_report.json")


if __name__ == "__main__":
    main()
