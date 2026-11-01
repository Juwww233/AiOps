# -*- coding: utf-8 -*-
"""检索评测：在同一评测集上对比 dense / sparse / hybrid / rerank 的文件级命中率。

判定口径（文件级）：
- 对每个 query 各模式召回 RETRIEVE_K 个分片，按顺序去重得到「文档名排名」
- Hit@k：排在前 k 的文档里出现任一 expected 文件
- MRR  ：第一个 expected 文件所处名次位置的倒数

输出：总体表格 + 按「语义 / 精确词」分组的表格。
运行前提：Milvus 已启动且 uploads 文档已入向量库。
用法：python eval/run_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger

logger.remove()

from app.services.vector_search_service import vector_search_service
from app.services.vector_store_manager import vector_store_manager

DATASET = Path(__file__).resolve().parent / "dataset.json"
REPORT = Path(__file__).resolve().parent / "report_latest.json"
RETRIEVE_K = 12
MAX_DOCS = 5
MODES = {
    "dense":  vector_search_service.dense_search_documents,
    "sparse": vector_search_service.sparse_search_documents,
    "hybrid": vector_search_service.hybrid_search_documents,
    "rerank": lambda q, k: vector_search_service.rerank_search_documents(q, k, recall_k=k),
}


def doc_rank_from_docs(docs):
    """按召回顺序把分片去重成 文档名排名（保留首次出现顺序）"""
    seen, rank = set(), []
    for d in docs:
        name = (d.metadata.get("_file_name") or "").lower()
        if not name or name in seen:
            continue
        seen.add(name)
        rank.append(name)
    return rank


def hit_at(rank, expected, k):
    return any(e in rank[:k] for e in expected)


def reciprocal_rank(rank, expected):
    for i, name in enumerate(rank, start=1):
        if name in expected:
            return 1.0 / i
    return 0.0


def _summary_on(results, idxs):
    """对给定 query 下标集合统计每个模式"""
    out = {}
    for mode in MODES:
        sub = [results[mode][i] for i in idxs]
        n = len(sub)
        h = {k: sum(r["hits"][k] for r in sub) / n for k in (1, 3, 5)}
        mrr = sum(r["rr"] for r in sub) / n
        out[mode] = {"hit@1": round(h[1], 4), "hit@3": round(h[3], 4),
                     "hit@5": round(h[5], 4), "mrr": round(mrr, 4)}
    return out


def print_table(label, summary):
    print(f"\n{label}")
    print(f"{'模式':<12}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")
    print("-" * 44)
    for mode in MODES:
        s = summary[mode]
        print(f"{mode:<12}{s['hit@1']:>8.3f}{s['hit@3']:>8.3f}{s['hit@5']:>8.3f}{s['mrr']:>8.4f}")


def main():
    from app.config import config
    config.rag_rerank_max_chars = 300  # 控制 LLM 重排 token

    ds = json.loads(DATASET.read_text(encoding="utf-8"))
    queries = ds["queries"]
    types = [q.get("type", "语义") for q in queries]
    print(f"评测集: {len(queries)} 条 query "
          f"(语义 {types.count('语义')} / 精确词 {types.count('精确词')})\n")

    files = vector_store_manager.list_indexed_files()
    available = {f["name"].lower() for f in files}
    print(f"向量库文件: {len(available)} 个 -> {sorted(available)}")
    if not available:
        print("!! 向量库为空，先上传文档再评测"); return

    results = {mode: [] for mode in MODES}
    detail_rows = []
    for qi, q in enumerate(queries):
        expected = [e.lower() for e in q["expected"]]
        row = {"q": q["question"], "type": types[qi],
               "expected": list(expected), "ranks": {}}
        for mode, fn in MODES.items():
            docs = fn(q["question"], RETRIEVE_K)
            rank = doc_rank_from_docs(docs)
            hits = {k: hit_at(rank, expected, k) for k in range(1, MAX_DOCS + 1)}
            results[mode].append({"hits": hits, "rr": reciprocal_rank(rank, expected)})
            row["ranks"][mode] = rank[:5]
        detail_rows.append(row)

    all_idx = list(range(len(queries)))
    print_table("■ 总体（全部 query）", _summary_on(results, all_idx))
    for t in ("语义", "精确词"):
        idxs = [i for i, x in enumerate(types) if x == t]
        if idxs:
            print_table(f"■ 类型 = {t}（n={len(idxs)}）", _summary_on(results, idxs))

    # 逐条明细（第一命中位置, 0=未命中）
    print("\n逐条第一命中位置（dense/sparse/hybrid/rerank，0=未命中）:")
    for row in detail_rows:
        def first_hit(mode):
            for i, n in enumerate(row["ranks"][mode], 1):
                if n in row["expected"]:
                    return i
            return 0
        print(f"  [{row['type']}] "
              f"{first_hit('dense')}/{first_hit('sparse')}/{first_hit('hybrid')}/{first_hit('rerank')} "
              f"{row['q'][:34]}")

    report = {"summary": {f"all": _summary_on(results, all_idx),
                          "语义": _summary_on(results, [i for i, x in enumerate(types) if x == '语义']),
                          "精确词": _summary_on(results, [i for i, x in enumerate(types) if x == '精确词'])},
              "detail": detail_rows}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {REPORT}")


if __name__ == "__main__":
    main()
