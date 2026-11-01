"""向量检索服务模块

支持检索模式（config.rag_search_mode 切换）：
- dense  : 纯稠密（text-embedding-v4 + L2 TopK）
- sparse : 纯 BM25 稀疏（服务器端 BM25 Function 分词打分）
- hybrid : 稠密 + 稀疏双路召回，RRF 融合（默认）
- rerank : hybrid 召回 N 条 -> LLM 重排 -> Top-K（三阶段流水线）
"""

import time
from typing import List

from langchain_core.documents import Document
from loguru import logger
from pymilvus import AnnSearchRequest, Collection, RRFRanker

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.rerank_service import rerank_service
from app.services.vector_embedding_service import vector_embedding_service


class VectorSearchService:
    """向量检索服务 - 负责从 Milvus 中搜索相似向量"""

    @staticmethod
    def _file_names(docs: List[Document]) -> List[str]:
        """按检索顺序返回去重后的文档名列表（便于日志）"""
        seen, names = set(), []
        for d in docs:
            n = d.metadata.get("_file_name") or "?"
            if n not in seen:
                seen.add(n)
                names.append(n)
        return names

    def __init__(self):
        """初始化向量检索服务"""
        logger.info("向量检索服务初始化完成")

    # ---------- 混合检索（dense + sparse + RRF）----------

    @staticmethod
    def _hits_to_documents(hits, search_type: str) -> List[Document]:
        """把 pymilvus 命中的结果转成 LangChain Document（带 retrieval_score 元数据）"""
        docs: List[Document] = []
        for hit in hits:
            content = hit.entity.get("content") or ""
            if not content:
                continue
            metadata = dict(hit.entity.get("metadata") or {})
            metadata["retrieval_mode"] = search_type
            metadata["retrieval_score"] = float(hit.distance)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _dense_request(self, query: str, limit: int) -> AnnSearchRequest:
        """稠密路：query 用 text-embedding-v4 向量化，L2 距离"""
        query_vector = vector_embedding_service.embed_query(query)
        return AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit,
        )

    def _sparse_request(self, query: str, limit: int) -> AnnSearchRequest:
        """稀疏路：query 原样传文本，Milvus BM25 Function 服务器端分词打分"""
        return AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=limit,
        )

    def hybrid_search_documents(
        self, query: str, top_k: int | None = None, per_route: int | None = None
    ) -> List[Document]:
        """稠密 + 稀疏双路召回，RRF 排序融合，返回 top_k 个文档"""
        top_k = top_k or config.rag_top_k
        per_route = per_route or max(top_k * 4, 10)

        t0 = time.perf_counter()
        collection: Collection = milvus_manager.get_collection()
        reqs = [
            self._dense_request(query, per_route),
            self._sparse_request(query, per_route),
        ]
        results = collection.hybrid_search(
            reqs=reqs,
            rerank=RRFRanker(k=60),  # Reciprocal Rank Fusion
            limit=top_k,
            output_fields=["content", "metadata"],
            consistency_level="Strong",  # 刚上传/删除后立即检索可见（无 flush）
        )
        docs = self._hits_to_documents(results[0], search_type="hybrid")
        logger.info(
            f"[hybrid] query='{query}' 双路各取{per_route} -> RRF 融合 {len(docs)} 条, "
            f"耗时{(time.perf_counter() - t0) * 1000:.0f}ms | {self._file_names(docs)}"
        )
        return docs

    def dense_search_documents(self, query: str, top_k: int | None = None) -> List[Document]:
        """纯稠密检索（等价旧版 langchain similarity_search）"""
        top_k = top_k or config.rag_top_k
        t0 = time.perf_counter()
        collection: Collection = milvus_manager.get_collection()
        query_vector = vector_embedding_service.embed_query(query)
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content", "metadata"],
            consistency_level="Strong",
        )
        docs = self._hits_to_documents(results[0], search_type="dense")
        logger.info(
            f"[dense] query='{query}' {len(docs)} 条, "
            f"耗时{(time.perf_counter() - t0) * 1000:.0f}ms | {self._file_names(docs)}"
        )
        return docs

    def sparse_search_documents(self, query: str, top_k: int | None = None) -> List[Document]:
        """纯 BM25 稀疏检索（query 文本直传，服务器分词打分）"""
        top_k = top_k or config.rag_top_k
        t0 = time.perf_counter()
        collection: Collection = milvus_manager.get_collection()
        results = collection.search(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=top_k,
            output_fields=["content", "metadata"],
            consistency_level="Strong",
        )
        docs = self._hits_to_documents(results[0], search_type="sparse")
        logger.info(
            f"[sparse] query='{query}' {len(docs)} 条, "
            f"耗时{(time.perf_counter() - t0) * 1000:.0f}ms | {self._file_names(docs)}"
        )
        return docs

    def rerank_search_documents(
        self,
        query: str,
        top_k: int | None = None,
        recall_k: int | None = None,
    ) -> List[Document]:
        """三阶段检索：hybrid 召回 recall_k 条 -> LLM 重排 -> 返回 top_k 条"""
        top_k = top_k or config.rag_top_k
        recall_k = recall_k or config.rag_rerank_recall_k

        # 召回段：稠密+稀疏 RRF，取稍大的候选集
        t0 = time.perf_counter()
        recall_docs = self.hybrid_search_documents(query, top_k=recall_k)
        recall_ms = (time.perf_counter() - t0) * 1000
        if not recall_docs:
            return []

        before = self._file_names(recall_docs[:top_k])

        # 重排段：LLM 按相关度排序，取 top_k
        t1 = time.perf_counter()
        reranked = rerank_service.rerank(query, recall_docs, top_k)
        rerank_ms = (time.perf_counter() - t1) * 1000
        after = self._file_names(reranked)

        logger.info(
            f"[rerank] query='{query}' 召回{len(recall_docs)}条({recall_ms:.0f}ms) -> "
            f"LLM重排返回{len(reranked)}条({rerank_ms:.0f}ms)\n"
            f"          召回top{top_k}: {before}\n"
            f"          重排top{top_k}: {after}"
        )
        return reranked

    def search_documents(self, query: str, top_k: int | None = None) -> List[Document]:
        """按 config.rag_search_mode 分发检索模式（dense / sparse / hybrid / rerank）"""
        mode = config.rag_search_mode
        if mode == "hybrid":
            return self.hybrid_search_documents(query, top_k)
        if mode == "sparse":
            return self.sparse_search_documents(query, top_k)
        if mode == "rerank":
            return self.rerank_search_documents(query, top_k)
        return self.dense_search_documents(query, top_k)


# 全局单例
vector_search_service = VectorSearchService()
