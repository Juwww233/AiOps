"""检索重排服务（第三段）

召回(hybrid/dense) -> 重排 -> Top-K。当前实现为 LLM Reranker：
用 ChatQwen 对召回片段按与查询的相关度重新排序（结构化输出排序编号）。

为什么用 LLM 而不是交叉编码器：
- 本环境无 DashScope rerank API、也未装本地 cross-encoder(bge/torch)。
- LLM 重排是免额外模型依赖、可直接用的方案；生产可替换为 bge-reranker 等。

接口保持 search_documents(query, docs, top_k) -> List[Document]，
便于将来无缝替换成 cross-encoder 实现。
"""
import time
from typing import List

from langchain_core.documents import Document
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config


class RerankOrder(BaseModel):
    """LLM 输出的文档排序结果（编号从 1 开始，对应输入顺序）"""
    order: List[int] = Field(
        ..., description="按相关度从高到低排列的文档编号数组，编号从1开始"
    )


class LLMReranker:
    """基于 LLM 的文档重排器"""

    def _build_llm(self) -> ChatQwen:
        return ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0,
            base_url=config.dashscope_api_base,
            enable_thinking=config.dashscope_enable_thinking,
        )

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """按相关度重排 docs，返回前 top_k 个。任何异常回退为原顺序。"""
        if not docs:
            return []
        top_k = min(top_k, len(docs))
        if len(docs) <= 1:
            return docs[:top_k]

        t0 = time.perf_counter()
        try:
            snippet = lambda i, d: (  # noqa: E731
                d.page_content[: config.rag_rerank_max_chars].replace("\n", " ")
            )
            numbered = "\n".join(f"[{i + 1}] {snippet(i, d)}" for i, d in enumerate(docs))

            llm = self._build_llm()
            structured = llm.with_structured_output(RerankOrder)
            result = structured.invoke(
                [
                    (
                        "system",
                        "你是一个检索重排器。根据文档片段与查询的相关程度，把文档从高到低排序。"
                        "只输出排序后的编号数组，不要输出解释。",
                    ),
                    (
                        "human",
                        f"查询：{query}\n\n候选文档片段：\n{numbered}\n\n"
                        f"请输出这 {len(docs)} 个文档按相关度从高到低排列的编号数组。",
                    ),
                ]
            )
            order = getattr(result, "order", None)
            if not order:
                return docs[:top_k]

            # 校验 + 去重 + 补全，保证返回完整顺序
            seen, ordered = set(), []
            for idx in order:
                i = int(idx) - 1
                if 0 <= i < len(docs) and i not in seen:
                    seen.add(i)
                    ordered.append(i)
            for i in range(len(docs)):
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)

            reranked = [docs[i] for i in ordered][:top_k]
            logger.info(
                f"LLM 重排完成: {len(docs)} 候选 -> top{top_k}, "
                f"耗时{(time.perf_counter() - t0) * 1000:.0f}ms"
            )
            logger.debug(f"LLM 原始输出顺序(索引1基): {order[:len(docs)]}")
            return reranked

        except Exception as e:
            logger.warning(f"LLM 重排失败，回退原顺序: {e}")
            return docs[:top_k]


# 全局单例
rerank_service = LLMReranker()
