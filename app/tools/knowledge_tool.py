"""知识检索工具 - 从向量数据库中检索相关信息"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_search_service import vector_search_service


def _retrieve(query: str, mode: str) -> Tuple[str, List[Document]]:
    """内部实现。mode: hybrid(混合召回, 内部/低延迟) | rerank(混合召回+LLM重排, 面向最终回答)"""
    try:
        logger.info(f"知识检索工具被调用: query='{query}', mode={mode}")

        if mode == "rerank":
            # 三阶段：混合召回 -> LLM 重排 -> Top-K（供最终回答引用）
            docs = vector_search_service.rerank_search_documents(
                query,
                top_k=config.rag_top_k,
                recall_k=config.rag_rerank_recall_k,
            )
        else:
            # 内部固定混合召回：dense + BM25 sparse 双路 + RRF，避免每个内部步骤都触发 LLM 重排
            docs = vector_search_service.hybrid_search_documents(query, config.rag_top_k)

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"检索到 {len(docs)} 个相关文档 (mode={mode})")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    内部统一使用混合召回（dense + BM25 稀疏 + RRF），固定不随 rag_search_mode 重排，
    供 Agent 内部步骤/经验召回使用，控制额外 LLM 调用开销。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    return _retrieve(query, "hybrid")


@tool(response_format="content_and_artifact")
def retrieve_knowledge_rerank(query: str) -> Tuple[str, List[Document]]:
    """从知识库中高精度检索相关信息来回答问题

    面向最终回答的知识查询：混合召回 + LLM 重排后再引用（当 rag_search_mode=hybrid 时退化为混合召回）。
    当用户的问题需要精确资料来组织最终答复时使用。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    mode = "rerank" if config.rag_search_mode == "rerank" else "hybrid"
    return _retrieve(query, mode)


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)
