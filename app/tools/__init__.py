"""工具模块 - 供 Agent 调用的各种工具"""

from app.tools.knowledge_tool import retrieve_knowledge, retrieve_knowledge_rerank
from app.tools.query_metrics_alerts import query_prometheus_alerts
from app.tools.time_tool import get_current_time

# 内部工具集（Agent 推理 / 经验召回 / Executor 步骤）：
# 知识检索固定走混合召回(dense+BM25 RRF)，不随 rag_search_mode 重排，控制额外 LLM 调用。
DEFAULT_LOCAL_AGENT_TOOLS = (
    retrieve_knowledge,
    get_current_time,
    query_prometheus_alerts,
)

# 面向最终回答的工具集：知识检索会按 rag_search_mode 走 rerank(混合召回+LLM 重排)。
# 仅绑定在直接面向用户答案的对话 Agent 上。
QA_LOCAL_AGENT_TOOLS = (
    retrieve_knowledge_rerank,
    get_current_time,
    query_prometheus_alerts,
)

__all__ = [
    "DEFAULT_LOCAL_AGENT_TOOLS",
    "QA_LOCAL_AGENT_TOOLS",
    "retrieve_knowledge",
    "retrieve_knowledge_rerank",
    "get_current_time",
    "query_prometheus_alerts",
]
