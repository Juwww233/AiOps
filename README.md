# AiOps

> 面向研发运维场景的 AI Agent 系统：RAG 知识库问答 + AIOps 智能诊断

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5-blueviolet.svg)](https://milvus.io/)

---

## ✨ 核心特性

| 能力 | 说明 |
|---|---|
| **RAG 知识库问答** | 文档上传 → 两级分片 → 向量化 → Milvus 入库；检索走「稠密 + 稀疏」双路召回 → RRF 融合 → LLM 重排 |
| **AIOps 智能诊断** | 基于 LangGraph 的 Planner-Executor-Replanner 三阶段编排，完成「取告警 → 排查 → 生成诊断报告」闭环 |
| **智能对话** | ReAct Agent + 多轮会话记忆 + SSE 流式输出 |
| **MCP 工具集成** | FastMCP 服务端（streamable-http），日志与监控工具统一注册给 Agent |
| **文件管理** | 知识库文件上传 / 列表 / 删除，按 Milvus 元数据聚合分片数 |
| **检索评估** | 自建运维评测集 + 公开中文检索基准（T2Retrieval），支持四路策略对比 |

---

## 🛠️ 技术栈

| 层 | 选型 |
|---|---|
| Web 框架 | FastAPI + SSE（sse-starlette） |
| Agent 编排 | LangChain + LangGraph（StateGraph / ToolNode / MemorySaver） |
| 对话模型 | 阿里云 DashScope `qwen3.8-27b`（OpenAI 兼容模式） |
| 向量模型 | DashScope `text-embedding-v4`（1024 维） |
| 向量库 | Milvus 2.5（稠密向量 + 服务器端 BM25 稀疏向量） |
| 工具协议 | MCP（FastMCP，streamable-http） |
| 日志 | Loguru（按天轮转） |

---

## 🏗️ 架构概览

```
                          ┌──────────────────────────────┐
  浏览器 (static/) ──────► │  FastAPI  :9900              │
                          │                              │
                          │  /api/chat, /chat_stream ──► ReAct Agent
                          │  /api/aiops ──────────────► LangGraph 图
                          │  /api/upload, /api/files ──► 分片/向量化
                          └───────────┬──────────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐
        │ CLS MCP   :8003  │ │ Monitor MCP :8004│ │ Milvus :19530│
        │ FastMCP · 5 工具 │ │ FastMCP · 2 工具 │ │ 稠密 + 稀疏  │
        └──────────────────┘ └──────────────────┘ └──────────────┘
                                                          ▲
                                          Mock Prometheus :9090
```

**AIOps 诊断流程**

```
        ┌─────────┐      固定边      ┌──────────┐      固定边      ┌───────────┐
 input ─►│ Planner ├────────────────►│ Executor ├────────────────►│ Replanner │
        └─────────┘                 └────┬─────┘                 └─────┬─────┘
             ▲                           │ 调工具（本地 + MCP）         │
             │                           ▼                             │ 条件边
             │                     ┌───────────┐                       │
             │                     │ ToolNode  │                       │
             │                     └───────────┘                       │
             └────────────── 还有剩余 plan 且无 response ◄─────────────┤
                                                                       ▼
                                                        有 response / plan 空 → END
```

---

## 🚀 快速开始

### 环境要求

- **Python 3.13**（见 `.python-version`）
- **uv**（推荐包管理）或 pip
- **Docker Desktop**（运行 Milvus）
- **阿里云 DashScope API Key**：[获取地址](https://bailian.console.aliyun.com/)

### 1. 安装依赖

```powershell
cd C:\Users\ASUS\Workspaceidea\AiOps

# 使用 uv（推荐）
uv sync

# 或使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. 配置 `.env`

```powershell
notepad .env
```

至少填写：

```ini
DASHSCOPE_API_KEY=sk-xxxxxxxx
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.8-27b
```

> ⚠️ `DASHSCOPE_API_BASE` 建议显式配置：`ChatQwen` 默认走新加坡站点（intl），部分 Key 只在主站可用。

### 3. 启动（Windows 一键）

```powershell
.\start-windows.bat
```

脚本依次完成：

```
[1/9] 检查 uv          [2/9] 检查 Python 版本
[3/9] 同步虚拟环境      [4/9] 启动 Milvus（Docker Compose）
[5/9] 启动 CLS MCP     [6/9] 启动 Monitor MCP
[7/9] 启动 Mock Prometheus (9090)
[8/9] 启动 FastAPI      [9/9] 上传 aiops-docs 文档到向量库
```

停止：`.\stop-windows.bat`

### 4. Linux / macOS

```bash
uv sync
make init     # Docker + 服务 + 文档
make start    # 启动
make stop     # 停止
```

### 访问

| 服务 | 地址 |
|---|---|
| Web 界面 | http://localhost:9900 |
| API 文档 | http://localhost:9900/docs |
| Milvus 管理界面（Attu） | http://localhost:8000 |

---

## 📡 API 接口

| 功能 | 方法 | 路径 | 说明 |
|---|---|---|---|
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| 清空会话 | POST | `/api/chat/clear` | 清理该会话的 checkpoint |
| 获取会话信息 | GET | `/api/chat/session/{session_id}` | 查历史消息 |
| AIOps 诊断 | POST | `/api/aiops` | Plan-Execute-Replan，SSE 流式 |
| 上传文件 | POST | `/api/upload` | 保存并自动建索引 |
| 文件列表 | GET | `/api/files` | 按 `_source` 聚合分片数 |
| 删除文件 | DELETE | `/api/files/{name}` | 删向量 + 删原文件 |
| 索引目录 | POST | `/api/index_directory` | 批量索引 |
| 健康检查 | GET | `/health` | Milvus 连接状态 |

### 请求示例

```bash
# 普通对话（注意字段名是大写 Id / Question）
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"CPU 使用率过高怎么排查"}'

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}' --no-buffer

# 文件列表
curl http://localhost:9900/api/files
```

---

## 📁 项目结构

```
AiOps/
├── app/
│   ├── main.py                          # FastAPI 入口（lifespan 中连接 Milvus）
│   ├── config.py                        # 配置（Pydantic Settings 读 .env）
│   ├── api/
│   │   ├── chat.py                      # 对话 / 流式 / 会话管理
│   │   ├── aiops.py                     # AIOps 诊断（SSE）
│   │   ├── file.py                      # 上传 / 列表 / 删除 / 目录索引
│   │   └── health.py                    # 健康检查
│   ├── agent/
│   │   ├── mcp_client.py                # MCP 客户端（MultiServerMCPClient）
│   │   └── aiops/
│   │       ├── planner.py               # ① 先检索经验 → 生成计划
│   │       ├── executor.py              # ② 只执行 plan[0]，内部小 ReAct 循环
│   │       ├── replanner.py             # ③ 决策：continue / replan / respond（含三层熔断）
│   │       ├── state.py                 # PlanExecuteState（past_steps 用 operator.add）
│   │       └── utils.py                 # 工具描述格式化
│   ├── services/
│   │   ├── aiops_service.py             # StateGraph 编排 + 条件边 + 事件格式化
│   │   ├── rag_agent_service.py         # 对话 Agent（create_agent + MemorySaver + 消息裁剪）
│   │   ├── document_splitter_service.py # 两级分片 + 小片段合并
│   │   ├── vector_embedding_service.py  # DashScope Embeddings（实现 LangChain 接口）
│   │   ├── vector_index_service.py      # 文件 → 分片 → 入库
│   │   ├── vector_store_manager.py      # Milvus VectorStore / 列表 / 删除
│   │   ├── vector_search_service.py     # dense / sparse / hybrid / rerank 四种检索
│   │   └── rerank_service.py            # LLM 重排（结构化输出排序）
│   ├── tools/
│   │   ├── knowledge_tool.py            # 知识库检索（内部 hybrid / QA 走 rerank）
│   │   ├── time_tool.py                 # 当前时间
│   │   └── query_metrics_alerts.py      # Prometheus 告警查询
│   ├── core/
│   │   └── milvus_client.py             # collection schema / 索引 / 迁移
│   ├── models/                          # Pydantic 模型
│   └── utils/logger.py                  # Loguru 配置
├── mcp_servers/
│   ├── cls_server.py                    # CLS 日志 MCP（:8003，5 个工具）
│   └── monitor_server.py                # 监控指标 MCP（:8004，2 个工具）
├── static/                              # 前端（index.html / app.js / styles.css）
├── aiops-docs/                          # 11 篇运维知识文档（启动时自动入库）
├── eval/
│   ├── dataset.json                     # 自建评测集（33 条，含类型标注）
│   ├── run_eval.py                      # 四路策略对比（Hit/MRR，按类型分层）
│   └── public/                          # 公开基准 T2Retrieval 评测脚本与结果
├── uploads/                             # 上传文件落盘目录
├── volumes/                             # Milvus 数据持久化（etcd/minio/milvus）
├── logs/                                # app_YYYY-MM-DD.log（按天轮转）
├── mock_prometheus.py                   # Mock 告警数据源（:9090，4 条固定告警）
├── vector-database.yml                  # Milvus Compose（etcd + minio + standalone + attu）
├── start-windows.bat / stop-windows.bat
├── Makefile
├── pyproject.toml
└── .env
```

---

## ⚙️ 配置说明

```ini
# ---- 应用 ----
APP_NAME=AiOps
DEBUG=True
HOST=0.0.0.0
PORT=9900

# ---- DashScope ----
DASHSCOPE_API_KEY=sk-xxxx
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.8-27b
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4

# ---- Milvus ----
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TIMEOUT=10000

# ---- RAG ----
RAG_TOP_K=3
RAG_MODEL=qwen3.8-27b
RAG_SEARCH_MODE=rerank        # dense | sparse | hybrid | rerank
RAG_RERANK_RECALL_K=8         # rerank 模式的召回窗口
RAG_RERANK_MAX_CHARS=500      # 重排时每片段截断长度

# ---- 分片 ----
CHUNK_MAX_SIZE=800            # 二级分割器实际用 2 倍（1600）
CHUNK_OVERLAP=100

# ---- MCP ----
MCP_CLS_TRANSPORT=streamable-http
MCP_CLS_URL=http://localhost:8003/mcp
MCP_MONITOR_TRANSPORT=streamable-http
MCP_MONITOR_URL=http://localhost:8004/mcp

# ---- Prometheus ----
PROMETHEUS_BASE_URL=http://127.0.0.1:9090
PROMETHEUS_REQUEST_TIMEOUT=10.0
```

---

## 🔍 检索设计

### 入库链路

```
上传 → 按 Markdown 标题切（h1/h2）
     → 递归字符二次切分（1600 字符 / overlap 100）
     → 合并 < 300 字符碎片
     → text-embedding-v4 生成 1024 维向量
     → 写入 Milvus（content 字段同时被 BM25 Function 生成稀疏向量）
```

### 检索链路

| 模式 | 说明 |
|---|---|
| `dense` | 纯稠密：L2 距离 TopK |
| `sparse` | 纯 BM25：query 文本直传，服务器端分词打分 |
| `hybrid` | 稠密 + 稀疏双路 → **RRF** 融合 |
| `rerank` | hybrid 召回 N 条 → **LLM 重排** → TopK |

**检索分层**：Agent 内部推理固定用 `hybrid`（低延迟），只有最终面向答案的检索走 `rerank`。

### Milvus 关键配置

```python
content 字段：VARCHAR(8000字节)，enable_analyzer=True，tokenizer=jieba
sparse 字段：SPARSE_FLOAT_VECTOR，由 BM25 Function 自动生成
sparse 索引：SPARSE_INVERTED_INDEX，metric_type 必须为 BM25（用 IP 会被拒绝）
dense 索引 ：IVF_FLAT，L2
```

---

## 📊 检索评估

### 自建运维评测集（`eval/`）

- **语料**：11 篇运维文档（含 1 篇纯干扰文档，不给 gold）
- **问题**：33 条，按类型标注（语义型 12 / 精确词型 21）
- **指标**：文件级 Hit@k、MRR

| 策略 | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| dense | 0.909 | 0.970 | 0.942 |
| sparse | 0.697 | 0.879 | 0.803 |
| hybrid | 0.818 | 0.970 | 0.897 |
| **rerank** | **0.909** | **1.000** | **0.950** |

### 公开基准 T2Retrieval（`eval/public/`）

- 数据集：MTEB / C-MTEB `T2Retrieval`（论文 [arXiv:2304.03679](https://arxiv.org/abs/2304.03679)，Apache-2.0）
- 子集：1 万语料 / 200 query / 1041 条标注，候选窗口统一 20、截断 10

| 策略 | nDCG@10 | MRR@10 | Recall@10 |
|---|---|---|---|
| dense | 0.9069 | 0.9490 | 0.8862 |
| sparse | 0.7725 | 0.8773 | 0.7533 |
| hybrid | 0.8866 | 0.9360 | 0.8874 |
| **rerank** | **0.9428** | **0.9690** | **0.9244** |

**结论**：混合召回的 Recall 略高但排序质量下降（RRF 只看名次不看分值）；引入 LLM 重排后 nDCG 反超稠密单路。

### 运行评测

```powershell
# 自建集（需 Milvus 已启动且文档已入库）
.venv\Scripts\python.exe eval\run_eval.py

# 公开基准（需先下载数据并向量化，脚本见 eval/public/）
.venv\Scripts\python.exe eval\public\run_public_eval.py
.venv\Scripts\python.exe eval\public\run_rerank_eval.py
.venv\Scripts\python.exe eval\public\run_baseline_samek.py
```

---

## 🤖 AIOps 诊断流程

```
1. Planner：检索知识库历史经验 → 结合工具列表生成结构化计划
2. Executor：只执行 plan[0]，内部「LLM 决策 → ToolNode 执行 → LLM 收尾」
3. Replanner：输出 continue / replan / respond
4. 熔断保护：
   · 已执行 ≥ 8 步 → 强制收尾
   · 已执行 ≥ 5 步 → 禁止重规划
   · replan 新步骤数 ≤ 剩余步骤数
5. 生成诊断报告 → SSE 推送 → 前端渲染
```

**告警数据源**：`mock_prometheus.py` 在 `127.0.0.1:9090` 返回 4 条固定告警（CPU 高 / 磁盘高 / 服务不可用 / 内存高）。生产环境替换为真实 Prometheus 即可，工具代码无需改动。

---

## 🐛 常见问题

### 1. Milvus 连接失败

```powershell
# 确认 Docker Desktop 已启动
docker ps

# 启动 Milvus
docker compose -f vector-database.yml up -d

# 若提示容器名冲突（旧容器残留）
docker rm -f milvus-attu milvus-standalone milvus-etcd milvus-minio
docker compose -f vector-database.yml up -d
```

### 2. `make` 命令不可用（Windows）

使用 `.\start-windows.bat` / `.\stop-windows.bat` 代替。

### 3. 端口被占用

```powershell
netstat -ano | findstr :9900
taskkill /F /PID <PID>
```

### 4. DashScope 报 `Free quota exhausted` / `Arrearage`

- 免费额度耗尽 → 控制台充值，或换有额度的模型
- `Arrearage` → 账户欠费，需充值后恢复

### 5. `qwen3.8-27b` 报 `Required body invalid`

该模型是思考模型，必须显式传 `enable_thinking=false`（项目已在 `config.dashscope_enable_thinking` 中处理）。

### 6. 日志在哪

```powershell
Get-Content logs\app_$(Get-Date -Format yyyy-MM-dd).log -Tail 50   # 主服务
type mcp_cls.log                                                   # CLS MCP
type mcp_monitor.log                                               # Monitor MCP
docker logs --tail 100 milvus-standalone                           # Milvus
```

---

## 📚 参考资源

- [LangGraph Plan-and-Execute 教程](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Milvus 稀疏向量与 BM25](https://milvus.io/docs/sparse_vector.md)
- [MCP 协议](https://modelcontextprotocol.io/)
- [阿里云百炼](https://bailian.console.aliyun.com/)
