# -*- coding: utf-8 -*-
"""Mock Prometheus —— 在 127.0.0.1:9090 模拟 Prometheus 告警 API。

用途：让 app/tools/query_metrics_alerts.py 的 `GET /api/v1/alerts` 有真实输入可返回，
从而把 AIOps「从告警到诊断报告」的链路真正跑通（真实环境中请替换为真实 Prometheus）。

返回的告警刻意与 knowledge 语料 / MCP mock 服务(data-sync-service)对齐，方便对账。
启动：python mock_prometheus.py   （start-windows.bat 也会自动拉起）
"""
import datetime
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Mock Prometheus", version="1.0.0")


def _now_minus(minutes: float) -> str:
    """返回 N 分钟前的 RFC3339(UTC) 字符串，模拟 activeAt。"""
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    return dt.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


def _alerts_payload():
    """返回与真实 Prometheus `/api/v1/alerts` 结构一致的告警列表。"""
    now = time.time()
    return {
        "status": "success",
        "data": {
            "alerts": [
                {
                    "labels": {
                        "alertname": "NodeCPUUsageHigh",
                        "severity": "critical",
                        "instance": "10.0.1.12:9100",
                        "job": "node-exporter",
                        "namespace": "prod",
                        "node": "node-12",
                    },
                    "annotations": {
                        "summary": "CPU 使用率持续高于 90%",
                        "description": "节点 node-12 (10.0.1.12) CPU 使用率 96.2% 超过阈值 90%，已持续约 15 分钟",
                    },
                    "state": "firing",
                    "activeAt": _now_minus(15),
                    "value": "1e+00",
                },
                {
                    "labels": {
                        "alertname": "NodeDiskUsageHigh",
                        "severity": "warning",
                        "instance": "10.0.1.12:9100",
                        "job": "node-exporter",
                        "namespace": "prod",
                        "node": "node-12",
                    },
                    "annotations": {
                        "summary": "根分区磁盘使用率超过 85%",
                        "description": "节点 node-12 根分区 / 使用率 91%，请检查 /var/log 日志与临时文件",
                    },
                    "state": "firing",
                    "activeAt": _now_minus(95),
                    "value": "1e+00",
                },
                {
                    "labels": {
                        "alertname": "ServiceDown",
                        "severity": "critical",
                        "instance": "10.0.1.12:8080",
                        "job": "data-sync-service",
                        "namespace": "prod",
                    },
                    "annotations": {
                        "summary": "data-sync-service 服务不可用",
                        "description": "data-sync-service 探活失败，最近 5 分钟 5xx 比例持续上升",
                    },
                    "state": "firing",
                    "activeAt": _now_minus(6),
                    "value": "1e+00",
                },
                {
                    "labels": {
                        "alertname": "NodeMemoryUsageHigh",
                        "severity": "warning",
                        "instance": "10.0.1.13:9100",
                        "job": "node-exporter",
                        "namespace": "prod",
                        "node": "node-13",
                    },
                    "annotations": {
                        "summary": "节点内存使用率偏高",
                        "description": "节点 node-13 内存使用率 88%，接近阈值，尚未触发 firing",
                    },
                    "state": "pending",
                    "activeAt": _now_minus(30),
                    "value": "1e+00",
                },
            ]
        },
    }


@app.get("/api/v1/alerts")
def alerts():
    """Prometheus Alerts HTTP API（与 query_metrics_alerts.py 对接）。"""
    return _alerts_payload()


@app.get("/-/healthy")
def healthy():
    """Prometheus 健康检查（工具只用到 alerts，这里保证探测也 200）。"""
    return PlainTextResponse("Prometheus Server is Healthy.\n")


if __name__ == "__main__":
    uvicorn.run("mock_prometheus:app", host="127.0.0.1", port=9090, log_level="info")
