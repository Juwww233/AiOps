# 数据库连接池耗尽告警处理方案

## 告警名称
DB_CONNECTION_POOL_EXHAUSTED

## 问题描述
应用日志持续出现拿不到数据库连接的异常，例如 HikariCP 的 `Connection is not available, request timed out after 30000ms` 或 Druid 的 `wait millis 30000, active 100, maxActive 100`，接口大面积超时。

## 排查步骤

### 步骤1: 获取当前时间
记录开始报错的时间点，回看该时刻前后的慢查询与长事务。

### 步骤2: 看连接池监控
查看应用监控里连接池的 active / idle / pending（等待获取数），确认是否 active 长期等于 maximumPoolSize。

### 步骤3: 查数据库侧会话
```sql
-- MySQL
SHOW PROCESSLIST;
-- PostgreSQL
SELECT * FROM pg_stat_activity WHERE state <> 'idle' ORDER BY state_change;
```
重点找长时间未提交的事务、长时间运行的慢查询，以及大量 Sleep 状态的空闲连接。

### 步骤4: 看应用线程栈是否卡在取连接
```
jstack <pid> | grep -B5 -A15 "getConnection"
```
大量线程阻塞在 `DriverManager.getConnection` / `HikariPool.getConnection`，说明都在等连接归还。

### 步骤5: 定位连接未归还的代码
检查是否所有 DAO/查询都走 try-with-resources 或 finally 关闭；排查线程池/异步任务里借了连接没还的情况。

## 常见原因分析

### 原因1: 慢SQL或长事务占住连接
一个事务里执行慢查询数分钟，期间连接一直被占用，池很快被占满。

### 原因2: 连接泄漏
代码某条路径获取连接后未 close，异常分支未释放，连接只借不还。

### 原因3: 连接池最大值配置过小
峰值并发超过 maximumPoolSize，大量请求排队等连接直至超时。

### 原因4: 突发流量
活动/秒杀等流量瞬间打满连接池。

## 处置建议
1. 临时调大 maximumPoolSize 或重启分批恢复，先止血。
2. 找到并 kill 掉占用的长事务/慢SQL，快速释放连接。
3. 根治：修连接泄漏、优化慢SQL、拆分长事务、为关键业务隔离独立连接池。
