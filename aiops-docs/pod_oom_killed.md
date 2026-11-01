# Pod内存超限被杀(OOMKilled)告警处理方案

## 告警名称
K8S_POD_OOMKILLED

## 问题描述
Kubernetes 中某个 Pod 反复重启，事件里显示 OOMKilled，容器被内核因超过内存 limit 杀掉，退出码为 137。

## 排查步骤

### 步骤1: 获取当前时间
记录 Pod 被杀的时间点，用于对齐监控与事件。

### 步骤2: 查看 Pod 事件与上次状态
```bash
kubectl describe pod <pod-name> -n <namespace>
```
重点看 Last State 是否为 OOMKilled、Exit Code 是否为 137，Events 里是否出现 Killed。

### 步骤3: 对比实际内存使用与 limit
```bash
kubectl top pod <pod-name> -n <namespace>
```
看当前 usage 是否贴近 requests/limits；再看容器 YAML 里 resources.limits.memory 配了多少。

### 步骤4: 进入容器看进程内存
```bash
kubectl exec -it <pod-name> -n <namespace> -- ps aux --sort=-rss
```
确认是哪个进程 RSS 大，Java 应用可结合 jmap/jstat 或 GC 日志判断堆外内存。

### 步骤5: 查应用侧监控
内存曲线是否为持续上涨（泄漏）还是瞬时尖峰（大对象/流量突增）。

## 常见原因分析

### 原因1: 内存泄漏
缓存、静态集合、未释放的连接对象持续堆积，RSS 只升不降，最终触顶被杀。

### 原因2: 缓存或连接池无上限增长
本地缓存不设上限、线程池/连接池按需膨胀，堆外(off-heap)内存尤其容易把容器打爆。

### 原因3: 一次性大对象
大查询结果集、大文件读取一次性进内存造成瞬时尖峰。

### 原因4: resources.limits 设置过小
应用合理峰值本身就接近 limit，被频繁压到边界。

## 处置建议
1. 先调大 resources.limits.memory（或移除不合理的小 limit）止血恢复。
2. 分析内存曲线：泄漏需 dump 分析；尖峰需限流/分批。
3. 为 Java 应用合理设置 -Xmx 与容器 limit 的比例，预留堆外空间。
