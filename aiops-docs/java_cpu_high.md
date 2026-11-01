# Java应用CPU使用率过高告警处理方案

## 告警名称
JVM_CPU_HIGH

## 问题描述
Java 进程占用 CPU 过高，例如 java 进程整体 CPU 超过 100%（多核超过 N 核），接口响应明显变慢或出现超时，需要定位到具体线程甚至代码行。

## 排查步骤

### 步骤1: 获取当前时间
记录告警时间，便于回看应用日志与监控曲线。

### 步骤2: 确认是哪个 Java 进程占 CPU
```bash
top -c
```
找到 CPU 占用最高的 java 进程，记录其 PID。

### 步骤3: 线程级定位
```bash
top -H -p <pid>
```
找到 CPU 最高的线程，记录其 TID（十进制）。

### 步骤4: 用 jstack 打印线程栈
```bash
# 十进制 TID 转十六进制（jstack 里 nid 是十六进制）
printf '%x\n' <TID>
jstack <pid> | grep -A 30 "nid=0x<十六进制>"
```
重点看线程处于什么状态：RUNNABLE 死循环、WAITING 锁等待、还是频繁 GC 线程。

### 步骤5: 观察 GC 频率
```bash
jstat -gcutil <pid> 1000
```
如果 FGC 频繁且 FGCT 很高，说明 Full GC 把 CPU 打满。

### 步骤6: 结合应用日志
搜索告警时间点前后的业务日志、慢日志、异常堆栈。

## 常见原因分析

### 原因1: 热点代码死循环
业务线程 while 空转或算法未收敛，线程栈停在某个业务方法反复执行。

### 原因2: 频繁 Full GC
堆内存太小或存在内存泄漏，GC 线程持续回收导致 CPU 飙升，伴随 FGC 高、Old 区打满。

### 原因3: 正则灾难性回溯
复杂正则匹配超长字符串时发生回溯爆炸，卡在 java.util.regex 的匹配方法。

### 原因4: 锁竞争与自旋忙等
多个线程竞争同一把锁，等待线程被反复唤醒，栈顶集中在 LockSupport.park 附近的高频 spin。

## 处置建议
1. 先保存现场：连续多次 jstack 到文件、必要时 jmap -dump:format=b 导出堆。
2. 定位热点后修复代码；Full GC 引起则调整 -Xmx 或排查泄漏。
3. 变更后灰度观察 GC 曲线与 CPU 是否回落。
