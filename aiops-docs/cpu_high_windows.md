# Windows服务器CPU使用率过高告警处理方案

## 告警名称
WIN_CPU_HIGH

## 问题描述
Windows Server 上 CPU 持续跑满，站点响应变慢。定位占用进程时用的工具和 Linux 完全不同（没有 top/vmstat），需要借助 tasklist、wmic、任务管理器或 PerfMon。

## 排查步骤

### 步骤1: 获取当前时间
记录告警时间，便于对齐 IIS 日志与应用日志。

### 步骤2: 用 tasklist 找高 CPU 进程
```cmd
tasklist | sort /R
```
或用任务管理器「详细信息」按 CPU 排序，记录占用最高的 PID 与进程名。

### 步骤3: 定位到 IIS 站点进程
```cmd
tasklist /fi "PID eq <pid>"
```
如果高 CPU 进程是 w3wp.exe，说明是某个 IIS 应用池，用 `appcmd list wp` 反查是哪个站点/应用池。

### 步骤4: 用 wmic/perfmon 细化
```cmd
wmic path Win32_PerfFormattedData_PerfProc_Process get Name,IDProcess,PercentProcessorTime
```
或用性能监视器（PerfMon）加计数器 Process\% Processor Time 观察。

### 步骤5: 结合应用日志
搜索该站点告警时间前后的请求日志、异常堆栈。

## 常见原因分析

### 原因1: W3WP（IIS 应用池）死循环/密集计算
站点代码里存在死循环或大计算量请求，w3wp.exe CPU 飙高。

### 原因2: 应用池回收异常
回收间隔配置不当或内存暴涨触发频繁回收。

### 原因3: 第三方扩展/模块占用
ISAPI 扩展、杀毒扫描等把 CPU 吃掉。

## 处置建议
1. 先重启对应 IIS 应用池临时止血（iisreset /apppool 或回收）。
2. 抓到 w3wp 后做 dump（procdump）分析热点。
3. 修复代码后灰度观察 CPU。
