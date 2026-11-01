# 运维常用命令速查表（Linux）

> 说明：本篇是把常见命令集中在一起的中性速查，不针对任何单一告警场景，避免和具体处理方案文档混淆。

## CPU 相关
- `top`：实时看 CPU 占用最高的进程。
- `vmstat 1`：看整体 CPU 的 user/sys/iowait 分布。
- `mpstat -P ALL 1`：逐核查看使用率。

## 内存相关
- `free -h`：看物理内存与 swap 使用。
- `ps aux --sort=-rss`：按物理内存排序找大进程。

## 磁盘相关
- `df -h`：看各分区使用率。
- `du -sh *`：看当前目录下各子目录大小。
- `lsof +L1`：找已删除但仍占用空间的文件。

## 日志相关
- `tail -f /var/log/xxx.log`：跟踪应用日志。
- `grep -i "error" /var/log/xxx.log | tail -50`：筛错误。

## 进程/端口
- `ss -lntp` 或 `netstat -lntp`：看端口监听。
- `kill -9 <pid>`：强制结束进程（慎用）。

## 网络
- `ping`：连通性。
- `mtr`：路径与丢包。
- `curl -v`：看 HTTP 请求细节。

> 速查表只列命令与一句用途，不展开具体故障的处理步骤。
