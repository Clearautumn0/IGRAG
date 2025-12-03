#!/bin/bash
# 进度查看快捷命令

alias check-progress='python /home/m025/qqw/IGRAG/check_build_progress.py'
alias watch-progress='python /home/m025/qqw/IGRAG/check_build_progress.py --watch'

echo "快捷命令已创建："
echo "  check-progress  - 查看一次进度"
echo "  watch-progress  - 持续监控进度"
echo ""
echo "使用前请运行: source .progress_alias.sh"
