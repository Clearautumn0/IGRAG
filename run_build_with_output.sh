#!/bin/bash
# 运行密集描述知识库构建脚本，带详细输出

cd /home/m025/qqw/IGRAG

echo "=========================================="
echo "  开始构建密集描述知识库"
echo "=========================================="
echo ""
echo "时间: $(date)"
echo "环境: IGRAG conda环境"
echo ""

# 临时设置日志级别为INFO以便查看进度
export PYTHONPATH=/home/m025/qqw/IGRAG:$PYTHONPATH

# 运行脚本并实时显示输出
conda run -n IGRAG python -c "
import sys
import logging

# 临时设置日志级别为INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入并运行主脚本
sys.path.insert(0, '/home/m025/qqw/IGRAG')
from scripts.build_dense_knowledge_base import main

if __name__ == '__main__':
    print('正在启动脚本...')
    main()
"
