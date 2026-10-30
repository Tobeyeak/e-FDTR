#!/bin/bash
# DEIM模型导出ONNX脚本

# 绝对路径
PYTHON="/opt/miniforge3/envs/DETR/bin/python"
EXPORT_PY="/home/tione/notebook/tjn/DETR/e-FDTR/export.py"
CONFIG_FILE="/home/tione/notebook/tjn/DETR/e-FDTR/configs/dfine/e-FDTR-n.yml"
CHECKPOINT="/home/tione/notebook/tjn/DETR/Z-test/deim3/best_stg1.pth"

echo "=========================================="
echo "DEIM模型导出ONNX格式"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "检查点: $CHECKPOINT"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 检查点文件不存在 - $CHECKPOINT"
    echo "请先训练模型或指定正确的检查点路径"
    exit 1
fi

# 执行ONNX导出命令
$PYTHON $EXPORT_PY \
    -c $CONFIG_FILE \
    -r $CHECKPOINT \
    --format onnx

# 检查导出结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "ONNX导出成功完成!"
    echo "=========================================="
else
    echo "=========================================="
    echo "ONNX导出失败!"
    echo "=========================================="
    exit 1
fi
