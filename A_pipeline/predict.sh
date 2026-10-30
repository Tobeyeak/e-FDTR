#!/bin/bash
# DEIM推理脚本

# 绝对路径
PYTHON="/opt/miniforge3/envs/DETR/bin/python"
TORCH_INF="/home/tione/notebook/tjn/DETR/e-FDTR/tools/inference/detect/torch_inf.py"
CONFIG_FILE="/home/tione/notebook/tjn/DETR/e-FDTR/configs/dfine/e-FDTR-n.yml"
CHECKPOINT="/home/tione/notebook/tjn/DETR/Z-test/deim3/best_stg1.pth"
INPUT_IMAGE="/home/tione/notebook/tjn/DETR/test_img/Helicopter"
OUTPUT_DIR="/home/tione/notebook/tjn/DETR/Pre/MergE"
DEVICE="cuda:0"

echo "=========================================="
echo "DEIM模型推理脚本"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "检查点: $CHECKPOINT"
echo "输入路径: $INPUT_IMAGE"
echo "设备: $DEVICE"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 检查点文件不存在 - $CHECKPOINT"
    exit 1
fi

if [ ! -e "$INPUT_IMAGE" ]; then
    echo "错误: 输入路径不存在 - $INPUT_IMAGE"
    exit 1
fi

if [ -d "$INPUT_IMAGE" ]; then
    echo "检测到输入为文件夹，将遍历其中的 jpg/jpeg/png/bmp 图像进行检测"
fi

mkdir -p "$OUTPUT_DIR"

# 执行推理命令
$PYTHON $TORCH_INF \
    -c "$CONFIG_FILE" \
    -r "$CHECKPOINT" \
    --input "$INPUT_IMAGE" \
    --device "$DEVICE" \
    -o "$OUTPUT_DIR"

echo "推理完成!"
