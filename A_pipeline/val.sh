#!/bin/bash
# DEIM模型测试脚本（仅测试不训练）

# 绝对路径
TORCHRUN="/opt/miniforge3/envs/DETR/bin/torchrun"
TRAIN_PY="/home/tione/notebook/tjn/DETR/e-FDTR/train.py"
CONFIG_FILE="/home/tione/notebook/tjn/DETR/e-FDTR/configs/dfine/e-FDTR-n.yml"
CHECKPOINT="/home/tione/notebook/tjn/DETR/Z-test/deim3/best_stg1.pth"
OUTPUT_DIR="/home/tione/notebook/tjn/DETR/Z-test/deim3-val"
GPU_IDS="0"
MASTER_PORT=29500
NUM_GPUS=1

echo "=========================================="
echo "DEIM模型测试模式"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "检查点: $CHECKPOINT"
echo "输出目录: $OUTPUT_DIR"
echo "使用GPU: $GPU_IDS"
echo "进程数: $NUM_GPUS"
echo "主端口: $MASTER_PORT"
echo "模式: 仅测试 (--test-only)"
echo "=========================================="

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 检查点文件不存在 - $CHECKPOINT"
    echo "请确保检查点路径正确"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_IDS

$TORCHRUN \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS \
    $TRAIN_PY \
    -c $CONFIG_FILE \
    --test-only \
    --yolo-metrice \
    --output-dir $OUTPUT_DIR \
    -r $CHECKPOINT

echo "测试完成!"
echo "YOLO指标已保存至: $OUTPUT_DIR/paper_data.txt"
