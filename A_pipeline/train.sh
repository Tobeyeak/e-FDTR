#!/bin/bash
# 训练脚本: e-FDTR-n 训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 绝对路径
TORCHRUN="/opt/miniforge3/envs/DETR/bin/torchrun"
TRAIN_PY="/home/tione/notebook/tjn/DETR/e-FDTR/train.py"
CONFIG_FILE="/home/tione/notebook/tjn/DETR/e-FDTR/configs/dfine/e-FDTR-n.yml"
OUTPUT_DIR="/home/tione/notebook/tjn/DETR/Z-test/deim"
PRETRAIN_WEIGHTS="/home/tione/notebook/tjn/DETR/e-FDTR/ckpt/deim_dfine_hgnetv2_n_coco_160e.pth"

# 训练参数
SEED=0
MASTER_PORT=7777
NUM_GPUS=4

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "开始训练配置: "
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "进程数: $NUM_GPUS"
echo "随机种子: $SEED"
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $CONFIG_FILE"
echo "预训练权重: $PRETRAIN_WEIGHTS"
echo "=========================================="

# 执行训练命令
$TORCHRUN \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS \
    $TRAIN_PY \
    -c $CONFIG_FILE \
    --seed=$SEED \
    --output-dir=$OUTPUT_DIR \
    --summary-dir=$OUTPUT_DIR/summary \
    -t $PRETRAIN_WEIGHTS \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "训练完成!"
