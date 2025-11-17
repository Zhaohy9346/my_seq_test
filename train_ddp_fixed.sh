#!/bin/bash

# 设置DDP环境变量
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 配置文件
CONFIG_FILE="${1:-configs/cod_dataset.yaml}"
NUM_GPUS="${2:-2}"
RESUME_WEIGHTS="${3:-}"

echo "=========================================="
echo "Training with DDP Fix"
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "Resume: $RESUME_WEIGHTS"
echo "=========================================="

if [ -z "$RESUME_WEIGHTS" ]; then
    # 从头开始训练
    python train_net.py \
        --num-gpu $NUM_GPUS \
        --config-file $CONFIG_FILE \
        SOLVER.IMS_PER_GPU 8 \
        INPUT.FT_SIZE_TRAIN 384 \
        INPUT.FT_SIZE_TEST 384
else
    # 从checkpoint恢复
    python train_net.py \
        --num-gpu $NUM_GPUS \
        --config-file $CONFIG_FILE \
        --resume \
        MODEL.WEIGHTS $RESUME_WEIGHTS \
        SOLVER.IMS_PER_GPU 8 \
        INPUT.FT_SIZE_TRAIN 384 \
        INPUT.FT_SIZE_TEST 384
fi