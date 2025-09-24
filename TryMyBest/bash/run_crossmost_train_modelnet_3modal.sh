#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2

# 设置分布式训练参数
WORLD_SIZE=3
BATCH_SIZE=32
NUM_WORKERS=8

# 设置输出目录
OUTPUT_DIR="outputs/modelnet40_crossmost_3modal"
mkdir -p $OUTPUT_DIR

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=29500 \
    train_CrossMoST_modelnet40_3modal.py \
    --config config/modelnet40_crossmost_3modal.yaml \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs 200 \
    --warmup_epochs 10 \
    --eval_freq 1 \
    --save_freq 10 \
    --lr 2e-4 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --layer_decay 0.75 \
    --model_ema \
    --model_ema_decay 0.9998 \
    --clip_grad 1.0 \
    --pin_memory \
    --wandb \
    --run_id "crossmost_3modal_$(date +%Y%m%d_%H%M%S)" 