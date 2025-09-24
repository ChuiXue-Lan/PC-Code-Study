#!/bin/bash

cd /home/cls2024/ltx/Replicate/TMB/

set -x

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

time=`date +%m-%d_%H-%M-%S`

# torchrun --nproc_per_node=4 --master_port=12345 train_3modal_ULIP.py --output_dir ./outputs/modelnet40_crossmost/pretrain/ --config ./configs/3modal_ULIP.yaml >outputs/modelnet40_crossmost/pretrain/$time.out 
# torchrun --nproc_per_node=4 --master_port=12345 train_3modal_ULIP.py --output_dir /data/Lan/outputs/CrossMoST-main/pretrain/ --config ./configs/3modal_ULIP.yaml >outputs/modelnet40_crossmost/pretrain/$time.out --resume /home/cls2024/ltx/Replicate/CrossMoST-main/outputs/modelnet40_crossmost/pretrain/checkpoint-last.pth
torchrun --nproc_per_node=1 --master_port=12345 train_3modal_ULIP.py --output_dir /data/Lan/outputs/TMB/pretrain/ --config ./configs/3modal_ULIP.yaml >outputs/modelnet40_crossmost/pretrain/$time.out