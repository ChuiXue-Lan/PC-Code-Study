#!/bin/bash

cd /home/cls2024/ltx/Replicate/CrossMoST-main/

set -x

export CUDA_VISIBLE_DEVICES=1
export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

time=`date +%m-%d_%H-%M-%S`

torchrun --nproc_per_node=1 --master_port=12345 train_CrossMoST_modelnet40.py --output_dir ./outputs/modelnet40_crossmost/mini_32/ --config ./configs/modelnet40_crossmost.yaml >outputs/modelnet40_crossmost/mini_32/$time.out 