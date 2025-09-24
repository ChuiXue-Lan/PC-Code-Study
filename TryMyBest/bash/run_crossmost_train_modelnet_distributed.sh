#!/bin/bash

cd /home/cls2024/ltx/Replicate/CrossMoST-main/

set -x

export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

time=`date +%m-%d_%H-%M-%S`

torchrun --nproc_per_node=3 --master_port=12345 ./train_CrossMoST_modelnet40.py --output_dir ./outputs/modelnet40_crossmost/distributed/ --config ./configs/modelnet40_crossmost.yaml >outputs/modelnet40_crossmost/$time.out 