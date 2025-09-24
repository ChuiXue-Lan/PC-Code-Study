#!/bin/bash

CUDA_VISIBLE_DEVICES=1

cd /data/Lan/code/CrossMoST-main/

set -x

export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

time=`date +%m-%d_%H-%M-%S`

python \
    -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 train_CrossMoST_modelnet40depth.py \
    --output_dir ./outputs/modelnet40_crossmost_h5/ \
    --config ./configs/modelnet40_crossmost_h5.yaml >outputs/modelnet40_crossmost_h5/$time.out