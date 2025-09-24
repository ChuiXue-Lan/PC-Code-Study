#!/bin/bash

# 设置CPU相关环境变量
export MKL_DEBUG_CPU_TYPE=5
export OPENBLAS_TARGET_ARCH=HASWELL
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 设置CUDA相关环境变量
export CUDA_VISIBLE_DEVICES=1
export NCCL_DEBUG=INFO
export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1
# 添加调试信息
export PYTHONFAULTHANDLER=1
export TORCH_USE_CUDA_DSA=0

# 切换到项目目录
cd "$(dirname "$0")"

time=`date +%m-%d_%H-%M-%S`

# 打印环境信息
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA version:"
python -c "import torch; print(torch.version.cuda)"
echo "GPU info:"
nvidia-smi

# 使用Python的调试模式运行
python -X faulthandler train_CrossMoST_modelnet40.py \
    --output_dir ./outputs/modelnet40_crossmost/ \
    --config ./configs/modelnet40_crossmost.yaml > outputs/modelnet40_crossmost/$time.out 2>&1