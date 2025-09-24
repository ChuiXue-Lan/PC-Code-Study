#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/5  14:41
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
# import sys
# import torch

# print("Python版本为：", sys.version)

# print("cuda是否可用:", torch.cuda.is_available())
# print("GPU个数:", torch.cuda.device_count())
# print("torch.__version__:", torch.__version__)
# print("torch.version.cuda:", torch.version.cuda)
# print('CUDA version:', torch.version.cuda)
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np

# 读取 NPY 文件
npy_file = "F:/Temp/test/shapenet_pc/02691156-1a04e3eab45ca15dd86060f189eb133.npy"
data = np.load(npy_file)

# 打印文件信息
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Data format:", data.dtype.byteorder)
print("First few points:")
print(data[:5])

# 保存为简单格式的 NPY 文件（确保是 little-endian）
data = data.astype('<f4')  # 转换为 little-endian float32
np.save("F:/Temp/test/shapenet_pc/converted.npy", data)



