#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/5  14:41
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-
import sys
import torch

print("Python版本为：", sys.version)

print("cuda是否可用:", torch.cuda.is_available())
print("GPU个数:", torch.cuda.device_count())
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print('CUDA version:', torch.version.cuda)


