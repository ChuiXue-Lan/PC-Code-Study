# Exponential Moving Average (EMA) of model updates
# References:
# Timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    
    模型指数滑动平均（Exponential Moving Average, EMA）类。

    主要作用：
    - 维护一个模型参数的滑动平均副本（self.ema），用于提升模型泛化能力和稳定性。
    - EMA常用于训练过程中，部分训练方案（如EfficientNet、MobileNet等）依赖EMA平滑权重以获得更优结果。
    - 通过设置decay参数控制滑动平均的更新速度，decay越大，历史权重影响越大。

    主要参数说明：
    - model: 需要进行EMA的原始模型。
    - decay: EMA的衰减系数，通常接近1（如0.9999），控制新旧权重的融合比例。
    - device: 可选，指定EMA模型存放的设备（如'cpu'），可节省显存，但需手动验证EMA权重。
    - resume: 可选，若指定则从已有的EMA权重字典恢复。

    主要方法说明：
    - __init__：初始化，深拷贝原始模型，设置为eval模式，冻结参数，必要时加载已有权重。
    - _load_checkpoint：从checkpoint字典加载EMA权重，自动处理DataParallel前缀问题。
    - update：用当前模型参数更新EMA权重，实现公式：
        ema_v = ema_v * decay + model_v * (1 - decay)
      支持自动处理DataParallel前缀和设备迁移。

    使用注意事项：
    - EMA模型参数不参与梯度计算。
    - EMA模型初始化、分配设备、分布式封装的顺序需谨慎。
    """

    def __init__(self, model, decay=0.9999, device='', resume=''):
        # 深拷贝一份模型用于EMA
        self.ema = deepcopy(model)
        self.ema.eval()  # EMA模型始终处于评估模式
        self.decay = decay
        self.device = device  # 可选，EMA模型存放的设备
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')  # 判断是否有module前缀（DataParallel）
        if resume:
            self._load_checkpoint(resume)
        # 冻结EMA模型参数，不参与梯度计算
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint):
        """
        从checkpoint字典加载EMA权重，自动处理DataParallel前缀。
        """
        assert isinstance(checkpoint, dict)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            # 如果EMA模型有module前缀但权重没有，则补上
            if self.ema_has_module:
                name = 'module.' + k if not k.startswith('module') else k
            else:
                name = k
            new_state_dict[name] = v
        self.ema.load_state_dict(new_state_dict)
        print("Loaded state_dict_ema")

    def update(self, model):
        """
        用当前模型参数更新EMA权重。
        """
        # 判断是否需要补module前缀
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                # EMA更新公式
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
            