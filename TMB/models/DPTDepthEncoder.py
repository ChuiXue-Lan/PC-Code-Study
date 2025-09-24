import torch
import torch.nn as nn
# from .dpt import DPT, _make_fusion_block, _make_encoder, forward_vit
from .dpt.models import DPT, _make_fusion_block
from .dpt.blocks import _make_encoder
from .dpt.vit import forward_vit
import torch.nn.functional as F


class DPTDepthEncoder(DPT):
    def __init__(
        self,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        output_level="path_1",  # 支持输出多个层级的特征
    ):
        super().__init__(
            head=nn.Identity(),  # 不需要head
            features=features,
            backbone=backbone,
            readout=readout,
            channels_last=channels_last,
            use_bn=use_bn,
            enable_attention_hooks=enable_attention_hooks,
        )
        self.output_level = output_level

    def forward(self, x):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        # x: [B, 1, H, W] => 需要升维到3通道才能喂入ViT（如果是ViT）
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # 修改：只通过 self.pretrained.activations 访问
        print("Before forward_vit - pretrained.activations keys:", self.pretrained.activations.keys())
        print("Before forward_vit - pretrained.activations id:", id(self.pretrained.activations))
        
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        
        print("After forward_vit - pretrained.activations keys:", self.pretrained.activations.keys())
        print("After forward_vit - pretrained.activations id:", id(self.pretrained.activations))

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.output_level == "path_1":
            out = path_1
        elif self.output_level == "path_2":
            out = path_2
        elif self.output_level == "path_3":
            out = path_3
        elif self.output_level == "path_4":
            out = path_4
        else:
            raise ValueError(f"Unsupported output level: {self.output_level}")
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)  # [B, C]
        return out

'''
使用示例：
depth_encoder = DPTDepthEncoder(backbone="vitb_rn50_384", output_level="path_1")

dummy_depth = torch.randn(2, 1, 384, 384)  # B×1×H×W
depth_feat = depth_encoder(dummy_depth)   # 输出为 B×C×H'×W'
print(depth_feat.shape)

'''

class DPTDepthEncoderWithMask(DPT):
    def __init__(
        self,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        output_level="path_1",
        masked=False,
    ):
        super().__init__(
            head=nn.Identity(),
            features=features,
            backbone=backbone,
            readout=readout,
            channels_last=channels_last,
            use_bn=use_bn,
            enable_attention_hooks=enable_attention_hooks,
        )
        self.output_level = output_level
        self.masked = masked

        if self.masked:
            self.mask_token = nn.Parameter(torch.randn(1, 1, 1))  # [1, 1, 1] for single channel
            nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):  # x: [B, 1, H, W], mask: [B, 1, H, W]
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        # 应用掩码（可选）
        if self.masked and mask is not None:
            # 替换被mask区域为 learnable token
            x = x * (1 - mask) + self.mask_token * mask

        # 复制通道数： [B, 1, H, W] -> [B, 3, H, W]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ViT主干特征提取
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.output_level == "path_1":
            out = path_1
        elif self.output_level == "path_2":
            out = path_2
        elif self.output_level == "path_3":
            out = path_3
        elif self.output_level == "path_4":
            out = path_4
        else:
            raise ValueError(f"Unsupported output level: {self.output_level}")
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)  # [B, C]
        return out
'''
depth_encoder = DPTDepthEncoderWithMask(
    backbone="vitb_rn50_384",
    output_level="path_1",
    masked=True
)

depth = torch.randn(2, 1, 384, 384)
mask = (torch.rand_like(depth) > 0.7).float()  # 30%区域为1，表示mask

feat = depth_encoder(depth, mask=mask)
print(feat.shape)  # [B, C, H', W']
'''