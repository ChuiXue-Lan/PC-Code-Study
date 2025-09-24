import torch
import torch.nn as nn
import torch.nn.functional as F
from .UResnetEncoder import UResnetEncoder

class DepthFeatureEncoder(nn.Module):
    def __init__(self, encoder: UResnetEncoder, proj_dim=512, use_multi_scale=False):
        super().__init__()
        self.encoder = encoder
        self.use_multi_scale = use_multi_scale
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if not use_multi_scale:
            self.proj = nn.Linear(encoder.out_channels, proj_dim)
        else:
            self.scales = len(self.encoder.down_layers) + 1  # +1 for initial conv
            total_channels = sum([
                encoder.ngf * (2 ** i) for i in range(self.scales)
            ])
            self.proj = nn.Linear(total_channels, proj_dim)

    def forward(self, x):  # x: [B, 1, H, W]
        feat, skips = self.encoder(x)

        if self.use_multi_scale:
            pooled_feats = []
            for skip in skips:
                pooled = self.pool(skip).squeeze(-1).squeeze(-1)  # [B, C]
                pooled_feats.append(pooled)
            x = torch.cat(pooled_feats, dim=1)  # [B, total_C]
        else:
            x = self.pool(feat).squeeze(-1).squeeze(-1)  # [B, C]

        return self.proj(x)  # [B, proj_dim]

'''
# 示例
# 创建深度图编码器
encoder = UResnetEncoder(ngf=32, n_blocks=1, n_down=2)
depth_encoder = DepthFeatureEncoder(encoder, proj_dim=512, use_multi_scale=True)

# 输入深度图张量
depth_map = torch.randn(4, 1, 256, 256).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
depth_feat = depth_encoder(depth_map)  # 输出: [4, 512]

'''
class DepthFeatureEncoderWithMask(nn.Module):
    def __init__(self, encoder: UResnetEncoder, proj_dim=512, use_multi_scale=False, masked=False):
        super().__init__()
        self.encoder = encoder
        self.use_multi_scale = use_multi_scale
        self.masked = masked
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if not use_multi_scale:
            self.proj = nn.Linear(encoder.out_channels, proj_dim)
        else:
            self.scales = len(self.encoder.down_layers) + 1
            total_channels = sum([
                encoder.ngf * (2 ** i) for i in range(self.scales)
            ])
            self.proj = nn.Linear(total_channels, proj_dim)

        if self.masked:
            # 假设使用一个 learnable token 替代被 mask 掉的区域
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1))  # shape: [1, 1, 1, 1] for broadcasting
            nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):  # x: [B, 1, H, W], mask: [B, 1, H, W]
        if self.masked and mask is not None:
            # 替换被 mask 区域为 learnable mask_token
            x = x * (1 - mask) + self.mask_token * mask

        feat, skips = self.encoder(x)

        if self.use_multi_scale:
            pooled_feats = []
            for skip in skips:
                pooled = self.pool(skip).squeeze(-1).squeeze(-1)  # [B, C]
                pooled_feats.append(pooled)
            x = torch.cat(pooled_feats, dim=1)  # [B, total_C]
        else:
            x = self.pool(feat).squeeze(-1).squeeze(-1)  # [B, C]

        return self.proj(x)  # [B, proj_dim]