import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthFeatureEncoder(nn.Module):
    def __init__(self, encoder, proj_dim=512, use_multi_scale=False, mask=True):
        super().__init__()
        self.encoder = encoder
        self.use_multi_scale = use_multi_scale
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mask = mask

        if not use_multi_scale:
            self.proj = nn.Linear(encoder.out_channels, proj_dim)
        else:
            self.scales = len(self.encoder.down_layers) + 1  # +1 for initial conv
            total_channels = sum([
                encoder.ngf * (2 ** i) for i in range(self.scales)
            ])
            self.proj = nn.Linear(total_channels, proj_dim)

        # MIM相关组件
        if self.mask:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.out_channels))
            torch.nn.init.normal_(self.mask_token, std=.02)
            self.mask_proj = nn.Linear(encoder.out_channels, proj_dim)

    def forward(self, x, mask=None):  # x: [B, 1, H, W]
        if self.mask and self.training:
            # 获取特征和跳跃连接
            feat, skips = self.encoder(x)
            B, C, H, W = feat.shape
            
            if mask is None:
                # 随机生成掩码
                mask_ratio = 0.75
                num_patches = H * W
                num_mask = int(mask_ratio * num_patches)
                mask = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)
                for i in range(B):
                    perm = torch.randperm(num_patches, device=x.device)
                    mask[i, perm[:num_mask]] = True
            
            # 展平特征
            feat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # 应用掩码
            mask_tokens = self.mask_token.expand(B, feat.shape[1], -1)
            w = mask.unsqueeze(-1).type_as(mask_tokens)
            masked_feat = feat * (1 - w) + mask_tokens * w
            
            # 重塑回原始形状
            masked_feat = masked_feat.transpose(1, 2).reshape(B, C, H, W)
            
            # 获取全局特征
            if self.use_multi_scale:
                pooled_feats = []
                for skip in skips:
                    pooled = self.pool(skip).squeeze(-1).squeeze(-1)  # [B, C]
                    pooled_feats.append(pooled)
                x = torch.cat(pooled_feats, dim=1)  # [B, total_C]
            else:
                x = self.pool(masked_feat).squeeze(-1).squeeze(-1)  # [B, C]
            
            # 投影到目标维度
            cls_feat = self.proj(x)  # [B, proj_dim]
            
            # MIM重建
            mim_feat = self.mask_proj(masked_feat.flatten(2).transpose(1, 2))  # [B, H*W, proj_dim]
            
            # 对齐损失
            align_feat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            return cls_feat, mim_feat, align_feat, mask
            
        else:
            # 正常前向传播
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