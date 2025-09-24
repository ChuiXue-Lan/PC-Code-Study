import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckResidualBlock(nn.Module):
    """优化为瓶颈残差块：1×1卷积降维→3×3卷积提特征→1×1卷积升维，减少参数量"""
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.InstanceNorm2d, dilation=1):
        super().__init__()
        mid_channels = out_channels // 4  # 瓶颈层通道数=输出通道数/4，减少计算量
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)  # 降维
        )
        self.conv2 = nn.Sequential(
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            # 加入空洞卷积， dilation>1时扩大感受野（无需下采样）
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                      padding=dilation, dilation=dilation, bias=False)
        )
        self.conv3 = nn.Sequential(
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)  # 升维
        )
        # shortcut分支：支持空洞卷积场景下的维度对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels or dilation != 1:
            self.shortcut = nn.Sequential(
                norm_layer(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + identity  # 残差连接保留原始信息
    
class UResnetEncoderV2(nn.Module):
    """优化后编码器：瓶颈残差块+空洞卷积分支+多尺度特征增强"""
    def __init__(self, ngf=32, n_blocks=2, n_down=2, norm_type='instance', use_dilation=True):
        super().__init__()
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.n_down = n_down
        self.use_dilation = use_dilation  # 是否启用空洞卷积分支
        # 自适应归一化：支持instance/layer两种类型
        self.norm_layer = nn.InstanceNorm2d if norm_type == 'instance' else nn.LayerNorm2d

        # 1. 初始卷积：新增BN+ReLU预激活，提升特征初始化稳定性
        self.first_conv = nn.Sequential(
            self.norm_layer(1),  # 对单通道深度图先归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(1, ngf, kernel_size=7, stride=1, padding=3, bias=False)
        )

        # 2. 下采样模块：交替使用普通下采样+空洞卷积（若启用）
        self.down_layers = nn.ModuleList()
        self.dilation_layers = nn.ModuleList()  # 新增空洞卷积分支
        # mult = 1
        # 通道进程：每次下采样通道数翻倍；第一层输入为ngf（来自first_conv输出）
        prev_channels = ngf
        for i in range(n_down):
            # mult *= 2
            out_channels = prev_channels * 2
            # 普通下采样残差块（步长2，降分辨率）
            down_block = BottleneckResidualBlock(
                # ngf * mult // 4, ngf * mult, stride=2, norm_layer=self.norm_layer, dilation=1
                prev_channels, out_channels, stride=2, norm_layer=self.norm_layer, dilation=1
            )
            self.down_layers.append(down_block)
            # 空洞卷积分支（步长1，保分辨率，扩感受野）
            if self.use_dilation:
                dilate_block = BottleneckResidualBlock(
                    # ngf * mult, ngf * mult, stride=1, norm_layer=self.norm_layer, dilation=2**(i+1)
                    out_channels, out_channels, stride=1, norm_layer=self.norm_layer, dilation=2**(i+1)
                )
                self.dilation_layers.append(dilate_block)
            # +
            prev_channels = out_channels

        # 3. 中间特征强化：堆叠瓶颈残差块，增加特征表达能力（原n_blocks=1→2）
        self.mid_layers = nn.ModuleList()
        # +
        current_channels = prev_channels  # 经过所有下采样后的通道数
        for _ in range(n_blocks):
            self.mid_layers.append(
                # BottleneckResidualBlock(ngf * mult, ngf * mult, stride=1, norm_layer=self.norm_layer)
                BottleneckResidualBlock(current_channels, current_channels, stride=1, norm_layer=self.norm_layer)
            )

        # 权重初始化：新增正交初始化选项，适配瓶颈结构
        self.apply(self._init_weights)
        # self.out_channels = ngf * mult
        self.out_channels = current_channels

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # 瓶颈结构用正交初始化，避免维度压缩导致的信息损失
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: [B,1,H,W]
        skips = []
        # 初始卷积+特征存储
        x = self.first_conv(x)
        skips.append(x)  # 存储初始特征（尺度1）

        # 下采样+空洞卷积分支（若启用）
        for i, down_block in enumerate(self.down_layers):
            x = down_block(x)
            # 空洞卷积增强：保分辨率，补充大尺度几何信息
            if self.use_dilation:
                x_dilate = self.dilation_layers[i](x)
                x = x + x_dilate  # 融合下采样特征与空洞卷积特征
            skips.append(x)  # 存储下采样后特征（尺度2→n_down+1）

        # 中间特征强化
        for mid_block in self.mid_layers:
            x = mid_block(x)

        return x, skips  # 主特征 + 多尺度跳连接特征
    
class DepthFeatureEncoderV2(nn.Module):
    """优化后特征投影：双池化（平均+最大）融合+分层投影，保留更多特征细节"""
    def __init__(self, encoder: UResnetEncoderV2, proj_dim=512, use_multi_scale=True):
        super().__init__()
        self.encoder = encoder
        self.use_multi_scale = use_multi_scale
        # 双池化：全局平均池化（保留整体趋势）+ 全局最大池化（保留局部峰值）
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        if not use_multi_scale:
            # 单尺度：双池化拼接后投影
            self.proj = nn.Linear(encoder.out_channels * 2, proj_dim)
        else:
            # 多尺度：各尺度双池化后拼接，再分层投影（减少维度爆炸）
            self.scales = len(self.encoder.down_layers) + 1
            self.scale_projs = nn.ModuleList()  # 各尺度独立投影层
            total_proj_dim = 0
            for i in range(self.scales):
                scale_channels = encoder.ngf * (2 ** i)
                # 每个尺度双池化后投影到固定子维度（如64维）
                self.scale_projs.append(nn.Linear(scale_channels * 2, 64))
                total_proj_dim += 64
            # 最终统一投影到目标维度
            self.final_proj = nn.Linear(total_proj_dim, proj_dim)

    def forward(self, x):  # x: [B,1,H,W]
        feat, skips = self.encoder(x)

        if self.use_multi_scale:
            scale_feats = []
            for i, skip in enumerate(skips):
                # 双池化融合：[B,C,H,W]→[B,C]
                avg_pooled = self.avg_pool(skip).squeeze(-1).squeeze(-1)
                max_pooled = self.max_pool(skip).squeeze(-1).squeeze(-1)
                pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, 2C]
                # 各尺度独立投影
                scale_feat = self.scale_projs[i](pooled)
                scale_feats.append(scale_feat)
            # 拼接所有尺度特征并最终投影
            x = torch.cat(scale_feats, dim=1)
            x = self.final_proj(x)
        else:
            # 单尺度双池化融合+投影
            avg_pooled = self.avg_pool(feat).squeeze(-1).squeeze(-1)
            max_pooled = self.max_pool(feat).squeeze(-1).squeeze(-1)
            x = torch.cat([avg_pooled, max_pooled], dim=1)
            x = self.proj(x)

        return x  # [B, proj_dim]