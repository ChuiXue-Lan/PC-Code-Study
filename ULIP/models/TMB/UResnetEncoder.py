import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if (stride != 1 or in_channels != out_channels)
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity


class UResnetEncoder(nn.Module):
    def __init__(self, ngf=32, n_blocks=1, n_down=2, norm_layer=nn.InstanceNorm2d):
        super(UResnetEncoder, self).__init__()
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.n_down = n_down
        self.norm_layer = norm_layer

        self.first_conv = nn.Conv2d(1, ngf, kernel_size=7, stride=1, padding=3, bias=False)

        self.down_layers = nn.ModuleList()
        mult = 1
        for i in range(n_down):
            mult *= 2
            block = BasicResidualBlock(ngf * mult // 2, ngf * mult, stride=2, norm_layer=norm_layer)
            self.down_layers.append(block)

        self.mid_layers = nn.ModuleList()
        for _ in range(n_blocks):
            self.mid_layers.append(
                BasicResidualBlock(ngf * mult, ngf * mult, stride=1, norm_layer=norm_layer)
            )

        self.apply(self._init_weights)
        self.out_channels = ngf * mult  # 最后一层通道数

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: [B, 1, H, W]
        skips = []

        x = self.first_conv(x)
        skips.append(x)

        for down_block in self.down_layers:
            x = down_block(x)
            skips.append(x)

        for mid_block in self.mid_layers:
            x = mid_block(x)

        return x, skips  # 返回主输出和多尺度特征
