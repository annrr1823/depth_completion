"""Sparse-to-Dense 风格基线：早期融合 RGBD + Encoder-Decoder。

参考: Ma & Karaman, "Sparse-to-Dense: Depth Prediction from Sparse Depth
Samples and a Single Image", ICRA 2018.

输入: concat(RGB, sparse_depth, validity_mask) -> [B,5,H,W]
输出: dense depth -> [B,1,H,W]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpProj(nn.Module):
    """论文常用的 UpProj 上采样模块（简化实现）。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        return self.relu(x + self.conv2(x))


class SparseToDenseNet(nn.Module):
    def __init__(self, in_channels: int = 5, base_channels: int = 64):
        super().__init__()
        c = base_channels
        # Encoder
        self.enc1 = nn.Sequential(ConvBNReLU(in_channels, c), ConvBNReLU(c, c))
        self.enc2 = nn.Sequential(ConvBNReLU(c, c * 2, s=2), ConvBNReLU(c * 2, c * 2))
        self.enc3 = nn.Sequential(ConvBNReLU(c * 2, c * 4, s=2), ConvBNReLU(c * 4, c * 4))
        self.enc4 = nn.Sequential(ConvBNReLU(c * 4, c * 8, s=2), ConvBNReLU(c * 8, c * 8))

        # Decoder
        self.up3 = UpProj(c * 8, c * 4)
        self.up2 = UpProj(c * 4, c * 2)
        self.up1 = UpProj(c * 2, c)
        self.head = nn.Conv2d(c, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.up3(e4) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up1(d2) + e1
        depth = F.relu(self.head(d1))  # 深度非负
        return depth
