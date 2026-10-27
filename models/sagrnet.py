"""SAGRNet: Sparse-Aware Guided Residual Network

作业新颖性设计说明
==================
Sparse-to-Dense 的经典做法是把 RGB 与稀疏深度在输入端直接拼接后做回归。
本模型在此基础上加入三点针对“稀疏深度补全”特性的改动：

1) 显式有效性掩码通道
   稀疏图中大量 0 既可能是“无测量”也可能是“深度恰为 0”。
   额外输入 validity mask，让网络区分“空”与“真零”。

2) 多尺度稀疏引导注入 (Sparse Guidance Injection)
   在 decoder 每个尺度把下采样后的 sparse depth / mask 与特征拼接，
   使深层特征在上采样过程中持续“锚定”已知深度点，减轻无纹理区域漂移。

3) 稀疏锚定残差预测 (Sparse-Anchored Residual)
   先用稀疏点的最近邻插值得到粗深度 D_coarse，网络预测残差 Δ，
   最终深度 D = softplus(D_coarse + Δ)。
   已知测量点附近预测被自然约束，边界处则由 RGB 纹理引导残差修正。

参考论文思想: Ma & Karaman, Sparse-to-Dense (ICRA 2018)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_to_dense import SparseToDenseNet


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


class SEBlock(nn.Module):
    """轻量通道注意力，强化对深度相关特征的选择。"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class RGBGuidedAttention(nn.Module):
    """用 RGB 特征生成空间注意力，引导深度特征贴合物体边界。"""

    def __init__(self, rgb_ch: int, depth_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(rgb_ch, depth_ch, 1, bias=False),
            nn.BatchNorm2d(depth_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        attn = self.proj(rgb_feat)
        return depth_feat * (1.0 + attn)


class SparseGuideFuse(nn.Module):
    """把当前尺度的 sparse depth + mask 注入特征。"""

    def __init__(self, feat_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNReLU(feat_ch + 2, feat_ch),
            SEBlock(feat_ch),
        )

    def forward(
        self, feat: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        sparse_r = F.interpolate(sparse, size=feat.shape[-2:], mode="nearest")
        mask_r = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")
        return self.fuse(torch.cat([feat, sparse_r, mask_r], dim=1))


def nearest_sparse_fill(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对稀疏深度做简单最近邻填充，得到粗深度先验 D_coarse。

    实现：对 mask 做多次 max-pool 扩张，把最近的非零深度传播出去。
    这是可微的近似（训练时可截断梯度仅作输入条件）。
    """
    # 使用 no_grad 的填充作为条件输入；网络学残差
    with torch.no_grad():
        depth = sparse.clone()
        m = mask.clone()
        # 迭代膨胀约覆盖全图（对 240x320，约 9~12 次足够启发式传播）
        for _ in range(12):
            # 取邻域最大 mask 位置的深度（近似 NN）
            pooled_m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
            # 对应深度：在扩展环上取邻域内 mask 加权 —— 简化为对 depth 做 max
            # 更稳妥：用平均池化在“新填入”区域
            neigh = F.avg_pool2d(depth * m, kernel_size=3, stride=1, padding=1)
            neigh_w = F.avg_pool2d(m, kernel_size=3, stride=1, padding=1).clamp_min(1e-6)
            filled = neigh / neigh_w
            new_region = (pooled_m > 0) & (m == 0)
            depth = torch.where(new_region, filled, depth)
            m = pooled_m
        # 仍未填到的位置用 batch 内有效深度中位数兜底
        if (m == 0).any():
            valid_vals = sparse[mask > 0]
            fallback = valid_vals.median() if valid_vals.numel() > 0 else torch.tensor(1.0, device=sparse.device)
            depth = torch.where(m > 0, depth, fallback.expand_as(depth))
    return depth


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(in_ch, out_ch),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class SAGRNet(nn.Module):
    """Sparse-Aware Guided Residual Network."""

    def __init__(self, in_channels: int = 5, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # 浅层 RGB 分支（用于边界注意力）
        self.rgb_stem = nn.Sequential(
            ConvBNReLU(3, c),
            ConvBNReLU(c, c),
        )

        # 早期融合主干（RGB + sparse + mask）
        self.stem = ConvBNReLU(in_channels, c)
        self.enc1 = nn.Sequential(ConvBNReLU(c, c), ConvBNReLU(c, c))
        self.down1 = ConvBNReLU(c, c * 2, s=2)
        self.enc2 = nn.Sequential(ConvBNReLU(c * 2, c * 2), ConvBNReLU(c * 2, c * 2))
        self.down2 = ConvBNReLU(c * 2, c * 4, s=2)
        self.enc3 = nn.Sequential(ConvBNReLU(c * 4, c * 4), ConvBNReLU(c * 4, c * 4))
        self.down3 = ConvBNReLU(c * 4, c * 8, s=2)
        self.enc4 = nn.Sequential(ConvBNReLU(c * 8, c * 8), ConvBNReLU(c * 8, c * 8))

        self.guide4 = SparseGuideFuse(c * 8)
        self.guide3 = SparseGuideFuse(c * 4)
        self.guide2 = SparseGuideFuse(c * 2)
        self.guide1 = SparseGuideFuse(c)

        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)

        self.rgb_attn = RGBGuidedAttention(c, c)
        self.residual_head = nn.Sequential(
            ConvBNReLU(c, c),
            nn.Conv2d(c, 1, 3, padding=1),
        )
        # 残差缩放，避免初期扰动过大
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B,5,H,W] = RGB(3) + sparse_depth(1) + validity_mask(1)
        Returns:
            dense depth [B,1,H,W]
        """
        assert x.shape[1] == 5, f"期望 5 通道输入，得到 {x.shape[1]}"
        rgb = x[:, :3]
        sparse = x[:, 3:4]
        mask = x[:, 4:5]

        coarse = nearest_sparse_fill(sparse, mask)

        rgb_feat = self.rgb_stem(rgb)

        f = self.stem(x)
        e1 = self.enc1(f)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        e4 = self.guide4(e4, sparse, mask)
        d3 = self.guide3(self.up3(e4, e3), sparse, mask)
        d2 = self.guide2(self.up2(d3, e2), sparse, mask)
        d1 = self.guide1(self.up1(d2, e1), sparse, mask)

        d1 = self.rgb_attn(rgb_feat, d1)
        residual = self.residual_head(d1) * self.res_scale

        # softplus 保证非负；coarse 提供稀疏锚定
        depth = F.softplus(coarse + residual)
        return depth


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "sagrnet").lower()
    in_ch = mcfg.get("in_channels", 5)
    base = mcfg.get("base_channels", 64)
    if name in ("sagrnet", "sagr"):
        return SAGRNet(in_channels=in_ch, base_channels=base)
    if name in ("sparse_to_dense", "s2d", "baseline"):
        return SparseToDenseNet(in_channels=in_ch, base_channels=base)
    raise ValueError(f"未知模型: {name}")


def pack_model_input(
    rgb: torch.Tensor,
    sparse_depth: torch.Tensor,
    sparse_mask: torch.Tensor,
) -> torch.Tensor:
    """将 batch 字段拼成网络输入 [B,5,H,W]。"""
    return torch.cat([rgb, sparse_depth, sparse_mask], dim=1)
