"""深度补全损失函数：仅在有效 GT 像素上计算，并可约束稀疏点一致性。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_mask(
    gt: torch.Tensor,
    gt_valid: Optional[torch.Tensor] = None,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> torch.Tensor:
    if gt_valid is not None:
        mask = gt_valid > 0.5
    else:
        mask = (gt > min_depth) & (gt <= max_depth) & torch.isfinite(gt)
    return mask


def masked_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    return F.l1_loss(pred[mask], gt[mask])


def masked_mse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[mask], gt[mask])


def masked_berhu(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reverse Huber (BerHu)，Sparse-to-Dense 论文常用。"""
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    diff = (pred[mask] - gt[mask]).abs()
    c = 0.2 * diff.max().detach().clamp_min(1e-6)
    berhu = torch.where(diff <= c, diff, (diff * diff + c * c) / (2.0 * c))
    return berhu.mean()


def rgb_edge_aware_smoothness(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RGB 梯度引导的深度平滑：在无纹理区域鼓励平滑，边缘处放宽。

    这是作业新颖性的一部分：利用 RGB 边缘保护深度不连续。
    """
    # rgb 已归一化，用其梯度幅值作权重
    grad_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    grad_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    rgb_dx = torch.mean(torch.abs(rgb[:, :, :, :-1] - rgb[:, :, :, 1:]), dim=1, keepdim=True)
    rgb_dy = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), dim=1, keepdim=True)

    weight_x = torch.exp(-rgb_dx)
    weight_y = torch.exp(-rgb_dy)

    if mask is not None:
        mx = mask[:, :, :, :-1] * mask[:, :, :, 1:]
        my = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        loss_x = (grad_dx * weight_x * mx).sum() / mx.sum().clamp_min(1.0)
        loss_y = (grad_dy * weight_y * my).sum() / my.sum().clamp_min(1.0)
    else:
        loss_x = (grad_dx * weight_x).mean()
        loss_y = (grad_dy * weight_y).mean()
    return loss_x + loss_y


class DepthCompletionLoss(nn.Module):
    """组合损失：主回归 + 稀疏点一致性 + RGB 边缘平滑。"""

    def __init__(
        self,
        loss_type: str = "l1",
        l1_weight: float = 1.0,
        mse_weight: float = 0.0,
        edge_weight: float = 0.1,
        sparse_consistency: float = 0.5,
        min_depth: float = 0.0,
        max_depth: float = 10.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type.lower()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.edge_weight = edge_weight
        self.sparse_consistency = sparse_consistency
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        rgb: torch.Tensor,
        gt_valid: Optional[torch.Tensor] = None,
        sparse_depth: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mask = _valid_mask(gt, gt_valid, self.min_depth, self.max_depth)

        if self.loss_type == "l1":
            main = masked_l1(pred, gt, mask)
        elif self.loss_type == "mse":
            main = masked_mse(pred, gt, mask)
        elif self.loss_type == "berhu":
            main = masked_berhu(pred, gt, mask)
        elif self.loss_type == "mixed":
            main = self.l1_weight * masked_l1(pred, gt, mask) + self.mse_weight * masked_mse(
                pred, gt, mask
            )
        else:
            raise ValueError(f"未知 loss type: {self.loss_type}")

        total = main if self.loss_type == "mixed" else (
            self.l1_weight * main if self.loss_type == "l1" else main
        )
        # 对 mse/berhu 也允许用 l1_weight 作总系数
        if self.loss_type in ("mse", "berhu"):
            total = main

        losses: Dict[str, torch.Tensor] = {"main": main.detach(), "total": total}

        if self.sparse_consistency > 0 and sparse_depth is not None and sparse_mask is not None:
            sm = sparse_mask > 0.5
            if sm.any():
                sparse_loss = F.l1_loss(pred[sm], sparse_depth[sm])
                total = total + self.sparse_consistency * sparse_loss
                losses["sparse"] = sparse_loss.detach()
                losses["total"] = total

        if self.edge_weight > 0:
            edge = rgb_edge_aware_smoothness(pred, rgb, mask.float())
            total = total + self.edge_weight * edge
            losses["edge"] = edge.detach()
            losses["total"] = total

        return losses


def build_loss(cfg: Dict[str, Any]) -> DepthCompletionLoss:
    tcfg = cfg.get("train", {}).get("loss", {})
    dcfg = cfg.get("data", {})
    return DepthCompletionLoss(
        loss_type=tcfg.get("type", "l1"),
        l1_weight=tcfg.get("l1_weight", 1.0),
        mse_weight=tcfg.get("mse_weight", 0.0),
        edge_weight=tcfg.get("edge_weight", 0.1),
        sparse_consistency=tcfg.get("sparse_consistency", 0.5),
        min_depth=dcfg.get("min_depth", 0.0),
        max_depth=dcfg.get("max_depth", 10.0),
    )
