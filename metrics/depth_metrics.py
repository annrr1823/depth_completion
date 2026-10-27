"""深度补全评测指标：RMSE、Abs Relative Difference（及常用的 δ 阈值）。"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _flat_valid(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_valid: Optional[torch.Tensor] = None,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gt_valid is not None:
        mask = gt_valid > 0.5
    else:
        mask = (gt > min_depth) & (gt <= max_depth) & torch.isfinite(gt)
    mask = mask & torch.isfinite(pred) & (pred > 0)
    return pred[mask], gt[mask]


def compute_rmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """RMSE = sqrt(mean((pred - gt)^2))，在有效像素上计算。"""
    return torch.sqrt(torch.mean((pred - gt) ** 2))


def compute_abs_rel(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Abs Relative Difference = mean(|pred - gt| / gt)。"""
    return torch.mean(torch.abs(pred - gt) / gt.clamp_min(1e-6))


def compute_delta(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 1.25) -> torch.Tensor:
    """δ < threshold 的像素比例。"""
    ratio = torch.max(pred / gt.clamp_min(1e-6), gt / pred.clamp_min(1e-6))
    return (ratio < threshold).float().mean()


class DepthMetrics:
    """可跨 batch 累积的深度指标。"""

    def __init__(self, min_depth: float = 0.0, max_depth: float = 10.0) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reset()

    def reset(self) -> None:
        self.sum_sq = 0.0
        self.sum_abs_rel = 0.0
        self.sum_delta1 = 0.0
        self.count = 0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        gt_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        p, g = _flat_valid(pred, gt, gt_valid, self.min_depth, self.max_depth)
        n = int(p.numel())
        if n == 0:
            return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}

        sq = torch.sum((p - g) ** 2).item()
        abs_rel = torch.sum(torch.abs(p - g) / g.clamp_min(1e-6)).item()
        ratio = torch.max(p / g.clamp_min(1e-6), g / p.clamp_min(1e-6))
        delta1 = torch.sum((ratio < 1.25).float()).item()

        self.sum_sq += sq
        self.sum_abs_rel += abs_rel
        self.sum_delta1 += delta1
        self.count += n

        return {
            "rmse": float((sq / n) ** 0.5),
            "abs_rel": float(abs_rel / n),
            "delta1": float(delta1 / n),
        }

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}
        return {
            "rmse": (self.sum_sq / self.count) ** 0.5,
            "abs_rel": self.sum_abs_rel / self.count,
            "delta1": self.sum_delta1 / self.count,
        }
