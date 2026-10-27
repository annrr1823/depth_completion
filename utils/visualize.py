"""深度补全结果可视化：稀疏输入 / 预测 / GT。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def _depth_display(depth: np.ndarray, vmax: Optional[float] = None) -> tuple[np.ndarray, float]:
    d = depth.copy().astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if vmax is None:
        vmax = float(np.percentile(d[valid], 95)) if valid.any() else 1.0
    d = np.clip(d, 0, vmax)
    d[~valid] = 0
    return d, vmax


def visualize_depth_triplet(
    rgb: np.ndarray | torch.Tensor,
    sparse: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    save_path: Optional[str | Path] = None,
    title: str = "",
    cmap: str = "plasma",
) -> plt.Figure:
    """并排展示 RGB、稀疏深度、预测深度、真实深度。

    便于分析物体边界（边缘是否对齐）和无纹理区域（墙面/地板是否平滑）。
    """
    rgb_np = _to_numpy(rgb)
    if rgb_np.dtype != np.uint8:
        # 假设已归一化到 [0,1] 或 ImageNet 均值附近；简单截断后展示
        if rgb_np.max() <= 1.5:
            rgb_show = np.clip(rgb_np, 0, 1)
        else:
            # 反归一化近似（仅可视化）
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb_show = np.clip(rgb_np * std + mean, 0, 1)
    else:
        rgb_show = rgb_np.astype(np.float32) / 255.0

    sparse_np = _to_numpy(sparse)
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)

    gt_show, vmax = _depth_display(gt_np)
    pred_show, _ = _depth_display(pred_np, vmax=vmax)
    sparse_show, _ = _depth_display(sparse_np, vmax=vmax)

    # 稀疏点放大便于观察
    sparse_vis = sparse_show.copy()
    ys, xs = np.where(sparse_np > 0)
    marker = np.zeros_like(sparse_vis)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 1), min(sparse_vis.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(sparse_vis.shape[1], x + 2)
        marker[y0:y1, x0:x1] = sparse_vis[y, x] if sparse_vis[y, x] > 0 else vmax * 0.5
    sparse_vis = np.maximum(sparse_vis, marker)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb_show)
    axes[0].set_title("RGB")
    axes[1].imshow(sparse_vis, cmap=cmap, vmin=0, vmax=vmax)
    axes[1].set_title(f"Sparse ({len(ys)} pts)")
    im = axes[2].imshow(pred_show, cmap=cmap, vmin=0, vmax=vmax)
    axes[2].set_title("Prediction")
    axes[3].imshow(gt_show, cmap=cmap, vmin=0, vmax=vmax)
    axes[3].set_title("Ground Truth")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.88, bottom=0.05)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_comparison_grid(
    batch: dict,
    pred: torch.Tensor,
    save_dir: str | Path,
    max_items: int = 8,
    prefix: str = "sample",
) -> list[Path]:
    """对一个 batch 保存多张对比图。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n = min(max_items, pred.shape[0])
    for i in range(n):
        path = save_dir / f"{prefix}_{i:03d}.png"
        visualize_depth_triplet(
            rgb=batch["rgb"][i],
            sparse=batch["sparse_depth"][i],
            pred=pred[i],
            gt=batch["gt_depth"][i],
            save_path=path,
            title=f"{prefix}_{i:03d}",
        )
        plt.close()
        paths.append(path)
    return paths
