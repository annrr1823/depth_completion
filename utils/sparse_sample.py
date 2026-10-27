"""从稠密深度图中随机采样稀疏深度点。"""

from __future__ import annotations

import numpy as np
import torch


def sample_sparse_depth(
    depth: np.ndarray | torch.Tensor,
    num_samples: int = 500,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """从有效深度像素中随机采样固定数量的稀疏点。

    Args:
        depth: [H, W] 或 [1, H, W] 稠密深度图（米）。
        num_samples: 采样点数，作业默认 500。
        min_depth / max_depth: 有效深度范围；超出视为无效。
        rng: 可选的 numpy Generator，保证可复现。

    Returns:
        sparse_depth: 与 depth 同形状，仅采样位置保留深度，其余为 0。
        valid_mask: 与 depth 同形状，采样位置为 1，其余为 0。
    """
    is_torch = isinstance(depth, torch.Tensor)
    device = depth.device if is_torch else None
    dtype = depth.dtype if is_torch else depth.dtype

    depth_np = depth.detach().cpu().numpy() if is_torch else np.asarray(depth)
    squeeze_channel = False
    if depth_np.ndim == 3 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
        squeeze_channel = True
    if depth_np.ndim != 2:
        raise ValueError(f"期望 depth 为 [H,W] 或 [1,H,W]，得到 {depth_np.shape}")

    h, w = depth_np.shape
    valid = (depth_np > min_depth) & (depth_np <= max_depth) & np.isfinite(depth_np)
    ys, xs = np.where(valid)
    n_valid = len(ys)

    sparse = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    if n_valid > 0:
        k = min(num_samples, n_valid)
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(n_valid, size=k, replace=False)
        sel_y, sel_x = ys[idx], xs[idx]
        sparse[sel_y, sel_x] = depth_np[sel_y, sel_x].astype(np.float32)
        mask[sel_y, sel_x] = 1.0

    out_sparse, out_mask = sparse, mask
    if squeeze_channel:
        out_sparse = out_sparse[None, ...]
        out_mask = out_mask[None, ...]

    if is_torch:
        out_sparse = torch.from_numpy(out_sparse).to(device=device, dtype=dtype)
        out_mask = torch.from_numpy(out_mask).to(device=device, dtype=dtype)
    return out_sparse, out_mask


def sample_sparse_depth_batch(
    depth: torch.Tensor,
    num_samples: int = 500,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 batch 深度图逐张采样。depth: [B,1,H,W] -> sparse/mask: [B,1,H,W]。"""
    spars, masks = [], []
    for i in range(depth.shape[0]):
        s, m = sample_sparse_depth(
            depth[i],
            num_samples=num_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        spars.append(s)
        masks.append(m)
    return torch.stack(spars, dim=0), torch.stack(masks, dim=0)
