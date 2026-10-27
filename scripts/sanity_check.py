"""不下载数据、不训练：检查稀疏采样、模型前向、损失与指标通路。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.depth_loss import DepthCompletionLoss
from metrics.depth_metrics import DepthMetrics, compute_abs_rel, compute_rmse
from models.sagrnet import SAGRNet, pack_model_input
from utils.sparse_sample import sample_sparse_depth
from utils.visualize import visualize_depth_triplet
import matplotlib.pyplot as plt


def test_sparse_sample() -> None:
    h, w = 120, 160
    depth = np.zeros((h, w), dtype=np.float32)
    # 模拟一块有效深度区域
    # 有效区域 80x100，用水平渐变深度填充
    depth[20:100, 30:130] = np.tile(
        np.linspace(1.0, 4.0, 100, dtype=np.float32)[None, :], (80, 1)
    )
    sparse, mask = sample_sparse_depth(depth, num_samples=500, rng=np.random.default_rng(0))
    n = int(mask.sum())
    assert n == 500, f"期望采样 500 点，得到 {n}"
    assert np.allclose(sparse[mask > 0], depth[mask > 0])
    print(f"[OK] sparse sample: {n} points")


def test_model_and_loss() -> None:
    device = torch.device("cpu")
    model = SAGRNet(in_channels=5, base_channels=32).to(device)
    criterion = DepthCompletionLoss(loss_type="l1", edge_weight=0.05, sparse_consistency=0.5)

    b, h, w = 2, 120, 160
    rgb = torch.randn(b, 3, h, w)
    gt = torch.rand(b, 1, h, w) * 4 + 0.5
    sparse = torch.zeros(b, 1, h, w)
    mask = torch.zeros(b, 1, h, w)
    for i in range(b):
        ys = torch.randint(0, h, (500,))
        xs = torch.randint(0, w, (500,))
        sparse[i, 0, ys, xs] = gt[i, 0, ys, xs]
        mask[i, 0, ys, xs] = 1.0

    x = pack_model_input(rgb, sparse, mask)
    pred = model(x)
    assert pred.shape == gt.shape
    losses = criterion(pred, gt, rgb, (gt > 0).float(), sparse, mask)
    assert torch.isfinite(losses["total"])
    print(f"[OK] model forward + loss: pred={tuple(pred.shape)} loss={losses['total'].item():.4f}")

    metrics = DepthMetrics()
    batch_m = metrics.update(pred, gt, (gt > 0).float())
    print(f"[OK] metrics batch: {batch_m}")
    print(f"[OK] metrics accum: {metrics.compute()}")

    # 公式自检
    p = pred[0, 0].reshape(-1)
    g = gt[0, 0].reshape(-1)
    print(f"[OK] rmse={compute_rmse(p, g).item():.4f} abs_rel={compute_abs_rel(p, g).item():.4f}")


def test_vis_smoke() -> None:
    h, w = 60, 80
    rgb = np.random.rand(h, w, 3).astype(np.float32)
    gt = np.linspace(1, 5, h * w).reshape(h, w).astype(np.float32)
    sparse, _ = sample_sparse_depth(gt, num_samples=100, rng=np.random.default_rng(1))
    pred = gt + np.random.randn(h, w).astype(np.float32) * 0.1
    out = ROOT / "outputs" / "vis" / "sanity_triplet.png"
    visualize_depth_triplet(rgb, sparse, pred, gt, save_path=out, title="sanity")
    plt.close("all")
    print(f"[OK] visualization saved to {out}")


if __name__ == "__main__":
    test_sparse_sample()
    test_model_and_loss()
    test_vis_smoke()
    print("\nAll sanity checks passed. 尚未下载数据 / 尚未训练。")
