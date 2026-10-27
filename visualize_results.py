"""独立可视化脚本：展示稀疏输入 / 预测 / GT，并简单分析边界与无纹理区域。

用法（训练出 checkpoint 之后）:
  python visualize_results.py --config configs/default.yaml \\
      --checkpoint outputs/checkpoints/sagrnet_nyu_500/best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import build_dataloader
from models.sagrnet import build_model, pack_model_input
from utils.misc import load_config, set_seed
from utils.visualize import visualize_depth_triplet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--out-dir", type=str, default="outputs/vis/analysis")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def analyze_error_map(
    rgb: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    save_path: Path,
) -> None:
    """误差图 + RGB，便于讨论物体边界 / 无纹理区域。"""
    err = np.abs(pred - gt)
    err[~valid] = 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(rgb, 0, 1) if rgb.max() <= 1.5 else rgb)
    axes[0].set_title("RGB")
    im1 = axes[1].imshow(err, cmap="magma")
    axes[1].set_title("|Pred - GT|")
    axes[2].imshow(gt, cmap="plasma")
    axes[2].set_title("GT Depth")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(
        "误差分析提示：边界处误差升高通常因深度不连续；"
        "墙面/地板等无纹理区若误差均匀偏大，说明稀疏点约束不足。"
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("experiment", {}).get("seed", 42))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(cfg, split="validation", download=False)
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    saved = 0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for batch in loader:
        rgb = batch["rgb"].to(device)
        sparse = batch["sparse_depth"].to(device)
        smask = batch["sparse_mask"].to(device)
        gt = batch["gt_depth"].to(device)
        pred = model(pack_model_input(rgb, sparse, smask))

        b = rgb.shape[0]
        for i in range(b):
            if saved >= args.num_samples:
                break
            rgb_np = rgb[i].cpu().numpy().transpose(1, 2, 0) * std + mean
            sparse_np = sparse[i, 0].cpu().numpy()
            pred_np = pred[i, 0].cpu().numpy()
            gt_np = gt[i, 0].cpu().numpy()
            valid = gt_np > cfg["data"].get("min_depth", 0.0)

            visualize_depth_triplet(
                rgb_np,
                sparse_np,
                pred_np,
                gt_np,
                save_path=out_dir / f"triplet_{saved:03d}.png",
                title=f"sample {saved}",
            )
            plt.close("all")
            analyze_error_map(
                rgb_np,
                pred_np,
                gt_np,
                valid,
                save_path=out_dir / f"error_{saved:03d}.png",
            )
            saved += 1
        if saved >= args.num_samples:
            break

    # 写一段可放进报告的分析模板
    note = out_dir / "analysis_notes.md"
    note.write_text(
        """# 可视化分析笔记（模板）

请结合 `triplet_*.png` 与 `error_*.png` 填写：

1. **物体边界**：预测深度是否与 RGB 边缘对齐？误差图在桌沿、门框等处是否形成细线状高亮？
2. **无纹理区域**：墙面、地板、天花板是否出现深度“鼓包”或整体偏大/偏小？稀疏点附近是否更准确？
3. **稀疏点密度影响**：500 点覆盖下，远距离小物体是否容易被平滑掉？
4. **与基线对比**（可选）：关闭残差锚定 / 多尺度注入后，边界与无纹理区谁更差？

结论（示例）：多尺度稀疏注入与残差锚定有助于无纹理区域稳定；RGB 注意力有助于边界对齐，但仍可能在强反光/遮挡处失效。
""",
        encoding="utf-8",
    )
    print(f"Saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
