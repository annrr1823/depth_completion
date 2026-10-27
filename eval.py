"""评测脚本（作业指标）：

  - RMSE (Root Mean Squared Error)
  - Abs Relative Difference = mean(|pred - gt| / gt)

仅在有效深度像素上计算（gt_valid / 深度范围 mask）。
同时可选保存稀疏输入 / 预测 / GT 对比图，便于报告可视化部分。

用法:
  # 解压验证集后
  python scripts/extract_nyu.py --cache-dir data/nyu_depth_v2 --splits val

  # 评测最新 checkpoint（默认找 outputs/checkpoints/<exp>/epoch_*.pth 最大编号）
  python eval.py --config configs/default.yaml

  # 或指定权重
  python eval.py --config configs/default.yaml \\
      --checkpoint outputs/checkpoints/sagrnet_nyu_500/epoch_049.pth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import build_dataloader
from metrics import DepthMetrics
from models.sagrnet import build_model, pack_model_input
from utils.misc import load_config, set_seed
from utils.visualize import save_comparison_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RGB-guided sparse depth completion")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="模型权重；为空则自动选择实验目录下最新 epoch_*.pth / best.pth",
    )
    p.add_argument("--split", type=str, default="validation", help="validation / train")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vis-dir", type=str, default="")
    p.add_argument("--num-vis", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=-1, help="覆盖配置中的 num_workers；-1 表示不覆盖")
    return p.parse_args()


def find_latest_checkpoint(cfg: dict) -> Path:
    exp = cfg.get("experiment", {}).get("name", "exp")
    out = Path(cfg.get("experiment", {}).get("output_dir", "outputs"))
    ckpt_dir = out / "checkpoints" / exp
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"未找到 checkpoint 目录: {ckpt_dir}")

    best = ckpt_dir / "best.pth"
    if best.exists():
        return best

    epochs = []
    for p in ckpt_dir.glob("epoch_*.pth"):
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        if m:
            epochs.append((int(m.group(1)), p))
    if not epochs:
        raise FileNotFoundError(f"{ckpt_dir} 下没有 best.pth 或 epoch_*.pth")
    epochs.sort(key=lambda x: x[0])
    return epochs[-1][1]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("experiment", {}).get("seed", 42))
    device = torch.device(args.device)

    if args.num_workers >= 0:
        cfg.setdefault("data", {})["num_workers"] = args.num_workers

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    exp = cfg.get("experiment", {}).get("name", "exp")
    vis_dir = Path(args.vis_dir) if args.vis_dir else Path("outputs") / "vis" / f"eval_{exp}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] device={device}", flush=True)
    print(f"[eval] checkpoint={ckpt_path}", flush=True)
    print(f"[eval] split={args.split}", flush=True)
    print("[eval] building dataloader...", flush=True)
    loader = build_dataloader(cfg, split=args.split, download=False)
    print(f"[eval] num_batches={len(loader)}", flush=True)

    print("[eval] loading model...", flush=True)
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    metrics = DepthMetrics(
        min_depth=cfg["data"].get("min_depth", 0.0),
        max_depth=cfg["data"].get("max_depth", 10.0),
    )

    vis_saved = False
    for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
        rgb = batch["rgb"].to(device, non_blocking=True)
        sparse = batch["sparse_depth"].to(device, non_blocking=True)
        smask = batch["sparse_mask"].to(device, non_blocking=True)
        gt = batch["gt_depth"].to(device, non_blocking=True)
        gt_valid = batch["gt_valid"].to(device, non_blocking=True)

        pred = model(pack_model_input(rgb, sparse, smask))
        metrics.update(pred, gt, gt_valid)

        if not vis_saved and args.num_vis > 0:
            save_comparison_grid(
                {k: v.cpu() for k, v in batch.items()},
                pred.cpu(),
                save_dir=vis_dir / "triplet",
                max_items=args.num_vis,
                prefix="eval",
            )
            vis_saved = True

    result = metrics.compute()
    # 作业要求的两项主指标
    report = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_valid_pixels": metrics.count,
        "RMSE": result["rmse"],
        "Abs_Relative_Difference": result["abs_rel"],
        # 附加参考（非作业硬性要求）
        "delta1": result["delta1"],
    }

    print("\n" + "=" * 56)
    print("作业评测指标（有效深度像素上）")
    print("=" * 56)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Split      : {args.split}")
    print(f"Valid px   : {metrics.count}")
    print(f"RMSE       : {result['rmse']:.6f}")
    print(f"Abs Rel    : {result['abs_rel']:.6f}")
    print(f"(ref) δ<1.25: {result['delta1']:.6f}")
    print("=" * 56)
    print("定义:")
    print("  RMSE    = sqrt(mean((pred - gt)^2))")
    print("  Abs Rel = mean(|pred - gt| / gt)")
    print("=" * 56)

    out_json = vis_dir / "metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[eval] metrics -> {out_json}")
    if args.num_vis > 0:
        print(f"[eval] visualizations -> {vis_dir / 'triplet'}")


if __name__ == "__main__":
    main()
