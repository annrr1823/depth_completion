"""训练入口。

当前阶段请勿直接大规模训练；先确认数据已下载：
  python scripts/download_nyu.py
然后再：
  python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# 保证以脚本方式运行时可找到本地包
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import build_dataloader
from losses.depth_loss import build_loss
from metrics import DepthMetrics
from models.sagrnet import build_model, pack_model_input
from utils.misc import AverageMeter, load_config, set_seed
from utils.visualize import save_comparison_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB-Guided Sparse Depth Completion - Train")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--download", action="store_true", help="允许自动下载数据集（默认关闭）")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--dry-run", action="store_true", help="只构建模型/损失并做一次假前向，不读数据")
    return p.parse_args()


def dry_run(cfg: dict, device: torch.device) -> None:
    """不依赖真实数据的通路检查。"""
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    h, w = cfg["data"]["image_height"], cfg["data"]["image_width"]
    rgb = torch.randn(2, 3, h, w, device=device)
    sparse = torch.zeros(2, 1, h, w, device=device)
    mask = torch.zeros(2, 1, h, w, device=device)
    # 放几个假稀疏点
    sparse[:, :, 10, 10] = 2.0
    sparse[:, :, 50, 80] = 3.5
    mask[:, :, 10, 10] = 1.0
    mask[:, :, 50, 80] = 1.0
    gt = torch.rand(2, 1, h, w, device=device) * 5 + 0.5
    gt_valid = (gt > 0).float()

    x = pack_model_input(rgb, sparse, mask)
    pred = model(x)
    losses = criterion(pred, gt, rgb, gt_valid, sparse, mask)
    print(f"[dry-run] input={tuple(x.shape)} pred={tuple(pred.shape)} loss={losses['total'].item():.4f}")
    print("[dry-run] OK — 模型与损失通路正常。")


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, epoch, writer):
    model.train()
    loss_meter = AverageMeter("loss")
    log_interval = cfg["train"].get("log_interval", 50)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar):
        rgb = batch["rgb"].to(device)
        sparse = batch["sparse_depth"].to(device)
        smask = batch["sparse_mask"].to(device)
        gt = batch["gt_depth"].to(device)
        gt_valid = batch["gt_valid"].to(device)

        x = pack_model_input(rgb, sparse, smask)
        pred = model(x)
        losses = criterion(pred, gt, rgb, gt_valid, sparse, smask)
        loss = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip = cfg["train"].get("grad_clip", 0)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_meter.update(loss.item(), rgb.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        global_step = epoch * len(loader) + step
        if writer is not None and step % log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, epoch, writer, vis_dir: Path):
    model.eval()
    metrics = DepthMetrics(
        min_depth=cfg["data"].get("min_depth", 0.0),
        max_depth=cfg["data"].get("max_depth", 10.0),
    )
    loss_meter = AverageMeter("val_loss")
    first_batch = None
    first_pred = None

    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False):
        rgb = batch["rgb"].to(device)
        sparse = batch["sparse_depth"].to(device)
        smask = batch["sparse_mask"].to(device)
        gt = batch["gt_depth"].to(device)
        gt_valid = batch["gt_valid"].to(device)

        pred = model(pack_model_input(rgb, sparse, smask))
        losses = criterion(pred, gt, rgb, gt_valid, sparse, smask)
        loss_meter.update(losses["total"].item(), rgb.size(0))
        metrics.update(pred, gt, gt_valid)

        if first_batch is None:
            first_batch = {k: v.cpu() for k, v in batch.items()}
            first_pred = pred.cpu()

    result = metrics.compute()
    if writer is not None:
        writer.add_scalar("val/loss", loss_meter.avg, epoch)
        for k, v in result.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    if first_batch is not None:
        save_comparison_grid(
            first_batch,
            first_pred,
            save_dir=vis_dir / f"epoch_{epoch:03d}",
            max_items=cfg.get("vis", {}).get("num_samples", 4),
            prefix=f"ep{epoch}",
        )
    return loss_meter.avg, result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("experiment", {}).get("seed", 42))
    device = torch.device(args.device)

    if args.dry_run:
        dry_run(cfg, device)
        return

    exp_name = cfg.get("experiment", {}).get("name", "exp")
    out_root = Path(cfg.get("experiment", {}).get("output_dir", "outputs"))
    ckpt_dir = out_root / "checkpoints" / exp_name
    vis_dir = out_root / "vis" / exp_name
    log_dir = out_root / "logs" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={device} config={args.config}", flush=True)
    print("[train] building train dataloader (indexing local tar may take a while)...", flush=True)
    train_loader = build_dataloader(cfg, split="train", download=args.download)
    print(f"[train] train batches/epoch ≈ {len(train_loader)}", flush=True)

    print("[train] building val dataloader...", flush=True)
    try:
        val_loader = build_dataloader(cfg, split="validation", download=args.download)
        print(f"[train] val batches ≈ {len(val_loader)}", flush=True)
    except FileNotFoundError as e:
        print(f"[train][warn] 验证集不可用，将跳过验证: {e}", flush=True)
        val_loader = None

    print("[train] building model...", flush=True)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    print("[train] start epochs", flush=True)
    lr = cfg["train"].get("lr", 1e-4)
    wd = cfg["train"].get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = cfg["train"].get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_rmse = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_rmse = ckpt.get("best_rmse", best_rmse)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    if SummaryWriter is None:
        print("[warn] tensorboard 未安装，跳过 SummaryWriter；可 pip install tensorboard")
        writer = None
    else:
        writer = SummaryWriter(log_dir=str(log_dir))
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg, epoch, writer
        )
        scheduler.step()

        if (epoch + 1) % cfg["train"].get("val_interval", 1) == 0:
            if val_loader is None:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f} (no val set)")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_rmse": best_rmse,
                        "cfg": cfg,
                    },
                    ckpt_dir / f"epoch_{epoch:03d}.pth",
                )
            else:
                val_loss, meters = validate(
                    model, val_loader, criterion, device, cfg, epoch, writer, vis_dir
                )
                print(
                    f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"RMSE={meters['rmse']:.4f} AbsRel={meters['abs_rel']:.4f}"
                )
                is_best = meters["rmse"] < best_rmse
                if is_best:
                    best_rmse = meters["rmse"]
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_rmse": best_rmse,
                        "metrics": meters,
                        "cfg": cfg,
                    },
                    ckpt_dir / ("best.pth" if is_best else f"epoch_{epoch:03d}.pth"),
                )
                if is_best:
                    print(f"  -> new best RMSE={best_rmse:.4f}, saved best.pth")

    if writer is not None:
        writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
