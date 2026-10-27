"""NYU Depth V2 深度补全 Dataset。

数据来源（作业指定）：
  https://huggingface.co/datasets/sayakpaul/nyu_depth_v2

该 HF 数据集由若干 tar 组成，内部是 .h5（字段 rgb / depth）。
新版 `datasets` 库已不再支持其加载脚本，因此本实现直接读取本地 tar/h5。

重要：索引时记录每个 h5 在 tar 内的字节偏移，训练时用 seek 随机读，
避免每次重新打开/扫描整个 2~3GB tar（否则进度条会长期停在 0%）。
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.sparse_sample import sample_sparse_depth
from .transforms import JointRandomHorizontalFlip, JointResize, ToTensorNormalize

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("需要安装 h5py: pip install h5py") from e


@dataclass(frozen=True)
class TarSample:
    tar_path: Path
    offset: int  # 成员数据在 tar 中的起始偏移
    size: int
    name: str


@dataclass(frozen=True)
class FileSample:
    """已解压的独立 .h5 路径（比 tar seek 更快）。"""
    path: Path


SampleItem = TarSample | FileSample


def _ensure_rgb_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"不支持的 RGB 类型: {type(image)}")


def _ensure_depth_np(depth: Any) -> np.ndarray:
    """统一为 float32 [H,W]，单位：米。"""
    if isinstance(depth, Image.Image):
        depth = np.array(depth)
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0] if depth.shape[-1] == 1 else depth[0] if depth.shape[0] == 1 else depth.mean(axis=-1)
    depth = depth.astype(np.float32)
    if np.isfinite(depth).any() and np.nanmax(depth) > 100:
        depth = depth / 1000.0
    return depth


def _load_h5_fileobj(fobj) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(fobj, "r") as f:
        rgb = np.array(f["rgb"])
        depth = np.array(f["depth"])
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb, depth


def _load_h5_bytes(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    return _load_h5_fileobj(io.BytesIO(raw))


def _load_h5_path(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    return _load_h5_fileobj(path)


def _read_tar_member(sample: TarSample) -> bytes:
    """按偏移直接读取 tar 成员，O(1) 随机访问。"""
    with open(sample.tar_path, "rb") as f:
        f.seek(sample.offset)
        return f.read(sample.size)


class NYUDepthCompletionDataset(Dataset):
    """NYU Depth V2 -> (RGB, sparse_depth, mask, gt_depth)。"""

    def __init__(
        self,
        split: str = "train",
        data_root: str | Path = "data/nyu_depth_v2",
        dataset_name: str = "sayakpaul/nyu_depth_v2",
        num_sparse: int = 500,
        height: int = 240,
        width: int = 320,
        min_depth: float = 0.0,
        max_depth: float = 10.0,
        download: bool = False,
        use_huggingface: bool = False,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> None:
        del dataset_name, download, use_huggingface

        assert split in ("train", "validation", "test", "val")
        if split in ("val", "test"):
            split = "validation"

        self.split = split
        self.num_sparse = num_sparse
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.seed = seed
        self.data_root = Path(data_root)

        self.resize = JointResize(height, width)
        self.to_tensor = ToTensorNormalize()
        self.is_train = split == "train"
        self.aug_flip = JointRandomHorizontalFlip(0.5) if self.is_train else None

        # 优先已解压 h5；否则回退到 tar 偏移随机读
        self.samples: List[SampleItem] = self._index_extracted(split)
        source = "extracted"
        if not self.samples:
            self.samples = self._index_local_tars(split)
            source = "tar"

        if max_samples is not None:
            self.samples = self.samples[: max(0, int(max_samples))]

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"在 {self.data_root} 下未找到 {split} 的 .h5 样本。\n"
                f"请先下载 tar，例如:\n"
                f"  export HF_ENDPOINT=https://hf-mirror.com\n"
                f"  python scripts/download_nyu.py --cache-dir {self.data_root}\n"
                f"可选加速: python scripts/extract_nyu.py --cache-dir {self.data_root}\n"
                f"期望路径类似: {self.data_root}/data/train-000000.tar"
            )
        print(
            f"[data] split={split} samples={len(self.samples)} source={source} root={self.data_root}",
            flush=True,
        )

    def _tar_dir(self) -> Path:
        cand = self.data_root / "data"
        return cand if cand.is_dir() else self.data_root

    def _index_extracted(self, split: str) -> List[SampleItem]:
        """扫描 extracted/{train|val}/**/*.h5。"""
        prefix = "train" if split == "train" else "val"
        base = self.data_root / "extracted" / prefix
        if not base.is_dir():
            # tar 内路径本身带 train/...，解压到 extracted/ 后也可能是 extracted/train/...
            alt = self.data_root / "extracted"
            if alt.is_dir():
                paths = sorted(alt.glob(f"{prefix}/**/*.h5"))
            else:
                return []
        else:
            paths = sorted(base.rglob("*.h5"))

        if not paths:
            # 再试：extracted 根下直接是 train/...
            paths = sorted((self.data_root / "extracted").glob(f"{prefix}/**/*.h5"))
        return [FileSample(path=p) for p in paths]

    def _index_local_tars(self, split: str) -> List[SampleItem]:
        tar_dir = self._tar_dir()
        prefix = "train" if split == "train" else "val"
        tars = sorted(tar_dir.glob(f"{prefix}-*.tar"))
        if not tars:
            tars = sorted(tar_dir.glob(f"{prefix}*.tar"))

        samples: List[SampleItem] = []
        for tar_path in tars:
            print(f"[data] indexing {tar_path.name} (record offsets)...", flush=True)
            with tarfile.open(tar_path, "r:") as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.endswith(".h5"):
                        samples.append(
                            TarSample(
                                tar_path=tar_path,
                                offset=int(m.offset_data),
                                size=int(m.size),
                                name=m.name,
                            )
                        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_raw(self, index: int) -> tuple[Image.Image, np.ndarray]:
        sample = self.samples[index]
        if isinstance(sample, FileSample):
            rgb, depth = _load_h5_path(sample.path)
        else:
            rgb, depth = _load_h5_bytes(_read_tar_member(sample))
        return _ensure_rgb_pil(rgb), _ensure_depth_np(depth)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rgb, depth = self._get_raw(index)

        depth = depth.copy()
        invalid = (~np.isfinite(depth)) | (depth <= self.min_depth) | (depth > self.max_depth)
        depth[invalid] = 0.0

        rgb, depth = self.resize(rgb, depth)
        if self.aug_flip is not None:
            rgb, depth = self.aug_flip(rgb, depth)

        rgb_t, gt_t = self.to_tensor(rgb, depth)

        rng = np.random.default_rng(self.seed + index * 9973)
        sparse, mask = sample_sparse_depth(
            gt_t,
            num_samples=self.num_sparse,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            rng=rng,
        )
        gt_valid = ((gt_t > self.min_depth) & (gt_t <= self.max_depth)).float()

        return {
            "rgb": rgb_t,
            "sparse_depth": sparse,
            "sparse_mask": mask,
            "gt_depth": gt_t,
            "gt_valid": gt_valid,
        }


def build_dataloader(
    cfg: Dict[str, Any],
    split: str = "train",
    download: bool = False,
) -> DataLoader:
    del download
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    ds = NYUDepthCompletionDataset(
        split=split,
        data_root=data_cfg.get("data_root", "data/nyu_depth_v2"),
        num_sparse=data_cfg.get("num_sparse", 500),
        height=data_cfg.get("image_height", 240),
        width=data_cfg.get("image_width", 320),
        min_depth=data_cfg.get("min_depth", 0.0),
        max_depth=data_cfg.get("max_depth", 10.0),
        seed=cfg.get("experiment", {}).get("seed", 42),
        max_samples=data_cfg.get("max_samples"),
    )

    is_train = split == "train"
    batch_size = train_cfg.get("batch_size", 8) if is_train else eval_cfg.get("batch_size", 8)
    num_workers = int(data_cfg.get("num_workers", 4))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
