"""图像与深度的联合变换（保持 RGB/Depth 空间对齐）。"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class JointResize:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(
        self, rgb: Image.Image, depth: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        rgb = rgb.resize((self.width, self.height), Image.BILINEAR)
        # 深度用最近邻，避免在边缘处插值出虚假深度
        depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
        depth_img = depth_img.resize((self.width, self.height), Image.NEAREST)
        depth = np.array(depth_img, dtype=np.float32)
        return rgb, depth


class JointRandomCrop:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(
        self, rgb: Image.Image, depth: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        w, h = rgb.size
        if h < self.height or w < self.width:
            # 过小则先 resize
            scale = max(self.height / h, self.width / w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            rgb = rgb.resize((nw, nh), Image.BILINEAR)
            depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
            depth = np.array(depth_img.resize((nw, nh), Image.NEAREST), dtype=np.float32)
            w, h = rgb.size

        top = np.random.randint(0, h - self.height + 1)
        left = np.random.randint(0, w - self.width + 1)
        rgb = TF.crop(rgb, top, left, self.height, self.width)
        depth = depth[top : top + self.height, left : left + self.width]
        return rgb, depth


class JointRandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, rgb: Image.Image, depth: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        if np.random.rand() < self.p:
            rgb = TF.hflip(rgb)
            depth = np.ascontiguousarray(np.fliplr(depth))
        return rgb, depth


class ToTensorNormalize:
    """RGB -> ImageNet 归一化；Depth -> float tensor [1,H,W]。"""

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __call__(
        self, rgb: Image.Image, depth: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_t = TF.to_tensor(rgb)  # [3,H,W], [0,1]
        rgb_t = TF.normalize(rgb_t, self.mean, self.std)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        return rgb_t, depth_t
