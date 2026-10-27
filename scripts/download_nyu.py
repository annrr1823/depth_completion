"""下载 NYU Depth V2 的 tar 分包（不依赖已废弃的 dataset loading script）。

数据仓库: https://huggingface.co/datasets/sayakpaul/nyu_depth_v2
布局:
  data/train-000000.tar ... train-000011.tar   (共 12 个)
  data/val-000000.tar   ... val-000001.tar     (共 2 个)

推荐（国内镜像）:
  export HF_ENDPOINT=https://hf-mirror.com
  pip install h5py huggingface_hub
  python scripts/download_nyu.py --cache-dir data/nyu_depth_v2

只下训练第一个分包做冒烟:
  python scripts/download_nyu.py --cache-dir data/nyu_depth_v2 --splits train --max-files 1
"""

from __future__ import annotations

import argparse
from pathlib import Path


TRAIN_FILES = [f"data/train-{i:06d}.tar" for i in range(12)]
VAL_FILES = [f"data/val-{i:06d}.tar" for i in range(2)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="sayakpaul/nyu_depth_v2")
    parser.add_argument("--cache-dir", type=str, default="data/nyu_depth_v2")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="逗号分隔: train / val",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="每个 split 最多下载几个 tar；0 表示全部",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("请先安装: pip install huggingface_hub") from e

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    files = []
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    if "train" in splits:
        files.extend(TRAIN_FILES)
    if "val" in splits or "validation" in splits:
        files.extend(VAL_FILES)

    if args.max_files and args.max_files > 0:
        # 按 split 截断
        truncated = []
        if "train" in splits:
            truncated.extend(TRAIN_FILES[: args.max_files])
        if "val" in splits or "validation" in splits:
            truncated.extend(VAL_FILES[: args.max_files])
        files = truncated

    print(f"将下载 {len(files)} 个文件到 {cache}")
    for rel in files:
        print(f"-> {rel}")
        path = hf_hub_download(
            repo_id=args.repo,
            repo_type="dataset",
            filename=rel,
            local_dir=str(cache),
        )
        print(f"   saved: {path}")

    print("Done.")
    print("依赖提醒: 训练读 h5 需要 `pip install h5py`")


if __name__ == "__main__":
    main()
