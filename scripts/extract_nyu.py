"""把 NYU Depth V2 的 tar 解压成独立 .h5，加速训练/评测读盘。

数据布局（下载后）:
  data/nyu_depth_v2/data/train-*.tar
  data/nyu_depth_v2/data/val-*.tar

解压后:
  data/nyu_depth_v2/extracted/train/**/*.h5
  data/nyu_depth_v2/extracted/val/**/*.h5

用法示例:
  # 只解压验证集（你已下载 val-000000.tar）
  python scripts/extract_nyu.py --cache-dir data/nyu_depth_v2 --splits val

  # 解压 train + val
  python scripts/extract_nyu.py --cache-dir data/nyu_depth_v2 --splits train,val

Dataset / eval 发现 extracted/ 后会自动优先使用解压文件。
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def extract_tar(tar_path: Path, out_root: Path, skip_existing: bool = True) -> int:
    print(f"[extract] {tar_path.name} -> {out_root}", flush=True)
    n_new, n_skip = 0, 0
    with tarfile.open(tar_path, "r:") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".h5")]
        total = len(members)
        for i, m in enumerate(members, 1):
            dest = out_root / m.name
            if skip_existing and dest.exists() and dest.stat().st_size > 0:
                n_skip += 1
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                if src is None:
                    continue
                with open(dest, "wb") as fout:
                    fout.write(src.read())
                n_new += 1
            if i % 200 == 0 or i == total:
                print(f"  progress {i}/{total} (new={n_new}, skip={n_skip})", flush=True)
    print(f"[extract] done {tar_path.name}: new={n_new}, skip={n_skip}", flush=True)
    return n_new


def main() -> None:
    p = argparse.ArgumentParser(description="Extract NYU Depth V2 tar archives to .h5 files")
    p.add_argument("--cache-dir", type=str, default="data/nyu_depth_v2")
    p.add_argument(
        "--splits",
        type=str,
        default="val",
        help="逗号分隔: train / val（默认只解压 val）",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="即使目标 h5 已存在也强制覆盖",
    )
    args = p.parse_args()

    root = Path(args.cache_dir)
    tar_dir = root / "data" if (root / "data").is_dir() else root
    out_root = root / "extracted"
    out_root.mkdir(parents=True, exist_ok=True)

    splits = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
    prefixes = []
    if "train" in splits:
        prefixes.append("train")
    if "val" in splits or "validation" in splits:
        prefixes.append("val")
    if not prefixes:
        raise SystemExit("请至少指定一个 split: train 或 val")

    total_new = 0
    for prefix in prefixes:
        tars = sorted(tar_dir.glob(f"{prefix}-*.tar"))
        if not tars:
            tars = sorted(tar_dir.glob(f"{prefix}*.tar"))
        if not tars:
            print(f"[warn] 未找到 {prefix} 的 tar，目录: {tar_dir}")
            continue
        for tar_path in tars:
            total_new += extract_tar(tar_path, out_root, skip_existing=not args.no_skip)

    # 统计解压结果
    for prefix in prefixes:
        n = len(list((out_root).glob(f"{prefix}/**/*.h5")))
        print(f"[extract] {prefix}: {n} h5 under {out_root / prefix}")
    print(f"[extract] 本轮新增 {total_new} 个文件。")


if __name__ == "__main__":
    main()
