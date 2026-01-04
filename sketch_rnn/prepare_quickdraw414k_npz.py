"""
将 QuickDraw414k 的坐标序列（`QuickDraw414k/coordinate_files/.../*.npy`）
转换成 sketch-rnn 训练脚本需要的 .npz 数据集格式。

### QuickDraw414k 的 npy（根据你给的打印与 draw_three 验证）
每行 4 列，形如：
  [x, y, pen_down, pen_up]

- **x, y**: 绝对坐标（0~255 左右）
- **pen_down / pen_up**: one-hot（大部分是 [1,0]，遇到分笔时为 [0,1]）
- 末尾常见一行 **[0,0,0,0]**（占位），随后是 **[-1,-1,-1,-1]** padding

你在 `debug.py` 里用第 4 列（pen_up）做 “next stroke” 标记能正确画出图，
因此这里默认用 **第 4 列作为 stroke-3 的第三列 pen_up**。

### sketch-rnn 需要的格式
npz 内包含 3 个字段：train / valid / test
每个字段是一个 list（dtype=object），元素为变长的 stroke-3 数组：
  [dx, dy, pen_up]
其中 **pen_up=1 表示从该点到下一点要抬笔（结束当前 stroke）**。
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np


def _read_list_file(list_path: str) -> List[str]:
    """读取 tiny_*_set.txt 这类列表文件，返回 png 文件名列表（不含 label）。"""
    items: List[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式通常为：xxx.png <label>
            parts = line.split()
            items.append(parts[0])
    return items


def _load_npy_sequence(npy_path: str) -> np.ndarray:
    seq = np.load(npy_path, encoding="latin1", allow_pickle=True)
    if getattr(seq, "dtype", None) is not None and seq.dtype == "object":
        seq = seq[0]
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] < 4:
        raise ValueError(f"Unexpected npy shape: {seq.shape} at {npy_path}")
    seq = seq[:, 0:4].astype(np.float32, copy=False)
    return seq


def _truncate_at_neg_one(seq: np.ndarray) -> np.ndarray:
    """遇到 -1（任意列）就截断。"""
    neg_mask = np.any(seq == -1, axis=1)
    if not np.any(neg_mask):
        return seq
    end = int(np.argmax(neg_mask))  # 第一个 True 的位置
    return seq[:end]


def _strip_trailing_zeros(seq: np.ndarray) -> np.ndarray:
    """
    QuickDraw414k 的 npy 末尾常见一个 [0,0,0,0] 行作为结束占位，
    这不是有效点（会给 SketchRNN 增加一个“静止点”），所以剔除末尾连续全 0 行。
    """
    if len(seq) == 0:
        return seq
    i = len(seq)
    while i > 0:
        row = seq[i - 1]
        if np.all(row == 0):
            i -= 1
            continue
        break
    return seq[:i]


def _to_stroke3(
    seq4: np.ndarray,
    coord_format: str,
) -> np.ndarray:
    """
    seq4: [N,4] -> stroke-3: [N,3] = [dx, dy, pen_up]
    coord_format: "offset" or "absolute"
    """
    x = seq4[:, 0].astype(np.float32, copy=False)
    y = seq4[:, 1].astype(np.float32, copy=False)
    # 约定：第 4 列是 pen_up (0/1)
    pen_up = seq4[:, 3].astype(np.float32, copy=False)

    if coord_format == "absolute":
        dx = np.empty_like(x)
        dy = np.empty_like(y)
        dx[0] = x[0]
        dy[0] = y[0]
        dx[1:] = x[1:] - x[:-1]
        dy[1:] = y[1:] - y[:-1]
    elif coord_format == "offset":
        dx = x
        dy = y
    else:
        raise ValueError(f"Unknown coord_format: {coord_format}")

    out = np.stack([dx, dy, pen_up], axis=1).astype(np.float32)
    return out


def _convert_split(
    list_path: str,
    coord_root: str,
    coord_format: str,
    limit: int | None = None,
) -> List[np.ndarray]:
    png_names = _read_list_file(list_path)
    if limit is not None:
        png_names = png_names[:limit]

    strokes: List[np.ndarray] = []
    for name in png_names:
        npy_name = name.replace(".png", ".npy")
        npy_path = os.path.join(coord_root, npy_name)
        seq4 = _load_npy_sequence(npy_path)
        seq4 = _truncate_at_neg_one(seq4)
        seq4 = _strip_trailing_zeros(seq4)
        if len(seq4) == 0:
            continue
        stroke3 = _to_stroke3(seq4, coord_format=coord_format)
        strokes.append(stroke3)
    return strokes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q414k_root", type=str, default="../QuickDraw414k", help="QuickDraw414k 根目录")
    p.add_argument("--train_list", type=str, default="../QuickDraw414k/picture_files/tiny_train_set.txt")
    p.add_argument("--valid_list", type=str, default="../QuickDraw414k/picture_files/tiny_val_set.txt")
    p.add_argument("--test_list", type=str, default="../QuickDraw414k/picture_files/tiny_test_set.txt")
    p.add_argument("--train_coord_root", type=str, default="../QuickDraw414k/coordinate_files/train")
    p.add_argument("--valid_coord_root", type=str, default="../QuickDraw414k/coordinate_files/val")
    p.add_argument("--test_coord_root", type=str, default="../QuickDraw414k/coordinate_files/test")
    p.add_argument("--out_npz", type=str, default="./quickdraw414k_sketchrnn.npz")
    # 你给的 npy 打印更像是绝对坐标（且 draw_three 也按 absolute 画），因此默认 absolute
    p.add_argument("--coord_format", type=str, choices=["offset", "absolute"], default="absolute",
                  help="npy 前两列是 offset 还是 absolute 坐标")
    p.add_argument("--limit", type=int, default=None, help="只转换前 N 条（调试用）")
    args = p.parse_args()

    # 允许只给 q414k_root 时自动拼路径（但默认值已是完整相对路径）
    _ = args.q414k_root

    print("Converting train...")
    train = _convert_split(args.train_list, args.train_coord_root, args.coord_format, args.limit)
    print(f"Train strokes: {len(train)}")

    print("Converting valid...")
    valid = _convert_split(args.valid_list, args.valid_coord_root, args.coord_format, args.limit)
    print(f"Valid strokes: {len(valid)}")

    print("Converting test...")
    test = _convert_split(args.test_list, args.test_coord_root, args.coord_format, args.limit)
    print(f"Test strokes: {len(test)}")

    out_dir = os.path.dirname(os.path.abspath(args.out_npz))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out_npz,
        train=np.array(train, dtype=object),
        valid=np.array(valid, dtype=object),
        test=np.array(test, dtype=object),
    )
    print(f"Saved: {args.out_npz}")


if __name__ == "__main__":
    main()


