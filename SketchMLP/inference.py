#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_single.py
单次推理：图片 + sketch -> 类别
所有路径直接写死在本文件里，无需命令行
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from Hyper_params import hp
from Networks5 import net
from Dataset import off2abs, draw_three
from SketchUtils import SketchUtil

# ========== 1. 需要改的路径 ==========
IMG_DIR     = r"./demo/img"      # 图片完整路径
SKETCH_DIR   = r"./demo/seq"   # sketch 坐标文件
WEIGHT_PATH   = r"./pretrain/QD414k.pkl"        # 预训练权重
CATEGORIES_TXT= r"./categories.txt"             # 可选，没有就留空 ""
GPU_ID        = 0                               # GPU 编号，<0 表示用 CPU
RESULT_TXT     = r"./demo.txt"
# =====================================


SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
def load_categories(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def sketch_transform(coord_path):
    seq = np.load(coord_path, encoding='latin1', allow_pickle=True)
    if seq.dtype == 'O':
        seq = seq[0]
    seq = seq[:, :3].astype(np.float32) if seq.shape[1] == 4 else seq.astype(np.float32)

    index_neg = np.where(seq == -1)[0]
    if len(index_neg) == 0:
        seq = off2abs(seq)
        seq[:, :2] /= 256.0
    else:
        idx = index_neg[0]
        seq[:idx] = off2abs(seq)[:idx]
        seq[:idx, :2] /= 256.0
        seq[idx:] = -1

    res = np.ones((hp.seq_len, 3), dtype=np.float32) * -1
    real_len = min(hp.seq_len, len(seq))
    res[:real_len] = seq[:real_len]

    sketch_img = draw_three(res, stroke_flag=0)
    return sketch_img, torch.tensor(res)

def img_transform(img_path):
    transform = T.Compose([
        T.Resize((hp.img_size, hp.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(Image.open(img_path).convert("RGB"))

def inference(model, img_path, sketch_path, categories=None):
    model.eval()
    with torch.no_grad():
        img_tensor = img_transform(img_path).unsqueeze(0)
        _, sketch_points = sketch_transform(sketch_path)
        sketch_seq = sketch_points.unsqueeze(0)
        if GPU_ID >= 0:
            img_tensor  = img_tensor.cuda()
            sketch_seq  = sketch_seq.cuda()

        logits, _, _, _ = model(img_tensor, sketch_seq)
        prob = torch.softmax(logits, dim=1)
        top5_p, top5_idx = torch.topk(prob, k=5, dim=1)
        top5_idx = top5_idx.cpu().numpy()[0]
        top5_p   = top5_p.cpu().numpy()[0]

    res = [os.path.basename(img_path), os.path.basename(sketch_path)]
    for idx, p in zip(top5_idx, top5_p):
        name = categories[idx] if categories else f"class_{idx}"
        res.append(f"{name}({p*100:.2f}%)")
    return " ".join(res)

def main():
    device = torch.device("cpu" if GPU_ID < 0 else f"cuda:{GPU_ID}")
    categories = load_categories(CATEGORIES_TXT)

    model = net()
    ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["net_state_dict"] if "net_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()
    print("==> 模型加载完成，开始批量推理...")

    img_names = [f for f in os.listdir(IMG_DIR)
                 if os.path.splitext(f.lower())[1] in SUPPORTED_EXT]
    with open(RESULT_TXT, "w", encoding="utf-8") as fw:
        for im in img_names:
            name, _ = os.path.splitext(im)
            sketch_path = os.path.join(SKETCH_DIR, name + ".npy")
            if not os.path.isfile(sketch_path):
                print(f"[WARN] 找不到对应 sketch: {sketch_path}，跳过")
                continue
            line = inference(model,
                             os.path.join(IMG_DIR, im),
                             sketch_path,
                             categories)
            print(line)
            fw.write(line + "\n")
    print(f"\n==> 结果已保存到 {RESULT_TXT}")

if __name__ == "__main__":
    main()