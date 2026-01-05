# visualize_compare.py
import os, json, glob, re, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from Dataset import get_dataloader
from Networks5 import net
from Hyper_params import hp
from metrics import accuracy
from torch.nn import CrossEntropyLoss

CKPT_DIR   = "ckpts-huge"
CACHE_FILE = "acc_cache.json"   # 缓存路径
IMG_SAVE   = "compare_curve.png"
BATCH_SIZE = 800                # 评估 batch，越大越快
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 1. 扫描 checkpoint
# -------------------------------------------------
def scan_ckpts(prefix):
    files = glob.glob(os.path.join(CKPT_DIR, f"{prefix}*.pkl"))
    files.sort(key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)))
    assert len(files) == 20, f"期望 20 个文件，实际 {len(files)} 个"
    return files

ckpts_A = scan_ckpts("414k_Tiny_MoE_epoch_")
ckpts_B = scan_ckpts("414k_Tiny_MoE_Deeper_epoch_")

# -------------------------------------------------
# 2. 评估单个 checkpoint（返回正确率 %）
# -------------------------------------------------
@torch.no_grad()
def eval_ckpt(model_path, net_struct):
    hp.net_struct = net_struct          # 动态切换深度
    model = net().to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True)['net_state_dict'])
    model.eval()

    _, _, val_loader = get_dataloader()
    # 把 batch_size 放大可显著提速；如显存吃紧可改小
    val_loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=val_loader.num_workers, pin_memory=True)

    top1, top5, n = 0., 0., 0
    for b in val_loader:
        img = b['sketch_img'].to(DEVICE, non_blocking=True)
        seq = b['sketch_points'].to(DEVICE, non_blocking=True)
        lab = b['sketch_label'].to(DEVICE, non_blocking=True)

        logits, *_ = model(img, seq)
        err1, err5 = accuracy(logits, lab, topk=(1, 5))
        batch = lab.size(0)
        top1 += (100 - err1.item()) * batch   # 直接累加正确率
        top5 += (100 - err5.item()) * batch
        n += batch
    return top1/n, top5/n

# -------------------------------------------------
# 3. 一次性评估 20 轮，带缓存
# -------------------------------------------------
def build_cache():
    if os.path.exists(CACHE_FILE):
        return json.load(open(CACHE_FILE))

    print(">>> 首次运行，评估 40 个 checkpoint（约几分钟）……")
    cache = {"Tiny_top1": [], "Tiny_top5": [],
             "Deep_top1": [], "Deep_top5": []}

    # 评估 A 组
    for ck in ckpts_A:
        acc1, acc5 = eval_ckpt(ck, [10,2,2,2,2])
        cache["Tiny_top1"].append(acc1)
        cache["Tiny_top5"].append(acc5)
        print(f"  A {os.path.basename(ck)}  Top-1={acc1:.2f}  Top-5={acc5:.2f}")

    # 评估 B 组
    for ck in ckpts_B:
        acc1, acc5 = eval_ckpt(ck, [10,2,3,3,2])
        cache["Deep_top1"].append(acc1)
        cache["Deep_top5"].append(acc5)
        print(f"  B {os.path.basename(ck)}  Top-1={acc1:.2f}  Top-5={acc5:.2f}")

    json.dump(cache, open(CACHE_FILE, "w"), indent=2)
    print(f">>> 评估完成，缓存已写入 {CACHE_FILE}")
    return cache

# -------------------------------------------------
# 4. 画图
# -------------------------------------------------
def plot_curve(data):
    epochs = np.arange(1, 21)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, data["Tiny_top1"],  marker='o', lw=2.2, label="Tiny Top-1")
    plt.plot(epochs, data["Tiny_top5"],  marker='s', lw=2.2, label="Tiny Top-5")
    plt.plot(epochs, data["Deep_top1"],  marker='o', lw=2.2, label="Deep-Tiny Top-1")
    plt.plot(epochs, data["Deep_top5"],  marker='s', lw=2.2, label="Deep-Tiny Top-5")

    plt.title("QuickDraw414k 20 Epochs Accuracy Comparison", fontsize=15)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.xlim(0.5, 20.5)
    plt.ylim(bottom=min(
        min(data["Tiny_top1"]), min(data["Tiny_top5"]),
        min(data["Deep_top1"]), min(data["Deep_top5"])
    ) - 1)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(IMG_SAVE, dpi=300)
    print(f">>> 曲线已保存至 {os.path.abspath(IMG_SAVE)}")
    plt.show()

# -------------------------------------------------
# 5. 主入口
# -------------------------------------------------
if __name__ == "__main__":
    hp.Dataset = "QuickDraw414k"
    plot_curve(build_cache())