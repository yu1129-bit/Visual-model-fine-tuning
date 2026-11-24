"""阶段1 补充：数据探索——每类数量/尺寸/长宽比统计、损坏检查、每类拼图、易混淆品种分析。"""
import csv
import os
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT_DIR = "/root/vision-finetune/splits"
OUT_DIR = "/root/vision-finetune/outputs"


def load_csv(name):
    with open(os.path.join(SPLIT_DIR, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = load_csv("train.csv")
    val = load_csv("val.csv")
    test = load_csv("test.csv")

    # ---- 1. 每类数量统计 ----
    per_class = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for r in train:
        per_class[r["class_name"]]["train"] += 1
    for r in val:
        per_class[r["class_name"]]["val"] += 1
    for r in test:
        per_class[r["class_name"]]["test"] += 1

    with open(os.path.join(OUT_DIR, "per_class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "train", "val", "test", "total"])
        for cls, d in sorted(per_class.items()):
            w.writerow([cls, d["train"], d["val"], d["test"],
                        d["train"] + d["val"] + d["test"]])

    # ---- 2. 尺寸与长宽比统计 ----
    sizes = []
    aspects = []
    broken = []
    non_rgb = []
    modes = Counter()
    for r in train:
        try:
            img = Image.open(r["image_path"])
            modes[img.mode] += 1
            img.load()
            if img.mode not in ("RGB", "L", "RGBA", "LA", "P"):
                non_rgb.append(r["image_path"])
            w, h = img.size
            sizes.append((w, h))
            aspects.append(w / h)
        except Exception:
            broken.append(r["image_path"])

    sizes = np.array(sizes)
    aspects = np.array(aspects)
    stats = {
        "total_train": len(train),
        "broken_images": len(broken),
        "non_rgb_images": len(non_rgb),
        "image_modes": dict(modes),
        "width": {"min": int(sizes[:, 0].min()), "max": int(sizes[:, 0].max()),
                  "mean": round(float(sizes[:, 0].mean()), 1)},
        "height": {"min": int(sizes[:, 1].min()), "max": int(sizes[:, 1].max()),
                   "mean": round(float(sizes[:, 1].mean()), 1)},
        "aspect_ratio": {"min": round(float(aspects.min()), 3),
                         "max": round(float(aspects.max()), 3),
                         "mean": round(float(aspects.mean()), 3)},
    }
    import json
    with open(os.path.join(OUT_DIR, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("数据统计:", json.dumps(stats, ensure_ascii=False, indent=2))

    # ---- 3. 每类抽 4 张拼图 ----
    by_class = defaultdict(list)
    for r in train:
        by_class[r["class_name"]].append(r)
    n_cls = len(by_class)
    cols = 8  # 每行 8 类
    rows = (n_cls + cols - 1) // cols
    fig, axes = plt.subplots(rows * 4, cols, figsize=(cols * 1.8, rows * 4 * 1.8))
    for i, (cls, items) in enumerate(sorted(by_class.items())):
        col = i % cols
        row_block = i // cols
        # 每类取前 4 张
        for j in range(4):
            ax = axes[row_block * 4 + j, col]
            if j < len(items):
                img = Image.open(items[j]["image_path"]).convert("RGB")
                ax.imshow(img)
                if j == 0:
                    ax.set_title(cls, fontsize=7)
            ax.axis("off")
    fig.suptitle(f"Oxford-IIIT Pet: 每类 4 张样本（共 {n_cls} 类）", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "class_grid.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"拼图已保存: class_grid.png（{n_cls} 类）")

    # ---- 4. 尺寸/长宽比分布图 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(sizes[:, 0], bins=30, alpha=0.7, label="width")
    axes[0].hist(sizes[:, 1], bins=30, alpha=0.7, label="height")
    axes[0].set_title("图像宽高分布")
    axes[0].set_xlabel("像素")
    axes[0].legend()
    axes[1].hist(aspects, bins=30, alpha=0.7)
    axes[1].set_title("长宽比分布")
    axes[1].set_xlabel("宽/高")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "size_dist.png"), dpi=120)
    plt.close(fig)

    print("\n阶段1数据探索完成")


if __name__ == "__main__":
    main()
