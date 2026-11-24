"""补充实验：
1. 阶段2 增强消融：无增强 / 基础增强 / 基础+RandAugment（用 V3 配置）
2. 阶段4 V2 特征对比：CLS token vs patch 平均池化（缓存特征后训 LogisticRegression）
3. 阶段5 LoRA r 消融：r=4/8/16

这些实验在基础训练完成后运行，结果写入 outputs/ablation_*.json。
"""
import os
import sys
import json
import argparse
import time

import numpy as np
import torch
from torch.cuda.amp import autocast

sys.path.insert(0, "/root/vision-finetune/src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["augment", "feature_pool", "lora_r"])
    ap.add_argument("--split-dir", default="/root/vision-finetune/splits")
    ap.add_argument("--output-dir", default="/root/vision-finetune/outputs")
    args = ap.parse_args()

    if args.task == "augment":
        run_augment_ablation(args)
    elif args.task == "feature_pool":
        run_feature_pool(args)
    elif args.task == "lora_r":
        run_lora_r(args)


# --------------------------------------------------------------------------- #
# 阶段2：增强消融
# --------------------------------------------------------------------------- #
def run_augment_ablation(args):
    from data import build_loaders
    from models import build_v3
    import torch.nn as nn
    import torch.optim as optim
    from train import train_one_epoch, evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for aa_name, aa_val in [("none", "none"), ("base", "base"), ("randaug", "rand-m9-mstd0.5-inc1")]:
        torch.manual_seed(42)
        model = build_v3().to(device)
        train_loader, val_loader, _ = build_loaders(
            args.split_dir, "vit_base_patch16_224.augreg_in21k_ft_in1k",
            64, aa=aa_val, num_workers=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
        scaler = torch.cuda.amp.GradScaler()
        # 只训 5 个 epoch 做对比（消融目的，不必跑满 15）
        best_val = 0.0
        for ep in range(5):
            train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss, val_t1, _ = evaluate(model, val_loader, criterion, device)
            best_val = max(best_val, val_t1)
        results[aa_name] = {"best_val_top1": round(best_val, 2)}
        print(f"增强[{aa_name}] best_val_top1={best_val:.2f}")
    with open(os.path.join(args.output_dir, "ablation_augment.json"), "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------------------- #
# 阶段4：CLS vs patch 平均池化
# --------------------------------------------------------------------------- #
def run_feature_pool(args):
    from data import build_transforms, PetDataset
    from models import build_feature_extractor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms(False, "vit_base_patch16_224.augreg_in21k_ft_in1k")

    train_ds = PetDataset(os.path.join(args.split_dir, "train.csv"), transform=transform)
    val_ds = PetDataset(os.path.join(args.split_dir, "val.csv"), transform=transform)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8)

    results = {}
    for pool in ["cls", "avg"]:
        model = build_feature_extractor(pool).to(device).eval()
        feats, labels = [], []
        with torch.no_grad():
            for x, y in train_loader:
                f = model(x.to(device))
                feats.append(f.cpu().numpy())
                labels.append(y.numpy())
        X = np.concatenate(feats)
        y = np.concatenate(labels)
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(X, y)
        # 验证
        vfeats, vlabels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vfeats.append(model(x.to(device)).cpu().numpy())
                vlabels.append(y.numpy())
        vX = np.concatenate(vfeats)
        vy = np.concatenate(vlabels)
        acc = accuracy_score(vy, clf.predict(vX))
        results[pool] = {"val_top1": round(float(acc) * 100, 2)}
        print(f"特征[{pool}] val_top1={acc*100:.2f}")
    with open(os.path.join(args.output_dir, "ablation_feature_pool.json"), "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------------------- #
# 阶段5：LoRA r 消融
# --------------------------------------------------------------------------- #
def run_lora_r(args):
    from data import build_loaders
    from models import build_v4, count_trainable_params
    import torch.nn as nn
    import torch.optim as optim
    from train import train_one_epoch, evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for r in [4, 8, 16]:
        torch.manual_seed(42)
        model = build_v4(r=r, alpha=2 * r).to(device)  # alpha=2r 保持 scaling 一致
        n_trainable = count_trainable_params(model)
        train_loader, val_loader, _ = build_loaders(
            args.split_dir, "vit_base_patch16_224.augreg_in21k_ft_in1k",
            64, num_workers=8)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=1e-3, weight_decay=0.05)
        scaler = torch.cuda.amp.GradScaler()
        best_val = 0.0
        peak_mem = 0.0
        for ep in range(15):
            train_one_epoch(model, train_loader, optimizer, criterion, scaler, device,
                            label_smoothing=0.1)
            val_loss, val_t1, _ = evaluate(model, val_loader, criterion, device)
            best_val = max(best_val, val_t1)
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / 1024**3)
        results[f"r={r}"] = {
            "best_val_top1": round(best_val, 2),
            "trainable_params": n_trainable,
            "peak_mem_gb": round(peak_mem, 2),
        }
        print(f"LoRA r={r}: val_top1={best_val:.2f} params={n_trainable:,} mem={peak_mem:.2f}GB")
    with open(os.path.join(args.output_dir, "ablation_lora_r.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
