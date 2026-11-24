"""评估：top-1/top-5/Macro-F1、每类准确率、混淆矩阵、猫狗大类判错统计。"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             top_k_accuracy_score)

from data import build_loaders
from models import build_model, build_v4, merge_lora_v4


@torch.no_grad()
def predict_all(model, loader, device, use_amp=True):
    model.eval()
    all_logits, all_targets = [], []
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        with autocast(enabled=use_amp):
            logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(target.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return logits, probs, preds, targets


def evaluate_model(model, test_loader, device, class_names, use_amp=True):
    logits, probs, preds, targets = predict_all(model, test_loader, device, use_amp)

    top1 = accuracy_score(targets, preds)
    top5 = top_k_accuracy_score(targets, logits, k=5, labels=list(range(logits.size(1))))
    macro_f1 = f1_score(targets, preds, average="macro")

    # 每类准确率
    per_class = {}
    for c in range(logits.size(1)):
        mask = targets == c
        if mask.sum() > 0:
            per_class[class_names[c]] = accuracy_score(targets[mask], preds[mask])

    # 混淆矩阵
    cm = confusion_matrix(targets, preds)

    # 猫狗大类判错：Oxford-IIIT Pet 前 12 类是猫，后 25 类是狗
    is_cat = targets < 12
    pred_is_cat = preds < 12
    cross_species = (is_cat != pred_is_cat).float().mean().item()

    return {
        "top1": float(top1),
        "top5": float(top5),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "cross_species_error": cross_species,
        "preds": preds.tolist(),
        "targets": targets.tolist(),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.variant == "v4":
        # LoRA：加载后合并权重再评估
        model = build_v4(r=args.lora_r, alpha=args.lora_alpha)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state"])
        model = merge_lora_v4(model)
        model = model.to(device)
    else:
        model = build_model(args.variant)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state"])
        model = model.to(device)

    _, _, test_loader = build_loaders(args.split_dir, args.model_name, args.batch_size)
    class_names = list(torch.load(args.ckpt, map_location=device).get("class_names", range(37)))

    result = evaluate_model(model, test_loader, device, class_names)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"test_{args.variant}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"top1={result['top1']:.4f} top5={result['top5']:.4f} "
          f"macro_f1={result['macro_f1']:.4f} cross_species_err={result['cross_species_error']:.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-name", default="vit_base_patch16_224.augreg_in21k_ft_in1k")
    ap.add_argument("--split-dir", default="./splits")
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    args = ap.parse_args()
    main(args)
