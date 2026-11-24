"""阶段6：Grad-CAM 可视化 + 错误分析。"""
import os
import sys
import json
import csv
import argparse
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, "./src")
from models import build_model, build_v4, merge_lora_v4
from data import build_transforms, PetDataset

OUT = "./outputs"
SPLIT = "./splits"


def load_model(variant, ckpt, device):
    if variant == "v4":
        m = build_v4(r=8, alpha=16)
        m.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
        m = merge_lora_v4(m)
    else:
        m = build_model(variant)
        m.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    return m.to(device).eval()


def get_target_layer(model, variant):
    if variant == "v1":
        return [model.layer4[-1]]
    else:
        return [model.blocks[-1].norm1]


def gradcam_visualize(variant, ckpt, device, n=10):
    model = load_model(variant, ckpt, device)

    # 从 test 集取前 n 张
    test_ds = PetDataset(os.path.join(SPLIT, "test.csv"),
                         transform=transforms.Compose([transforms.Resize(248), transforms.CenterCrop(224), transforms.ToTensor()]))
    os.makedirs(os.path.join(OUT, "gradcam", variant), exist_ok=True)

    if variant == "v1":
        # ResNet 用 GradCAM
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        for i in range(n):
            img_tensor, label = test_ds[i]
            pil_img = Image.open(test_ds.rows[i]["image_path"]).convert("RGB")
            orig = np.array(pil_img).astype(np.float32) / 255.0
            input_tensor = img_tensor.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(label)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            import cv2
            h, w = orig.shape[:2]
            grayscale_cam = cv2.resize(grayscale_cam, (w, h))
            vis = show_cam_on_image(orig, grayscale_cam, use_rgb=True)
            plt.imsave(os.path.join(OUT, "gradcam", variant, f"cam_{i:03d}_label{label}.png"), vis)
    else:
        # ViT 用注意力图（取最后一个 block 的 CLS 对其他 patch 的注意力）
        attn_hooks = {}
        def make_hook(name):
            def hook(module, inp, out):
                # 需要 forward 时捕获 attention 权重，这里用 forward hook 拿不到，改用下面方式
                pass
            return hook

        # 直接在 forward 中提取：重新实现一个提取注意力的前向
        def get_attention_map(model, x):
            B = x.shape[0]
            x = model.patch_embed(x)
            cls_token = model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + model.pos_embed
            x = model.pos_drop(x)
            for blk in model.blocks:
                # 手动计算 attention
                B, N, C = x.shape
                qkv = blk.attn.qkv(x).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)
                x = blk(x)  # 正常走一个 block
                last_attn = attn
            # 取 CLS token 对所有 patch 的注意力（去掉 CLS 自身）
            attn_map = last_attn.mean(dim=1)[:, 0, 1:]  # (B, N_patch)
            return attn_map

        for i in range(n):
            img_tensor, label = test_ds[i]
            pil_img = Image.open(test_ds.rows[i]["image_path"]).convert("RGB")
            orig = np.array(pil_img).astype(np.float32) / 255.0
            input_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                attn_map = get_attention_map(model, input_tensor)  # (1, 196)
            # reshape 成 14x14 再放大
            side = int(np.sqrt(attn_map.shape[1]))
            attn_grid = attn_map.reshape(side, side).cpu().numpy()
            attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
            import cv2
            h, w = orig.shape[:2]
            attn_grid = cv2.resize(attn_grid, (w, h))
            # 叠加
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_grid), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255.0
            vis = 0.5 * heatmap + 0.5 * orig
            vis = vis / vis.max()
            plt.imsave(os.path.join(OUT, "gradcam", variant, f"attn_{i:03d}_label{label}.png"), vis)
    print(f"{variant} 可视化完成（{n} 张）")


def error_analysis(variant, ckpt, device, n=20):
    """分析错分图片，返回错分类型统计。"""
    from evaluate import predict_all
    from data import build_loaders
    model = load_model(variant, ckpt, device)
    _, _, test_loader = build_loaders(SPLIT, "vit_base_patch16_224.augreg_in21k_ft_in1k", 64)
    logits, probs, preds, targets = predict_all(model, test_loader, device, use_amp=False)
    preds = preds.numpy()
    targets = targets.numpy()

    # 类别名
    class_names = []
    with open(os.path.join(SPLIT, "test.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            class_names.append(r["class_name"])
    # 去重保持顺序
    seen = []
    for c in class_names:
        if c not in seen:
            seen.append(c)
    class_names = seen

    # 找错分
    wrong = np.where(preds != targets)[0]
    cat_classes = set(range(12))  # 前12是猫
    cross_species = 0
    wrong_details = []
    for idx in wrong[:n]:
        is_cross = (targets[idx] < 12) != (preds[idx] < 12)
        if is_cross:
            cross_species += 1
        wrong_details.append({
            "index": int(idx),
            "true": class_names[targets[idx]],
            "pred": class_names[preds[idx]],
            "cross_species": bool(is_cross),
            "true_id": int(targets[idx]),
            "pred_id": int(preds[idx]),
        })

    result = {
        "variant": variant,
        "total_wrong": int(len(wrong)),
        "cross_species_among_wrong": cross_species,
        "wrong_details": wrong_details,
    }
    with open(os.path.join(OUT, f"error_{variant}.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"{variant} 错误分析完成：共错 {len(wrong)} 张，其中跨猫狗大类 {cross_species} 张（前{n}张）")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-cam", type=int, default=10)
    ap.add_argument("--n-err", type=int, default=20)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gradcam_visualize(args.variant, args.ckpt, device, args.n_cam)
    error_analysis(args.variant, args.ckpt, device, args.n_err)
