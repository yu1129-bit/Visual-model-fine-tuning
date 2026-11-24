"""可视化：Grad-CAM（ResNet）/ 注意力图（ViT）、训练曲线、混淆矩阵、拼图。"""

import os
import argparse
import json
import math

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import build_model, build_v4


def draw_training_curves(history, save_path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_top1"], label="train")
    axes[1].plot(epochs, history["val_top1"], label="val")
    axes[1].set_title(f"{title} - Top-1 Acc")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def draw_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def draw_grid(sample_images, labels, save_path, cols=4):
    n = len(sample_images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()
    for i, (img, lab) in enumerate(zip(sample_images, labels)):
        axes[i].imshow(img)
        axes[i].set_title(lab, fontsize=8)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_gradcam(model, model_type, image_paths, save_dir, class_idx=None):
    """对一组图片生成 Grad-CAM / 注意力图。"""
    os.makedirs(save_dir, exist_ok=True)

    if model_type == "vit":
        # ViT 用最后一个 block 的注意力，或用 GradCAM 作用于 blocks[-1].norm1
        target_layers = [model.blocks[-1].norm1]
    else:
        target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, p in enumerate(image_paths):
        from torchvision import transforms
        pil_img = Image.open(p).convert("RGB")
        orig = np.array(pil_img).astype(np.float32) / 255.0
        inp = transforms.Compose([
            transforms.Resize(248),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])(pil_img).unsqueeze(0)

        targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
        grayscale_cam = cam(input_tensor=inp, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(orig, grayscale_cam, use_rgb=True)
        plt.imsave(os.path.join(save_dir, f"cam_{i:03d}.png"), visualization)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.variant == "v4":
        model = build_v4()
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state"])
        model = model.merge_and_unload()
    else:
        model = build_model(args.variant)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state"])
    model = model.to(device).eval()

    # 训练曲线（从 metrics json 读）
    metrics = json.load(open(args.metrics)) if args.metrics else None
    if metrics:
        draw_training_curves(metrics["history"], os.path.join(args.output_dir, "curves.png"),
                             args.variant.upper())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metrics", default="")
    ap.add_argument("--output-dir", default="./outputs")
    args = ap.parse_args()
    main(args)
