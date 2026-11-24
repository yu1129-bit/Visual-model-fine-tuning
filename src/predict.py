"""推理：输入图片路径或文件夹，输出 top-3 类别与置信度。V4 演示 merge_and_unload。"""

import os
import argparse
import glob

import torch
import timm
from torchvision import transforms

from models import build_model, build_v4, resolve_data_config
from data import PetDataset


@torch.no_grad()
def predict_image(model, image_path, transform, device, class_names):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    logits = model(inp)
    probs = torch.softmax(logits, dim=1)[0]
    top3 = torch.topk(probs, 3)
    result = []
    for val, idx in zip(top3.values, top3.indices):
        result.append((class_names[idx.item()], round(val.item(), 4)))
    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.variant == "v4":
        model = build_v4(r=args.lora_r, alpha=args.lora_alpha)
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model = model.merge_and_unload()
        print("[v4] LoRA 权重已 merge_and_unload")
    else:
        model = build_model(args.variant)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(248),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 类别名（37 类猫狗品种）
    from torchvision.datasets import OxfordIIITPet
    class_names = OxfordIIITPet(root=args.data_root, split="trainval", download=False).classes

    # 输入可以是文件或文件夹
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(args.input, e))
        paths.sort()

    for p in paths:
        top3 = predict_image(model, p, transform, device, class_names)
        print(f"\n{p}")
        for rank, (name, conf) in enumerate(top3, 1):
            print(f"  {rank}. {name}: {conf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="图片路径或文件夹")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()
    main(args)
