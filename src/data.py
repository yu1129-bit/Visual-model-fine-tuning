"""数据准备、划分、增强与 DataLoader。

职责：
1. 下载 Oxford-IIIT Pet 数据集
2. 从 trainval 按类别分层划出 10% 验证集，保存 splits/{train,val,test}.csv
3. 数据统计（每类数量、尺寸、长宽比），损坏/非 RGB 图片检查
4. 增强管道：训练 RandomResizedCrop(224)+翻转+RandAugment；验证/测试 Resize(248)+CenterCrop(224)
5. 归一化均值/方差/输入尺寸用 timm.data.resolve_data_config 读取，不手写
"""

import os
import csv
import argparse
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


DATA_ROOT = os.environ.get("PET_DATA_ROOT", "./data")
SPLITS_DIR = "./splits"


# --------------------------------------------------------------------------- #
# 数据集下载与划分
# --------------------------------------------------------------------------- #
def download_pets(root: str = DATA_ROOT):
    """下载 Oxford-IIIT Pet，返回 trainval/test 两个 torchvision 目录。"""
    from torchvision.datasets import OxfordIIITPet

    os.makedirs(root, exist_ok=True)
    # trainval 用于我们自己分层划分；test 是官方测试集
    trainval = OxfordIIITPet(root=root, split="trainval", download=True)
    test = OxfordIIITPet(root=root, split="test", download=True)
    return trainval, test


def stratified_split(trainval, val_ratio: float = 0.1, seed: int = 42):
    """按类别分层划分 trainval -> train/val。

    返回 train_indices, val_indices。
    """
    rng = np.random.default_rng(seed)
    labels = np.array(trainval._labels)  # OxfordIIITPet 的 _labels 是 1-37
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def write_splits(trainval, test, train_idx, val_idx, split_dir: str = SPLITS_DIR):
    """把划分结果写成 splits/{train,val,test}.csv。

    列：image_path, class_id(1-37), class_name, split
    """
    os.makedirs(split_dir, exist_ok=True)
    class_names = trainval.classes  # 37 个品种名

    def _rows(indices, split_name):
        rows = []
        for i in indices:
            path, label = trainval._images[i], trainval._labels[i]
            # torchvision 的 _labels 是 0-indexed（0-36）
            rows.append([path, int(label), class_names[label], split_name])
        return rows

    def _write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "class_id", "class_name", "split"])
            w.writerows(rows)

    _write(os.path.join(split_dir, "train.csv"), _rows(train_idx, "train"))
    _write(os.path.join(split_dir, "val.csv"), _rows(val_idx, "val"))

    # 官方 test 集
    test_rows = [
        [test._images[i], int(test._labels[i]),
         class_names[test._labels[i]], "test"]
        for i in range(len(test))
    ]
    _write(os.path.join(split_dir, "test.csv"), test_rows)
    return split_dir


# --------------------------------------------------------------------------- #
# 数据探索 / 统计
# --------------------------------------------------------------------------- #
def inspect_images(image_paths, n_workers: int = 8):
    """检查损坏图片、非 RGB（灰度/RGBA/其他模式）图片，统计尺寸与长宽比。

    返回 stats 字典。
    """
    n_broken, n_nonrgb = 0, []
    sizes = []
    aspects = []
    modes = Counter()

    for p in image_paths:
        try:
            img = Image.open(p)
            modes[img.mode] += 1
            img.load()  # 触发真实解码，检测损坏
            if img.mode not in ("RGB", "L", "RGBA", "LA"):
                n_nonrgb.append(p)
            sizes.append(img.size)
            w, h = img.size
            aspects.append(w / h)
        except Exception:
            n_broken += 1

    return {
        "total": len(image_paths),
        "broken": n_broken,
        "non_rgb": n_nonrgb,
        "modes": dict(modes),
        "sizes": sizes,
        "aspects": aspects,
    }


# --------------------------------------------------------------------------- #
# Dataset / 增强
# --------------------------------------------------------------------------- #
class PetDataset(Dataset):
    """从 CSV 划分文件加载的数据集。"""

    def __init__(self, csv_path, transform=None):
        self.transform = transform
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image_path"]).convert("RGB")
        label = int(r["class_id"])  # 已经是 0-indexed（0-36）
        if self.transform:
            img = self.transform(img)
        return img, label


def resolve_data_config(model_name: str):
    """用 timm 读取与预训练权重一致的归一化均值/方差/输入尺寸。"""
    data_cfg = timm.data.resolve_data_config({}, model=model_name)
    return data_cfg


def build_transforms(train: bool, model_name: str, aa: str = "rand-m9-mstd0.5-inc1"):
    """构建训练或验证/测试增强。归一化参数从 timm 读取。"""
    data_cfg = resolve_data_config(model_name)
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    # 输入尺寸：ViT-B/16 与 ResNet-18 训练都用 224
    size = 224

    normalize = transforms.Normalize(mean=mean, std=std)

    if train:
        # 可选：无增强 / 基础增强 / 基础增强 + RandAugment（消融用 aa 参数控制）
        layers = [transforms.RandomResizedCrop(size, scale=(0.08, 1.0))]
        layers.append(transforms.RandomHorizontalFlip())
        if aa == "none":
            pass
        elif aa == "base":
            pass  # 基础增强 = RandomResizedCrop + 翻转
        else:
            from torchvision.transforms import RandAugment
            layers.append(RandAugment(num_ops=2, magnitude=9))
        layers += [transforms.ToTensor(), normalize]
        return transforms.Compose(layers)
    else:
        # 验证/测试：Resize(248) + CenterCrop(224)
        return transforms.Compose([
            transforms.Resize(248),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])


def build_loaders(csv_dir: str, model_name: str, batch_size: int,
                  aa: str = "rand-m9-mstd0.5-inc1", num_workers: int = 8,
                  pin_memory: bool = True):
    """构建 train/val/test 三个 DataLoader。"""
    train_ds = PetDataset(os.path.join(csv_dir, "train.csv"),
                          transform=build_transforms(True, model_name, aa))
    val_ds = PetDataset(os.path.join(csv_dir, "val.csv"),
                        transform=build_transforms(False, model_name))
    test_ds = PetDataset(os.path.join(csv_dir, "test.csv"),
                         transform=build_transforms(False, model_name))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def prepare_data(args):
    """阶段 1 入口：下载 + 划分 + 保存 splits。"""
    trainval, test = download_pets(args.root)
    train_idx, val_idx = stratified_split(trainval, args.val_ratio, args.seed)
    write_splits(trainval, test, train_idx, val_idx, args.split_dir)
    print(f"[prepare] train={len(train_idx)} val={len(val_idx)} test={len(test)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-dir", default=SPLITS_DIR)
    args = ap.parse_args()
    prepare_data(args)


if __name__ == "__main__":
    main()
