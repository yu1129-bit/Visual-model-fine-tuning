"""模型构建：V1/V2/V3/V4 四组模型 + LoRA 注入。

LoRA 使用自定义实现（不依赖 peft），针对 timm ViT 的注意力 qkv 线性层注入低秩旁路。
"""

import math

import timm
import torch
import torch.nn as nn

NUM_CLASSES = 37


# --------------------------------------------------------------------------- #
# V1：ResNet-18 随机初始化，从零训练
# --------------------------------------------------------------------------- #
def build_v1():
    model = timm.create_model("resnet18", pretrained=False, num_classes=NUM_CLASSES)
    return model


# --------------------------------------------------------------------------- #
# V2：ViT-B/16 线性探测（冻结主干，只训分类头）
# --------------------------------------------------------------------------- #
def build_v2():
    model = timm.create_model(
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True, num_classes=NUM_CLASSES,
    )
    # 冻结全部主干
    for p in model.parameters():
        p.requires_grad = False
    # 只让分类头可训练
    for p in model.head.parameters():
        p.requires_grad = True
    return model


# --------------------------------------------------------------------------- #
# V3：ViT-B/16 全量微调
# --------------------------------------------------------------------------- #
def build_v3():
    model = timm.create_model(
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True, num_classes=NUM_CLASSES,
    )
    return model


# --------------------------------------------------------------------------- #
# V4：ViT-B/16 + LoRA（自定义 LoRA 实现，注入注意力层的 qkv）
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """在某个 Linear 层上加低秩旁路：out = W x + (alpha/r) * B A x。

    冻结原权重 W，只训练 A、B。
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.linear = linear  # 原层（冻结）
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_features = linear.in_features
        out_features = linear.out_features
        # A: (in, r)，B: (r, out)
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 初始化：A 用 kaiming 均匀，B 置零（使初始旁路为 0）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # 冻结原权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        out = self.linear(x)
        delta = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return out + delta * self.scaling


def build_v4(r: int = 8, alpha: int = 16, dropout: float = 0.1):
    model = timm.create_model(
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True, num_classes=NUM_CLASSES,
    )
    # 注入 LoRA 到所有 qkv 层
    for name, module in model.named_modules():
        if name.endswith("qkv") and isinstance(module, nn.Linear):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
    # 分类头保持可训练（LoRA 层只冻结了 qkv，其余主干需要冻结）
    for n, p in model.named_parameters():
        if "head" not in n and "lora_A" not in n and "lora_B" not in n:
            p.requires_grad = False
    return model


def merge_lora_v4(model):
    """把 LoRA 旁路合并回原权重（merge_and_unload 的等价实现）。"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            merged = module.linear.weight.data + (module.lora_A @ module.lora_B).t() * module.scaling
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], module.linear)  # 换回原 Linear
            module.linear.weight.data = merged
    return model


# --------------------------------------------------------------------------- #
# 特征提取器（V2 线性探测缓存特征用）
# --------------------------------------------------------------------------- #
class FeatureExtractor(nn.Module):
    """提取 ViT 主干特征（CLS token 或 patch 平均池化）。"""

    def __init__(self, model_name: str, pool: str = "cls"):
        super().__init__()
        assert pool in ("cls", "avg")
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0,  # 无分类头
        )
        self.pool = pool

    def forward(self, x):
        if self.pool == "cls":
            return self.backbone(x)  # num_classes=0 时 timm 返回 pool 后的特征
        # patch 平均池化：forward_features 取 token，去掉 CLS 再平均
        x = self.backbone.forward_features(x)
        return x[:, 1:, :].mean(dim=1)


def build_feature_extractor(pool: str = "cls"):
    return FeatureExtractor("vit_base_patch16_224.augreg_in21k_ft_in1k", pool=pool)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(variant: str, **kwargs):
    if variant == "v1":
        return build_v1()
    if variant == "v2":
        return build_v2()
    if variant == "v3":
        return build_v3()
    if variant == "v4":
        return build_v4(**kwargs)
    raise ValueError(f"unknown variant {variant}")
