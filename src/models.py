"""模型构建：V1/V2/V3/V4 四组模型 + LoRA 注入。"""

import timm
import torch.nn as nn
from peft import LoraConfig, get_peft_model

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
# V4：ViT-B/16 + LoRA
# --------------------------------------------------------------------------- #
def build_v4(r: int = 8, alpha: int = 16, dropout: float = 0.1):
    model = timm.create_model(
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True, num_classes=NUM_CLASSES,
    )
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv"],  # timm ViT 注意力层的 qkv 合并在一个 Linear 里
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
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
