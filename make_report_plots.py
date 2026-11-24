"""生成最终报告所需的所有图表：训练曲线、混淆矩阵、结果对比。"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"


def plot_training_curves():
    """四组模型的训练/验证曲线（用 seed=42 的 metrics）。"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for col, v in enumerate(["v1", "v2", "v3", "v4"]):
        m = json.load(open(f"{OUT}/metrics_{v}_seed42.json"))
        h = m["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0, col].plot(epochs, h["train_loss"], label="train", color="#e74c3c")
        axes[0, col].plot(epochs, h["val_loss"], label="val", color="#3498db")
        axes[0, col].set_title(f"{v.upper()} Loss")
        axes[0, col].set_xlabel("epoch")
        axes[0, col].legend(fontsize=7)
        axes[1, col].plot(epochs, h["train_top1"], label="train", color="#e74c3c")
        axes[1, col].plot(epochs, h["val_top1"], label="val", color="#3498db")
        axes[1, col].set_title(f"{v.upper()} Top-1 Acc")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].legend(fontsize=7)
    fig.suptitle("Training Curves (seed=42)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/training_curves.png", dpi=130)
    plt.close(fig)
    print("training_curves.png 已生成")


def plot_confusion_matrices():
    """四组模型的混淆矩阵（归一化）。"""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for col, v in enumerate(["v1", "v2", "v3", "v4"]):
        d = json.load(open(f"{OUT}/test_{v}_seed42.json"))
        cm = np.array(d["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = axes[col].imshow(cm_norm, cmap="Blues", aspect="auto")
        axes[col].set_title(f"{v.upper()} Confusion Matrix\n(test top1={d['top1']*100:.1f}%)")
        axes[col].set_xlabel("predicted")
        axes[col].set_ylabel("true")
        fig.colorbar(im, ax=axes[col], fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrices.png", dpi=130)
    plt.close(fig)
    print("confusion_matrices.png 已生成")


def plot_comparison():
    """结果对比柱状图。"""
    models = ["V1\nfrom scratch", "V2\nlinear", "V3\nfull", "V4\nLoRA"]
    top1_means = [36.90, 89.78, 92.69, 92.45]
    top1_stds = [3.62, 0.28, 0.38, 0.45]
    params = [11195493, 28453, 85827109, 323365]
    mems = [1.65, 0.99, 5.31, 3.62]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(models))
    axes[0].bar(x, top1_means, yerr=top1_stds, capsize=5, color=["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Test Top-1 (%)")
    axes[0].set_title("Test Top-1 Accuracy")
    axes[0].set_ylim(0, 100)

    axes[1].bar(x, np.log10(params), color=["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Trainable Params (log10)")
    axes[1].set_title("Trainable Parameters (log scale)")

    axes[2].bar(x, mems, color=["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"])
    axes[2].set_xticks(x); axes[2].set_xticklabels(models)
    axes[2].set_ylabel("Peak GPU Mem (GB)")
    axes[2].set_title("Peak GPU Memory")
    fig.tight_layout()
    fig.savefig(f"{OUT}/comparison.png", dpi=130)
    plt.close(fig)
    print("comparison.png 已生成")


if __name__ == "__main__":
    plot_training_curves()
    plot_confusion_matrices()
    plot_comparison()
    print("全部图表生成完成")
