"""训练脚本：统一入口，支持 V1/V2/V3/V4，含 AMP、梯度检查点、峰值显存记录、早停。"""

import os
import json
import time
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from data import build_loaders
from models import build_model, count_trainable_params, count_total_params


# --------------------------------------------------------------------------- #
# 训练工具
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch))
    return res


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(state, path):
    torch.save(state, path)


# --------------------------------------------------------------------------- #
# 训练循环
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    use_amp=True, grad_accum=1, label_smoothing=0.0):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    imgs_per_sec = AverageMeter()

    optimizer.zero_grad()
    t0 = time.time()
    for i, (images, target) in enumerate(loader):
        images, target = images.to(device), target.to(device)

        with autocast(enabled=use_amp):
            output = model(images)
            if label_smoothing > 0:
                loss = F.cross_entropy(output, target, label_smoothing=label_smoothing)
            else:
                loss = criterion(output, target)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item() * grad_accum, bs)
        top1.update(acc1.item(), bs)
        imgs_per_sec.update(bs / (time.time() - t0))
        t0 = time.time()

    return losses.avg, top1.avg, imgs_per_sec.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        with autocast(enabled=use_amp):
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        bs = images.size(0)
        losses.update(loss.item(), bs)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)
    return losses.avg, top1.avg, top5.avg


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.variant, r=args.lora_r, alpha=args.lora_alpha)
    model = model.to(device)

    if args.grad_checkpointing:
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(True)

    n_trainable = count_trainable_params(model)
    n_total = count_total_params(model)
    print(f"[{args.variant}] trainable={n_trainable:,} total={n_total:,} "
          f"({100 * n_trainable / max(n_total, 1):.2f}%)")

    train_loader, val_loader, _ = build_loaders(
        args.split_dir, args.model_name, args.batch_size,
        aa=args.aa, num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()

    # 优化器与学习率分层
    if args.variant in ("v1",):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    else:
        # V2/V3/V4 用 AdamW；主干与分类头可用不同 lr
        head_params = []
        body_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "head" in name or "classifier" in name:
                head_params.append(p)
            else:
                body_params.append(p)
        param_groups = []
        if body_params:
            param_groups.append({"params": body_params, "lr": args.lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": args.lr * args.head_lr_ratio})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # 调度：warmup + cosine
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=args.amp)

    best_val_top1 = 0.0
    history = {"train_loss": [], "train_top1": [], "val_loss": [],
               "val_top1": [], "lr": [], "imgs_per_sec": []}
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        loss, t1, ips = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            use_amp=args.amp, grad_accum=args.grad_accum,
            label_smoothing=args.label_smoothing,
        )
        val_loss, val_t1, val_t5 = evaluate(model, val_loader, criterion, device, use_amp=args.amp)
        scheduler.step()

        history["train_loss"].append(loss)
        history["train_top1"].append(t1)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_t1)
        history["lr"].append(get_lr(optimizer))
        history["imgs_per_sec"].append(ips)

        print(f"[{args.variant}][seed={args.seed}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={loss:.4f} train_top1={t1:.2f} val_loss={val_loss:.4f} "
              f"val_top1={val_t1:.2f} lr={get_lr(optimizer):.2e} ips={ips:.1f}")

        if val_t1 > best_val_top1:
            best_val_top1 = val_t1
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_top1": best_val_top1,
                "seed": args.seed,
                "variant": args.variant,
            }, os.path.join(args.output_dir, f"best_{args.variant}_seed{args.seed}.pt"))

    # 峰值显存
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0.0

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "best_val_top1": best_val_top1,
        "peak_mem_gb": round(peak_mem, 2),
        "history": history,
    }
    with open(os.path.join(args.output_dir, f"metrics_{args.variant}_seed{args.seed}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.variant}][seed={args.seed}] DONE best_val_top1={best_val_top1:.2f} "
          f"peak_mem={peak_mem:.2f}GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--model-name", default="vit_base_patch16_224.augreg_in21k_ft_in1k")
    ap.add_argument("--split-dir", default="./splits")
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr-ratio", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--aa", default="rand-m9-mstd0.5-inc1")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    args = ap.parse_args()
    main(args)
