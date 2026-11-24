#!/usr/bin/env bash
# 一键跑通：四组模型 × 3 个种子
set -e

cd "$(dirname "$0")/.."

SEEDS="42 123 2024"

run() {
  local variant=$1
  local config=$2
  local extra=${3:-}
  for s in $SEEDS; do
    echo "======================================================"
    echo "==> $variant  seed=$s"
    echo "======================================================"
    # 从 yaml 读关键超参（简单解析）
    python src/train.py \
      --variant "$variant" \
      --seed "$s" \
      --config "configs/$config" \
      $extra || exit 1
  done
}

# V1 ResNet-18 从零
run v1 v1_scratch.yaml "--epochs 100 --batch-size 128 --lr 0.1 --weight-decay 5e-4 --warmup-epochs 5 --label-smoothing 0.0"

# V2 线性探测
run v2 v2_linear.yaml "--epochs 50 --batch-size 128 --lr 1e-3 --weight-decay 0.0 --label-smoothing 0.0"

# V3 全量微调
run v3 v3_full.yaml "--epochs 15 --batch-size 64 --lr 3e-5 --weight-decay 0.05 --warmup-epochs 1 --label-smoothing 0.1"

# V4 LoRA
run v4 v4_lora.yaml "--epochs 15 --batch-size 64 --lr 1e-3 --weight-decay 0.05 --warmup-epochs 1 --label-smoothing 0.1"

echo "==> 全部完成"
