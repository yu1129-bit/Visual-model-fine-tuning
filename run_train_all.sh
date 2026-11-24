#!/usr/bin/env bash
# 正式训练：四组模型 × 3 种子，全部后台执行
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export PET_DATA_ROOT=/root/vision-finetune/data

cd /root/vision-finetune/src

SPLIT=/root/vision-finetune/splits
OUT=/root/vision-finetune/outputs
LOG=/root/vision-finetune/outputs/train_all.log

echo "========== 训练开始 $(date) ==========" | tee -a $LOG

for seed in 42 123 2024; do
  echo "----- V1 ResNet18 seed=$seed -----" | tee -a $LOG
  python train.py --variant v1 --seed $seed --epochs 100 --batch-size 128 \
    --lr 0.1 --weight-decay 5e-4 --warmup-epochs 5 --label-smoothing 0.0 \
    --split-dir $SPLIT --output-dir $OUT --model-name resnet18 --num-workers 8 \
    2>&1 | tee -a $LOG
done

for seed in 42 123 2024; do
  echo "----- V2 linear seed=$seed -----" | tee -a $LOG
  python train.py --variant v2 --seed $seed --epochs 50 --batch-size 128 \
    --lr 1e-3 --weight-decay 0.0 --warmup-epochs 1 --label-smoothing 0.0 \
    --split-dir $SPLIT --output-dir $OUT --num-workers 8 \
    2>&1 | tee -a $LOG
done

for seed in 42 123 2024; do
  echo "----- V3 full seed=$seed -----" | tee -a $LOG
  python train.py --variant v3 --seed $seed --epochs 15 --batch-size 64 \
    --lr 3e-5 --weight-decay 0.05 --warmup-epochs 1 --label-smoothing 0.1 \
    --split-dir $SPLIT --output-dir $OUT --num-workers 8 \
    2>&1 | tee -a $LOG
done

for seed in 42 123 2024; do
  echo "----- V4 LoRA seed=$seed -----" | tee -a $LOG
  python train.py --variant v4 --seed $seed --epochs 15 --batch-size 64 \
    --lr 1e-3 --weight-decay 0.05 --warmup-epochs 1 --label-smoothing 0.1 \
    --split-dir $SPLIT --output-dir $OUT --lora-r 8 --lora-alpha 16 --num-workers 8 \
    2>&1 | tee -a $LOG
done

echo "========== 训练全部结束 $(date) ==========" | tee -a $LOG
