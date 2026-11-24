#!/usr/bin/env bash
# 对每个模型的 3 个种子分别做测试集评估，得到均值±std
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export PET_DATA_ROOT=/root/vision-finetune/data
cd /root/vision-finetune/src

SPLIT=/root/vision-finetune/splits
OUT=/root/vision-finetune/outputs

for v in v1 v2 v3 v4; do
  for s in 42 123 2024; do
    echo "===== 评估 $v seed=$s ====="
    python evaluate.py --variant $v \
      --ckpt $OUT/best_${v}_seed${s}.pt \
      --split-dir $SPLIT --output-dir $OUT \
      --batch-size 64 --lora-r 8 --lora-alpha 16 --seed $s 2>&1 | grep 'top1\|saved'
  done
done
echo "===== 全部种子评估完成 ====="
