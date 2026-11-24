#!/usr/bin/env bash
# 阶段6：测试集评估（4组，各用 seed=42 的 best checkpoint）
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export PET_DATA_ROOT=/root/vision-finetune/data
cd /root/vision-finetune/src

SPLIT=/root/vision-finetune/splits
OUT=/root/vision-finetune/outputs

for v in v1 v2 v3 v4; do
  echo "===== 评估 $v ====="
  python evaluate.py --variant $v \
    --ckpt $OUT/best_${v}_seed42.pt \
    --split-dir $SPLIT --output-dir $OUT \
    --batch-size 64 --lora-r 8 --lora-alpha 16
done
echo "===== 全部评估完成 ====="
