#!/usr/bin/env bash
# 阶段 1：下载数据 + 划分 + 保存 splits
set -e

cd "$(dirname "$0")/.."

echo "==> 下载 Oxford-IIIT Pet 并划分数据"
python src/data.py \
  --root ./data \
  --val-ratio 0.1 \
  --seed 42 \
  --split-dir ./splits

echo "==> 划分完成，文件如下："
ls -la splits/
