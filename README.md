# 视觉模型微调全流程（图像分类）

在 Oxford-IIIT Pet（37 类猫狗品种）上跑通「数据划分 → 增强 → 预训练模型 → 四种微调策略 → 评估/可视化/推理」完整链路，对比预训练收益与不同微调方式在精度、显存、训练时间上的取舍。

## 四组模型

| 编号 | 模型 | 训练方式 | 目的 |
|------|------|---------|------|
| V1 | ResNet-18 | 随机初始化从零训练 | 无预训练下限 |
| V2 | ViT-B/16 | 线性探测（冻结主干只训头） | 预训练特征质量 |
| V3 | ViT-B/16 | 全量微调 | 精度上限 |
| V4 | ViT-B/16 + LoRA | 参数高效微调（~1% 参数） | 逼近 V3 |

## 环境

```bash
pip install -r requirements.txt
```

- Python 3.10+，PyTorch 2.x，timm，peft，scikit-learn
- 单张 24GB 显卡，全程 AMP 混合精度

## 一键复现

```bash
# 1. 数据准备与划分（阶段 1）
bash scripts/01_prepare_data.sh

# 2. 跑四组 × 3 种子（阶段 2-5）
bash scripts/02_run_all.sh

# 3. 评估（阶段 6）
python src/evaluate.py --variant v3 --ckpt outputs/best_v3_seed42.pt

# 4. 推理
python src/predict.py --variant v4 --ckpt outputs/best_v4_seed42.pt --input path/to/img.jpg
```

## 目录结构

```
vision-finetune/
├── README.md
├── requirements.txt
├── configs/            # v1~v4 的 yaml 配置
├── splits/             # train/val/test 划分
├── scripts/            # 01 数据准备 / 02 全量训练
├── src/                # data/models/train/evaluate/visualize/predict
├── outputs/            # 日志、指标、图表（不含权重）
└── report.md           # 实验报告
```

模型权重不入库。
