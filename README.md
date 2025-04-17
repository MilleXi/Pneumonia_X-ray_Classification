# Pneumonia_X-ray_Classification
胸部X线图像肺炎分类系统

## 项目概述

本项目实现了一个基于深度学习的胸部X线图像肺炎分类系统，用于自动识别儿科患者的胸部X线图像中是否存在肺炎（正常/肺炎）。系统采用了多种先进的深度学习模型，包括ResNet50、EfficientNet、Vision Transformer (ViT)和Swin Transformer，并支持模型集成。

## 数据集说明

本项目使用的数据集来自广州市妇女儿童医学中心一至五岁儿科患者的胸部X线图像。

来源: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

数据集结构如下：

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

- `NORMAL`：正常胸部X光片，显示清晰的肺部，没有异常混浊区域
- `PNEUMONIA`：肺炎胸部X光片，包括细菌性肺炎和病毒性肺炎

## 技术路线

本项目采用的技术路线包括：

1. **数据处理与预处理**
   - 数据清洗：确保所有图像质量合格
   - 数据增强：旋转、翻转、随机裁剪、缩放、颜色增强、噪声加法等
   - 数据划分：训练集、验证集、测试集

2. **模型选择与预训练**
   - 支持多种预训练模型：ResNet50、EfficientNet、ViT、Swin Transformer
   - 迁移学习：使用在ImageNet上预训练的模型并进行微调

3. **模型训练策略**
   - 损失函数：交叉熵损失函数（支持类别权重）
   - 优化器：AdamW
   - 学习率调度：ReduceLROnPlateau
   - 正则化技术：Dropout、早停等

4. **多模型集成**
   - 支持软投票法（Soft Voting）
   - 支持加权平均结合各模型的预测结果

5. **后处理与可解释性**
   - 错误分析与模型改进

6. **模型评估**
   - 准确率（Accuracy）、精确度（Precision）、召回率（Recall）、F1-score
   - AUC-ROC曲线

## 项目结构

```
├── dataset.py      # 数据加载和预处理
├── model.py        # 模型定义
├── train.py        # 模型训练
├── evaluate.py     # 模型评估
├── utils.py        # 工具函数
├── config.py       # 配置参数
├── main.py         # 主程序入口
└── README.md       # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- timm
- scikit-learn
- pandas
- matplotlib
- seaborn
- opencv-python

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python main.py --mode train --model_name resnet --data_dir ./chest_xray --num_epochs 30
```

### 评估模型

```bash
python main.py --mode evaluate --model_name resnet --checkpoint ./checkpoints/best_model.pth
```

### 训练并评估模型

```bash
python main.py --mode train_evaluate --model_name resnet
```

### 使用模型集成

```bash
python main.py --mode train_evaluate --model_name ensemble --ensemble_models resnet,efficientnet
```

### 可视化模型预测

```bash
python main.py --mode visualize --checkpoint ./checkpoints/best_model.pth
```

## 命令行参数

- `--data_dir`：数据集目录
- `--output_dir`：输出目录
- `--model_name`：模型名称（'resnet', 'efficientnet', 'vit', 'swin', 'ensemble'）
- `--pretrained`：是否使用预训练权重
- `--num_epochs`：训练轮数
- `--batch_size`：批量大小
- `--learning_rate`：学习率
- `--mode`：执行模式（'train', 'evaluate', 'train_evaluate', 'visualize'）
- `--checkpoint`：用于评估的检查点路径
- `--use_ensemble`：是否使用模型集成
- `--ensemble_models`：集成模型列表，用逗号分隔

## 实验结果

在测试集上，各模型的性能如下：

| 模型 | 准确率 | 精确度 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|--------|-----|
| ResNet50 | 89 | 89 | 89 | 89 | 95 |
| EfficientNet | 76 | 79 | 76 | 77 | 85 |
| ViT | 73 | 77 | 73 | 73 | 83 |
| Swin Transformer |  81  | 82 | 81 | 81 | 91 |
| 模型集成 | 87 | 88 | 87 | 87 | 95 |

注：此处模型集成的是 ResNet50 与 EfficientNet

## 参考文献

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
2. Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning (pp. 6105-6114).
3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
4. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 10012-10022).
