"""
配置文件：包含项目的所有配置参数
"""

# 数据处理配置
DATA_CONFIG = {
    'data_dir': './chest_xray',  # 数据集目录
    'img_size': 224,  # 图像大小
    'batch_size': 32,  # 批量大小
    'num_workers': 4,  # 数据加载的工作线程数
    'train_ratio': 0.7,  # 训练集比例
    'val_ratio': 0.15,  # 验证集比例
    'test_ratio': 0.15,  # 测试集比例
}

# 模型配置
MODEL_CONFIG = {
    # 单个模型配置
    'resnet': {
        'model_name': 'resnet50',
        'pretrained': True,
        'freeze_backbone': False
    },
    'efficientnet': {
        'model_name': 'efficientnet-b0',
        'pretrained': True
    },
    'vit': {
        'model_name': 'vit_base_patch16_224',
        'pretrained': True
    },
    'swin': {
        'model_name': 'swin_base_patch4_window7_224',
        'pretrained': True
    },
    
    # 集成模型配置
    'ensemble': {
        'models': ['resnet', 'efficientnet'],
        'weights': [0.6, 0.4]  # 模型权重，默认等权
    },
    
    # 通用模型配置
    'num_classes': 2,  # 类别数量
    'dropout_rate': 0.3,  # Dropout率
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 30,  # 训练轮数
    'learning_rate': 1e-4,  # 初始学习率
    'weight_decay': 1e-5,  # 权重衰减
    'early_stopping_patience': 7,  # 早停的等待轮数
    'scheduler_patience': 3,  # 学习率调度器的等待轮数
    'scheduler_factor': 0.5,  # 学习率衰减因子
    'use_class_weights': True,  # 是否使用类别权重
    'clip_grad_norm': 1.0,  # 梯度裁剪范数
}

# 评估配置
EVAL_CONFIG = {
    'save_predictions': True,  # 是否保存预测结果
    'visualize_samples': 10,  # 可视化样本数量
    'confusion_matrix': True,  # 是否绘制混淆矩阵
    'roc_curve': True,  # 是否绘制ROC曲线
    'pr_curve': True,  # 是否绘制PR曲线
    'grad_cam': True,  # 是否使用Grad-CAM
}

# 输出配置
OUTPUT_CONFIG = {
    'output_dir': './output',  # 输出目录
    'save_model': True,  # 是否保存模型
    'save_history': True,  # 是否保存训练历史
    'log_interval': 10,  # 日志记录间隔（批次数）
}

# 系统配置
SYSTEM_CONFIG = {
    'seed': 42,  # 随机种子
    'device': 'cuda',  # 设备类型（'cuda'或'cpu'）
    'precision': 'float32',  # 精度类型（'float32'或'float16'）
    'debug': False,  # 是否开启调试模式
}

# 类别名称
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']