import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
from datetime import datetime


def set_seed(seed=42):
    """
    设置所有随机种子以确保可重复性
    
    参数:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    
    参数:
        history (dict): 包含训练历史的字典
        save_path (str, optional): 保存路径
    """
    # 创建一个2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制损失
    axs[0, 0].plot(history['train_loss'], label='Train Loss')
    axs[0, 0].plot(history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Model Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # 绘制准确率
    axs[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axs[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axs[0, 1].set_title('Model Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    
    # 绘制精确度和召回率
    axs[1, 0].plot(history['train_precision'], label='Train Precision')
    axs[1, 0].plot(history['val_precision'], label='Validation Precision')
    axs[1, 0].plot(history['train_recall'], label='Train Recall')
    axs[1, 0].plot(history['val_recall'], label='Validation Recall')
    axs[1, 0].set_title('Precision and Recall')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].legend()
    
    # 绘制F1和AUC
    axs[1, 1].plot(history['train_f1'], label='Train F1')
    axs[1, 1].plot(history['val_f1'], label='Validation F1')
    axs[1, 1].plot(history['train_auc'], label='Train AUC')
    axs[1, 1].plot(history['val_auc'], label='Validation AUC')
    axs[1, 1].set_title('F1 Score and AUC')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_model_results(results, model_name, experiment_name, save_dir='./results'):
    """
    保存模型结果
    
    参数:
        results (dict): 包含评估结果的字典
        model_name (str): 模型名称
        experiment_name (str): 实验名称
        save_dir (str): 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建结果字典
    results_dict = {
        'model_name': model_name,
        'experiment_name': experiment_name,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': {}
    }
    
    # 添加指标
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            results_dict['metrics'][metric] = float(value)
    
    # 保存为JSON
    with open(os.path.join(save_dir, f'{experiment_name}_{model_name}_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"结果已保存到 {os.path.join(save_dir, f'{experiment_name}_{model_name}_results.json')}")


def compare_models(results_list, save_path=None):
    """
    比较多个模型的性能
    
    参数:
        results_list (list): 包含多个模型结果的列表，每个元素都是一个字典
        save_path (str, optional): 保存路径
    """
    # 提取模型名称和指标
    model_names = [result['model_name'] for result in results_list]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    # 创建DataFrame
    data = []
    for result in results_list:
        row = [result['model_name']]
        for metric in metrics:
            row.append(result['metrics'].get(metric, 0))
        data.append(row)
    
    df = pd.DataFrame(data, columns=['Model'] + metrics)
    
    # 绘制比较图
    plt.figure(figsize=(15, 10))
    
    # 转换为适合seaborn的格式
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Value')
    
    # 绘制柱状图
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 打印表格比较
    print("Model Performance Comparison:")
    print(df.to_string(index=False))
    
    return df


def check_data_distribution(dataloaders, save_path=None):
    """
    检查数据分布情况
    
    参数:
        dataloaders (dict): 包含训练、验证和测试数据加载器的字典
        save_path (str, optional): 保存路径
    """
    # 获取每个数据集的类别分布
    class_counts = {}
    
    for split, dataloader in dataloaders.items():
        # 获取数据集
        dataset = dataloader.dataset
        labels = np.array(dataset.labels)
        
        # 获取类别映射（反向映射，从数字到名称）
        class_mapping = {v: k for k, v in dataset.classes.items()}
        
        # 计算每个类别的样本数
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_counts[split] = {class_mapping[int(label)]: counts[i] for i, label in enumerate(unique_labels)}
    
    # 创建DataFrame
    data = []
    for split, counts in class_counts.items():
        for class_name, count in counts.items():
            data.append([split, class_name, count])
    
    df = pd.DataFrame(data, columns=['Split', 'Class', 'Count'])
    
    # 可视化 (using English labels to avoid font rendering issues)
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='Class', y='Count', hue='Split', data=df)
    plt.title('Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    
    # 在每个条形上显示具体数值
    for p in chart.patches:
        chart.annotate(format(p.get_height(), '.0f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 打印比例信息 (using English labels to avoid font rendering issues)
    print("Dataset Class Distribution:")
    for split, counts in class_counts.items():
        total = sum(counts.values())
        print(f"\n{split} set:")
        for class_name, count in counts.items():
            print(f"  {class_name}: {count} ({count/total*100:.2f}%)")
    
    return df


def visualize_batch(dataloader, num_images=8, save_path=None):
    """
    可视化一批数据
    
    参数:
        dataloader: 数据加载器
        num_images (int): 要可视化的图像数量
        save_path (str, optional): 保存路径
    """
    # 获取一批数据
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # 获取类别名称
    class_names = {v: k for k, v in dataloader.dataset.classes.items()}
    
    # 设置图像网格
    rows = 2
    cols = num_images // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    
    # 显示图像
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # 将图像转换回原始形式（取消标准化）
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = images[i].numpy().transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f"Class: {class_names[labels[i].item()]}")
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def check_model_complexity(model, input_size=(3, 224, 224)):
    """
    检查模型复杂度
    
    参数:
        model: PyTorch模型
        input_size (tuple): 输入大小 (channels, height, width)
    """
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"不可训练参数数量: {total_params - trainable_params:,}")
    
    # 创建一个示例输入
    batch_size = 1
    x = torch.randn(batch_size, *input_size)
    
    # 计算浮点运算次数 (FLOPs)
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,))
        print(f"每次前向传播的浮点运算次数 (FLOPs): {flops/1e9:.2f} G")
    except ImportError:
        print("无法计算FLOPs，请安装thop库: pip install thop")
    
    # 模型大小
    model_size_mb = total_params * 4 / (1024 * 1024)  # 假设每个参数为4字节
    print(f"模型大小: {model_size_mb:.2f} MB")


def create_experiment_dir(base_dir="./experiments"):
    """
    创建实验目录
    
    参数:
        base_dir (str): 基础目录
        
    返回:
        experiment_dir (str): 实验目录路径
    """
    # 创建基础目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 使用当前时间创建唯一的实验目录名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    # 创建实验目录
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    
    print(f"创建实验目录: {experiment_dir}")
    
    return experiment_dir


def save_config(config, save_path):
    """
    保存配置文件
    
    参数:
        config (dict): 配置字典
        save_path (str): 保存路径
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"配置已保存到 {save_path}")