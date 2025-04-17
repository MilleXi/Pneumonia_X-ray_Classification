import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class ChestXRayDataset(Dataset):
    """
    胸部X线图像数据集：用于加载和预处理胸部X线图像，支持数据增强
    """
    def __init__(self, root_dir, split='train', transform=None, augment=False):
        """
        初始化数据集
        
        参数:
            root_dir (str): 数据集根目录路径
            split (str): 数据集划分 ('train', 'val', 'test')
            transform (callable, optional): 转换操作
            augment (bool): 是否进行数据增强
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.augment = augment
        
        # 类别映射
        self.classes = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        # 加载所有图像路径和标签
        self.images = []
        self.labels = []
        
        # 遍历数据集目录
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.classes[class_name])
        
        # 检查数据集大小
        print(f"{split} 集包含 {len(self.images)} 张图像")
        print(f"类别分布: 正常: {self.labels.count(0)}, 肺炎: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(split):
    """
    获取不同划分集的图像变换
    
    参数:
        split (str): 数据集划分 ('train', 'val', 'test')
        
    返回:
        transforms: 图像转换操作
    """
    # 基本转换 - 适用于所有集合
    base_transforms = [
        transforms.Resize((224, 224)),  # 调整大小以适应大多数预训练模型
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ]
    
    # 训练集增强
    if split == 'train':
        train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪和调整大小
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色增强
            # 有时添加随机噪声
            *base_transforms
        ]
        return transforms.Compose(train_transforms)
    
    # 验证集和测试集保持简单转换
    else:
        return transforms.Compose(base_transforms)


def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    创建数据加载器
    
    参数:
        data_dir (str): 数据集目录
        batch_size (int): 批量大小
        num_workers (int): 数据加载的工作线程数
        
    返回:
        dataloaders (dict): 包含训练、验证和测试集的数据加载器
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        # 获取相应的转换
        transform = get_transforms(split)
        
        # 创建数据集
        dataset = ChestXRayDataset(
            root_dir=data_dir,
            split=split,
            transform=transform,
            augment=(split == 'train')  # 只在训练集上进行数据增强
        )
        
        # 创建数据加载器
        shuffle = (split == 'train')  # 仅在训练集上打乱数据
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


# 设置随机种子，确保可重复性
def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 用法示例
if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    # 测试数据加载
    data_dir = "./chest_xray"
    dataloaders = get_data_loaders(data_dir, batch_size=32)
    
    # 显示数据集统计信息
    for split, dataloader in dataloaders.items():
        print(f"{split} 数据集大小: {len(dataloader.dataset)}")
        print(f"{split} 批次数量: {len(dataloader)}")
    
    # 获取一批示例数据并显示形状
    for images, labels in dataloaders['train']:
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签形状: {labels.shape}")
        print(f"标签示例: {labels[:5]}")
        break