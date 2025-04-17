import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torch.nn import CrossEntropyLoss


class ResNetModel(nn.Module):
    """
    基于ResNet50的肺炎检测模型
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        """
        初始化模型

        参数:
            num_classes (int): 类别数量
            pretrained (bool): 是否使用预训练权重
            freeze_backbone (bool): 是否冻结主干网络参数
        """
        super(ResNetModel, self).__init__()
        # 加载预训练的ResNet50
        self.model = models.resnet50(pretrained=pretrained)
        
        # 冻结参数（可选）
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 修改最后的全连接层以适应我们的分类任务
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),  # 添加dropout以减少过拟合
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class EfficientNetModel(nn.Module):
    """
    基于EfficientNet的肺炎检测模型
    """
    def __init__(self, num_classes=2, pretrained=True, model_name='efficientnet_b0'):
        """
        初始化模型

        参数:
            num_classes (int): 类别数量
            pretrained (bool): 是否使用预训练权重
            model_name (str): EfficientNet的具体型号
        """
        super(EfficientNetModel, self).__init__()
        # 使用timm库加载预训练的EfficientNet
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class VisionTransformerModel(nn.Module):
    """
    基于Vision Transformer (ViT)的肺炎检测模型
    """
    def __init__(self, num_classes=2, pretrained=True, model_name='vit_base_patch16_224'):
        """
        初始化模型

        参数:
            num_classes (int): 类别数量
            pretrained (bool): 是否使用预训练权重
            model_name (str): ViT的具体型号
        """
        super(VisionTransformerModel, self).__init__()
        # 使用timm库加载预训练的ViT
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class SwinTransformerModel(nn.Module):
    """
    基于Swin Transformer的肺炎检测模型
    """
    def __init__(self, num_classes=2, pretrained=True, model_name='swin_base_patch4_window7_224'):
        """
        初始化模型

        参数:
            num_classes (int): 类别数量
            pretrained (bool): 是否使用预训练权重
            model_name (str): Swin Transformer的具体型号
        """
        super(SwinTransformerModel, self).__init__()
        # 使用timm库加载预训练的Swin Transformer
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class ModelEnsemble(nn.Module):
    """
    模型集成类，用于组合多个模型的预测
    """
    def __init__(self, models, weights=None):
        """
        初始化模型集成

        参数:
            models (list): 模型列表
            weights (list, optional): 每个模型的权重，默认为等权重
        """
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        # 如果没有提供权重，则使用等权重
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            # 归一化权重
            weights_sum = sum(weights)
            self.weights = torch.tensor([w / weights_sum for w in weights])
    
    def forward(self, x):
        # 收集每个模型的输出
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 加权平均所有模型的输出
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += output * self.weights[i]
        
        return ensemble_output


def get_loss_fn(class_weights=None):
    """
    获取损失函数，可选择带有类别权重的交叉熵损失

    参数:
        class_weights (tensor, optional): 类别权重

    返回:
        loss_fn: 损失函数
    """
    if class_weights is not None:
        return CrossEntropyLoss(weight=class_weights)
    else:
        return CrossEntropyLoss()


def create_model(model_name, num_classes=2, pretrained=True):
    """
    创建指定类型的模型

    参数:
        model_name (str): 模型名称 ('resnet', 'efficientnet', 'vit', 'swin')
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重

    返回:
        model: 创建的模型
    """
    if model_name == 'resnet':
        return ResNetModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet':
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vit':
        return VisionTransformerModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'swin':
        return SwinTransformerModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")


def create_ensemble(model_names, num_classes=2, pretrained=True, weights=None):
    """
    创建模型集成

    参数:
        model_names (list): 模型名称列表
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重
        weights (list, optional): 每个模型的权重

    返回:
        ensemble: 模型集成
    """
    models = []
    for model_name in model_names:
        model = create_model(model_name, num_classes, pretrained)
        models.append(model)
    
    return ModelEnsemble(models, weights)


# 测试模型
if __name__ == "__main__":
    # 创建一个示例输入
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试各个模型
    for model_name in ['resnet', 'efficientnet', 'vit', 'swin']:
        model = create_model(model_name)
        print(f"测试 {model_name} 模型...")
        output = model(x)
        print(f"输出形状: {output.shape}")
    
    # 测试模型集成
    ensemble = create_ensemble(['resnet', 'efficientnet'])
    output = ensemble(x)
    print(f"集成模型输出形状: {output.shape}")