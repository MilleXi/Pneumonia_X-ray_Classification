import os
import time
import copy
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataset import get_data_loaders, set_seed
from model import create_model, create_ensemble, get_loss_fn

class Trainer:
    """
    模型训练器：负责模型训练、验证和测试
    """
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler=None, 
                 device=None, num_epochs=25, early_stopping_patience=10,
                 save_dir='./checkpoints'):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            dataloaders: 包含训练集和验证集的数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器 (可选)
            device: 训练设备 (CPU或GPU)
            num_epochs: 训练轮数
            early_stopping_patience: 早停的等待轮数
            save_dir: 模型保存目录
        """
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = save_dir
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 将模型移动到设备
        self.model = self.model.to(self.device)
        
    def train_model(self):
        """
        训练模型
        
        返回:
            model: 训练好的最佳模型
            history: 训练历史记录
        """
        since = time.time()
        
        # 初始化最佳验证准确率和对应的模型权重
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')
        best_epoch = 0
        
        # 训练历史
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': []
        }
        
        # 早停计数器
        no_improvement = 0
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                else:
                    self.model.eval()   # 设置模型为评估模式
                
                running_loss = 0.0
                all_preds = []
                all_labels = []
                all_probs = []
                
                # 遍历数据
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 梯度清零
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        
                        # 计算预测概率和类别
                        probs = F.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        
                        # 反向传播和优化（仅在训练阶段）
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.detach().cpu().numpy())
                
                # 计算平均损失和指标
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                all_probs = np.array(all_probs)
                
                # 计算各种评估指标
                epoch_acc = accuracy_score(all_labels, all_preds)
                epoch_precision = precision_score(all_labels, all_preds, average='weighted')
                epoch_recall = recall_score(all_labels, all_preds, average='weighted')
                epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
                
                # 计算AUC（对于二分类）
                if len(np.unique(all_labels)) == 2:
                    try:
                        epoch_auc = roc_auc_score(all_labels, all_probs[:, 1])
                    except:
                        epoch_auc = 0.5  # 处理异常情况
                else:
                    # 多分类情况下使用micro平均
                    try:
                        epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                    except:
                        epoch_auc = 0.5  # 处理异常情况
                
                # 记录历史
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)
                history[f'{phase}_precision'].append(epoch_precision)
                history[f'{phase}_recall'].append(epoch_recall)
                history[f'{phase}_f1'].append(epoch_f1)
                history[f'{phase}_auc'].append(epoch_auc)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f} '
                      f'Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')
                
                # 如果是验证阶段且当前模型更好，则保存模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    no_improvement = 0
                    
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                    print(f'Saved new best model to best_model.pth, validation accuracy: {best_acc:.4f}')
                
                # 更新学习率调度器
                if phase == 'val' and self.scheduler is not None:
                    self.scheduler.step(epoch_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f'Current learning rate: {current_lr:.8f}')
            
            # 检查是否有改进
            if phase == 'val' and epoch > 0:
                if history['val_acc'][-1] <= best_acc:
                    no_improvement += 1
                    print(f'Validation performance not improved for {no_improvement} epochs')
                    if no_improvement >= self.early_stopping_patience:
                        print(f'Early stopping! No improvement for {self.early_stopping_patience} epochs')
                        break
            
            # 每个epoch结束后保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth'))
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best validation accuracy: {best_acc:.4f}, achieved at epoch {best_epoch+1}')
        
        # 加载最佳模型权重
        self.model.load_state_dict(best_model_wts)
        return self.model, history
    
    def test_model(self, test_loader=None):
        """
        测试模型
        
        参数:
            test_loader: 测试数据加载器，如果为None则使用self.dataloaders中的'test'
            
        返回:
            results: 包含各种评估指标的字典
        """
        if test_loader is None:
            test_loader = self.dataloaders.get('test')
            if test_loader is None:
                raise ValueError("No test dataloader provided")
        
        # 设置为评估模式
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 转为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 计算AUC
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        
        # 返回结果
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        # 打印结果
        print(f'Test set results:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 score: {f1:.4f}')
        print(f'AUC: {auc:.4f}')
        
        return results


def main():
    """
    主函数
    """
    # 设置随机种子
    set_seed(42)
    
    # 配置参数
    data_dir = "./chest_xray"
    batch_size = 32
    num_epochs = 30
    model_name = 'resnet'  # 可选: 'resnet', 'efficientnet', 'vit', 'swin'
    num_classes = 2
    learning_rate = 1e-4
    early_stopping_patience = 7
    save_dir = './checkpoints'
    
    # 获取数据加载器
    dataloaders = get_data_loaders(data_dir, batch_size=batch_size)
    
    # 创建模型
    model = create_model(model_name, num_classes=num_classes, pretrained=True)
    
    # 设置训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 计算类别权重（如果需要处理不平衡问题）
    train_dataset = dataloaders['train'].dataset
    labels = np.array(train_dataset.labels)
    class_weights = torch.tensor([1.0 / (labels == i).sum() for i in range(num_classes)])
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    # 损失函数和优化器
    criterion = get_loss_fn(class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 创建训练器并训练
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir
    )
    
    # 训练模型
    model, history = trainer.train_model()
    
    # 测试模型
    results = trainer.test_model(dataloaders['test'])
    
    # 打印最终结果
    print(f"最终测试结果:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
    
    return model, history, results


if __name__ == "__main__":
    main()