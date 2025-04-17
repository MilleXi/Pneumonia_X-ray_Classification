import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import get_data_loaders, ChestXRayDataset, get_transforms
from model import create_model, create_ensemble
from train import Trainer


class ModelEvaluator:
    """
    模型评估器：用于评估模型性能和生成可视化
    """
    def __init__(self, model, dataloader, device=None, class_names=None):
        """
        初始化评估器
        
        参数:
            model: 已训练的模型
            dataloader: 测试数据加载器
            device: 计算设备
            class_names: 类别名称
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 设置类别名称
        self.class_names = class_names if class_names is not None else ['NORMAL', 'PNEUMONIA']
    
    def predict(self):
        """
        生成模型预测
        
        返回:
            predictions, labels, probabilities: 预测类别、真实标签和预测概率
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def plot_confusion_matrix(self, save_path=None):
        """
        绘制混淆矩阵
        
        参数:
            save_path: 保存路径，如果为None则显示图像
        """
        # 获取预测和标签
        predictions, labels, _ = self.predict()
        
        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        # 创建DataFrame以便使用seaborn绘图
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """
        绘制ROC曲线
        
        参数:
            save_path: 保存路径，如果为None则显示图像
        """
        # 获取预测和标签
        _, labels, probabilities = self.predict()
        
        # 对于二分类问题
        if len(self.class_names) == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend(loc="lower right")
        
        # 对于多分类问题
        else:
            plt.figure(figsize=(10, 8))
            for i in range(len(self.class_names)):
                fpr, tpr, _ = roc_curve((labels == i).astype(int), probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curve')
            plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """
        绘制精确率-召回率曲线
        
        参数:
            save_path: 保存路径，如果为None则显示图像
        """
        # 获取预测和标签
        _, labels, probabilities = self.predict()
        
        # 对于二分类问题
        if len(self.class_names) == 2:
            precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
            pr_auc = auc(recall, precision)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
        
        # 对于多分类问题 - 每个类别单独绘制
        else:
            plt.figure(figsize=(10, 8))
            for i in range(len(self.class_names)):
                precision, recall, _ = precision_recall_curve((labels == i).astype(int), probabilities[:, i])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f'{self.class_names[i]} (AUC = {pr_auc:.2f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Multi-class Precision-Recall Curve')
            plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def print_classification_report(self):
        """
        打印分类报告
        """
        # 获取预测和标签
        predictions, labels, _ = self.predict()
        
        # 打印分类报告
        report = classification_report(labels, predictions, target_names=self.class_names)
        print("Classification Report:")
        print(report)
        
        return report
    
    def visualize_model_predictions(self, num_images=6, save_dir=None):
        """
        可视化模型预测
        
        参数:
            num_images: 要可视化的图像数量
            save_dir: 保存目录，如果为None则显示图像
        """
        # 创建保存目录
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        # 设置为评估模式
        self.model.eval()
        
        all_images = []
        all_labels = []
        all_preds = []
        
        # 获取一批图像和标签
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                all_images.append(inputs.cpu())
                all_labels.append(labels.cpu())
                
                # 预测
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu())
        
        # 将所有数据连接起来
        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        
        # 选择要可视化的图像索引
        indices = np.random.choice(len(all_images), min(num_images, len(all_images)), replace=False)
        
        # 设置图像网格
        fig, axs = plt.subplots(2, 3, figsize=(15, 10)) if num_images >= 6 else plt.subplots(1, num_images, figsize=(15, 5))
        axs = axs.flatten()
        
        for i, idx in enumerate(indices):
            img = all_images[idx]
            label = all_labels[idx].item()
            pred = all_preds[idx].item()
            
            # 将图像转回原始形式（取消标准化）
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img.numpy().transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # 显示图像
            axs[i].imshow(img)
            axs[i].set_title(f'True: {self.class_names[label]}\nPredict: {self.class_names[pred]}')
            axs[i].axis('off')
            
            # 设置颜色来表示正确或错误的预测
            if label == pred:
                axs[i].set_title(f'True: {self.class_names[label]}\nPredict: {self.class_names[pred]}', color='green')
            else:
                axs[i].set_title(f'True: {self.class_names[label]}\nPredict: {self.class_names[pred]}', color='red')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'model_predictions.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def test_model(self, test_loader=None):
        """
        测试模型
        
        参数:
            test_loader: 测试数据加载器，如果为None则使用self.dataloader
                
        返回:
            results: 包含各种评估指标的字典
        """
        if test_loader is None:
            test_loader = self.dataloader
        
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
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
        print(f'测试集结果:')
        print(f'准确率: {accuracy:.4f}')
        print(f'精确率: {precision:.4f}')
        print(f'召回率: {recall:.4f}')
        print(f'F1分数: {f1:.4f}')
        print(f'AUC: {auc:.4f}')
        
        return results
    
    def analyze_errors(self, save_path=None):
        """
        分析模型的错误预测
        
        参数:
            save_path: 保存路径，如果为None则显示图像
        """
        # 获取预测和标签
        predictions, labels, probabilities = self.predict()
        
        # 找出错误预测的样本
        error_indices = np.where(predictions != labels)[0]
        
        if len(error_indices) == 0:
            print("No samples with prediction errors!")
            return
        
        # 获取错误样本的数据
        error_types = []
        for idx in error_indices:
            true_label = self.class_names[labels[idx]]
            pred_label = self.class_names[predictions[idx]]
            error_types.append(f"{true_label} -> {pred_label}")
        
        # 计算每种错误类型的频率
        error_counts = pd.Series(error_types).value_counts()
        
        # 可视化错误类型分布
        plt.figure(figsize=(12, 6))
        error_counts.plot(kind='bar', color='coral')
        plt.title('Error Prediction Type Distribution')
        plt.xlabel('Error Type (True -> Predicted)')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # 输出错误率
        error_rate = len(error_indices) / len(labels) * 100
        print(f"Error rate: {error_rate:.2f}%")
        
        # 对于每种错误类型，计算平均预测概率
        error_probs = {}
        for error_type in error_counts.index:
            type_indices = [i for i, et in enumerate(error_types) if et == error_type]
            error_idx = [error_indices[i] for i in type_indices]
            avg_prob = np.mean([probabilities[idx, predictions[idx]] for idx in error_idx])
            error_probs[error_type] = avg_prob
        
        print("Average prediction probability for error types:")
        for error_type, avg_prob in error_probs.items():
            print(f"{error_type}: {avg_prob:.2f}")


def main():
    """
    主函数
    """
    # 配置参数
    data_dir = "./chest_xray"
    model_name = 'resnet'  # 可选: 'resnet', 'efficientnet', 'vit', 'swin'
    num_classes = 2
    class_names = ['NORMAL', 'PNEUMONIA']
    checkpoint_path = './checkpoints/best_model.pth'
    results_dir = './results'
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取测试数据加载器
    dataloaders = get_data_loaders(data_dir, batch_size=32)
    test_loader = dataloaders['test']
    
    # 创建模型
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    
    # 加载训练好的模型权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 创建评估器
    evaluator = ModelEvaluator(model, test_loader, device, class_names)
    
    # 打印分类报告
    evaluator.print_classification_report()
    
    # 绘制混淆矩阵
    evaluator.plot_confusion_matrix(save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    # 绘制ROC曲线
    evaluator.plot_roc_curve(save_path=os.path.join(results_dir, 'roc_curve.png'))
    
    # 绘制精确率-召回率曲线
    evaluator.plot_precision_recall_curve(save_path=os.path.join(results_dir, 'pr_curve.png'))

    # 可视化模型预测
    evaluator.visualize_model_predictions(num_images=6, save_dir=results_dir)

    # 分析错误
    evaluator.analyze_errors(save_path=os.path.join(results_dir, 'error_analysis.png'))
    
    print(f"评估结果已保存到 {results_dir}")


if __name__ == "__main__":
    main()