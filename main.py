import os
import argparse
import json
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_data_loaders, set_seed
from model import create_model, create_ensemble, get_loss_fn
from train import Trainer
from evaluate import ModelEvaluator
from utils import (
    plot_training_history, 
    save_model_results, 
    check_data_distribution, 
    visualize_batch,
    check_model_complexity,
    create_experiment_dir,
    save_config
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='胸部X线图像肺炎检测')
    
    # 基本配置
    parser.add_argument('--data_dir', type=str, default='./chest_xray', help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--experiment_name', type=str, default='pneumonia_detection', help='实验名称')
    
    # 数据加载配置
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='resnet', 
                        choices=['resnet', 'efficientnet', 'vit', 'swin', 'ensemble'],
                        help='模型名称')
    parser.add_argument('--pretrained', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')
    parser.add_argument('--use_ensemble', action='store_true', help='是否使用模型集成')
    parser.add_argument('--ensemble_models', type=str, default='resnet,efficientnet',
                        help='集成模型列表，用逗号分隔')
    
    # 训练配置
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--early_stopping', type=int, default=7, help='早停的等待轮数')
    parser.add_argument('--use_class_weights', action='store_true', help='是否使用类别权重')
    
    # 执行模式
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'train_evaluate', 'visualize'],
                        help='执行模式')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='用于评估的检查点路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    experiment_dir = create_experiment_dir(args.output_dir)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    
    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    save_config(vars(args), config_path)
    
    # 获取数据加载器
    dataloaders = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # 检查数据分布
    check_data_distribution(
        dataloaders, 
        save_path=os.path.join(results_dir, 'data_distribution.png')
    )
    
    # 可视化一批数据
    visualize_batch(
        dataloaders['train'], 
        num_images=8, 
        save_path=os.path.join(results_dir, 'batch_visualization.png')
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    if args.model_name == 'ensemble' or args.use_ensemble:
        # 使用模型集成
        ensemble_model_names = args.ensemble_models.split(',')
        model = create_ensemble(
            model_names=ensemble_model_names,
            num_classes=args.num_classes,
            pretrained=args.pretrained
        )
    else:
        # 使用单个模型
        model = create_model(
            model_name=args.model_name,
            num_classes=args.num_classes,
            pretrained=args.pretrained
        )
    
    # 检查模型复杂度
    check_model_complexity(model)
    
    # 训练模式
    if args.mode in ['train', 'train_evaluate']:
        # 计算类别权重（如果需要）
        if args.use_class_weights:
            train_dataset = dataloaders['train'].dataset
            labels = train_dataset.labels
            class_counts = [labels.count(i) for i in range(args.num_classes)]
            class_weights = torch.tensor([
                1.0 / (count / len(labels)) for count in class_counts
            ], device=device)
            
            # 打印类别权重
            print(f"类别权重: {class_weights}")
        else:
            class_weights = None
        
        # 创建损失函数
        criterion = get_loss_fn(class_weights)
        
        # 创建优化器
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # 创建学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping,
            save_dir=checkpoint_dir
        )
        
        # 训练模型
        model, history = trainer.train_model()
        
        # 绘制训练历史
        plot_training_history(
            history, 
            save_path=os.path.join(results_dir, 'training_history.png')
        )
        
        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))
        
        # 使用最佳模型进行评估
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
    
    # 评估模式
    if args.mode in ['evaluate', 'train_evaluate']:
        # 如果指定了检查点，则加载检查点
        if args.checkpoint and args.mode == 'evaluate':
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        # 创建评估器
        class_names = list(dataloaders['train'].dataset.classes.keys())
        evaluator = ModelEvaluator(
            model=model,
            dataloader=dataloaders['test'],
            device=device,
            class_names=class_names
        )
        
        # 打印分类报告
        evaluator.print_classification_report()
        
        # 绘制混淆矩阵
        evaluator.plot_confusion_matrix(
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )
        
        # 绘制ROC曲线
        evaluator.plot_roc_curve(
            save_path=os.path.join(results_dir, 'roc_curve.png')
        )
        
        # 绘制精确率-召回率曲线
        evaluator.plot_precision_recall_curve(
            save_path=os.path.join(results_dir, 'pr_curve.png')
        )
        
        # 可视化模型预测
        evaluator.visualize_model_predictions(
            num_images=6, 
            save_dir=results_dir
        )
        
        
        # 分析错误
        evaluator.analyze_errors(
            save_path=os.path.join(results_dir, 'error_analysis.png')
        )
        
        # 运行测试并获取结果
        test_results = evaluator.test_model(dataloaders['test'])
        
        # 保存结果
        save_model_results(
            results=test_results,
            model_name=args.model_name,
            experiment_name=args.experiment_name,
            save_dir=results_dir
        )
    
    # 可视化模式
    if args.mode == 'visualize':
        # 创建评估器
        class_names = list(dataloaders['train'].dataset.classes.keys())
        evaluator = ModelEvaluator(
            model=model,
            dataloader=dataloaders['test'],
            device=device,
            class_names=class_names
        )
        
        # 可视化模型预测
        evaluator.visualize_model_predictions(
            num_images=10, 
            save_dir=results_dir
        )
    
    print(f"实验完成，结果保存在 {experiment_dir}")


if __name__ == "__main__":
    main()