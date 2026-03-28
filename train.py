"""
训练脚本
包含完整的训练循环、损失函数、验证和模型保存
"""
import os
# 修复OpenMP库冲突问题（必须在导入torch之前设置）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt

# 尝试导入TensorBoard（如果不可用则禁用）
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("警告: TensorBoard未安装，将禁用TensorBoard日志记录")
    print("安装命令: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from unet_model import UNet, UNetSmall, count_parameters
from dataset import create_data_loaders


# ========== 损失函数定义 ==========

class DiceLoss(nn.Module):
    """
    Dice Loss - 适合高度不平衡的分割任务

    原理：
    Dice系数衡量预测和真实标签的重叠程度
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice

    优点：
    1. 对类别不平衡不敏感（直接优化重叠区域）
    2. 对小目标友好（本任务中干扰源位置占比很小）
    3. 数值稳定
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        参数:
            pred: (B, 1, H, W) 预测概率（已经过sigmoid）
            target: (B, 1, H, W) 真实标签 [0, 1]

        返回:
            loss: 标量
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题

    原理：
    对容易分类的样本降低权重，让模型聚焦于难分类的样本
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    优点：
    1. 自动降低易分类样本的权重
    2. γ参数控制难易样本的权重差异
    3. 适合极度不平衡的数据
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        """
        参数:
            alpha: 平衡因子（对正样本的权重）
            gamma: 聚焦参数（越大，对难样本权重越高）
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        参数:
            pred: (B, 1, H, W) 原始logits（未经sigmoid）
            target: (B, 1, H, W) 真实标签 [0, 1]

        返回:
            loss: 标量
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        p_t = torch.exp(-bce_loss)  # 预测概率
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    组合损失：Dice Loss + Focal Loss

    原因：
    - Dice Loss优化重叠区域（全局优化）
    - Focal Loss处理类别不平衡（像素级优化）
    - 组合使用效果通常最佳
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred_logits, target):
        """
        参数:
            pred_logits: (B, 1, H, W) 原始logits
            target: (B, 1, H, W) 真实标签

        返回:
            loss: 标量
        """
        # Dice需要概率
        pred_prob = torch.sigmoid(pred_logits)
        dice = self.dice_loss(pred_prob, target)

        # Focal需要logits
        focal = self.focal_loss(pred_logits, target)

        return self.dice_weight * dice + self.focal_weight * focal


# ========== 训练器类 ==========

class Trainer:
    """训练器"""

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 device,
                 save_dir: str = './checkpoints',
                 log_dir: str = './runs'):
        """
        参数:
            model: U-Net模型
            train_loader: 训练集DataLoader
            val_loader: 验证集DataLoader
            criterion: 损失函数
            optimizer: 优化器
            device: 设备（cuda或cpu）
            save_dir: 模型保存目录
            log_dir: TensorBoard日志目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard（可选）
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0

        for batch_idx, (X, Y) in enumerate(self.train_loader):
            X = X.to(self.device)
            Y = Y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, Y)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] - Loss: {loss.item():.6f}")

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> float:
        """验证"""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for X, Y in self.val_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)

                outputs = self.model(X)
                loss = self.criterion(outputs, Y)
                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        # 保存最新模型
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  → 最佳模型已保存至 {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']

    def train(self, num_epochs: int, scheduler=None):
        """完整训练流程"""
        print("="*60)
        print("开始训练")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.val_loader)}")
        print(f"总轮数: {num_epochs}")
        print("="*60)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            print(f"\nEpoch [{epoch}/{num_epochs}]")
            print("-"*60)

            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # 学习率调度
            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start

            print("-"*60)
            print(f"Epoch [{epoch}/{num_epochs}] 完成 - 用时: {epoch_time:.2f}s")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  学习率: {current_lr:.6f}")

            # TensorBoard记录（如果可用）
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best)

        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"训练完成！总用时: {total_time/60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print("="*60)

        if self.writer is not None:
            self.writer.close()

    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存至: {save_path}")
        else:
            plt.show()

        plt.close()


# ========== 主训练函数 ==========

def main():
    """主训练函数"""
    # ========== 超参数配置 ==========
    CONFIG = {
        # 数据
        'data_dir': '../unet_dataset',
        'batch_size': 8,
        'train_ratio': 0.8,
        'use_augmentation': True,
        'num_workers': 0,  # Windows上建议设为0

        # 模型
        'model_type': 'unet',  # 'unet' 或 'unet_small'
        'in_channels': 3,
        'out_channels': 1,

        # 训练
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,

        # 损失函数
        'loss_type': 'combined',  # 'dice', 'focal', 'combined', 'bce_weighted'
        'dice_weight': 0.5,
        'focal_weight': 0.5,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,

        # 学习率调度
        'use_scheduler': True,
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,

        # 保存路径
        'save_dir': './checkpoints',
        'log_dir': './runs',

        # 其他
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("="*60)
    print("配置信息")
    print("="*60)
    for key, value in CONFIG.items():
        print(f"{key:20s}: {value}")
    print("="*60)

    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])

    # ========== 加载数据 ==========
    print("\n加载数据集...")
    train_loader, val_loader = create_data_loaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        train_ratio=CONFIG['train_ratio'],
        num_workers=CONFIG['num_workers'],
        use_augmentation=CONFIG['use_augmentation'],
        seed=CONFIG['seed']
    )

    # ========== 创建模型 ==========
    print("\n创建模型...")
    if CONFIG['model_type'] == 'unet':
        model = UNet(
            in_channels=CONFIG['in_channels'],
            out_channels=CONFIG['out_channels']
        )
    elif CONFIG['model_type'] == 'unet_small':
        model = UNetSmall(
            in_channels=CONFIG['in_channels'],
            out_channels=CONFIG['out_channels']
        )
    else:
        raise ValueError(f"未知模型类型: {CONFIG['model_type']}")

    print(f"模型参数数量: {count_parameters(model):,}")

    # ========== 创建损失函数 ==========
    print("\n创建损失函数...")
    if CONFIG['loss_type'] == 'dice':
        criterion = DiceLoss()
        print("使用损失: Dice Loss")
    elif CONFIG['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'])
        print("使用损失: Focal Loss")
    elif CONFIG['loss_type'] == 'combined':
        criterion = CombinedLoss(
            dice_weight=CONFIG['dice_weight'],
            focal_weight=CONFIG['focal_weight'],
            focal_alpha=CONFIG['focal_alpha'],
            focal_gamma=CONFIG['focal_gamma']
        )
        print("使用损失: Combined Loss (Dice + Focal)")
    elif CONFIG['loss_type'] == 'bce_weighted':
        # 带权重的BCE（正样本权重更高）
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
        print("使用损失: Weighted BCE Loss")
    else:
        raise ValueError(f"未知损失函数类型: {CONFIG['loss_type']}")

    # ========== 创建优化器 ==========
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # ========== 学习率调度器 ==========
    scheduler = None
    if CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=CONFIG['scheduler_factor'],
            patience=CONFIG['scheduler_patience'],
            verbose=True
        )
        print("使用学习率调度器: ReduceLROnPlateau")

    # ========== 创建训练器 ==========
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=CONFIG['device'],
        save_dir=CONFIG['save_dir'],
        log_dir=CONFIG['log_dir']
    )

    # ========== 开始训练 ==========
    trainer.train(num_epochs=CONFIG['num_epochs'], scheduler=scheduler)

    # ========== 绘制训练曲线 ==========
    trainer.plot_training_history(
        save_path=os.path.join(CONFIG['save_dir'], 'training_history.png')
    )


if __name__ == "__main__":
    main()
