"""
PyTorch数据集和数据加载器
用于加载生成的训练样本
"""
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional

# 修复OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class JammerLocalizationDataset(Dataset):
    """干扰源定位数据集"""

    def __init__(self, data_dir: str, transform=None):
        """
        参数:
            data_dir: 数据集目录（包含 .pkl 文件）
            transform: 可选的数据增强变换
        """
        self.data_dir = data_dir
        self.transform = transform

        # 加载所有 .pkl 文件路径
        self.sample_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.pkl')
        ])

        if len(self.sample_files) == 0:
            raise ValueError(f"在 {data_dir} 中未找到任何 .pkl 样本文件")

        print(f"加载数据集: {len(self.sample_files)} 个样本")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        返回:
            X: (3, H, W) 特征张量
            Y: (1, H, W) 标签张量
        """
        # 加载样本
        with open(self.sample_files[idx], 'rb') as f:
            sample_data = pickle.load(f)

        X = sample_data['X']  # (3, H, W)
        Y = sample_data['Y']  # (1, H, W)

        # 转换为 torch.Tensor
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

        # 应用数据增强（如果有）
        if self.transform is not None:
            X, Y = self.transform(X, Y)

        return X, Y

    def get_sample_info(self, idx) -> dict:
        """获取样本的元数据"""
        with open(self.sample_files[idx], 'rb') as f:
            sample_data = pickle.load(f)
        return {
            'true_theta': sample_data['true_theta'],
            'true_phi': sample_data['true_phi']
        }


class DataAugmentation:
    """
    数据增强类（可选）

    可能的增强方式：
    1. 水平翻转（flip along phi axis）
    2. 添加高斯噪声
    3. 随机缩放
    """

    def __init__(self, h_flip_prob=0.5, noise_std=0.02):
        """
        参数:
            h_flip_prob: 水平翻转概率
            noise_std: 高斯噪声标准差
        """
        self.h_flip_prob = h_flip_prob
        self.noise_std = noise_std

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用数据增强

        参数:
            X: (3, H, W) 特征张量
            Y: (1, H, W) 标签张量

        返回:
            增强后的 X, Y
        """
        # 1. 水平翻转（沿phi轴，即宽度维度）
        if torch.rand(1).item() < self.h_flip_prob:
            X = torch.flip(X, dims=[2])  # 翻转宽度维度
            Y = torch.flip(Y, dims=[2])

        # 2. 添加高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(X) * self.noise_std
            X = X + noise
            X = torch.clamp(X, 0, 1)  # 确保在[0, 1]范围内

        return X, Y


def create_data_loaders(data_dir: str,
                       batch_size: int = 8,
                       train_ratio: float = 0.8,
                       num_workers: int = 0,
                       use_augmentation: bool = False,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和验证集的DataLoader

    参数:
        data_dir: 数据集目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        num_workers: 数据加载线程数
        use_augmentation: 是否使用数据增强
        seed: 随机种子

    返回:
        train_loader, val_loader
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建完整数据集
    transform = DataAugmentation() if use_augmentation else None
    full_dataset = JammerLocalizationDataset(data_dir, transform=transform)

    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"数据集划分: 训练集 {train_size} 个, 验证集 {val_size} 个")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def visualize_sample(X: torch.Tensor, Y: torch.Tensor, save_path: Optional[str] = None):
    """
    可视化单个样本

    参数:
        X: (3, H, W) 特征张量
        Y: (1, H, W) 标签张量
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 转换为numpy数组
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()

    # 显示三个特征通道
    titles = ['SINR Map', 'Elite Neighbor Score', 'Bottom Neighbor Score', 'Ground Truth']
    for i in range(3):
        im = axes[i].imshow(X[i], cmap='viridis', aspect='auto')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Phi')
        axes[i].set_ylabel('Theta')
        plt.colorbar(im, ax=axes[i])

    # 显示标签
    im = axes[3].imshow(Y[0], cmap='hot', aspect='auto')
    axes[3].set_title(titles[3])
    axes[3].set_xlabel('Phi')
    axes[3].set_ylabel('Theta')
    plt.colorbar(im, ax=axes[3])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    else:
        plt.show()

    plt.close()


# 测试代码
if __name__ == "__main__":
    import sys

    # 测试数据集加载
    print("="*60)
    print("测试数据集加载")
    print("="*60)

    # 假设数据集在这个路径（需要先运行 data_generation.py）
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'unet_dataset')

    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录 {data_dir} 不存在")
        print("请先运行 data_generation.py 生成数据集")
        sys.exit(1)

    # 创建数据加载器
    try:
        train_loader, val_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=4,
            train_ratio=0.8,
            use_augmentation=True,
            seed=42
        )

        print(f"\n训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")

        # 测试加载一个批次
        print("\n测试加载一个批次...")
        for X_batch, Y_batch in train_loader:
            print(f"X_batch shape: {X_batch.shape}")
            print(f"Y_batch shape: {Y_batch.shape}")
            print(f"X_batch 值范围: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
            print(f"Y_batch 值范围: [{Y_batch.min():.4f}, {Y_batch.max():.4f}]")

            # 可视化第一个样本
            vis_dir = os.path.join(parent_dir, 'unet_dataset_visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            visualize_sample(
                X_batch[0],
                Y_batch[0],
                save_path=os.path.join(vis_dir, 'sample_visualization.png')
            )
            break

        print("\n" + "="*60)
        print("数据集测试完成！")
        print("="*60)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
