"""
推理脚本
用于使用训练好的U-Net模型进行干扰源定位预测
"""
import os
# 修复OpenMP库冲突问题（必须在导入torch之前设置）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import sys
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from unet_model import UNet, UNetSmall
from data_generation import CoordinateMapper, FeatureCalculator, DatasetGenerator

# 导入仿真模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from accuracy6_Bottom3 import angular_distance
except ImportError:
    print("警告: 无法导入angular_distance，使用简化版本")
    def angular_distance(theta1, phi1, theta2, phi2):
        """简化的角度距离计算"""
        return np.sqrt((theta1 - theta2)**2 + (phi1 - phi2)**2)


class JammerLocalizationPredictor:
    """干扰源定位预测器"""

    def __init__(self,
                 model_path: str,
                 grid_size: Tuple[int, int] = (64, 64),
                 theta_range: Tuple[float, float] = (-60, 60),
                 phi_range: Tuple[float, float] = (-90, 90),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        参数:
            model_path: 训练好的模型路径
            grid_size: 网格大小
            theta_range: theta范围
            phi_range: phi范围
            device: 设备
        """
        self.device = device
        self.grid_size = grid_size
        self.mapper = CoordinateMapper(grid_size, theta_range, phi_range)

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"预测器初始化完成")
        print(f"  - 设备: {self.device}")
        print(f"  - 网格大小: {grid_size}")
        print(f"  - 模型路径: {model_path}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # 从state_dict推断模型架构（简单检查第一层的通道数）
        # 这里假设使用标准UNet，实际使用时可能需要配置
        model = UNet(in_channels=3, out_channels=1)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        print(f"成功加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
        return model

    def predict(self, X_tensor: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        预测干扰源位置

        参数:
            X_tensor: (3, H, W) 特征张量（已归一化）

        返回:
            predicted_theta: 预测的theta
            predicted_phi: 预测的phi
            prob_mask: (H, W) 概率掩码
        """
        # 转换为torch tensor并添加batch维度
        if isinstance(X_tensor, np.ndarray):
            X_tensor = torch.from_numpy(X_tensor).float()

        X_tensor = X_tensor.unsqueeze(0)  # (1, 3, H, W)
        X_tensor = X_tensor.to(self.device)

        # 前向传播
        with torch.no_grad():
            logits = self.model(X_tensor)  # (1, 1, H, W)
            prob_mask = torch.sigmoid(logits)  # 转换为概率

        # 转换为numpy
        prob_mask = prob_mask.squeeze().cpu().numpy()  # (H, W)

        # 找到最大概率的位置
        max_idx = np.argmax(prob_mask)
        pred_row, pred_col = np.unravel_index(max_idx, prob_mask.shape)

        # 转换回角度坐标
        predicted_theta, predicted_phi = self.mapper.grid_to_coords(pred_row, pred_col)

        return predicted_theta, predicted_phi, prob_mask

    def visualize_prediction(self,
                           X_tensor: np.ndarray,
                           prob_mask: np.ndarray,
                           pred_theta: float,
                           pred_phi: float,
                           true_theta: Optional[float] = None,
                           true_phi: Optional[float] = None,
                           save_path: Optional[str] = None):
        """
        可视化预测结果

        参数:
            X_tensor: (3, H, W) 输入特征
            prob_mask: (H, W) 预测概率掩码
            pred_theta, pred_phi: 预测位置
            true_theta, true_phi: 真实位置（可选）
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 第一行：输入特征
        titles = ['SINR Map', 'Elite Neighbor Score', 'Bottom Neighbor Score']
        for i in range(3):
            im = axes[0, i].imshow(X_tensor[i], cmap='viridis', aspect='auto')
            axes[0, i].set_title(titles[i], fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel('Phi')
            axes[0, i].set_ylabel('Theta')
            plt.colorbar(im, ax=axes[0, i])

        # 第二行：预测结果
        # 1. 概率掩码
        im = axes[1, 0].imshow(prob_mask, cmap='hot', aspect='auto')
        axes[1, 0].set_title('Predicted Heatmap', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Phi')
        axes[1, 0].set_ylabel('Theta')
        plt.colorbar(im, ax=axes[1, 0])

        # 标注预测点
        pred_row, pred_col = self.mapper.coords_to_grid(pred_theta, pred_phi)
        axes[1, 0].plot(pred_col, pred_row, 'g*', markersize=20, markeredgecolor='white',
                       markeredgewidth=2, label=f'Pred ({pred_theta:.1f}°, {pred_phi:.1f}°)')

        if true_theta is not None and true_phi is not None:
            true_row, true_col = self.mapper.coords_to_grid(true_theta, true_phi)
            axes[1, 0].plot(true_col, true_row, 'bo', markersize=15,
                          markerfacecolor='none', markeredgewidth=3,
                          label=f'True ({true_theta:.1f}°, {true_phi:.1f}°)')

        axes[1, 0].legend(loc='upper right')

        # 2. 叠加显示（在SINR图上）
        axes[1, 1].imshow(X_tensor[0], cmap='viridis', aspect='auto', alpha=0.7)
        im = axes[1, 1].imshow(prob_mask, cmap='hot', aspect='auto', alpha=0.5)
        axes[1, 1].set_title('SINR + Prediction Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Phi')
        axes[1, 1].set_ylabel('Theta')

        axes[1, 1].plot(pred_col, pred_row, 'g*', markersize=20, markeredgecolor='white',
                       markeredgewidth=2, label=f'Pred ({pred_theta:.1f}°, {pred_phi:.1f}°)')

        if true_theta is not None and true_phi is not None:
            axes[1, 1].plot(true_col, true_row, 'bo', markersize=15,
                          markerfacecolor='none', markeredgewidth=3,
                          label=f'True ({true_theta:.1f}°, {true_phi:.1f}°)')

        axes[1, 1].legend(loc='upper right')

        # 3. 误差信息
        axes[1, 2].axis('off')
        info_text = f"Prediction Results\n{'='*40}\n\n"
        info_text += f"Predicted Position:\n"
        info_text += f"  θ = {pred_theta:.2f}°\n"
        info_text += f"  φ = {pred_phi:.2f}°\n\n"

        if true_theta is not None and true_phi is not None:
            error = angular_distance(pred_theta, pred_phi, true_theta, true_phi)
            info_text += f"True Position:\n"
            info_text += f"  θ = {true_theta:.2f}°\n"
            info_text += f"  φ = {true_phi:.2f}°\n\n"
            info_text += f"Localization Error:\n"
            info_text += f"  {error:.2f}°\n\n"

            if error < 5.0:
                info_text += "Status: ✓ Excellent\n"
            elif error < 10.0:
                info_text += "Status: ✓ Good\n"
            elif error < 15.0:
                info_text += "Status: ⚠ Acceptable\n"
            else:
                info_text += "Status: ✗ Poor\n"

        info_text += f"\nMax Probability: {prob_mask.max():.4f}\n"
        info_text += f"Grid Position: ({pred_row}, {pred_col})"

        axes[1, 2].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存至: {save_path}")
        else:
            plt.show()

        plt.close()


def predict_from_simulation(predictor: JammerLocalizationPredictor,
                           adaptive_array,
                           cache,
                           base_ga_params: dict,
                           true_theta: float,
                           true_phi: float,
                           elite_percentile: float = 30.0,
                           bottom_percentile: float = 30.0,
                           save_dir: Optional[str] = None):
    """
    从仿真数据进行完整的预测流程

    参数:
        predictor: 预测器对象
        adaptive_array: 相控阵对象
        cache: 缓存对象
        base_ga_params: GA参数
        true_theta, true_phi: 真实干扰源位置
        elite_percentile, bottom_percentile: 百分比参数
        save_dir: 保存目录

    返回:
        predicted_theta, predicted_phi, error
    """
    print(f"\n{'='*60}")
    print(f"预测干扰源位置: ({true_theta:.1f}°, {true_phi:.1f}°)")
    print(f"{'='*60}")

    # 1. 生成特征张量
    generator = DatasetGenerator(
        grid_size=predictor.grid_size,
        theta_range=(-60, 60),
        phi_range=(-90, 90),
        gaussian_sigma=1.5
    )

    print("生成特征张量...")
    X_tensor, Y_tensor = generator.generate_single_sample(
        true_theta, true_phi,
        adaptive_array, cache, base_ga_params,
        elite_percentile, bottom_percentile
    )

    # 2. 预测
    print("执行预测...")
    pred_theta, pred_phi, prob_mask = predictor.predict(X_tensor)

    # 3. 计算误差
    error = angular_distance(pred_theta, pred_phi, true_theta, true_phi)

    print(f"\n预测结果:")
    print(f"  预测位置: ({pred_theta:.2f}°, {pred_phi:.2f}°)")
    print(f"  真实位置: ({true_theta:.2f}°, {true_phi:.2f}°)")
    print(f"  定位误差: {error:.2f}°")

    # 4. 可视化
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'prediction_theta{true_theta:.1f}_phi{true_phi:.1f}.png')
    else:
        save_path = None

    predictor.visualize_prediction(
        X_tensor, prob_mask,
        pred_theta, pred_phi,
        true_theta, true_phi,
        save_path=save_path
    )

    return pred_theta, pred_phi, error


# ========== 主函数 ==========

def main():
    """测试预测器"""
    import pandas as pd
    from accuracy6_Bottom3 import (
        AdaptiveNullingPhasedArray,
        PhasedArrayCache
    )

    # 配置
    MODEL_PATH = './checkpoints/best_model.pth'
    GRID_SIZE = (64, 64)
    THETA_RANGE = (-60, 60)
    PHI_RANGE = (-90, 90)

    # 初始化仿真环境
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    DUMMY_CSV_FILENAME = os.path.join(parent_dir, "Phi_all.csv")

    if not os.path.exists(DUMMY_CSV_FILENAME):
        pd.DataFrame([
            {'theta_deg': t, 'phi_deg': p, 'gain_db': 0}
            for t in range(-90, 91, 5)
            for p in range(-90, 91, 5)
        ]).to_csv(DUMMY_CSV_FILENAME, index=False)

    adaptive_array = AdaptiveNullingPhasedArray(DUMMY_CSV_FILENAME, 8, 8, 0.5, 14)
    cache = PhasedArrayCache()

    base_ga_params = {
        'sample_points_per_null': 5,
        'phase_resolution_deg': 5.6,
        'num_generations': 50,
        'population_size': 40,
        'verbose': False,
        'max_retries': 2,
        'min_fitness_threshold': 38.0,
        'max_reinforce_rounds': 2
    }

    # 创建预测器
    print("="*60)
    print("初始化预测器")
    print("="*60)
    predictor = JammerLocalizationPredictor(
        model_path=MODEL_PATH,
        grid_size=GRID_SIZE,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE
    )

    # 测试几个位置
    test_positions = [
        (30.0, 45.0),
        (-35.0, -60.0),
        (40.0, 0.0),
        (-45.0, 75.0),
        (50.0, -30.0)
    ]

    results = []
    for i, (theta, phi) in enumerate(test_positions, 1):
        print(f"\n\n{'#'*60}")
        print(f"测试案例 {i}/{len(test_positions)}")
        print(f"{'#'*60}")

        pred_theta, pred_phi, error = predict_from_simulation(
            predictor,
            adaptive_array,
            cache,
            base_ga_params,
            true_theta=theta,
            true_phi=phi,
            save_dir='./predictions'
        )

        results.append({
            'true_theta': theta,
            'true_phi': phi,
            'pred_theta': pred_theta,
            'pred_phi': pred_phi,
            'error': error
        })

    # 统计结果
    print("\n\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    errors = [r['error'] for r in results]
    avg_error = np.mean(errors)
    success_rate = sum(1 for e in errors if e < 15.0) / len(errors) * 100

    print(f"平均误差: {avg_error:.2f}°")
    print(f"成功率 (误差 < 15°): {success_rate:.1f}%")
    print("\n详细结果:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. True: ({r['true_theta']:6.1f}°, {r['true_phi']:6.1f}°) → "
              f"Pred: ({r['pred_theta']:6.1f}°, {r['pred_phi']:6.1f}°) | "
              f"Error: {r['error']:5.2f}°")
    print("="*60)


if __name__ == "__main__":
    main()
