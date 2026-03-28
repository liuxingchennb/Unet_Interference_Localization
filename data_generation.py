"""
数据集生成模块
功能：
1. 调用现有仿真代码生成搜索结果
2. 为所有搜索点计算三个特征（SINR, elite_neighbor_score, bottom_neighbor_score）
3. 将稀疏搜索结果"光栅化"到固定大小网格
4. 生成标签（真实干扰源位置的2D高斯热点）
"""
import numpy as np
import os
import sys
from typing import Tuple, List, Optional
import pickle

# 添加父目录到路径以导入仿真模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有仿真代码的必要组件
try:
    from accuracy6_Bottom3 import (
        AdaptiveNullingPhasedArray,
        AdaptiveNullingSimulationEnvironment,
        PhasedArrayCache,
        grid_search_for_jammer,
        SearchResult,
        angular_distance
    )
except ImportError:
    print("警告: 无法导入仿真模块，请确保 accuracy6_Bottom3.py 在正确路径")


class CoordinateMapper:
    """坐标映射器：在角度坐标和网格索引之间转换"""

    def __init__(self,
                 grid_size: Tuple[int, int] = (64, 64),
                 theta_range: Tuple[float, float] = (-60, 60),
                 phi_range: Tuple[float, float] = (-90, 90)):
        """
        参数:
            grid_size: (height, width) 网格大小
            theta_range: theta角度范围 (min, max)
            phi_range: phi角度范围 (min, max)
        """
        self.grid_h, self.grid_w = grid_size
        self.theta_min, self.theta_max = theta_range
        self.phi_min, self.phi_max = phi_range

        # 计算每个像素对应的角度范围
        self.theta_step = (self.theta_max - self.theta_min) / self.grid_h
        self.phi_step = (self.phi_max - self.phi_min) / self.grid_w

    def coords_to_grid(self, theta: float, phi: float) -> Tuple[int, int]:
        """
        将角度坐标映射到网格索引

        参数:
            theta: theta角度
            phi: phi角度

        返回:
            (row, col): 网格索引
        """
        # theta -> row (注意Y轴反转：较小的theta对应较大的row索引)
        row = int((self.theta_max - theta) / self.theta_step)
        row = np.clip(row, 0, self.grid_h - 1)

        # phi -> col
        col = int((phi - self.phi_min) / self.phi_step)
        col = np.clip(col, 0, self.grid_w - 1)

        return row, col

    def grid_to_coords(self, row: int, col: int) -> Tuple[float, float]:
        """
        将网格索引映射回角度坐标（像素中心）

        参数:
            row: 行索引
            col: 列索引

        返回:
            (theta, phi): 角度坐标
        """
        # row -> theta (像素中心)
        theta = self.theta_max - (row + 0.5) * self.theta_step

        # col -> phi (像素中心)
        phi = self.phi_min + (col + 0.5) * self.phi_step

        return theta, phi


class FeatureCalculator:
    """特征计算器：计算精英邻居和底部邻居分数"""

    @staticmethod
    def calculate_neighbor_features(search_results: List[SearchResult],
                                   elite_percentile: float = 30.0,
                                   bottom_percentile: float = 30.0) -> dict:
        """
        为所有搜索点计算邻居特征

        参数:
            search_results: 搜索结果列表
            elite_percentile: 精英点百分比阈值
            bottom_percentile: 底部点百分比阈值

        返回:
            字典 {(theta, phi): {'sinr_db': float,
                                'elite_neighbor_score': int,
                                'bottom_neighbor_score': int}}
        """
        if not search_results:
            return {}

        # 1. 排序并筛选精英点和底部点
        sorted_results = sorted(search_results, key=lambda r: r.sinr_db, reverse=True)
        num_elite = max(1, int(len(sorted_results) * (elite_percentile / 100.0)))
        num_bottom = max(1, int(len(sorted_results) * (bottom_percentile / 100.0)))

        elite_results = sorted_results[:num_elite]
        bottom_results = sorted_results[-num_bottom:]

        # 2. 创建字典
        elite_dict = {(res.null_theta, res.null_phi): (res.sinr_db, res.null_width)
                     for res in elite_results}
        bottom_set = {(res.null_theta, res.null_phi): res.null_width
                     for res in bottom_results}

        # 3. 为所有点计算邻居特征（不仅是精英点）
        all_features = {}

        for res in search_results:
            current_theta = res.null_theta
            current_phi = res.null_phi
            current_width = res.null_width

            # 统计精英邻居和底部邻居
            elite_neighbor_count = 0
            bottom_neighbor_count = 0

            # 检查精英邻居
            for (elite_theta, elite_phi), (_, elite_width) in elite_dict.items():
                if (elite_theta, elite_phi) == (current_theta, current_phi):
                    continue

                if FeatureCalculator._is_neighbor(
                    current_theta, current_phi, current_width,
                    elite_theta, elite_phi, elite_width
                ):
                    elite_neighbor_count += 1

            # 检查底部邻居
            for (bottom_theta, bottom_phi), bottom_width in bottom_set.items():
                if FeatureCalculator._is_neighbor(
                    current_theta, current_phi, current_width,
                    bottom_theta, bottom_phi, bottom_width
                ):
                    bottom_neighbor_count += 1

            all_features[(current_theta, current_phi)] = {
                'sinr_db': res.sinr_db,
                'elite_neighbor_score': elite_neighbor_count,
                'bottom_neighbor_score': bottom_neighbor_count
            }

        return all_features

    @staticmethod
    def _is_neighbor(theta1: float, phi1: float, width1: float,
                    theta2: float, phi2: float, width2: float) -> bool:
        """
        判断两个网格点是否为邻居（简化版，可根据原代码的复杂逻辑调整）

        参数:
            theta1, phi1, width1: 第一个点的坐标和宽度
            theta2, phi2, width2: 第二个点的坐标和宽度

        返回:
            True if 是邻居, False otherwise
        """
        # 常规区域判断（简化版）
        avg_width = (width1 + width2) / 2
        theta_diff = abs(theta1 - theta2)
        phi_diff = abs(phi1 - phi2)

        # 判断是否在1.5个格子宽度内
        theta_is_adjacent = (theta_diff < avg_width * 1.5) and (theta_diff > avg_width * 0.5 or theta_diff < 0.1)
        phi_is_adjacent = (phi_diff < avg_width * 1.5) and (phi_diff > avg_width * 0.5 or phi_diff < 0.1)

        # 8邻居判断
        if theta_is_adjacent and phi_is_adjacent:
            return True
        elif theta_diff < 0.1 and phi_is_adjacent:
            return True
        elif phi_diff < 0.1 and theta_is_adjacent:
            return True

        return False


class DatasetGenerator:
    """数据集生成器：完整的数据生成流程"""

    def __init__(self,
                 grid_size: Tuple[int, int] = (64, 64),
                 theta_range: Tuple[float, float] = (-60, 60),
                 phi_range: Tuple[float, float] = (-90, 90),
                 gaussian_sigma: float = 1.5):
        """
        参数:
            grid_size: 输出网格大小
            theta_range: theta搜索范围
            phi_range: phi搜索范围
            gaussian_sigma: 标签高斯分布的标准差（像素单位）
        """
        self.grid_size = grid_size
        self.mapper = CoordinateMapper(grid_size, theta_range, phi_range)
        self.gaussian_sigma = gaussian_sigma

    def generate_single_sample(self,
                             true_theta: float,
                             true_phi: float,
                             adaptive_array,
                             cache,
                             base_ga_params: dict,
                             elite_percentile: float = 30.0,
                             bottom_percentile: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成单个训练样本

        参数:
            true_theta: 真实干扰源theta
            true_phi: 真实干扰源phi
            adaptive_array: 相控阵对象
            cache: 缓存对象
            base_ga_params: GA参数字典
            elite_percentile: 精英点百分比
            bottom_percentile: 底部点百分比

        返回:
            X_tensor: (3, H, W) 特征张量
            Y_tensor: (1, H, W) 标签张量
        """
        print(f"\n生成样本: 干扰源位置 ({true_theta:.1f}°, {true_phi:.1f}°)")

        # 1. 运行仿真获取搜索结果
        main_beam_theta, main_beam_phi = 0.0, 0.0

        env = AdaptiveNullingSimulationEnvironment(adaptive_array, 1e-12, 1e-15)
        env.add_jammer_hidden(true_theta, true_phi, 5e-12)

        search_range_theta = (-60, 60)
        search_range_phi = (-90, 90)

        search_results = grid_search_for_jammer(
            adaptive_array, main_beam_theta, main_beam_phi, env,
            search_range_theta, search_range_phi, cache, **base_ga_params
        )

        if not search_results:
            raise ValueError("搜索结果为空，无法生成样本")

        print(f"  - 获得 {len(search_results)} 个搜索点")

        # 2. 计算所有点的特征
        all_features = FeatureCalculator.calculate_neighbor_features(
            search_results, elite_percentile, bottom_percentile
        )

        # 3. 光栅化到网格
        X_tensor = self._rasterize_features(search_results, all_features)

        # 4. 生成标签
        Y_tensor = self._generate_label(true_theta, true_phi)

        print(f"  - 生成特征张量: {X_tensor.shape}, 标签张量: {Y_tensor.shape}")

        return X_tensor, Y_tensor

    def _rasterize_features(self,
                          search_results: List[SearchResult],
                          all_features: dict) -> np.ndarray:
        """
        将稀疏搜索结果光栅化到固定网格

        参数:
            search_results: 搜索结果列表
            all_features: 特征字典

        返回:
            (3, H, W) 特征张量
        """
        h, w = self.grid_size

        # 初始化三个通道
        sinr_map = np.zeros((h, w), dtype=np.float32)
        elite_map = np.zeros((h, w), dtype=np.float32)
        bottom_map = np.zeros((h, w), dtype=np.float32)

        # 统计值范围用于归一化
        sinr_values = []
        elite_values = []
        bottom_values = []

        # 填充网格
        for res in search_results:
            theta, phi = res.null_theta, res.null_phi
            row, col = self.mapper.coords_to_grid(theta, phi)

            features = all_features.get((theta, phi))
            if features is None:
                continue

            sinr_map[row, col] = features['sinr_db']
            elite_map[row, col] = features['elite_neighbor_score']
            bottom_map[row, col] = features['bottom_neighbor_score']

            sinr_values.append(features['sinr_db'])
            elite_values.append(features['elite_neighbor_score'])
            bottom_values.append(features['bottom_neighbor_score'])

        # 归一化
        # SINR: min-max归一化到[0, 1]
        if len(sinr_values) > 0:
            sinr_min, sinr_max = np.min(sinr_values), np.max(sinr_values)
            if sinr_max > sinr_min:
                sinr_map = np.where(sinr_map > 0,
                                   (sinr_map - sinr_min) / (sinr_max - sinr_min),
                                   0)

        # Elite邻居: 除以最大可能值（8个邻居）
        elite_map = elite_map / 8.0

        # Bottom邻居: 除以最大可能值（8个邻居）
        bottom_map = bottom_map / 8.0

        # 堆叠成 (3, H, W)
        X_tensor = np.stack([sinr_map, elite_map, bottom_map], axis=0).astype(np.float32)

        return X_tensor

    def _generate_label(self, true_theta: float, true_phi: float) -> np.ndarray:
        """
        生成标签：在真实位置创建2D高斯热点

        参数:
            true_theta: 真实theta
            true_phi: 真实phi

        返回:
            (1, H, W) 标签张量
        """
        h, w = self.grid_size

        # 找到真实位置的网格索引
        true_row, true_col = self.mapper.coords_to_grid(true_theta, true_phi)

        # 创建2D高斯分布
        Y_tensor = np.zeros((h, w), dtype=np.float32)

        # 创建网格坐标
        y, x = np.ogrid[0:h, 0:w]

        # 2D高斯公式
        gaussian = np.exp(-((x - true_col)**2 + (y - true_row)**2) / (2 * self.gaussian_sigma**2))

        # 归一化到[0, 1]
        if gaussian.max() > 0:
            gaussian = gaussian / gaussian.max()

        Y_tensor = gaussian

        # 添加通道维度 (1, H, W)
        Y_tensor = Y_tensor[np.newaxis, :, :].astype(np.float32)

        return Y_tensor

    def generate_dataset(self,
                        num_samples: int,
                        output_dir: str,
                        adaptive_array,
                        cache,
                        base_ga_params: dict,
                        jammer_theta_range: Tuple[float, float] = (27, 52),
                        jammer_phi_range: Tuple[float, float] = (-90, 90),
                        elite_percentile: float = 30.0,
                        bottom_percentile: float = 30.0):
        """
        批量生成数据集

        参数:
            num_samples: 生成样本数
            output_dir: 输出目录
            adaptive_array: 相控阵对象
            cache: 缓存对象
            base_ga_params: GA参数
            jammer_theta_range: 干扰源theta范围（绝对值）
            jammer_phi_range: 干扰源phi范围
            elite_percentile: 精英点百分比
            bottom_percentile: 底部点百分比
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n开始生成 {num_samples} 个训练样本...")
        print(f"输出目录: {output_dir}")

        for i in range(num_samples):
            print(f"\n{'='*60}")
            print(f"样本 {i+1}/{num_samples}")
            print(f"{'='*60}")

            # 随机生成干扰源位置
            true_theta = np.random.uniform(jammer_theta_range[0], jammer_theta_range[1])
            true_theta *= np.random.choice([-1, 1])  # 随机正负
            true_phi = np.random.uniform(jammer_phi_range[0], jammer_phi_range[1])

            try:
                X_tensor, Y_tensor = self.generate_single_sample(
                    true_theta, true_phi,
                    adaptive_array, cache, base_ga_params,
                    elite_percentile, bottom_percentile
                )

                # 保存样本
                sample_data = {
                    'X': X_tensor,
                    'Y': Y_tensor,
                    'true_theta': true_theta,
                    'true_phi': true_phi
                }

                filename = os.path.join(output_dir, f'sample_{i:04d}.pkl')
                with open(filename, 'wb') as f:
                    pickle.dump(sample_data, f)

                print(f"  ✓ 样本已保存: {filename}")

            except Exception as e:
                print(f"  ✗ 生成样本失败: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"数据集生成完成！")
        print(f"{'='*60}")


# 示例使用
if __name__ == "__main__":
    import pandas as pd

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

    # 创建数据集生成器
    generator = DatasetGenerator(
        grid_size=(64, 64),
        theta_range=(-60, 60),
        phi_range=(-90, 90),
        gaussian_sigma=1.5
    )

    # 生成数据集
    generator.generate_dataset(
        num_samples=100,  # 建议至少500-1000个样本
        output_dir=os.path.join(parent_dir, 'unet_dataset'),
        adaptive_array=adaptive_array,
        cache=cache,
        base_ga_params=base_ga_params,
        jammer_theta_range=(27, 52),
        jammer_phi_range=(-90, 90),
        elite_percentile=30.0,
        bottom_percentile=30.0
    )
