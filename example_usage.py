"""
完整使用示例
展示从数据生成到训练再到推理的完整流程
"""
import os
import sys
import numpy as np
import pandas as pd

# 修复OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


def example_1_generate_small_dataset():
    """
    示例1: 生成小规模测试数据集（10个样本）
    用于快速测试整个流程
    """
    print("\n" + "="*80)
    print("示例1: 生成小规模测试数据集")
    print("="*80)

    from data_generation import DatasetGenerator
    from accuracy6_Bottom3 import AdaptiveNullingPhasedArray, PhasedArrayCache

    # 初始化仿真环境
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

    # 生成小规模数据集
    output_dir = os.path.join(parent_dir, 'unet_dataset_small')
    generator.generate_dataset(
        num_samples=10,  # 小规模测试：只生成10个样本
        output_dir=output_dir,
        adaptive_array=adaptive_array,
        cache=cache,
        base_ga_params=base_ga_params,
        jammer_theta_range=(27, 52),
        jammer_phi_range=(-90, 90),
        elite_percentile=30.0,
        bottom_percentile=30.0
    )

    print(f"\n✓ 数据集已生成至: {output_dir}")
    print("下一步: 运行 example_2_visualize_dataset() 查看数据")


def example_2_visualize_dataset():
    """
    示例2: 可视化数据集样本
    """
    print("\n" + "="*80)
    print("示例2: 可视化数据集样本")
    print("="*80)

    from dataset import JammerLocalizationDataset, visualize_sample
    import pickle

    # 加载数据集
    data_dir = os.path.join(parent_dir, 'unet_dataset_small')

    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录 {data_dir} 不存在")
        print("请先运行 example_1_generate_small_dataset()")
        return

    dataset = JammerLocalizationDataset(data_dir)

    # 可视化前3个样本
    vis_dir = os.path.join(parent_dir, 'dataset_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    for i in range(min(3, len(dataset))):
        X, Y = dataset[i]
        info = dataset.get_sample_info(i)

        print(f"\n样本 {i}:")
        print(f"  真实位置: θ={info['true_theta']:.2f}°, φ={info['true_phi']:.2f}°")
        print(f"  特征张量: {X.shape}")
        print(f"  标签张量: {Y.shape}")

        save_path = os.path.join(vis_dir, f'sample_{i}.png')
        visualize_sample(X, Y, save_path)

    print(f"\n✓ 可视化结果已保存至: {vis_dir}")


def example_3_test_model_architecture():
    """
    示例3: 测试U-Net模型架构
    """
    print("\n" + "="*80)
    print("示例3: 测试U-Net模型架构")
    print("="*80)

    import torch
    from unet_model import UNet, UNetSmall, count_parameters

    # 创建标准U-Net
    model = UNet(in_channels=3, out_channels=1)
    print(f"\n标准U-Net:")
    print(f"  参数数量: {count_parameters(model):,}")

    # 创建轻量级U-Net
    model_small = UNetSmall(in_channels=3, out_channels=1)
    print(f"\n轻量级U-Net:")
    print(f"  参数数量: {count_parameters(model_small):,}")

    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        output = model(dummy_input)
        prob_mask = torch.sigmoid(output)

    print(f"\n前向传播测试:")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  概率掩码范围: [{prob_mask.min():.4f}, {prob_mask.max():.4f}]")

    print("\n✓ 模型架构测试通过")


def example_4_quick_training():
    """
    示例4: 快速训练测试（5轮）
    用于验证训练流程是否正常
    """
    print("\n" + "="*80)
    print("示例4: 快速训练测试（5轮）")
    print("="*80)

    import torch
    import torch.optim as optim
    from unet_model import UNet
    from dataset import create_data_loaders
    from train import Trainer, CombinedLoss

    # 检查数据集
    data_dir = os.path.join(parent_dir, 'unet_dataset_small')
    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录 {data_dir} 不存在")
        print("请先运行 example_1_generate_small_dataset()")
        return

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=4,
        train_ratio=0.8,
        num_workers=0,
        use_augmentation=False,
        seed=42
    )

    # 创建模型
    model = UNet(in_channels=3, out_channels=1)

    # 创建损失函数和优化器
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 创建训练器
    save_dir = os.path.join(parent_dir, 'unet_jammer_localization', 'checkpoints_test')
    log_dir = os.path.join(parent_dir, 'unet_jammer_localization', 'runs_test')

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir
    )

    # 快速训练5轮
    print("\n开始快速训练（5轮）...")
    trainer.train(num_epochs=5)

    # 绘制训练曲线
    trainer.plot_training_history(
        save_path=os.path.join(save_dir, 'training_history_test.png')
    )

    print(f"\n✓ 快速训练完成")
    print(f"  模型保存至: {save_dir}")


def example_5_inference_test():
    """
    示例5: 推理测试
    使用训练好的模型进行预测
    """
    print("\n" + "="*80)
    print("示例5: 推理测试")
    print("="*80)

    import torch
    import pickle
    from inference import JammerLocalizationPredictor
    from accuracy6_Bottom3 import angular_distance

    # 检查模型文件
    model_path = os.path.join(parent_dir, 'unet_jammer_localization', 'checkpoints_test', 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        print("请先运行 example_4_quick_training()")
        return

    # 创建预测器
    predictor = JammerLocalizationPredictor(
        model_path=model_path,
        grid_size=(64, 64),
        theta_range=(-60, 60),
        phi_range=(-90, 90)
    )

    # 加载一个测试样本
    data_dir = os.path.join(parent_dir, 'unet_dataset_small')
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    if not sample_files:
        print(f"错误: 在 {data_dir} 中未找到测试样本")
        return

    sample_path = os.path.join(data_dir, sample_files[0])
    with open(sample_path, 'rb') as f:
        sample_data = pickle.load(f)

    X_tensor = sample_data['X']
    true_theta = sample_data['true_theta']
    true_phi = sample_data['true_phi']

    print(f"\n测试样本:")
    print(f"  真实位置: θ={true_theta:.2f}°, φ={true_phi:.2f}°")

    # 预测
    pred_theta, pred_phi, prob_mask = predictor.predict(X_tensor)

    # 计算误差
    error = angular_distance(pred_theta, pred_phi, true_theta, true_phi)

    print(f"\n预测结果:")
    print(f"  预测位置: θ={pred_theta:.2f}°, φ={pred_phi:.2f}°")
    print(f"  定位误差: {error:.2f}°")

    # 可视化
    save_dir = os.path.join(parent_dir, 'unet_jammer_localization', 'predictions_test')
    os.makedirs(save_dir, exist_ok=True)

    predictor.visualize_prediction(
        X_tensor, prob_mask,
        pred_theta, pred_phi,
        true_theta, true_phi,
        save_path=os.path.join(save_dir, 'prediction_test.png')
    )

    print(f"\n✓ 推理测试完成")
    print(f"  可视化结果已保存至: {save_dir}")


def example_6_full_pipeline():
    """
    示例6: 完整流程演示
    从仿真到预测的端到端流程
    """
    print("\n" + "="*80)
    print("示例6: 完整流程演示（端到端）")
    print("="*80)

    print("\n此示例展示如何将U-Net集成到现有仿真代码中")
    print("完整流程:")
    print("  1. 运行仿真 → 获取search_results")
    print("  2. 计算三个特征 → 光栅化到网格")
    print("  3. U-Net预测 → 获得干扰源位置")
    print("  4. 与真实位置对比 → 计算误差")

    print("\n代码示例:")
    print("-" * 60)

    code = '''
from inference import JammerLocalizationPredictor
from data_generation import DatasetGenerator
from accuracy6_Bottom3 import (
    AdaptiveNullingPhasedArray,
    AdaptiveNullingSimulationEnvironment,
    PhasedArrayCache,
    grid_search_for_jammer,
    angular_distance
)

# 1. 初始化仿真环境
adaptive_array = AdaptiveNullingPhasedArray(...)
cache = PhasedArrayCache()
env = AdaptiveNullingSimulationEnvironment(...)
env.add_jammer_hidden(true_theta, true_phi, power)

# 2. 运行网格搜索（与原代码相同）
search_results = grid_search_for_jammer(
    adaptive_array, main_beam_theta, main_beam_phi, env,
    search_range_theta, search_range_phi, cache, **ga_params
)

# 3. 生成特征张量（替换原有的analyze_with_dynamic_neighborhood）
generator = DatasetGenerator(grid_size=(64, 64))
X_tensor, _ = generator.generate_single_sample(
    true_theta, true_phi,
    adaptive_array, cache, ga_params
)

# 4. U-Net预测（替换原有的启发式公式）
predictor = JammerLocalizationPredictor(
    model_path='./checkpoints/best_model.pth',
    grid_size=(64, 64)
)
pred_theta, pred_phi, prob_mask = predictor.predict(X_tensor)

# 5. 计算误差
error = angular_distance(pred_theta, pred_phi, true_theta, true_phi)
print(f"定位误差: {error:.2f}°")
'''

    print(code)
    print("-" * 60)

    print("\n关键改动点（在 accuracy6_Bottom3.py 中）:")
    print("  原代码 (第1044-1050行):")
    print("    est_theta, est_phi, error = analyze_with_dynamic_neighborhood(...)")
    print("\n  新代码:")
    print("    X_tensor, _ = generator.generate_single_sample(...)")
    print("    est_theta, est_phi, _ = predictor.predict(X_tensor)")


def main_menu():
    """主菜单"""
    examples = {
        '1': ('生成小规模测试数据集（10个样本）', example_1_generate_small_dataset),
        '2': ('可视化数据集样本', example_2_visualize_dataset),
        '3': ('测试U-Net模型架构', example_3_test_model_architecture),
        '4': ('快速训练测试（5轮）', example_4_quick_training),
        '5': ('推理测试', example_5_inference_test),
        '6': ('完整流程演示', example_6_full_pipeline),
        '0': ('运行全部示例', None)
    }

    print("\n" + "="*80)
    print("U-Net干扰源定位优化系统 - 使用示例")
    print("="*80)
    print("\n选择示例运行:")

    for key, (desc, _) in examples.items():
        if key != '0':
            print(f"  {key}. {desc}")
    print(f"  0. 运行全部示例（按顺序）")
    print(f"  q. 退出")

    while True:
        choice = input("\n请输入选项 (1-6, 0, q): ").strip()

        if choice == 'q':
            print("退出")
            break
        elif choice == '0':
            # 运行全部示例
            for key in ['1', '2', '3', '4', '5', '6']:
                _, func = examples[key]
                if func:
                    func()
                    input("\n按Enter继续...")
            break
        elif choice in examples and choice != '0':
            _, func = examples[choice]
            if func:
                func()
            input("\n按Enter返回菜单...")
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    # 可以直接运行特定示例，或显示菜单
    import sys

    if len(sys.argv) > 1:
        # 命令行参数运行
        example_num = sys.argv[1]
        examples = {
            '1': example_1_generate_small_dataset,
            '2': example_2_visualize_dataset,
            '3': example_3_test_model_architecture,
            '4': example_4_quick_training,
            '5': example_5_inference_test,
            '6': example_6_full_pipeline
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"未知示例: {example_num}")
            print("用法: python example_usage.py [1-6]")
    else:
        # 交互式菜单
        main_menu()
