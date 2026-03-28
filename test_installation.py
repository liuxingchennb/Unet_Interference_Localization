"""
安装测试脚本
验证所有依赖是否正确安装，以及OpenMP问题是否已修复
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_imports():
    """测试所有必要的导入"""
    print("="*60)
    print("测试1: 检查Python库导入")
    print("="*60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False

    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas 导入失败: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib 导入失败: {e}")
        return False

    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy 导入失败: {e}")
        return False

    try:
        from torch.utils.tensorboard import SummaryWriter
        print(f"✓ TensorBoard")
    except ImportError as e:
        print(f"✗ TensorBoard 导入失败: {e}")
        return False

    return True


def test_module_imports():
    """测试项目模块导入"""
    print("\n" + "="*60)
    print("测试2: 检查项目模块导入")
    print("="*60)

    try:
        from unet_model import UNet, UNetSmall
        print("✓ unet_model.py")
    except ImportError as e:
        print(f"✗ unet_model.py 导入失败: {e}")
        return False

    try:
        from dataset import JammerLocalizationDataset
        print("✓ dataset.py")
    except ImportError as e:
        print(f"✗ dataset.py 导入失败: {e}")
        return False

    try:
        from data_generation import DatasetGenerator, CoordinateMapper
        print("✓ data_generation.py")
    except ImportError as e:
        print(f"✗ data_generation.py 导入失败: {e}")
        return False

    try:
        from train import Trainer, DiceLoss, FocalLoss, CombinedLoss
        print("✓ train.py")
    except ImportError as e:
        print(f"✗ train.py 导入失败: {e}")
        return False

    try:
        from inference import JammerLocalizationPredictor
        print("✓ inference.py")
    except ImportError as e:
        print(f"✗ inference.py 导入失败: {e}")
        return False

    return True


def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*60)
    print("测试3: 测试U-Net模型创建")
    print("="*60)

    try:
        import torch
        from unet_model import UNet, UNetSmall, count_parameters

        # 测试标准U-Net
        model = UNet(in_channels=3, out_channels=1)
        param_count = count_parameters(model)
        print(f"✓ 标准U-Net创建成功")
        print(f"  - 参数数量: {param_count:,}")

        # 测试前向传播
        dummy_input = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  - 前向传播成功: {dummy_input.shape} → {output.shape}")

        # 测试轻量级U-Net
        model_small = UNetSmall(in_channels=3, out_channels=1)
        param_count_small = count_parameters(model_small)
        print(f"✓ 轻量级U-Net创建成功")
        print(f"  - 参数数量: {param_count_small:,}")

        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinate_mapper():
    """测试坐标映射"""
    print("\n" + "="*60)
    print("测试4: 测试坐标映射器")
    print("="*60)

    try:
        from data_generation import CoordinateMapper

        mapper = CoordinateMapper(
            grid_size=(64, 64),
            theta_range=(-60, 60),
            phi_range=(-90, 90)
        )

        # 测试坐标转换
        theta, phi = 30.0, 45.0
        row, col = mapper.coords_to_grid(theta, phi)
        theta_back, phi_back = mapper.grid_to_coords(row, col)

        print(f"✓ 坐标转换测试")
        print(f"  - 原始坐标: θ={theta:.1f}°, φ={phi:.1f}°")
        print(f"  - 网格索引: row={row}, col={col}")
        print(f"  - 恢复坐标: θ={theta_back:.1f}°, φ={phi_back:.1f}°")
        print(f"  - 误差: Δθ={abs(theta-theta_back):.2f}°, Δφ={abs(phi-phi_back):.2f}°")

        return True
    except Exception as e:
        print(f"✗ 坐标映射测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n" + "="*60)
    print("测试5: 测试损失函数")
    print("="*60)

    try:
        import torch
        from train import DiceLoss, FocalLoss, CombinedLoss

        # 创建测试数据
        pred = torch.randn(4, 1, 64, 64)
        target = torch.rand(4, 1, 64, 64)

        # 测试Dice Loss
        dice_loss = DiceLoss()
        pred_prob = torch.sigmoid(pred)
        loss_dice = dice_loss(pred_prob, target)
        print(f"✓ Dice Loss: {loss_dice.item():.6f}")

        # 测试Focal Loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        loss_focal = focal_loss(pred, target)
        print(f"✓ Focal Loss: {loss_focal.item():.6f}")

        # 测试Combined Loss
        combined_loss = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
        loss_combined = combined_loss(pred, target)
        print(f"✓ Combined Loss: {loss_combined.item():.6f}")

        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openmp_fix():
    """测试OpenMP问题是否已修复"""
    print("\n" + "="*60)
    print("测试6: 测试OpenMP库冲突修复")
    print("="*60)

    try:
        # 检查环境变量
        if os.environ.get('KMP_DUPLICATE_LIB_OK') == 'TRUE':
            print("✓ KMP_DUPLICATE_LIB_OK 环境变量已设置")
        else:
            print("⚠ KMP_DUPLICATE_LIB_OK 环境变量未设置")

        # 尝试导入可能冲突的库
        import torch
        import numpy as np
        import matplotlib.pyplot as plt

        # 创建简单操作测试
        a = np.random.randn(10, 10)
        b = torch.randn(10, 10)

        print("✓ NumPy和PyTorch可以同时使用，无冲突")

        # 测试matplotlib（可能触发OpenMP）
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(a)
        plt.close(fig)

        print("✓ Matplotlib可以正常使用，无冲突")

        return True
    except Exception as e:
        print(f"✗ OpenMP测试失败: {e}")
        print("\n建议: 确保所有脚本开头都设置了 os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'")
        return False


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# U-Net干扰源定位系统 - 安装验证测试")
    print("#"*60)

    tests = [
        test_imports,
        test_module_imports,
        test_model_creation,
        test_coordinate_mapper,
        test_loss_functions,
        test_openmp_fix
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {test_name}")

    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n" + "🎉"*20)
        print("所有测试通过！系统已正确安装，可以开始使用了！")
        print("🎉"*20)
        print("\n下一步:")
        print("  1. 运行示例: python example_usage.py")
        print("  2. 生成数据: python data_generation.py")
        print("  3. 训练模型: python train.py")
    else:
        print("\n⚠️ 部分测试失败，请检查:")
        print("  1. 确保所有依赖已安装: pip install -r requirements.txt")
        print("  2. 查看 TROUBLESHOOTING.md 获取帮助")
        print("  3. 检查Python版本 >= 3.7")


if __name__ == "__main__":
    main()
