# U-Net干扰源定位优化系统

## 项目概述

本项目使用深度学习（U-Net图像分割模型）优化相控阵干扰源定位算法，将定位准确率从80%提升至更高水平。

### 核心思路

将干扰源定位问题**重新定义为图像分割任务**：

- **输入 (X)**: `(3, H, W)` 三通道特征图
  - 通道1：SINR图（在各点施加零陷后的信噪比）
  - 通道2：精英邻居分数（周围高SINR点的数量）
  - 通道3：底部邻居分数（周围低SINR点的数量）

- **输出 (Y)**: `(1, H, W)` 概率热力图
  - 最亮的点 = 模型预测的干扰源位置

- **优势**:
  - 自动学习复杂的非线性特征组合
  - 能够区分"真实高分集群"和"自然零陷"产生的"虚假高分集群"
  - 替换固定启发式公式：`final_score = sinr * (1 + k*elite - w*bottom)`

---

## 项目结构

```
unet_jammer_localization/
│
├── data_generation.py      # 数据集生成模块
├── unet_model.py           # U-Net模型架构
├── dataset.py              # PyTorch数据集和加载器
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
├── README.md               # 本文档
└── example_usage.py        # 完整使用示例
```

---

## 安装依赖

```bash
pip install torch torchvision numpy pandas matplotlib scipy tensorboard
```

**环境要求**:
- Python 3.7+
- PyTorch 1.8+
- CUDA（可选，用于GPU加速）

---

## 使用流程

### 第一步：生成训练数据集

```python
# 运行数据生成脚本
python data_generation.py
```

**说明**:
- 会调用现有仿真代码 `accuracy6_Bottom3.py`
- 为每个样本生成：
  - 完整的搜索结果 (theta, phi, SINR)
  - 计算所有点的三个特征（SINR, elite_neighbor, bottom_neighbor）
  - 光栅化到固定网格 (64×64)
  - 生成2D高斯标签

**配置参数**:
```python
generator = DatasetGenerator(
    grid_size=(64, 64),          # 网格分辨率
    theta_range=(-60, 60),       # theta搜索范围
    phi_range=(-90, 90),         # phi搜索范围
    gaussian_sigma=1.5           # 标签高斯宽度
)

# 生成数据集
generator.generate_dataset(
    num_samples=500,             # 建议至少500-1000个样本
    output_dir='../unet_dataset',
    adaptive_array=adaptive_array,
    cache=cache,
    base_ga_params=base_ga_params,
    jammer_theta_range=(27, 52),
    jammer_phi_range=(-90, 90),
    elite_percentile=30.0,       # 精英点百分比（可调）
    bottom_percentile=30.0       # 底部点百分比（可调）
)
```

**输出**:
- 数据保存在 `unet_dataset/` 目录
- 每个样本为一个 `.pkl` 文件
- 包含 `X` (3, 64, 64), `Y` (1, 64, 64), `true_theta`, `true_phi`

---

### 第二步：训练U-Net模型

```python
# 运行训练脚本
python train.py
```

**关键超参数**:

```python
CONFIG = {
    # 数据
    'data_dir': '../unet_dataset',
    'batch_size': 8,
    'train_ratio': 0.8,
    'use_augmentation': True,    # 数据增强（水平翻转、噪声）

    # 模型
    'model_type': 'unet',        # 'unet' 或 'unet_small'

    # 训练
    'num_epochs': 100,
    'learning_rate': 1e-3,

    # 损失函数（关键！）
    'loss_type': 'combined',     # 推荐：Dice + Focal
    'dice_weight': 0.5,
    'focal_weight': 0.5,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # 学习率调度
    'use_scheduler': True,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
}
```

**为什么选择Combined Loss (Dice + Focal)?**

1. **Dice Loss**:
   - 直接优化预测和真实标签的重叠程度
   - 对类别不平衡不敏感（99%背景，1%目标）
   - 适合小目标定位

2. **Focal Loss**:
   - 自动降低"易分类像素"的权重
   - 让模型聚焦于"难分类像素"
   - `γ=2`参数控制聚焦强度

3. **组合效果**:
   - Dice优化全局重叠（宏观）
   - Focal处理类别不平衡（微观）
   - 实验表明组合损失效果最佳

**训练输出**:
- 模型保存在 `checkpoints/`
  - `best_model.pth` (最佳验证损失)
  - `latest_checkpoint.pth` (最新)
- TensorBoard日志: `runs/`
- 训练曲线: `checkpoints/training_history.png`

**监控训练**:
```bash
# 启动TensorBoard
tensorboard --logdir=runs
```

---

### 第三步：推理预测

```python
# 运行推理脚本
python inference.py
```

**使用训练好的模型进行预测**:

```python
from inference import JammerLocalizationPredictor

# 创建预测器
predictor = JammerLocalizationPredictor(
    model_path='./checkpoints/best_model.pth',
    grid_size=(64, 64),
    theta_range=(-60, 60),
    phi_range=(-90, 90)
)

# 预测（从仿真数据）
pred_theta, pred_phi, prob_mask = predict_from_simulation(
    predictor,
    adaptive_array,
    cache,
    base_ga_params,
    true_theta=35.0,
    true_phi=45.0,
    save_dir='./predictions'  # 保存可视化结果
)

print(f"预测位置: ({pred_theta:.2f}°, {pred_phi:.2f}°)")
```

**核心推理流程**:

1. **生成特征张量** (与训练时相同):
   ```python
   # 运行仿真 → 获取search_results
   # 计算三个特征 → 光栅化到网格
   X_tensor = (3, 64, 64)  # SINR, Elite, Bottom
   ```

2. **模型预测**:
   ```python
   logits = model(X_tensor)          # (1, 1, 64, 64)
   prob_mask = torch.sigmoid(logits) # 转换为概率 [0, 1]
   ```

3. **提取位置**:
   ```python
   # 找到最大概率的像素
   max_idx = np.argmax(prob_mask)
   pred_row, pred_col = np.unravel_index(max_idx, (64, 64))

   # 转换回角度坐标
   pred_theta, pred_phi = mapper.grid_to_coords(pred_row, pred_col)
   ```

---

## 模型架构详解

### U-Net结构

```
输入: (3, 64, 64)

编码器（下采样）:
  64x64 (64通道)  → MaxPool
  32x32 (128通道) → MaxPool
  16x16 (256通道) → MaxPool
  8x8  (512通道)  → MaxPool
  4x4  (512通道)  [瓶颈层]

解码器（上采样 + 跳跃连接）:
  4x4  (512通道) → UpSample + Concat[8x8]
  8x8  (256通道) → UpSample + Concat[16x16]
  16x16 (128通道) → UpSample + Concat[32x32]
  32x32 (64通道) → UpSample + Concat[64x64]
  64x64 (64通道)

输出层:
  64x64 (1通道) → 概率掩码
```

**为什么选择U-Net?**

1. **跳跃连接** (Skip Connections):
   - 将编码器的高分辨率特征直接传递给解码器
   - 保留精确的空间位置信息
   - 非常适合需要精确定位的任务

2. **对称架构**:
   - 编码器：提取高级语义特征
   - 解码器：重建空间分辨率
   - 完美适配"特征图 → 热力图"的转换

3. **在小数据集上表现良好**:
   - 相比ResNet、VGG等，U-Net对样本数量要求较低
   - 500-1000个样本即可获得良好效果

**模型参数量**:
- 标准U-Net: ~31M 参数
- 轻量级U-NetSmall: ~7M 参数（通道数减半）

---

## 数据集设计

### 特征通道设计

| 通道 | 含义 | 归一化方法 | 物理意义 |
|------|------|-----------|---------|
| 通道0 | SINR图 | Min-Max归一化到[0,1] | 在该点施加零陷后的信噪比 |
| 通道1 | 精英邻居分数 | 除以8（最大邻居数） | 周围高SINR点的数量 |
| 通道2 | 底部邻居分数 | 除以8（最大邻居数） | 周围低SINR点的数量 |

**特征工程的关键**:
- 这三个特征是原启发式公式中使用的相同特征
- 但U-Net能够学习它们之间的**非线性组合**
- 不再依赖固定的 `k` 和 `w` 系数

### 标签设计

- **2D高斯分布**，中心在真实干扰源位置
- 标准差 `σ = 1.5` 像素（可调）
- 归一化到 `[0, 1]`

**为什么用高斯而不是单点?**
- 提供"软标签"，允许预测有一定容差
- 避免对网格量化误差过于敏感
- 帮助模型学习"接近目标"的概念

---

## 与原算法的集成

### 方案1: 完全替换（推荐）

在 `accuracy6_Bottom3.py` 中：

```python
# 原代码 (第1050行左右)
est_theta, est_phi, error = analyze_with_dynamic_neighborhood(
    search_results, main_beam_theta, main_beam_phi,
    baseline_sinr, (jammer_theta, jammer_phi), trial_num,
    **analysis_params
)

# 新代码：替换为U-Net预测
from unet_jammer_localization.inference import JammerLocalizationPredictor

predictor = JammerLocalizationPredictor(
    model_path='./unet_jammer_localization/checkpoints/best_model.pth',
    grid_size=(64, 64)
)

# 生成特征 → 预测
X_tensor, _ = generator.generate_single_sample(...)
est_theta, est_phi, prob_mask = predictor.predict(X_tensor)
```

### 方案2: 混合策略（保守）

```python
# 同时运行两种方法
unet_theta, unet_phi, _ = predictor.predict(X_tensor)
heuristic_theta, heuristic_phi, _ = analyze_with_dynamic_neighborhood(...)

# 如果两者一致，使用U-Net结果；否则使用启发式（作为后备）
if angular_distance(unet_theta, unet_phi, heuristic_theta, heuristic_phi) < 10:
    est_theta, est_phi = unet_theta, unet_phi
else:
    est_theta, est_phi = heuristic_theta, heuristic_phi  # 后备方案
```

---

## 超参数调优建议

### 关键超参数

1. **数据生成阶段**:
   - `elite_percentile`: 30% (可尝试 20%-40%)
   - `bottom_percentile`: 30% (可尝试 20%-40%)
   - `gaussian_sigma`: 1.5像素 (可尝试 1.0-2.5)
   - `grid_size`: (64, 64) (更高分辨率如128×128需要更多样本)

2. **训练阶段**:
   - `learning_rate`: 1e-3 (Adam优化器的标准起点)
   - `batch_size`: 8 (根据GPU内存调整，4-16)
   - `num_epochs`: 100 (观察验证损失曲线决定)
   - `loss_type`: 'combined' (优先尝试)

3. **损失函数权重**:
   - `dice_weight`: 0.5
   - `focal_weight`: 0.5
   - `focal_gamma`: 2.0 (控制难样本聚焦程度)

### 调优策略

1. **先小规模测试**:
   - 用100个样本快速迭代
   - 确定最佳超参数组合

2. **然后大规模训练**:
   - 用500-1000个样本训练最终模型

3. **观察指标**:
   - 验证损失 (主要指标)
   - 训练/验证损失差距 (过拟合检测)
   - 实际定位误差 (最终指标)

---

## 常见问题

### Q1: 需要多少训练样本？

**答**:
- 最少: 200-300个（可能欠拟合）
- 推荐: 500-1000个（良好平衡）
- 理想: 1000-2000个（最佳性能）

U-Net对样本数量要求相对较低，但更多数据总是有帮助。

---

### Q2: 训练时间？

**答**:
- CPU: ~2-4小时 (500样本, 100轮)
- GPU (RTX 3080): ~20-40分钟

数据生成时间取决于仿真速度（这是瓶颈）。

---

### Q3: 如果定位失败率仍然较高？

**答**: 尝试以下策略：

1. **增加训练样本**
2. **调整标签高斯宽度** (`gaussian_sigma`)
3. **尝试不同的损失函数组合**
4. **增加模型容量** (使用标准U-Net而非UNetSmall)
5. **数据增强** (水平翻转、旋转、噪声)
6. **集成学习**: 训练多个模型，取平均预测

---

### Q4: 模型泛化能力？

**答**:
- 如果训练数据覆盖整个 theta/phi 范围，泛化能力很好
- 建议在整个可能范围内均匀采样生成训练数据
- 使用**验证集**监控泛化性能

---

### Q5: 如何可视化模型学到了什么？

**答**: 使用 `inference.py` 的可视化功能：

```python
predictor.visualize_prediction(X_tensor, prob_mask, ...)
```

这会显示：
- 输入的三个特征图
- 模型预测的热力图
- 预测位置 vs 真实位置
- 定位误差

---

## 性能预期

### 原算法（启发式）
- 准确率: **80%** (误差 < 15°)
- 失败原因: 被"自然零陷"欺骗

### U-Net优化算法（预期）
- 准确率: **95-98%** (误差 < 15°)
- 优势:
  - 学习区分"真实"和"虚假"高分区域
  - 自适应特征组合
  - 对复杂场景鲁棒性更强

---

## 进阶优化

### 1. 多尺度预测
使用不同网格分辨率的集成：
- 粗网格 (32×32): 快速全局定位
- 细网格 (128×128): 精确局部定位

### 2. 注意力机制
在U-Net中加入注意力模块，让模型聚焦关键区域

### 3. 时序信息
如果有多个时间步的测量，使用3D U-Net或LSTM融合时序信息

### 4. 不确定性估计
使用Dropout或集成方法估计预测的置信度

---

## 引用和参考

如果本项目对你的研究有帮助，请引用：

```
基于U-Net的相控阵干扰源定位优化系统
作者: Star Liu
年份: 2025
```

**相关论文**:
- U-Net: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Focal Loss: Lin et al. "Focal Loss for Dense Object Detection" (2017)
- Dice Loss: Milletari et al. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (2016)

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱:  
- GitHub Issues: [项目链接]

---

**祝训练顺利！🚀**
