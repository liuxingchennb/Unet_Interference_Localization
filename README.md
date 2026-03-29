# U-Net Interference Localization

基于 U-Net 的单干扰源角度定位项目，用于将传统启发式搜索结果进一步映射为二维概率热力图，从而提升 8×8 单元相控阵场景下的干扰源定位精度。

## 项目亮点

- 问题形式：将干扰源角度定位重写为二维热力图预测任务
- 输入特征：SINR 图、精英邻居分数图、底部邻居分数图
- 输出结果：干扰源位置概率热力图，取峰值点作为最终角度估计
- 应用场景：8×8 单元相控阵、单个干扰源定位
- 当前目标：把启发式方法升级为可训练、可复现、可扩展的深度学习定位流程

## 当前结果

根据本项目当前实验设置，在单个干扰源定位任务中：

- 启发式算法是基础定位方案
- 使用 U-Net 对特征图进行学习后，误差小于 `7.5°` 的准确度可提升到 `98%+`

说明：

- 上述结果基于当前仿真流程与实验配置
- 结果会受到样本分布、网格分辨率、损失函数和训练数据规模影响
- 公开仓库当前主要提供代码与流程，默认不包含训练权重文件

## 方法概览

传统做法通常直接在搜索结果上使用固定公式进行打分，例如：

```python
final_score = sinr * (1 + k * elite - w * bottom)
```

这个项目的核心思路是，不再手工固定 `k`、`w` 的组合方式，而是让网络从数据中学习这三个特征之间的非线性关系。

### 输入

模型输入为形状 `(3, H, W)` 的三通道特征图：

1. `SINR map`
2. `Elite-neighbor score map`
3. `Bottom-neighbor score map`

### 输出

模型输出为形状 `(1, H, W)` 的概率热力图：

- 热力图峰值位置对应预测的干扰源角度
- 真实标签使用二维高斯分布构造，而不是单点标签

### 为什么选择 U-Net

- 保留空间位置信息，适合“角度网格 -> 热力图”任务
- Skip Connection 对小目标定位更友好
- 在中小规模数据集上也较容易训练出稳定结果

## 仓库内容

```text
unet_jammer_localization/
├── data_generation.py      # 数据生成与特征构建
├── dataset.py              # PyTorch 数据集与 DataLoader
├── unet_model.py           # U-Net / U-Net Small 模型定义
├── train.py                # 训练入口
├── inference.py            # 推理与可视化
├── example_usage.py        # 从生成数据到推理的完整示例
├── QUICKSTART.md           # 快速上手说明
├── PROJECT_SUMMARY.md      # 项目总结
├── TROUBLESHOOTING.md      # 常见问题排查
└── FIXES_APPLIED.md        # 修复记录
```

## 依赖环境

- Python 3.7+
- PyTorch 1.8+
- NumPy / SciPy / Pandas / Matplotlib
- TensorBoard（可选，建议安装）

安装方式：

```bash
pip install -r requirements.txt
```

如果需要 GPU 版本 PyTorch，请按你的 CUDA 版本从 [PyTorch 官网](https://pytorch.org/) 安装对应包。

## 重要说明

这个仓库中的训练与推理流程依赖你原有的仿真代码，尤其是父目录中的 `accuracy6_Bottom3.py`。

当前代码里以下模块直接依赖该文件：

- `data_generation.py`
- `inference.py`
- `example_usage.py`

因此，公开仓库目前更准确地说是：

- 一个可独立阅读和复用的 U-Net 定位模块
- 一个与现有启发式/仿真工程集成的训练推理原型
- 不是完全脱离原仿真环境即可直接运行的一键式独立项目

如果后续你希望把它做成更完整的公开项目，建议继续补两部分：

1. 将 `accuracy6_Bottom3.py` 中训练所需的最小仿真接口抽离出来
2. 提供一小份可公开的示例数据或 demo 权重

## 快速开始

### 1. 生成数据

```bash
python data_generation.py
```

默认示例配置会：

- 构建 `64 × 64` 角度网格
- 生成三通道输入特征
- 为每个样本生成二维高斯标签

脚本中的默认示例参数包括：

- `theta_range=(-60, 60)`
- `phi_range=(-90, 90)`
- `jammer_theta_range=(27, 52)`
- `jammer_phi_range=(-90, 90)`
- `gaussian_sigma=1.5`

生成后的数据默认写入父目录下的 `unet_dataset/`。

### 2. 训练模型

```bash
python train.py
```

当前默认训练配置位于 [train.py](train.py) 中，核心参数包括：

- 数据目录：`../unet_dataset`
- 模型：`unet`
- 批大小：`8`
- 训练轮数：`100`
- 学习率：`1e-3`
- 损失函数：`combined`，即 Dice Loss + Focal Loss

训练输出默认保存在：

- `./checkpoints`
- `./runs`

训练结束后通常会得到：

- `best_model.pth`
- `latest_checkpoint.pth`
- `training_history.png`

### 3. 推理预测

```bash
python inference.py
```

推理阶段会：

1. 根据仿真搜索结果生成与训练一致的三通道特征图
2. 使用训练好的 U-Net 输出概率热力图
3. 读取热力图峰值并映射回角度坐标

核心预测接口是 [inference.py](inference.py) 中的 `JammerLocalizationPredictor`。

### 4. 运行完整示例

```bash
python example_usage.py
```

这个脚本包含：

- 小规模数据生成
- 样本可视化
- 模型结构测试
- 快速训练
- 预测示例

适合在你现有工程环境中快速验证整条链路。

## 训练设计

### 标签设计

- 使用二维高斯热力图作为监督信号
- 比单点标签更平滑
- 对网格离散误差更鲁棒

### 损失函数

默认采用 `CombinedLoss`：

- Dice Loss：优化预测区域与真实区域的重叠
- Focal Loss：缓解类别不平衡，强化难样本学习

这套组合尤其适合“小目标 + 大背景”的定位问题。

### 模型版本

当前提供两个模型：

- `UNet`
- `UNetSmall`

其中标准 `UNet` 是默认版本。

## 仓库中未包含的内容

为了避免仓库体积过大以及 GitHub 大文件限制，以下内容默认未纳入版本库：

- 训练得到的 `.pth` 权重文件
- TensorBoard 日志
- 测试预测图片
- `__pycache__`
- 动态缓存文件

如果你需要公开发布训练结果，推荐使用以下方式之一：

- GitHub Release：适合发布固定版本权重
- Git LFS：适合把大模型文件像代码一样长期托管在仓库体系中

## 适合公开展示的后续增强

如果你准备继续把这个仓库打磨成作品集或论文配套仓库，下一步优先建议是：

1. 加一张方法流程图
2. 加一张预测热力图示例
3. 补一个 `LICENSE`
4. 补一个 `assets/` 目录专门放 README 图片
5. 发布一版可直接下载的预训练权重

## 相关文档

- [QUICKSTART.md](QUICKSTART.md)
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- [FIXES_APPLIED.md](FIXES_APPLIED.md)

## License

当前仓库还没有单独添加开源许可证文件。如果你计划让别人复用代码，建议后续补充 `LICENSE`。
