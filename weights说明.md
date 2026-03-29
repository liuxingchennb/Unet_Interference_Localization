# Weights Notes

## 文件

- `best_model.pth`

## 说明

这份权重来自当前仓库内可用的 U-Net 最佳权重文件：

- 路径：`checkpoints_test/best_model.pth`
- 文件大小：约 `207 MB`

需要特别说明的是，这个权重对应的是仓库中的“快速训练测试”流程，用于验证训练与推理链路可以正常跑通，不应直接当作最终论文级或正式实验权重理解。

## 对应训练配置

该权重对应的训练流程主要来自 [example_usage.py](example_usage.py) 中的快速训练示例：

- 数据集：`unet_dataset_small`
- 数据规模：`10` 个样本的小规模测试集
- 模型：`UNet`
- 输入通道数：`3`
- 输出通道数：`1`
- Batch Size：`4`
- Train / Val 划分：`0.8 / 0.2`
- 数据增强：`False`
- 优化器：`Adam`
- 学习率：`1e-3`
- 损失函数：`CombinedLoss`
- Dice 权重：`0.5`
- Focal 权重：`0.5`
- 训练轮数：`5`
- 保存目录：`checkpoints_test/`

## 对应结果

这份权重的用途是：

- 验证数据生成、训练、推理、可视化流程是否连通
- 作为一个可下载的 demo 权重，方便别人快速理解项目结构

这份权重不代表 README 中提到的正式目标结果，即：

- 单个干扰源定位任务中
- 定位误差小于 `7.5°`
- 准确度可达到 `98%+`

如果后续需要发布“正式实验权重”，建议单独再发一个 Release，并在说明中补充：

- 训练样本数量
- 训练轮数
- 验证集设置
- 误差阈值定义
- 最终准确率统计口径

## 使用方式

下载后可配合 [inference.py](inference.py) 中的 `JammerLocalizationPredictor` 使用，例如：

```python
predictor = JammerLocalizationPredictor(
    model_path="./best_model.pth",
    grid_size=(64, 64),
    theta_range=(-60, 60),
    phi_range=(-90, 90)
)
```

## 依赖提醒

当前推理与数据生成流程依赖外部仿真代码，尤其是父目录中的 `accuracy6_Bottom3.py`。因此即使下载了这份权重，也需要配套仿真环境才能完整复现本项目流程。
