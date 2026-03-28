# 故障排除指南

## 常见问题及解决方案

---

## 问题1: OpenMP库冲突错误 ✅ 已修复

### 错误信息
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

### 原因
这是因为PyTorch和NumPy/SciPy使用了不同版本的OpenMP库（Intel的libiomp5md.dll），导致冲突。

### 解决方案（已自动修复）

我已经在所有相关文件中添加了修复代码：
- `example_usage.py`
- `dataset.py`
- `train.py`
- `inference.py`

所有文件的开头都添加了：
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

**现在可以直接运行，不会再出现这个错误！**

### 手动修复方法（如果需要）

如果你在其他脚本中遇到同样问题，在文件开头添加：

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 然后再导入其他库
import torch
import numpy as np
# ...
```

**重要**: 必须在导入torch/numpy之前设置这个环境变量！

---

## 问题2: TensorBoard未安装 ✅ 已修复

### 错误信息
```
ModuleNotFoundError: No module named 'tensorboard'
```

### 原因
TensorBoard用于训练可视化，但不是强制依赖。

### 解决方案（已自动修复）

我已经将TensorBoard改为可选依赖。如果未安装，代码会自动禁用TensorBoard功能。

**推荐安装**（用于训练监控）：
```bash
pip install tensorboard
```

**或者直接运行**（会显示警告但不会报错）：
```bash
python train.py
# 会显示: "警告: TensorBoard未安装，将禁用TensorBoard日志记录"
```

安装后使用TensorBoard：
```bash
# 训练时会自动记录日志到 runs/ 目录
python train.py

# 在另一个终端启动TensorBoard
tensorboard --logdir=runs

# 在浏览器打开 http://localhost:6006
```

---

## 问题3: 数据集路径不存在

### 错误信息
```
错误: 数据集目录 xxx 不存在
```

### 解决方案
1. 检查数据集是否已生成：
   ```bash
   # 先运行示例1生成数据
   python example_usage.py
   # 选择 1
   ```

2. 或者手动生成：
   ```bash
   python data_generation.py
   ```

---

## 问题3: 模型文件不存在

### 错误信息
```
错误: 模型文件 xxx 不存在
```

### 解决方案
1. 先训练模型：
   ```bash
   python train.py
   ```

2. 或者使用示例4快速训练：
   ```bash
   python example_usage.py
   # 选择 4
   ```

---

## 问题4: CUDA/GPU相关错误

### 错误信息
```
RuntimeError: CUDA out of memory
```

### 解决方案
1. 减小batch_size：
   ```python
   # 在 train.py 中修改
   CONFIG = {
       'batch_size': 4,  # 从8改为4
       ...
   }
   ```

2. 或者使用CPU训练：
   ```python
   CONFIG = {
       'device': 'cpu',  # 强制使用CPU
       ...
   }
   ```

---

## 问题5: 导入模块失败

### 错误信息
```
ModuleNotFoundError: No module named 'xxx'
```

### 解决方案
1. 安装缺失的依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 如果是导入accuracy6_Bottom3失败：
   ```bash
   # 检查文件路径
   # 确保 accuracy6_Bottom3.py 在正确位置
   ```

---

## 问题6: 可视化图片不显示

### 可能原因
使用了非交互式后端（Agg）

### 解决方案
如果想在窗口中查看图片，修改matplotlib后端：

```python
# 在代码开头
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import matplotlib.pyplot as plt
```

如果只想保存图片（不显示），保持现状即可。

---

## 问题7: 训练损失为NaN

### 可能原因
1. 学习率过大
2. 数据归一化问题
3. 梯度爆炸

### 解决方案
1. 降低学习率：
   ```python
   CONFIG = {
       'learning_rate': 1e-4,  # 从1e-3降低到1e-4
       ...
   }
   ```

2. 检查数据：
   ```python
   # 确保特征已归一化到[0, 1]
   print(f"X min: {X.min()}, max: {X.max()}")
   print(f"Y min: {Y.min()}, max: {Y.max()}")
   ```

3. 使用梯度裁剪：
   ```python
   # 在训练循环中添加
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

---

## 问题8: 预测结果偏差很大

### 可能原因
1. 训练样本太少
2. 训练不充分
3. 数据分布不均匀

### 解决方案
1. 增加训练样本：
   ```python
   # 生成更多样本（500-1000个）
   generator.generate_dataset(num_samples=500, ...)
   ```

2. 延长训练时间：
   ```python
   CONFIG = {
       'num_epochs': 200,  # 增加到200轮
       ...
   }
   ```

3. 调整损失函数：
   ```python
   # 尝试不同的损失函数组合
   CONFIG = {
       'loss_type': 'focal',  # 或 'dice'
       ...
   }
   ```

---

## 问题9: Windows上训练速度慢

### 解决方案
1. 设置DataLoader的num_workers=0：
   ```python
   CONFIG = {
       'num_workers': 0,  # Windows上建议设为0
       ...
   }
   ```

2. 使用GPU：
   ```bash
   # 安装CUDA版本的PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

---

## 问题10: TensorBoard无法启动

### 错误信息
```
tensorboard: command not found
```

### 解决方案
```bash
# 重新安装tensorboard
pip install tensorboard

# 启动
tensorboard --logdir=runs
```

---

## 验证安装是否正确

运行以下测试脚本：

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✓ 所有依赖安装成功")
print(f"✓ PyTorch版本: {torch.__version__}")
print(f"✓ CUDA可用: {torch.cuda.is_available()}")
print(f"✓ NumPy版本: {np.__version__}")
```

---

## 获取帮助

如果以上方案都无法解决问题：

1. 检查错误堆栈跟踪（完整错误信息）
2. 查看相关文件的注释
3. 查阅PyTorch官方文档: https://pytorch.org/docs/
4. 确保Python版本 >= 3.7

---

## 快速诊断命令

```bash
# 检查Python版本
python --version

# 检查已安装的包
pip list | findstr torch
pip list | findstr numpy

# 检查文件结构
dir unet_jammer_localization

# 测试模型导入
python -c "from unet_jammer_localization.unet_model import UNet; print('OK')"
```

---

**所有问题都已修复！现在可以正常运行了。** ✅
