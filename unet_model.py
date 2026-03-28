"""
U-Net模型架构
用于图像分割任务的深度学习模型

为什么选择U-Net？
1. 保留空间位置信息：通过跳跃连接（skip connections）将编码器的特征直接传递给解码器
2. 精确定位：非常适合需要精确像素级定位的任务
3. 在小数据集上表现良好：相比其他深度网络，U-Net对训练样本数量要求较低
4. 对称架构：编码器提取特征，解码器重建空间分辨率，非常适合本任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块：MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块：UpSample -> Concat -> DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 使用双线性插值上采样 or 转置卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        参数:
            x1: 来自解码器的特征（低分辨率）
            x2: 来自编码器的跳跃连接特征（高分辨率）
        """
        x1 = self.up(x1)

        # 如果尺寸不匹配，进行padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        # 沿通道维度拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net模型

    架构：
        输入: (batch_size, 3, 64, 64) - 三通道特征图
        输出: (batch_size, 1, 64, 64) - 单通道概率掩码

    编码器路径（下采样）:
        64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4

    解码器路径（上采样 + 跳跃连接）:
        4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
    """

    def __init__(self, in_channels=3, out_channels=1, bilinear=True):
        """
        参数:
            in_channels: 输入通道数（3：SINR, Elite, Bottom）
            out_channels: 输出通道数（1：概率掩码）
            bilinear: True使用双线性插值上采样，False使用转置卷积
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 编码器（下采样路径）
        self.inc = DoubleConv(in_channels, 64)     # 64x64 -> 64x64 (64通道)
        self.down1 = Down(64, 128)                  # 64x64 -> 32x32 (128通道)
        self.down2 = Down(128, 256)                 # 32x32 -> 16x16 (256通道)
        self.down3 = Down(256, 512)                 # 16x16 -> 8x8 (512通道)

        # 瓶颈层
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)      # 8x8 -> 4x4 (512通道)

        # 解码器（上采样路径 + 跳跃连接）
        self.up1 = Up(1024, 512 // factor, bilinear)  # 4x4 -> 8x8
        self.up2 = Up(512, 256 // factor, bilinear)   # 8x8 -> 16x16
        self.up3 = Up(256, 128 // factor, bilinear)   # 16x16 -> 32x32
        self.up4 = Up(128, 64, bilinear)              # 32x32 -> 64x64

        # 输出层
        self.outc = OutConv(64, out_channels)         # 64x64 -> 64x64 (1通道)

    def forward(self, x):
        """
        前向传播

        参数:
            x: (batch_size, 3, 64, 64) 输入特征张量

        返回:
            logits: (batch_size, 1, 64, 64) 原始输出（未经sigmoid）
        """
        # 编码器路径 + 保存特征用于跳跃连接
        x1 = self.inc(x)      # (B, 64, 64, 64)
        x2 = self.down1(x1)   # (B, 128, 32, 32)
        x3 = self.down2(x2)   # (B, 256, 16, 16)
        x4 = self.down3(x3)   # (B, 512, 8, 8)
        x5 = self.down4(x4)   # (B, 512, 4, 4)

        # 解码器路径 + 跳跃连接
        x = self.up1(x5, x4)  # (B, 256, 8, 8)
        x = self.up2(x, x3)   # (B, 128, 16, 16)
        x = self.up3(x, x2)   # (B, 64, 32, 32)
        x = self.up4(x, x1)   # (B, 64, 64, 64)

        # 输出层
        logits = self.outc(x) # (B, 1, 64, 64)

        return logits

    def predict(self, x):
        """
        推理模式：返回概率掩码

        参数:
            x: (batch_size, 3, 64, 64) 输入特征张量

        返回:
            prob_mask: (batch_size, 1, 64, 64) 概率掩码 [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            prob_mask = torch.sigmoid(logits)
        return prob_mask


class UNetSmall(nn.Module):
    """
    轻量级U-Net（如果数据量较小或计算资源有限）

    通道数减半：32 -> 64 -> 128 -> 256 -> 512
    """

    def __init__(self, in_channels=3, out_channels=1, bilinear=True):
        super(UNetSmall, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # 瓶颈层
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        # 解码器
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)

        # 输出层
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            prob_mask = torch.sigmoid(logits)
        return prob_mask


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = UNet(in_channels=3, out_channels=1)
    print("="*60)
    print("U-Net 模型架构")
    print("="*60)
    print(model)
    print("\n" + "="*60)
    print(f"可训练参数数量: {count_parameters(model):,}")
    print("="*60)

    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, 64)
    print(f"\n测试输入形状: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)
        prob_mask = torch.sigmoid(output)

    print(f"输出 logits 形状: {output.shape}")
    print(f"概率掩码形状: {prob_mask.shape}")
    print(f"概率掩码值范围: [{prob_mask.min():.4f}, {prob_mask.max():.4f}]")

    # 测试轻量级模型
    print("\n" + "="*60)
    print("轻量级 U-Net 模型")
    print("="*60)
    model_small = UNetSmall(in_channels=3, out_channels=1)
    print(f"可训练参数数量: {count_parameters(model_small):,}")

    with torch.no_grad():
        output_small = model_small(dummy_input)
    print(f"输出形状: {output_small.shape}")
    print("="*60)
