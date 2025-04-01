import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）：
    由两个 3x3 的卷积层组成，并使用跳跃连接（Skip Connection）。
    作用：提高深层网络的训练效果，避免梯度消失问题。
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 处理通道不匹配的情况

    def forward(self, x):
        identity = x  # 记录输入值，以便跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)  # 调整输入尺寸以匹配输出

        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积 + BN + ReLU
        out = self.bn2(self.conv2(out))  # 第二个卷积 + BN
        out += identity  # 残差连接
        return F.relu(out)  # 经过 ReLU 激活后输出


class ResNet18(nn.Module):
    """
    ResNet-18 结构：
    - 采用 `nn.Sequential` 组织整个网络结构
    - 适用于分类任务，默认输出 1000 个类别（可调整 `num_classes`）
    """

    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 初始大卷积层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 最大池化层，减少计算量
            self._make_layer(64, 64, 2),  # 第一层残差块（两个 64 通道的块）
            self._make_layer(64, 128, 2, stride=2),  # 第二层残差块（输入 64，输出 128）
            self._make_layer(128, 256, 2, stride=2),  # 第三层残差块（输入 128，输出 256）
            self._make_layer(256, 512, 2, stride=2),  # 第四层残差块（输入 256，输出 512）
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，转换为 1x1 特征图
        )
        self.fc = nn.Linear(512, num_classes)  # 最终的全连接层

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        构建残差块层（包含多个 ResidualBlock）。
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param blocks: 该层包含多少个残差块
        :param stride: 第一个残差块的步长（用于下采样）
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # 当输入通道数与输出通道数不一致时，需要使用 1x1 卷积进行调整
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]  # 第一个残差块（可能包含通道匹配）
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))  # 后续的残差块

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播逻辑。
        """
        x = self.features(x)  # 经过所有特征提取层
        x = torch.flatten(x, 1)  # 将 1x1 的特征图展平为向量
        x = self.fc(x)  # 通过全连接层进行分类
        return x
