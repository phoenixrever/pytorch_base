import torch

'''
每个卷积kernel 只会生成一个channel
要生成多个channel,就需要多个卷积kernel

'''


# 定义输入通道数和输出通道数
in_channels, out_channels = 5, 10

# 定义输入图像的宽度和高度
width, height = 100, 100

# 定义卷积核（滤波器）的大小（3x3）
kernel_size = 3

# 定义批量大小（batch_size），即一次处理的样本数量
batch_size = 1

# 创建一个随机输入张量，形状为 (batch_size, in_channels, width, height)
# 这里的 torch.randn() 生成服从标准正态分布的随机数
input = torch.randn(batch_size, in_channels, width, height)

# 定义一个 2D 卷积层
# Conv2d 的参数：(输入通道数, 输出通道数, 卷积核大小)
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

# 通过卷积层处理输入数据 torch.nn.Conv2d 期望的输入格式是 (batch_size, in_channels, height, width)，即 4D(Dimension)(维) 张量 所有randn定义了4维张量
# 张量（Tensor） 是 多维数组，是 PyTorch 和 深度学习 中数据的基本结构。它类似于 NumPy 数组（ndarray），但支持 GPU 加速 和 自动求导。
output = conv_layer(input)

# 打印输入张量的形状
print("Input shape:", input.shape)  # (1, 5, 100, 100)

# 打印输出张量的形状
print("Output shape:", output.shape)  # 由于 padding 默认为 0，输出尺寸将缩小
# 计算输出的空间维度：
# Output_size = (Input_size - Kernel_size) / Stride + 1
# 这里 stride=1（默认值），所以输出大小 = (100 - 3) / 1 + 1 = 98
# 形状应为 (1, 10, 98, 98)

# 打印卷积层的权重形状
print("Conv layer weight shape:", conv_layer.weight.shape)  # (10, 5, 3, 3)
# 权重张量的形状解释：
# (输出通道数, 输入通道数, 卷积核高度, 卷积核宽度)
