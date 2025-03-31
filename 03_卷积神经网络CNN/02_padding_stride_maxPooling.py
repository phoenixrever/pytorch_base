import torch  # 导入 PyTorch

'''
3x3的卷积核要保证卷积之后的大小不不变padding就是3/2=1 
5x5 就是 5/2=2

padding就是增加卷积之后的尺寸
相对的 stride(步长) 就是减少卷积之后的尺寸

torch.nn.MaxPool2d 作用是：
    对输入进行「最大池化（Max Pooling）」，减少数据尺寸，同时保留重要特征。
    它会取局部区域的最大值，这样能 增强特征，减少不重要的信息。
    
maxPooling 就是比如把4x4的图像分成2x2的4组 每组取个最大值，最后生成的为2x2的图像，缩小了一半，注意是在同一个channel中

最大池化的工作原理
    kernel_size=2 表示用 2×2 的窗口 在输入数据上滑动，每次取窗口内的 最大值，然后输出。
    这样可以 降低数据尺寸（减少计算量）并 增强特征的鲁棒性（去掉不重要的信息）。

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
output = maxpooling_layer(input)
print(output)

MaxPool2d 的作用总结
    减少计算量：减少数据尺寸，提高 CNN 计算效率。
    保留重要特征：只保留局部区域的最大值，减少噪声干扰。
    提升模型的平移不变性：即使输入稍微移动，最大值仍能有效表示特征。
    📌 在 CNN 里，MaxPooling 通常用于卷积层后，降低数据尺寸，同时保留最显著的特征！ 🚀
'''

# 创建一个 5x5 的输入数据（列表格式）
input = [
    3, 4, 6, 5, 7,
    2, 4, 6, 8, 2,
    1, 6, 7, 8, 4,
    9, 7, 4, 6, 2,
    3, 7, 5, 4, 1
]

# 将输入数据转换为 PyTorch 张量，并调整形状为 (batch_size=1, channels=1, height=5, width=5)
input = torch.Tensor(input).view(1, 1, 5, 5)

# 创建一个 2D 卷积层，输入通道数=1，输出通道数=1，卷积核大小=3x3，padding=1（保持尺寸不变），不使用偏置
'''
在数学计算中，偏置就是一个额外的值，它可以让结果整体上移或下移，而不改变计算的主要模式。

### **什么时候用偏置？**

✅ **需要偏置的情况**

- **小数据集或简单任务**：可能需要偏置来调整数值范围
- **没有批量归一化（BatchNorm）**：如果没有额外的层来调整数值，偏置可以帮助学习

❌ **不需要偏置的情况**

- **CNN 网络里通常不加偏置**（因为 BatchNorm 会自动调整数值）
- **自定义卷积核进行特定计算（如边缘检测）**：不需要偏置，否则会影响结果

'''
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

'''
stride=2  5x5的图像就卷积后变成了2x2
'''
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)


# 定义 3x3 的卷积核（权重），数值从 1 到 9
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(
    1, 1, 3, 3)  # (out_channels=1, in_channels=1, height=3, width=3)

'''
在 PyTorch 中，torch.nn.Conv2d 默认会随机初始化卷积核（权重），如果不手动赋值，卷积核的值是不确定的。
因此，这一步的作用是 替换掉随机初始化的卷积核，使用我们自定义的卷积核(数值从 1 到 9)进行计算。
'''
conv_layer.weight.data = kernel.data  # 将自定义的卷积核赋值给卷积层的权重

# 对输入数据应用卷积层，得到输出
output = conv_layer(input)

# 打印卷积后的结果
print(output)
