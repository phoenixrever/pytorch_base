import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 ResNet  模块

"""
- **如果网络需要学习新的特征**，它可以通过 F(x)学习。
- **如果网络不需要改变输入**，它可以直接跳过（即F(x) = 0 时， y=x）
"""


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）：
    由两个 3x3 的卷积层组成，输入和输出通道数相同。
    通过残差连接（跳跃连接）使梯度更容易传播，
    缓解深度网络中的梯度消失问题。
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        # 第一层 3x3 卷积，保持输入通道数不变
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # 第二层 3x3 卷积，保持输入通道数不变
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        前向传播过程：
        1. 先经过第一层卷积并使用 ReLU 激活函数
        2. 再经过第二层卷积
        3. 将输入 x 和卷积后的 y 相加（跳跃连接）
        4. 再经过 ReLU 激活函数
        """
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)  # 残差连接后再 ReLU


class Net(nn.Module):
    """
    卷积神经网络（包含残差块）
    结构：
    1. 5x5 卷积层（输入通道 1，输出通道 16）
    2. 最大池化层（池化窗口 2x2）
    3. 残差块（输入输出通道 16）
    4. 5x5 卷积层（输入通道 16，输出通道 32）
    5. 最大池化层（池化窗口 2x2）
    6. 残差块（输入输出通道 32）
    7. 全连接层（输入 512，输出 10）
    """

    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积：输入通道 1，输出通道 16，卷积核 5x5
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        # 第二层卷积：输入通道 16，输出通道 32，卷积核 5x5
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        # 2x2 最大池化层
        self.mp = nn.MaxPool2d(2)
        # 第一个残差块（输入输出通道 16）
        self.rblock1 = ResidualBlock(16)
        # 第二个残差块（输入输出通道 32）
        self.rblock2 = ResidualBlock(32)
        # 全连接层，输入 512，输出 10
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        """
        前向传播过程：
        1. 经过第一层卷积 -> ReLU -> 最大池化（降维）
        2. 通过第一个残差块（保持尺寸和通道数）
        3. 经过第二层卷积 -> ReLU -> 最大池化（降维）
        4. 通过第二个残差块（保持尺寸和通道数）
        5. 将特征图展平（batch_size, 32, 4, 4） → (batch_size, 512)
        6. 经过全连接层输出类别概率
        """
        in_size = x.size(0)  # 获取 batch_size
        x = self.mp(F.relu(self.conv1(x)))  # 卷积 → ReLU → 池化
        x = self.rblock1(x)  # 通过第一个残差块
        x = self.mp(F.relu(self.conv2(x)))  # 卷积 → ReLU → 池化
        x = self.rblock2(x)  # 通过第二个残差块
        x = x.view(in_size, -1)  # 展平
        x = self.fc(x)  # 通过全连接层
        return x


# 创建模型实例
model = Net()

# 下载并加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为 PyTorch 张量Ｃ(Chanel)ｘＨｘＷ，并归一化到 `[0,1]`，数据范围 [0,255] -> [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化后的值大约在 [-0.42, 2.82]
])

script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
data_dir = os.path.join(script_dir, "data")
train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 设置计算设备
# 如果有GPU（cuda:0）则使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 将模型移至指定设备


def train(epoch):
    """
    训练一个epoch

    参数:
    - epoch: 当前训练的轮次
    """
    # 用于累计损失值
    running_loss = 0.0

    # 枚举训练数据加载器中的每个批次
    # enumerate返回索引和数据，从0开始计数
    for batch_idx, data in enumerate(train_loader, 0):
        # 解包数据，得到输入和目标
        inputs, target = data

        # 设备转移：
        inputs, target = inputs.to(device), target.to(device)

        # 清空梯度缓存
        # 优化器在每次迭代前需要将梯度清零，否则梯度会累加
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 参数更新

        # 前向传播：计算模型的预测输出
        outputs = model(inputs)

        # 计算损失：预测值与真实值之间的差异
        # criterion通常是交叉熵损失函数，适用于分类问题
        loss = criterion(outputs, target)

        # 反向传播：计算梯度
        loss.backward()

        # 参数更新：使用优化器根据梯度更新模型参数
        optimizer.step()

        # 累加批次损失
        running_loss += loss.item()

        # 每300个批次打印一次训练状态
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (
                epoch + 1,              # 当前epoch
                batch_idx + 1,          # 当前批次
                running_loss / 2000))   # 平均损失
            # 重置累计损失
            running_loss = 0.0


def test():
    """
    在测试集上评估模型性能
    """
    # 正确分类的样本数
    correct = 0
    # 总样本数
    total = 0

    # 使用torch.no_grad()上下文管理器
    # 在评估阶段不需要计算梯度，可以节省内存并加速计算
    with torch.no_grad():
        # 遍历测试数据集
        for data in test_loader:
            # 解包数据
            inputs, target = data

            # 设备转移：
            inputs, target = inputs.to(device), target.to(device)

            # 前向传播，获取预测输出
            outputs = model(inputs)

            # 获取预测的类别
            # torch.max返回每行的最大值及其索引
            # dim=1表示在第二个维度（类别维度）上求最大值
            # 返回值中 '_' 是最大值，predicted是对应的索引（类别）
            _, predicted = torch.max(outputs.data, dim=1)

            # 累加总样本数
            total += target.size(0)

            # 累加正确预测的样本数
            # (predicted == target)创建一个布尔张量
            # .sum()计算True的数量
            # .item()将单元素张量转换为Python数值
            correct += (predicted == target).sum().item()

    # 打印准确率
    print('Accuracy on test set: %d %% [%d/%d]' % (
        100 * correct / total,  # 准确率百分比
        correct,                # 正确分类的样本数
        total))                 # 总样本数


for epoch in range(10):
    train(epoch)
    test()
