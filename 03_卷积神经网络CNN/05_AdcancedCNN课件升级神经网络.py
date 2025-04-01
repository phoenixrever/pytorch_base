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

# 定义 InceptionA 模块

"""
为什么这样设计？

1x1 卷积：减少计算量，同时用于提取低级特征。

5x5 卷积：用于捕捉更大范围的信息，但计算开销较大，所以先用 1x1 降维。

两层 3x3 代替 5x5：相比于 5x5，两个 3x3 卷积能提取更丰富的特征，并且计算量更低。

池化层分支：用于获取全局特征，提高模型的鲁棒性。



为什么 Inception 适合这个任务？
不同尺度的特征提取：

  1x1、3x3、5x5 的组合可以同时捕获局部和全局信息，提高模型的泛化能力。

计算效率优化：

  1x1 卷积降维降低计算量。

  3x3+3x3 代替 5x5 进一步减少计算成本。

提升模型的表达能力：

  每个输入都会经过多个分支，不同分支提取的信息互补，提高特征表示能力。



"""


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()

        # 1x1 卷积分支（用于降维和提取局部特征）
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 5x5 卷积分支（先 1x1 降维，再 5x5 提取特征）
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)  # 先用1x1降维
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)  # 5x5提取较大范围特征

        # 3x3 双层卷积分支（使用两层 3x3 卷积代替 5x5，减少计算量）
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)  # 先用1x1降维
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)  # 3x3 提取局部特征
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)  # 继续 3x3 进一步提取信息

        # 池化 + 1x1 卷积分支 （用于全局信息）
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        # 1x1 分支
        branch1x1 = self.branch1x1(x)

        # 5x5 分支
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # 3x3 分支
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        # 平均池化 + 1x1 分支
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 拼接所有分支的输出（通道维度 dim=1）
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

# 定义整个网络


"""
为什么这样设计？

卷积层提取特征：

  conv1 提取基础特征，生成 10 个特征图。

  conv2 进一步提取特征，并进入 Inception 结构。

InceptionA 增强特征表达：

  incep1 通过多个分支并行提取不同尺度的特征，并输出 88 通道（16+24+24+24）。

  incep2 进一步提取特征，提高模型的能力。

池化层降维：  

  self.mp 通过最大池化减少特征图的大小，降低计算成本，并保留主要特征。

全连接层进行分类：

  self.fc 作用是将 1408 维的特征映射到最终的 10 类分类任务上。
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 第一层卷积（输入通道为 1，输出通道为 10）
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # 第一个 InceptionA 模块，输入通道为 10
        self.incep1 = InceptionA(in_channels=10)

        # 第二层卷积（需要匹配前一层的输出通道）
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # InceptionA 输出通道数为 88（16+24+24+24）

        # 第二个 InceptionA 模块，输入通道为 20
        self.incep2 = InceptionA(in_channels=20)

        # 最大池化层
        self.mp = nn.MaxPool2d(2)

        # 全连接层（1408 需要根据实际输入尺寸调整）
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)  # 获取 batch size

        # 第一层卷积 + ReLU + 最大池化
        x = F.relu(self.mp(self.conv1(x)))

        # 第一个 InceptionA
        x = self.incep1(x)

        # 第二层卷积 + ReLU + 最大池化
        x = F.relu(self.mp(self.conv2(x)))

        # 第二个 InceptionA
        x = self.incep2(x)

        # 展平为全连接层输入
        x = x.view(in_size, -1)

        # 全连接层分类
        x = self.fc(x)

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
