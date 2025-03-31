import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader


class Net(torch.nn.Module):
    """
    定义一个简单的卷积神经网络
    继承自torch.nn.Module基类，是所有神经网络模块的基类
    """

    def __init__(self):
        """
        初始化网络结构
        在这里定义网络的各个层
        """
        # 调用父类的初始化方法，这一步是必须的
        super(Net, self).__init__()

        # 第一个卷积层：
        # 输入通道数=1（灰度图像）
        # 输出通道数=10（10个不同的特征图）
        # kernel_size=5（5x5的卷积核）
        # 对于MNIST数据集(28x28)，卷积后的尺寸变为(28-5+1)x(28-5+1) = 24x24
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)

        # 第二个卷积层：
        # 输入通道数=10（来自conv1的输出）
        # 输出通道数=20（20个不同的特征图）
        # kernel_size=5（5x5的卷积核）
        # 在经过第一次池化后尺寸为12x12，卷积后变为(12-5+1)x(12-5+1) = 8x8
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)

        # 最大池化层：
        # kernel_size=2（2x2的池化窗口）
        # 作用：将特征图的尺寸减半，同时保留最重要的特征
        # 每次执行后，特征图的高度和宽度都减半
        self.pooling = torch.nn.MaxPool2d(2)

        # 全连接层：
        # 输入特征数=320（计算方法：20个通道 × 4×4的特征图 = 320）
        # 输出特征数=10（对应10个数字类别）
        # 对应的维度变换过程如下所示：
        # 𝑘𝑒𝑟𝑛𝑒𝑙 = 2 × 2
        # (𝑏𝑎𝑡𝑐ℎ, 20, 8, 8) → 经过池化 → (𝑏𝑎𝑡𝑐ℎ, 20, 4, 4) → 展平 → (𝑏𝑎𝑡𝑐ℎ, 320)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        """
        定义前向传播路径
        参数x是输入数据
        """
        # 获取当前批次的样本数量
        # x的形状为[batch_size, channels, height, width]
        '''
            labels.size(0) 获取 labels 张量的第 0 维的大小，也就是 batch_size。
            labels = torch.tensor([1, 2, 0, 1, 2, 1, 0, 2, ...])  # 共 64 个元素
            print(labels.size(0))  # 输出: 64
        '''
        batch_size = x.size(0)

        # 第一个卷积块：卷积+池化+ReLU激活
        # 输入：(batch, 1, 28, 28)，即原始图像
        # 卷积后：(batch, 10, 24, 24)
        # 池化后：(batch, 10, 12, 12)
        # ReLU后保持形状不变，但激活了非线性特性
        x = F.relu(self.pooling(self.conv1(x)))

        # 第二个卷积块：卷积+池化+ReLU激活
        # 输入：(batch, 10, 12, 12)
        # 卷积后：(batch, 20, 8, 8)
        # 池化后：(batch, 20, 4, 4)
        # ReLU后形状不变
        x = F.relu(self.pooling(self.conv2(x)))

        # 展平操作，将3D特征图转为1D向量
        # 输入：(batch, 20, 4, 4)
        # 输出：(batch, 320)
        # -1表示自动计算该维度的大小，保持元素总数不变
        x = x.view(batch_size, -1)  # flatten

        # 全连接层，进行最终分类
        # 输入：(batch, 320)
        # 输出：(batch, 10)，对应10个类别的得分
        x = self.fc(x)

        # 返回最终输出
        # 注意：通常不在这里应用softmax，因为交叉熵损失函数会在内部处理
        return x


# 创建模型实例
model = Net()

# 下载并加载 MNIST 数据集
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
# 应该在这里添加：model = model.to(device)，将模型移至指定设备


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

        # 这里应该添加设备转移：
        # inputs, target = inputs.to(device), target.to(device)

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

            # 应该添加设备转移：
            # inputs, target = inputs.to(device), target.to(device)

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

# 在主程序中应该添加以下内容：
# 1. 定义损失函数：criterion = nn.CrossEntropyLoss()
# 2. 定义优化器：optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 3. 训练循环：for epoch in range(10): train(epoch)
# 4. 测试模型：test()
