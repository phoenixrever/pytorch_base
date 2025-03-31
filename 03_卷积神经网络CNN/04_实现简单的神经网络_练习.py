import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

""" 
1. 使用Sequential模块组织网络结构
现代PyTorch代码通常使用nn.Sequential来组织相关的网络层，这使得代码更加整洁和模块化：

feature_extractor包含所有用于特征提取的卷积层、激活函数和池化层
classifier包含扁平化操作和最终的分类层

2. 直接使用内置的Flatten层
不再需要手动使用view()来展平特征图，而是使用PyTorch内置的nn.Flatten()层，这使代码更加简洁。
3. 更规范的训练/测试流程

使用model.train()和model.eval()明确切换模型状态
数据加载器使用更清晰的解包方式：for inputs, targets in train_loader
标准化了数据预处理流程，添加了MNIST常用的均值和标准差归一化

4. 更清晰的模型结构

将网络分为特征提取和分类两部分，更符合最新的深度学习架构设计理念
ReLU激活函数作为独立层，而不是函数调用，这使得模型结构更明确

5. 优化器更新
使用Adam优化器替代了传统的SGD，这通常能获得更好的收敛性能。
这种写法更加符合当前PyTorch的使用习惯，更易于阅读和维护，也更容易扩展到复杂的模型架构
"""


class Net(nn.Module):
    """
    定义一个简单的卷积神经网络
    使用现代PyTorch风格实现
    """

    def __init__(self):
        """
        初始化网络结构
        使用顺序模块组织网络层次
        """
        super(Net, self).__init__()

        # 特征提取部分：包含两个卷积块
        self.feature_extractor = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),  # 输入(1,28,28) -> 输出(10,24,24)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层 (10,24,24) -> (10,12,12)

            # 第二个卷积块
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),  # 输入(10,12,12) -> 输出(20,8,8)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层 (20,8,8) -> (20,4,4)
        )

        # 分类器部分：包含全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平操作 (20,4,4) -> (320)
            nn.Linear(in_features=20*4*4, out_features=10)  # 全连接层进行分类
        )

    def forward(self, x):
        """
        定义前向传播
        """
        # 通过特征提取器
        x = self.feature_extractor(x)

        # 通过分类器
        x = self.classifier(x)

        return x


# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 下载并加载 MNIST 数据集
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
data_dir = os.path.join(script_dir, "data")
train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

# 加载自定义数据
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建模型实例
model = Net().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练函数


def train(epoch):
    model.train()  # 设置为训练模式
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 将数据移到指定设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累计损失
        running_loss += loss.item()

        # 打印训练状态
        if batch_idx % 300 == 299:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss/300:.3f}')
            running_loss = 0.0

# 测试函数


def test():
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for inputs, targets in test_loader:
            # 将数据移到指定设备
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)

            # 获取预测结果
            _, predicted = outputs.max(1)

            # 统计准确率
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 打印测试结果
    print(f'Accuracy on test set: {100.0 * correct / total:.2f}% [{correct}/{total}]')


# 训练模型
if __name__ == "__main__":
    num_epochs = 5
    for epoch in range(num_epochs):
        train(epoch)
        test()
