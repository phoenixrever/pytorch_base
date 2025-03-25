import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设置超参数
batch_size = 64  # 每个批次的样本数
learning_rate = 0.01  # 学习率
num_epochs = 5  # 训练的轮数

"""
归一化相当于把所有图片的亮度调整到一个标准，使得模型关注的是 形状、边缘等特征，而不是亮度本身。
具体见文档 https://phoenixhell.notion.site/1be76e0374d380e7b254fcc43b16e777
"""

# 数据预处理，将图像转换为张量，并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量，并归一化到 `[0,1]`，数据范围 [0,255] -> [0,1]
    transforms.Normalize((0.5,), (0.5,))  # 进一步归一化到 `[-1,1]`，提升训练效果和收敛速度。
])

# 下载并加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# 使用 DataLoader 加载数据集
# batch_size=4 每次从test_data中取64个数据进行打包 dataset[0] dataset[1] dataset[2] dataset[3]
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 定义一个简单的全连接神经网络


class NeuralNet(nn.Module):
    def __init__(self):
        '''
          super().__init__() 必须调用，否则 nn.Module 的功能不会生效。
          不写 super().__init__()，模型会缺失 to(), state_dict() 等功能，可能会直接报错。
        '''
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 输入层到隐藏层，MNIST 图片是 28x28 像素
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(128, 10)  # 隐藏层到输出层，10 个类别

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将 28x28 的图像展平为 1D 向量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 创建模型实例
model = NeuralNet()

# 选择损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器

# 训练模型


def train():
    model.train()  # 设置为训练模式
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# 测试模型


def test():
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，提高测试效率
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# 运行训练和测试
train()
test()
