import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

"""
 CNN 的核心思想是：

    卷积层（Conv2d） 提取局部特征。

    池化层（MaxPool2d） 降低特征图的尺寸，减少计算量，提高模型的泛化能力。

    全连接层（Linear） 将提取到的特征映射到具体的类别。


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

        nn.Module 是 PyTorch 中的一个特殊类，它有自己的初始化逻辑，因此在子类 Net 中重写 __init__ 方法时，
        需要调用 super(Net, self).__init__() 来确保 nn.Module 的初始化逻辑被正确执行。

        和 Java 不同的是：
            - Python 不会自动调用父类 `__init__()`
            - 如果想要执行父类的初始化逻辑，必须手动调用 `super().__init__()`
        """
        super(Net, self).__init__()

        # 特征提取部分：包含两个卷积块
        self.feature_extractor = nn.Sequential(
            # 第一个卷积块
            # 输入通道数=1（灰度图像）
            # 输出通道数=10（10个不同的特征图）
            # kernel_size=5（5x5的卷积核）
            # 对于MNIST数据集(28x28)，卷积后的尺寸变为(28-5+1)x(28-5+1) = 24x24
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),  # 输入(1,28,28) -> 输出(10,24,24)

            # ReLU后保持形状不变，但激活了非线性特性
            # ReLU（Rectified Linear Unit，修正线性单元）主要是为了引入非线性，使神经网络能够学习更复杂的映射关系。
            nn.ReLU(),  # 激活函数

            # 最大池化层：
            # kernel_size=2（2x2的池化窗口）
            # 作用：将特征图的尺寸减半 除以kernel_size=2，同时保留最重要的特征
            # 每次执行后，特征图的高度和宽度都减半
            nn.MaxPool2d(kernel_size=2),  # 池化层 (10,24,24) -> (10,12,12)

            # 第二个卷积块
            # 输入通道数=10（来自conv1的输出）
            # 输出通道数=20（20个不同的特征图）
            # kernel_size=5（5x5的卷积核）
            # 在经过第一次池化后尺寸为12x12，卷积后变为(12-5+1)x(12-5+1) = 8x8
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),  # 输入(10,12,12) -> 输出(20,8,8)

            nn.ReLU(),  # 激活函数

            nn.MaxPool2d(kernel_size=2),  # 池化层 (20,8,8) -> (20,4,4)
        )

        # 分类器部分：包含全连接层
        '''
            x = x.view(-1, 28*28)  # 将 28x28 的图像展平为 1D 向量
            view() 主要用于改变张量的形状（shape），但不会更改数据本身。
                view(-1, 28*28) 是把 28×28 变成 784，变成二维矩阵 (batch_size, 784)
                PyTorch 自动推导 batch 维度， 使用x里面的 batch_size


            view() 替代方案
                x = x.reshape(-1, 28*28)
                x = x.flatten(start_dim=1)
                    flatten(start_dim=1) 从第 1 维开始展平，等价于 view(-1, 28*28)。


                x 是一个 MNIST 数据的 Tensor，形状是 [batch_size, 1, 28, 28]：
                x.shape  # torch.Size([64, 1, 28, 28])
                执行：x = x.flatten(start_dim=1)
                相当于：
                    保持 batch 维度（64）不变。
                    把后面的 [1, 28, 28] 展平成 [1 × 28 × 28] = [784]。
                所以，x 的新形状变成：
                    torch.Size([64, 784]) 等价于： x = x.view(64, 784)
        '''
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平操作 (20,4,4) -> (320)

            # 全连接层 进行最终分类：
            # 输入特征数=320（计算方法：20个通道 × 4×4的特征图 = 320）
            # 输出特征数=10（对应10个数字类别）
            # 对应的维度变换过程如下所示：
            # 𝑘𝑒𝑟𝑛𝑒𝑙 = 2 × 2
            # (𝑏𝑎𝑡𝑐ℎ, 20, 8, 8) → 经过池化 → (𝑏𝑎𝑡𝑐ℎ, 20, 4, 4) → 展平 → (𝑏𝑎𝑡𝑐ℎ, 320)
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
print(f"current device: {device}")


# 数据预处理，将图像转换为张量，并进行归一化
'''
归一化的作用：
1. **让数据范围变小，训练更快、更稳定。**
2. **避免数值太大导致梯度爆炸或收敛慢。**
3. **减少不同图片的亮度影响，让模型学到更通用的特征。**

归一化计算可以 在 GPU 上并行加速，因为 PyTorch 里的 Normalize() 是张量操作，它可以直接在 CUDA 核心 上运行，比 CPU 快得多。

如何选择 mean 和 std？
一种方法是 手动指定，比如 MNIST 数据用 (0.5, 0.5)，让数据变成 [-1, 1]。
另一种方法是 计算整个数据集的均值和标准差：
    mean = dataset.data.float().mean() / 255  # 计算均值
    std = dataset.data.float().std() / 255    # 计算标准差
    print(mean, std)  # MNIST 的标准值是 0.1307, 0.3081

Normalize((0.1307,), (0.3081,)) 不是 简单地归一化到 [0,1] 或 [-1,1]，而是标准化（Standardization），
即将数据转换为均值为 0、标准差为 1 的分布。

什么时候用哪种
- **训练 MNIST 模型** 👉 `Normalize((0.1307,), (0.3081,))` ✅ **推荐**
- **一般图像处理或简单归一化** 👉 `Normalize((0.5,), (0.5,))` 适用于非 MNIST 图像

★(0.1307,) 在 Python 里面是一个元素的元组（tuple）
'''
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

# 定义损失函数
'''
  定义损失函数（均方误差 MSE），使用 sum 进行累加 y hat -y 求平方，再把所有值加起来 MSELoss也是继承自Modul
  如果 loss.backward()，PyTorch 会根据计算图自动计算梯度，不需要手动求导。

  - **`reduction='mean'`（默认）** → **计算均值**（推荐，适用于标准梯度下降）。
  - **`reduction='sum'`** → **计算总和**（适用于 batch 大小变化时保持权重一致）。
  - **`reduction='none'`** → **不聚合，返回逐元素损失值**（适用于自定义损失计算）。

    PyTorch 的 CrossEntropyLoss() 自带 Softmax
        CrossEntropyLoss() = log_softmax() + NLLLoss()（负对数似然损失）
    这样神经网络的最后一部是不用做激活非线性变换的 生成概率分布的，这一步已经包含在交叉熵损失里面了
    具体可以看看课件

'''
criterion = nn.CrossEntropyLoss()

# 定义优化器（替代手动更新 w）
'''
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)  # 随机梯度下降优化器

如何调整这些参数

学习率 (lr)：
    如果训练的损失下降得很慢，可以尝试增大学习率。
    如果训练过程中损失波动较大或不稳定，尝试减小学习率。

动量 (momentum)：
    如果训练过程中出现了震荡或不收敛的情况，可以尝试使用较高的动量（如 momentum=0.9）。
    如果你认为优化过快导致了过拟合或收敛过早，可以使用较低的动量值（如 momentum=0.5）。 尤其是在较简单的任务中，如 MNIST。


你还可以尝试其他优化器，例如 Adam，它会自动调整学习率，并且通常能比 SGD 更加高效

什么时候选择 Adam？
    如果你没有对学习率进行细致调整的经验，或者你的任务是标准的图像分类任务（如 MNIST、CIFAR-10 等），并且模型较为复杂，那么 Adam 是一个很好的选择。
    对于不太复杂的任务，或者模型已经收敛较快时，可能可以尝试 SGD 或者 SGD + 动量（Momentum） 来进一步优化。

虽然 Adam（Adaptive Moment Estimation） 优化器具有自适应的学习率调整机制，但它仍然需要一个初始学习率（lr）作为基准。
Adam 会根据每个参数的梯度和二阶矩（梯度的平方）的估计动态地调整每个参数的学习率，但这个动态调整是相对于初始的学习率而言的。
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练函数


def train(epoch):
    model.train()  # 设置为训练模式
    # 用于累计损失值
    running_loss = 0.0
    # for epoch in range(num_epochs): 我在main里面写了 方便观察
    # 枚举训练数据加载器中的每个批次
    # enumerate返回索引和数据，从0开始计数
    # batch_idx：当前批次的索引（从 0 开始递增）。
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 将数据移到指定设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 清空梯度
        # 优化器在每次迭代前需要将梯度清零，否则梯度会累加
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

        '''
        解释：
            :：格式化说明符的起始部分，表示对后面的变量进行格式化。
            .4f：
                f：表示浮点数格式（fixed-point）。即将数字格式化为浮动小数点的表示。
                .4：指定小数点后显示的位数。数字 4 表示显示 四位小数。
        
        '''
        # 每100批mini-batch 输出loss 100批样本的loss平均值
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], 当前轮 Loss: {loss.item():.4f}, 100批平均 Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

# 测试函数


def test():
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    '''
    with torch.no_grad(): 表示在该代码块内的所有操作都不会计算梯度，离开该块后，梯度计算会恢复。
        在评估（测试）时，不需要进行梯度计算，因为我们不打算进行反向传播。
        关闭梯度计算会节省 显存 和 计算资源，从而加快推理速度并降低内存使用。
    
    with 语句是一种常用于资源管理的 Python 语法，这里用来确保在代码块内禁用梯度计算，执行完后自动恢复。
    with 语句的核心思想是 自动清理资源，无需显式地调用 close() 或 release() 等方法。
    
    # 使用 `with`
        with open('example.txt', 'r') as file:
            content = file.read()
        # 文件会在这里自动关闭，无需显式调用 file.close()

    '''
    with torch.no_grad():  # 关闭梯度计算，提高测试效率,测试时不需要计算梯度
        for inputs, targets in test_loader:  # 没有 enumerate，不关心 batch 索引。
            # 将数据移到指定设备
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)

            # 获取预测结果
            '''
            torch.max(outputs, 1) 返回 outputs 张量在 第 1 维（类别维度） 上的最大值和索引。具体来说：
                outputs 是一个形状为 [batch_size, num_classes] 的张量，表示每个样本对每个类别的预测得分。
                
                outputs 的形状：[batch_size, num_classes]，比如 [64, 10]（假设 batch size=64，类别数=10）。
                dim=1（第二个参数）：在 类别维度（num_classes） 上取最大值，即找出每个样本预测的类别索引。
                
                torch.max(outputs, 1) 返回两个值：
                    第一个返回值是每个样本在各个类别上的最大得分。
                    第二个返回值是最大得分所在的索引（即 预测类别）。
                    
                我们关心的是第二个返回值（predicted），它表示模型对每个样本的 预测类别。
            
            '''
            _, predicted = outputs.max(1)

            # 统计准确率
            '''
                labels.size(0) 获取 labels 张量的第 0 维的大小，也就是 batch_size。
                    labels = torch.tensor([1, 2, 0, 1, 2, 1, 0, 2, ...])  # 共 64 个元素
                    print(labels.size(0))  # 输出: 64
            '''
            total += targets.size(0)
            '''
                (predicted == labels) 或者 predicted.eq(targets) 是一个布尔张量，它表示预测类别是否与真实标签相同。

                    predicted == labels 的结果是一个形状为 [batch_size] 的布尔张量，表示每个样本的预测是否正确。

                    .sum() 会对布尔值进行求和，True 会被视为 1，False 会被视为 0。因此，.sum() 计算出当前批次中正确预测的样本数。

                    .item() 会将结果从一个单独的张量转换成 Python 数值，便于累加到 correct 变量。

                correct 累加了每个批次中正确预测的样本数。最终，correct 将包含模型在测试集上正确预测的总样本数。
            '''
            correct += predicted.eq(targets).sum().item()

    # 打印测试结果
    print(f'Accuracy on test set: {100.0 * correct / total:.2f}% [{correct}/{total}]')


# 训练模型
if __name__ == "__main__":
    num_epochs = 5
    # 正常情况下 for 写在 train 里面但是我要观察每轮的 test情况 所以再这写
    for epoch in range(num_epochs):
        train(epoch)
        # 在每个 epoch 结束后测试模型 可以看到每轮的 test 准确率上升情况
        test()

'''
训练过程中准确率先上升，然后在第 3 轮（Epoch 3）达到 98.53% → 98.44% 的小幅下降，之后又上升到 98.80%，最后降到 98.69%。

1. 过拟合
你训练的 Loss 下降了很多，但测试集准确率没有同步上升，这可能说明模型已经对训练集过拟合。

表现：

训练 Loss 在 5 轮内大幅下降，从 1.7125 降到 0.0862。
但测试集准确率 98.80% → 98.69% 出现轻微下降。

解决方案：
尝试添加 L2 正则化（weight_decay），减少过拟合：

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


增加 Dropout，在模型结构中增加：
self.dropout = nn.Dropout(p=0.5)  # 50% Dropout
减少训练轮数，可能 3~4 轮就够了。


.误差范围
98.53% → 98.44% → 98.80% → 98.69%
0.1%~0.2% 的浮动在深度学习中是正常的，属于统计误差。
一般不会太影响实际表现，特别是当测试集只有 10,000 张图片时，100 张图片的误差就会造成 1% 的波动。

第二次跑了下 全部上升的 确实是正常波动

****************将来实际跑的时候每次test达到最高点的时候就存盘，这样就能拿到准确率最高的哪个模型****************


什么时候需要额外的全连接层？
如果你的任务更复杂，比如：
    处理的是高分辨率图像（如 CIFAR-10、ImageNet）
    需要学到更丰富的特征（如复杂物体识别）
    训练数据量较大，可以承受更深的网络

这时候可以加入 额外的全连接层，例如：

nn.Linear(20*4*4, 100),  # 额外全连接层
nn.ReLU(),
nn.Linear(100, 50),
nn.ReLU(),
nn.Linear(50, 10)  # 最终分类
这种设计会增加模型的非线性能力，但训练起来也更慢，需要更多数据避免过拟合。


加全连接层 vs. 加卷积层，如何决定
全连接层的作用是 全局信息整合和分类，适用于：
    特征已经足够丰富，但模型的分类能力不足。

    数据集较小，需要更多参数去增强表达能力。

    任务要求复杂，比如：
        需要学习高维度的抽象特征（比如 ImageNet）。
        需要融合多个不同的特征（比如多模态任务）。



卷积层的作用是 提取局部特征，适用于：

    特征不足，模型学不到有用的信息。

    数据集比较复杂，图像包含更丰富的模式（如 CIFAR-10、ImageNet）。

    你发现：
        训练时准确率很低（表示特征不足）。
        你加了全连接层但效果没提升（说明特征不够）。


如何判断该加哪一种？

可以按以下步骤决定：

先加全连接层，看看效果
    如果准确率明显提升，说明原本分类能力不够。
    如果准确率几乎没变，说明特征不足 → 需要加卷积层。

检查训练误差
    训练误差高：模型太简单，应该加卷积层提取更多特征。
    训练误差低，但测试误差高：模型过拟合，应该减少全连接层或者加 Dropout。

观察特征图
    可以用 torchvision.utils.make_grid() 观察 CNN 提取的特征图。
    如果特征图看起来很随机，说明卷积层没学到东西 → 加深卷积层。
    
    
 经验法则
情况	                               方案
训练集小，特征已经够了，但分类能力不足	 加全连接层
训练集大，特征不够丰富	                加卷积层
训练误差高（模型太弱）	                加卷积层
训练误差低但测试误差高（过拟合）	     减少全连接层，增加 Dropout

总结
先试全连接层：如果分类能力不够，增加全连接层。
如果不够，再试卷积层：如果特征不足，增加卷积层。
不要一开始就堆太多全连接层，可能会过拟合或无效。
'''
