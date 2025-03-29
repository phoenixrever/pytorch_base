import os
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
    transforms.ToTensor(),  # 将PIL图像转换为 PyTorch 张量Ｃ(Chanel)ｘＨｘＷ，并归一化到 `[0,1]`，数据范围 [0,255] -> [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化后的值大约在 [-0.42, 2.82]
])

# 下载并加载 MNIST 数据集
'''
    D:/codepytorch_base 你是在这个目录下运行脚本的
    print(os.getcwd())
'''
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
data_dir = os.path.join(script_dir, "data")
train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

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
        '''
            中间加 ReLU（Rectified Linear Unit，修正线性单元）主要是为了引入非线性，使神经网络能够学习更复杂的映射关系。
            ReLU 的主要作用是引入非线性，使神经网络能够学习更复杂的模式，否则网络就会退化成一个简单的线性变换，无法发挥深度学习的优势。
            
            为什么选择 ReLU 而不是其他激活函数
                梯度消失问题较少：相比 sigmoid 和 tanh，ReLU 在正区间的梯度是 1，不容易出现梯度消失问题。
                计算简单：只需要取最大值（max(0, x)），计算量比 sigmoid 和 tanh 小。
                更深的网络更稳定：ReLU 在深层神经网络中表现较好，而 sigmoid 和 tanh 在深度增加时容易出现梯度消失，导致训练困难。
            `ReLU` 不是返回 **0-1** 的数，它的定义是：

            ReLU(x)=max(0,x)
            - 如果输入是正数，则输出不变。
            - 如果输入是负数，则输出变为 0。
            - 它不会把输出压缩到 0 到 1 之间，而是**保持正数部分不变，负数变成 0**。
            
           所以，经过 ReLU 后，输出仍然是 128 维，但其中的负数部分变成了 0，正数部分保持不变。
           
           一般来说，当你的神经网络有多个线性层（Linear）时，就应该在它们之间加上 ReLU 或其他非线性激活函数，这样才能让网络学到复杂的特征。
           
           
           隐藏层神经元128
           一般选择：
            小模型：64 或 128
            中等模型：256 或 512
            大模型：1024 及以上
                隐藏层太小（如 32） → 学习能力不够，分类效果可能变差。
                隐藏层太大（如 1024） → 计算量增加，可能会过拟合。
                
            增加隐藏层 vs. 增加神经元个数
                方式	                影响
                增加隐藏层数量	          提取更高级特征，提高模型表达能力，但计算量增加，可能导致梯度消失/爆炸
                增加单层神经元个数	      使单层学习能力更强，适用于简单任务，但模型可能不够深，难以学习复杂特征

            3. 选择合适的层数和神经元
                👉 如果数据简单（如 MNIST）
                     1~2 层隐藏层，每层 64~128 个神经元就足够。
                👉 如果数据复杂（如 CIFAR-10）
                    3~4 层隐藏层，每层 128~512 个神经元。
                👉 如果是深度学习（如人脸识别、语音识别）
                    5 层以上，可能还需要 CNN 或 Transformer。
                    
            **结论**
            - **层数越多，学习能力越强，但计算量增加，可能过拟合。**
            - **神经元个数越多，每层学习能力越强，但层数少可能不够深。**
            - **简单任务（MNIST）可以用 `1~2 层`，复杂任务可以用 `3~4 层`。**
        '''
        self.fc1 = nn.Linear(28*28, 128)  # 输入层到隐藏层，MNIST 图片是 28x28 像素
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(128, 10)  # 隐藏层到输出层，10 个类别

    def forward(self, x):
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
        # x = x.view(-1, 28*28)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 创建模型实例
model = NeuralNet()

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
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题

# 定义优化器（替代手动更新 w）
# torch.optim.SGD 是 PyTorch 提供的随机梯度下降（SGD, Stochastic Gradient Descent）优化器。
# 它用于自动更新模型的参数，避免我们手动计算 w -= 0.01 * w.grad 这样的操作。
'''
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
'''
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)  # 随机梯度下降优化器

'''
虽然 Adam（Adaptive Moment Estimation） 优化器具有自适应的学习率调整机制，但它仍然需要一个初始学习率（lr）作为基准。
Adam 会根据每个参数的梯度和二阶矩（梯度的平方）的估计动态地调整每个参数的学习率，但这个动态调整是相对于初始的学习率而言的。
'''
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型

'''
结论：
训练时：如果你没有调用 model.train()，默认会处于训练模式，这对训练过程没有太大影响。

评估时：如果你没有调用 model.eval()，那么模型在评估时的表现可能会受到影响，尤其是 Dropout 和 Batch Normalization 层的行为。

强烈建议：
即使在训练过程中没有显式调用 model.train()，在测试或评估阶段，最好始终使用 model.eval()，以确保 Dropout 和 Batch Normalization 在评估时的行为符合预期。

'''


def train():
    model.train()  # 设置为训练模式
    running_loss = 0.0
    for epoch in range(num_epochs):
        '''
            enumerate(train_loader)：
                batch_idx：当前批次的索引（从 0 开始递增）。
                    batch_idx 是 当前批次（batch）的索引，它表示当前迭代到的 mini-batch 是第几批。

                (images, labels)：当前批次的数据和对应的标签。

            train_loader 里的数据有 10,000 张图片。batch_size = 64（每次取 64 张）。 总共 10000 / 64 ≈ 156 个 batch。

            例
                当前 batch 索引（batch_idx）: 100, 图片批次大小: torch.Size([64, 1, 28, 28])
        '''
        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

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

# 测试模型


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
        for images, labels in test_loader:  # 没有 enumerate，不关心 batch 索引。
            outputs = model(images)
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
            _, predicted = torch.max(outputs, 1)  # 找到每个样本的最大输出概率的类别（即预测类别）
            # print (predicted)
            '''
                labels.size(0) 获取 labels 张量的第 0 维的大小，也就是 batch_size。
                    labels = torch.tensor([1, 2, 0, 1, 2, 1, 0, 2, ...])  # 共 64 个元素
                    print(labels.size(0))  # 输出: 64
            '''
            total += labels.size(0)  # labels.size(0) 获取当前批次中样本的数量（即批次大小），
            '''
                (predicted == labels) 是一个布尔张量，它表示预测类别是否与真实标签相同。

                    predicted == labels 的结果是一个形状为 [batch_size] 的布尔张量，表示每个样本的预测是否正确。

                    .sum() 会对布尔值进行求和，True 会被视为 1，False 会被视为 0。因此，.sum() 计算出当前批次中正确预测的样本数。

                    .item() 会将结果从一个单独的张量转换成 Python 数值，便于累加到 correct 变量。

                correct 累加了每个批次中正确预测的样本数。最终，correct 将包含模型在测试集上正确预测的总样本数。
            '''
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# 运行训练和测试
train()
test()  # Test Accuracy: 93.88%

# 93.88就上不去了 需要更多的 方法 比如图像特征提取 TODO 使用 CNN (卷积神经网络) 自动特征提取
