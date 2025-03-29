import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类，继承自 PyTorch 的 Dataset

"""
这边不用double，是因为 游戏显卡2080 等只支持float32位的浮点数

比较贵的显卡才支持double类型的数据

第一个：表示所有行，第二： 后面-1 表示读取到-1 即最后一列不要

-1表示只读取-1列 【-1】 中括号保证是矩阵 不加的话拿出来是向量，x是矩阵 得保持一致

"""


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 读取 CSV 文件，并转换为 NumPy 数组
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)

        # 数据集的长度
        self.len = xy.shape[0]

        # 特征数据（输入）
        self.x_data = torch.from_numpy(xy[:, :-1])

        # 标签数据（输出）
        self.y_data = torch.from_numpy(xy[:, [-1]])

    '''
        # __getitem__ 是一个魔术方法（特殊方法），用于实现对象的索引访问（如 obj[key]）。当对象被索引时，Python 会自动调用这个方法。
        # 这边的0就是访问test_data 的第一个数据 说白了 魔术方法 就是把类作为数组用
    '''

    def __getitem__(self, index):
        # 获取指定索引的数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回数据集的大小
        return self.len


# 创建数据集对象
'''
 这边文件找不到 是因为 D:/codepytorch_base 你是在这个目录下运行脚本的 淡然找不到
 print(os.getcwd())
 获取脚本所在目录 __file__ REPL 下不可用
'''
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
diabetes_path = os.path.join(script_dir, "data/diabetes.csv.gz")

dataset = DiabetesDataset(diabetes_path)

"""
torch.utils.data.DataLoader 是 PyTorch 提供的一个数据加载器，它用于批量加载数据，提高训练效率，并支持多线程加载。
- `dataset`：数据集对象
- `batch_size=32`：每次加载 32 个(行)样本
    batch_size=32 每次从test_data中取32个(行)数据进行打包 dataset[0] dataset[1] dataset[2] dataset[3]

- `shuffle=True`：每个 epoch 之前打乱数据
- `num_workers=2`：使用 2 个子进程加载数据


"""
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

# 定义神经网络模型


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 定义三层全连接网络
        self.linear1 = torch.nn.Linear(8, 6)  # 输入层 -> 隐藏层1
        self.linear2 = torch.nn.Linear(6, 4)  # 隐藏层1 -> 隐藏层2
        self.linear3 = torch.nn.Linear(4, 1)  # 隐藏层2 -> 输出层
        self.sigmoid = torch.nn.Sigmoid()     # 激活函数

    """
        注意这边调用的sigmod和前面不一样，前面调用的是nn.Functional下的函数。
        这次直接是nn.sigmoid,它是一个运行模块继承自Module，可以直接作为一个神经元
        这边都写x是一个惯例，类似上面的层级 o1 o2 y hat 都用x 代替
    """

    def forward(self, x):
        # 前向传播
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 创建模型实例
model = Model()

# 定义损失函数（二元交叉熵损失）
'''
  定义损失函数（均方误差 MSE），使用 sum 进行累加 y hat -y 求平方，再把所有值加起来 MSELoss也是继承自Modul
  如果 loss.backward()，PyTorch 会根据计算图自动计算梯度，不需要手动求导。
  
  - **`reduction='mean'`（默认）** → **计算均值**（推荐，适用于标准梯度下降）。
  - **`reduction='sum'`** → **计算总和**（适用于 batch 大小变化时保持权重一致）。
  - **`reduction='none'`** → **不聚合，返回逐元素损失值**（适用于自定义损失计算）。
'''
criterion = torch.nn.BCELoss()


# 定义优化器（随机梯度下降）
# 定义优化器（替代手动更新 w）
# torch.optim.SGD 是 PyTorch 提供的随机梯度下降（SGD, Stochastic Gradient Descent）优化器。
# 它用于自动更新模型的参数，避免我们手动计算 w -= 0.01 * w.grad 这样的操作。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型 if __name__ == '__main__':  windows下 num_workers 多进程 会出错
if __name__ == '__main__':
    for epoch in range(100):  # 训练 100 轮
        for i, data in enumerate(train_loader, 0):
            '''
            每次for循环迭代都是取的batch_size = 32行数据
            # 1. 准备数据 data是 Dataset 类 __getitem__ 方法返回的内容。
            #  __getitem__ return self.x_data[index], self.y_data[index] 返回的是一个 元组 (inputs, labels)，

                如果 batch_size = 32，则：
                    inputs.shape = (32, 8) （32 行，每行 8 个特征）
                    labels.shape = (32, 1) （32 行，每行 1 个标签）
            '''
            inputs, labels = data  # 拆分输入和标签
            print(inputs.shape)
            print(labels)

            # 2. 前向传播 注意这边的输入并不是按行的 而是矩阵乘法 输入是个矩阵 输出还是一个矩阵 是整体
            y_pred = model(inputs)  # 计算模型输出
            loss = criterion(y_pred, labels)  # 计算损失
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

            # 3. 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度

            # 4. 更新参数
            optimizer.step()  # 应用梯度更新模型参数
