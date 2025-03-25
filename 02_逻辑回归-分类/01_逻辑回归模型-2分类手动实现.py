import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 训练数据 torch.Tensor 是老式写法不推荐
# torch.tensor([[0], [0], [1]]) 生成了 torch.int64，而 torch.Tensor([[0], [0], [1]]) 生成了 torch.float32
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0], [0], [1]], dtype=torch.float32)

# 定义一个简单的逻辑回归模型 和线性模型的区别就是加了激活函数


class LogisticRegression(nn.Module):
    def __init__(self):
        '''
          super().__init__() 必须调用，否则 nn.Module 的功能不会生效。
          不写 super().__init__()，模型会缺失 to(), state_dict() 等功能，可能会直接报错。
        '''
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # 和线性模型的区别就是加了激活函数
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


# 创建模型实例
model = LogisticRegression()

# 选择损失函数和优化器
criterion = nn.BCELoss()  # 适用于二分类问题，要求模型的输出是概率值（0~1），所以需要手动加 sigmoid
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型 1000 轮
for epoch in range(1000):
    # 1. 前向传播，计算预测值
    # model 也是继承Module的__call__  方法,对象加括号 () 相当于调用该对象的 __call__ 方法,最终会调用 forward 方法
    y_pred = model(x_data)
    # 2. 计算损失
    loss = criterion(y_pred, y_data)
    # 打印当前轮数和损失值
    print(epoch, loss.item())

    # 3. 反向传播和梯度更新
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

# 训练结束后，打印最终的权重和偏置
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# 使用训练好的模型进行预测
x_test = torch.tensor([[4]], dtype=torch.float32)
y_pred = model(x_test)
# 是一个旧的 API 返回 y_pred 的数据，但仍保持原有的形状，不参与梯度计算
# detach()：返回一个新的张量，该张量 不再与计算图相关联，不会追踪梯度，确保不破坏计算图。 分离计算图，然后转为 NumPy 数组
print('预测值:', y_pred.detach().numpy())
