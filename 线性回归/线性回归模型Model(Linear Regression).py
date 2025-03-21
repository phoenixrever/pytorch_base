import torch

'''
继承torch.nn.Module 是 PyTorch 中所有神经网络模型的基类，提供了参数管理、模型存储等功能。

'''
# 定义线性回归模型
# 继承torch.nn.Module 是 PyTorch 中所有神经网络模型的基类，提供了参数管理、模型存储等功能。


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        # 调用 torch.nn.Module 的构造方法，初始化父类。
        # Python 2 需要 super(类名, self).__init__() 明确指定类名。
        # super(LinearRegressionModel, self).__init__()
        super().__init__()

        '''
          定义一个线性层（Linear 层）。 类的后面加括号相当于new了一个对象 linear就是一个Linear对象的实例
              线性层的计算公式是：
                y=wx+b
                
              其中：
              w 是权重（weight），模型训练时会自动更新。
              b 是偏置（bias），用于调整输出，使模型更灵活。
              
              linear对象里面就包含了权重w和偏置b两个参数，Linear也是继承与Model的也能自动进行反向传播
          
          (1, 1)参数的含义：
              torch.nn.Linear(in_features, out_features) 创建一个全连接层（FC层），这里：
                
              in_features=1：输入特征的维度是 1（即每个输入样本只有一个数值）。
                比如：
                x = torch.tensor([[2.0], [3.0], [4.0]])
                x 的形状是 (3, 1)，即 3 个样本，每个样本有 1 个特征。
                
                如果 in_features=2，那么每个输入样本必须是两个数（比如 [x1, x2] 这样的输入）。

              out_features=1：输出特征的维度是 1（即模型输出也是一个数值）。
                因为我们希望模型输出也是一个数，即：
                y = wx + b
                如果 out_features=2，那么模型就会输出两个数，变成：
                [y1​,y2​]=W⋅x+B
                其中 W 就是一个2x1的矩阵，意味着我们拟合的是两个输出值的情况，而不是一元线性回归。
          '''
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        '''
        在 Python 语法中，对象加括号 () 相当于调用该对象的 __call__ 方法，
        而在 PyTorch 中，torch.nn.Module 的所有子类（包括 torch.nn.Linear）都重载了 __call__ 方法，
        使得对象本身可以像函数一样被调用。

        在 PyTorch 的 torch.nn.Module 源码中：
          # *args 和 **kwargs 只是用来接收多个参数的语法糖，把它们收集成一个元组(1,2,3,4) 和一个字典}{'x':1,'y':2}
          # 比如 fun(1,2,3,x=1,y=2) args 代表1 2 3 不确定的参数，kwargs 代表 x=1 y=2 键值对参数
          # 当你只传了一个 x 时：Python 解释器会自动把 x 放入 *args 里
          def __call__(self, *args: Any, **kwargs: Any)
            # 这边会调用我自己重写的 forward 方法
            return self.forward(*input, **kwargs)

        可以看到，__call__ 方法最终会调用 forward 方法，所以：
        y_pred = self.linear(x)  等价于  y_pred = self.linear.forward(x)

        '''
        y_pred = self.linear(x)
        return y_pred


'''
x_data = torch.tensor([1.0, 2.0, 3.0])  # shape: (3,)
创建的是 一维张量，形状是 (3,)

但 PyTorch torch.nn.Linear(in_features, out_features) 需要输入二维张量，即 (batch_size, in_features) 的形状。
当你把 x_data 传入 model(x_data) 时： 
  torch.nn.Linear(1,1) 期望的输入形状是 (batch_size, 1)
  但你传入的 x_data 是 (3,)，形状不匹配
  PyTorch 试图把它当作 (1, 3)，即 1 行 3 列
  Linear(1,1) 只有 1 个输入特征，但 (1,3) 有 3 个输入特征
  导致矩阵乘法维度不匹配，触发错误
  
(1,3) 和 (1,1) 不能相乘，所以报错。


正确的写法 
x_data = torch.tensor([[1.0], [2.0], [3.0]])  # shape: (3,1)

即：
tensor([[1.0], 
        [2.0], 
        [3.0]])  # 形状: (3,1)
变成了二维张量 (3,1)，满足 torch.nn.Linear(1,1) 的输入要求。

矩阵乘法 (3,1) × (1,1) = (3,1) 计算正常 中间的1必须相等才能乘  
矩阵 A 的列数必须等于矩阵 B 的行数，才能进行乘法。详细见文档 矩阵乘法

'''
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# 创建模型实例
model = LinearRegressionModel()

'''
  定义损失函数（均方误差 MSE），使用 sum 进行累加 y hat -y 求平方，再把所有值加起来 MSELoss也是继承自Modul
  如果 loss.backward()，PyTorch 会根据计算图自动计算梯度，不需要手动求导。
  
  - **`reduction='mean'`（默认）** → **计算均值**（推荐，适用于标准梯度下降）。
  - **`reduction='sum'`** → **计算总和**（适用于 batch 大小变化时保持权重一致）。
  - **`reduction='none'`** → **不聚合，返回逐元素损失值**（适用于自定义损失计算）。
'''
criterion = torch.nn.MSELoss(reduction='sum')


# 定义优化器（随机梯度下降 SGD）对权重和偏置做优化(更新)，学习率 lr=0.01
# model.parameters() 返回递归遍历模型的所有可学习参数 调用linear.parameters() 获取 linear.weight 和 linear.bias
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型 1000 轮
for epoch in range(1000):
    # 1. 前向传播，计算预测值 model 也是继承Module的__call__ 最终调用forward
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
x_test = torch.tensor([[4.0]])
y_pred = model(x_test)
# 是一个旧的 API 返回 y_pred 的数据，但仍保持原有的形状，不参与梯度计算
# detach()：返回一个新的张量，该张量 不再与计算图相关联，不会追踪梯度，确保不破坏计算图。 分离计算图，然后转为 NumPy 数组
print('预测值:', y_pred.detach().numpy())
