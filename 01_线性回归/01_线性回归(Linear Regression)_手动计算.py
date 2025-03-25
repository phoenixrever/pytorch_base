import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0], requires_grad=True)


def forward(x):
    return w * x  # *被重载 x 自动转化未tensor类型


def loss(x, y):
    y_pred = forward(x)  # y hat
    return (y_pred - y) ** 2


# 前向传播 forward(4).item() 就是预测(predict) x 为4.0的时候 y的值是多少。
print('predict (before training)', 4, forward(4).item())

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        # 随时函数计算完成后会把梯度值存入 w 里面
        l = loss(x_val, y_val)
        # 模型会自动计算出每个权重的梯度。
        l.backward()
        # 通过梯度下降法来调整权重
        # w.data 是 w 的 实际数据（即权重的数值）
        # w.grad 是 梯度（或导数），它表示 损失函数 对 权重 w 的变化率。这个值是一个tensor
        # w.grad.data 是 w 对应的 梯度 的 数据data部分，表示当前的梯度值。
        # 0.01 是 学习率，它决定了每次调整权重的步长。学习率越小，权重变化越小，训练得越慢；学习率越大，权重变化越大，训练得越快（但可能不稳定）。
        # 梯度给出的变化速率可能非常大，因此我们不能直接用梯度来调整权重，而是需要乘一个小的常数来控制每次更新的幅度。这个常数就是 学习率。
        '''
        假设我们有一个简单的模型，权重 w 和损失函数之间的关系如下：

          假设损失函数对 w 的梯度（w.grad.data）是正数，比如 0.2。
          这表示，如果我们 增加 权重 w，损失会增大。因此，我们要 减小 权重来减少损失。所以用 减号：w.data -= 0.01 * 0.2，使得 w 变小，损失减小。
          如果梯度为负数，比如 -0.2：

          这表示，如果我们 增加 权重 w，损失会减小。为了继续减小损失，我们需要 增加 权重，所以 减号 再次会起作用：
          w.data -= 0.01 * (-0.2)，结果相当于 w.data += 0.01 * 0.2，即增加权重。
          
          所以 减号 是确保我们始终沿着梯度的 反方向 更新权重。
        '''
        w.data -= 0.01 * w.grad.data
        # 梯度清零 确保每次计算的梯度仅与当前小批量数据相关。
        w.grad.data.zero_()

    print('progress:', epoch, l.item())


print('predict (after training)', 4, forward(4).item())
