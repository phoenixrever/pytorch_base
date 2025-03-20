import torch

# 训练数据
x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([2.0, 4.0, 6.0])

# 初始化参数 w，并启用自动梯度计算 创建一个张量 w，并告诉 PyTorch “我要对它求导”（requires_grad=True）。
w = torch.tensor(1.0, requires_grad=True)

# 定义前向传播（预测）函数


def forward(x):
    return w * x  # PyTorch 会自动处理张量计算

# 定义均方误差损失函数


def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2


# 在训练前测试预测结果
print('Predict (before training):', 4, forward(torch.tensor(4.0)).item())

# 定义优化器（替代手动更新 w）
# torch.optim.SGD 是 PyTorch 提供的随机梯度下降（SGD, Stochastic Gradient Descent）优化器。
# 它用于自动更新模型的参数，避免我们手动计算 w -= 0.01 * w.grad 这样的操作。
optimizer = torch.optim.SGD([w], lr=0.01)  # 采用随机梯度下降（SGD）

# 训练过程
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        # 计算预测值
        y_pred = forward(x_val)
        # 计算损失
        l = loss(y_pred, y_val)

        # 反向传播计算梯度
        l.backward()

        # 使用 SGD 更新 w更新权重 （优化器会自动调整 w，不需要手动 `w.data -= ...`）
        optimizer.step()

        # 梯度清零（防止梯度累积）
        optimizer.zero_grad()

    # 每 10 轮打印一次损失
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {l.item()}')

# 训练后测试预测结果
print('Predict (after training):', 4, forward(torch.tensor(4.0)).item())

'''
Predict (before training): 4 4.0
Epoch 0: Loss = 7.315943717956543
Epoch 10: Loss = 0.017410902306437492
Epoch 20: Loss = 4.143271507928148e-05
Epoch 30: Loss = 9.874406714516226e-08
Epoch 40: Loss = 2.3283064365386963e-10
Epoch 50: Loss = 9.094947017729282e-13
Epoch 60: Loss = 9.094947017729282e-13
Epoch 70: Loss = 9.094947017729282e-13
Epoch 80: Loss = 9.094947017729282e-13
Epoch 90: Loss = 9.094947017729282e-13
Predict (after training): 4 7.999998569488525


Epoch 0:   Loss = 7.3e+00  （7.3）
Epoch 10:  Loss = 1.7e-02  （0.017）
Epoch 20:  Loss = 4.1e-05  （0.000041）
Epoch 30:  Loss = 9.8e-08  （0.000000098）
Epoch 40:  Loss = 2.3e-10  （0.00000000023）
Epoch 50:  Loss = 9.1e-13  （0.00000000000091）
Epoch 60:  Loss = 9.1e-13  （和上面几乎一样）
Epoch 70:  Loss = 9.1e-13  
Epoch 80:  Loss = 9.1e-13  
Epoch 90:  Loss = 9.1e-13  



'''
