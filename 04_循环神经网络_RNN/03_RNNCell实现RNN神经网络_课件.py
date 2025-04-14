import torch

# 参数定义
input_size = 4       # 输入向量维度（one-hot 编码的长度）
hidden_size = 4      # 隐藏层维度
batch_size = 1       # 一次只输入一个序列

# 字符索引和对应字符
idx2char = ['e', 'h', 'l', 'o']

# 训练数据（用数字表示字符）
x_data = [1, 0, 2, 2, 3]   # 输入序列：h, e, l, l, o
y_data = [3, 1, 2, 3, 2]   # 目标输出：o, h, l, o, l

# 独热编码查找表（手动定义）
one_hot_lookup = [
    [1, 0, 0, 0],  # e
    [0, 1, 0, 0],  # h
    [0, 0, 1, 0],  # l
    [0, 0, 0, 1],  # o
]

# 把 x_data 转换为独热向量
x_one_hot = [one_hot_lookup[x] for x in x_data]

# 转换成 PyTorch 的 Tensor，形状为 (seq_len, batch_size, input_size) seq_len = 5：因为输入了 5 个字符（h, e, l, l, o）。
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)

# 标签也转换为 Tensor，并 reshape 为 (seq_len, 1)
# view(-1, 1)：这会将 labels 的形状调整为 (seq_len, 1)，即每个标签都是一个独立的列向量。在这个例子中，y_data 有 5 个标签，所以形状最终会是 (5, 1)。
labels = torch.LongTensor(y_data).view(-1, 1)

# 定义模型


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()  # 注意这里要加括号！
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 使用 RNNCell（单步RNN）
        self.rnncell = torch.nn.RNNCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )

    def forward(self, input, hidden):
        # 每步更新隐藏状态
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        # 初始化隐藏状态为 0（形状为 batch_size × hidden_size）
        return torch.zeros(self.batch_size, self.hidden_size)


# 实例化模型
net = Model(input_size, hidden_size, batch_size)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# 训练循环
for epoch in range(15):  # 总共训练15次
    loss = 0
    optimizer.zero_grad()        # 梯度清零
    hidden = net.init_hidden()   # 每轮都重新初始化隐藏状态

    print('Predicted string: ', end='')

    # 逐步输入字符
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)       # 前向传播
        loss += criterion(hidden, label)  # 累加损失

        _, idx = hidden.max(dim=1)        # 取最大值对应的索引作为预测
        print(idx2char[idx.item()], end='')  # 显示预测字符

    loss.backward()       # 反向传播
    optimizer.step()      # 更新参数
    print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))
