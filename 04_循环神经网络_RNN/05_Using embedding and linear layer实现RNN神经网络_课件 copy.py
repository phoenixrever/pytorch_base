import torch

'''
### 总结

- **One-hot 编码**：每个字符都有一个唯一的编码，编码之间的差别是固定的。它无法体现字符之间的语义相似性。
- **Embedding 层**：通过上下文学习，Embedding 层可以将语义相似的字符映射到相似的向量空间中，这使得字符之间的关系得到体现，
例如 `'h'` 和 `'e'` 的向量会相对接近，因为它们常出现在类似的上下文中。

'''

# 定义参数
num_class = 4  # 类别数（即输出字符的种类数量，'e', 'h', 'l', 'o'）
input_size = 4  # 输入特征的大小，表示字符的索引的总数（One-hot编码的长度）
hidden_size = 8  # 隐藏层大小
embedding_size = 10  # 嵌入层的输出大小
num_layers = 2  # RNN层数
batch_size = 1  # 每次训练的批次大小
seq_len = 5  # 序列长度，即输入的字符序列长度

# 字符索引和字符对应的字典
idx2char = ['e', 'h', 'l', 'o']

# 输入数据：每个字符的索引（batch_size, seq_len）
x_data = [[1, 0, 2, 2, 3]]  # 对应字符 'h', 'e', 'l', 'l', 'o'

# 标签数据：目标字符的索引（batch * seq_len）
y_data = [3, 1, 2, 3, 2]  # 对应目标字符 'o', 'h', 'l', 'o', 'l'

# 转换为PyTorch的Tensor
inputs = torch.LongTensor(x_data)  # 输入数据
labels = torch.LongTensor(y_data)  # 标签数据

# 构建模型类


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 嵌入层（embedding layer），用于将输入的索引转换为稠密向量表示
        self.emb = torch.nn.Embedding(input_size, embedding_size)

        # RNN层：将嵌入层的输出传递到RNN
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)

        # 全连接层：将RNN的输出映射到指定类别数
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # 初始化隐藏状态（全为零）
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)

        # 通过嵌入层转换输入
        x = self.emb(x)  # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)

        # 传递输入到RNN层
        x, _ = self.rnn(x, hidden)  # (batch_size, seq_len, hidden_size)

        # 通过全连接层输出
        x = self.fc(x)  # (batch_size, seq_len, num_class)

        # reshape 输出为 (batch_size * seq_len, num_class)，以便计算损失
        return x.view(-1, num_class)


# 创建模型实例
net = Model()

# 定义损失函数：交叉熵损失函数用于多分类任务
criterion = torch.nn.CrossEntropyLoss()

# 定义优化器：Adam优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# 训练模型
for epoch in range(15):  # 训练15轮
    optimizer.zero_grad()  # 清空之前的梯度

    # 获取模型的输出
    outputs = net(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播
    loss.backward()

    # 更新模型参数
    optimizer.step()

    # 获取预测的最大值索引
    _, idx = outputs.max(dim=1)

    # 将索引转换为字符
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')

    # 打印当前的训练损失
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
