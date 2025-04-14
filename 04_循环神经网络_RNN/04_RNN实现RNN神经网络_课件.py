import torch

# 定义参数
input_size = 4  # 输入特征的维度（One-hot编码的长度）
hidden_size = 4  # 隐藏层的大小
num_layers = 1  # RNN的层数
batch_size = 1  # 每次训练的批次大小
seq_len = 5  # 序列的长度（在这个例子中是5，表示 "hello"）

# idx2char 用于将索引转换回字符
idx2char = ['e', 'h', 'l', 'o']

# x_data 是输入字符的索引
x_data = [1, 0, 2, 2, 3]  # 对应字符 'h', 'e', 'l', 'l', 'o'

# y_data 是目标字符的索引
y_data = [3, 1, 2, 3, 2]  # 对应目标字符 'o', 'h', 'l', 'o', 'l'

# One-hot 编码字典
one_hot_lookup = [[1, 0, 0, 0],  # 'e'
                  [0, 1, 0, 0],  # 'h'
                  [0, 0, 1, 0],  # 'l'
                  [0, 0, 0, 1]]  # 'o'

# 将 x_data 中的每个索引转换为 One-hot 编码
x_one_hot = [one_hot_lookup[x] for x in x_data]

# 转换为 PyTorch Tensor，形状为 (seq_len, batch_size, input_size)
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)

# 将标签数据转换为 LongTensor，形状为 (seq_len, 1)
labels = torch.LongTensor(y_data).view(seq_len, 1)

# 构建模型类


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        # 初始化参数
        self.num_layers = num_layers  # RNN层数
        self.batch_size = batch_size  # 批次大小
        self.input_size = input_size  # 输入特征大小
        self.hidden_size = hidden_size  # 隐藏层大小

        # 定义一个 RNN 模型
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers)

    def forward(self, input):
        # 初始化隐藏状态为零
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        # 将输入传入RNN，并得到输出
        out, _ = self.rnn(input, hidden)

        # 将输出展平为 (seq_len * batch_size, hidden_size)，以便后续处理
        return out.view(-1, self.hidden_size)

    def init_hidden(self):
        # 用来初始化隐藏状态
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


# 创建模型实例
net = Model(input_size, hidden_size, batch_size, num_layers)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 用于分类任务的交叉熵损失
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)  # Adam优化器

# 训练模型
for epoch in range(15):  # 训练15轮
    optimizer.zero_grad()  # 清空之前的梯度

    # 获取模型的输出
    outputs = net(inputs)

    # 计算损失
    loss = criterion(outputs, labels.view(-1))  # CrossEntropyLoss需要将标签展平

    # 反向传播
    loss.backward()

    # 更新模型参数
    optimizer.step()

    # 获取模型预测的字符索引
    _, idx = outputs.max(dim=1)

    # 将索引转换为字符
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')

    # 打印当前的训练损失
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
