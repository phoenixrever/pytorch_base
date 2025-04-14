import torch

# ========= 配置参数 =========
batch_size = 1      # 一次处理几个样本，这里只处理一个
seq_len = 3         # 序列长度，也就是时间步数 = 3
input_size = 4      # 每个时间步输入的特征维度是 4
hidden_size = 2     # RNN 隐藏状态的维度是 2
num_layers = 1      # RNN 堆叠层数，这里只用一层（最简单的情况）
'''
## 🧱 num_layers 为什么要用多层？

### ✅ 好处：

- **浅层学基础信息**（比如语音节奏、词语顺序）
- **中间层学上下文逻辑**（前后句之间的因果）
- **高层学抽象语义**（情感、主旨、意图等）

类似于 CNN 中的卷积层越往后提取越高级特征的原理。
层数越多，网络越深，学到的“理解”越复杂，但计算也更重，容易过拟合
'''

# ========= 定义 RNN 网络 =========
# torch.nn.RNN 表示一个多层的标准 RNN（非 LSTM、非 GRU）
cell = torch.nn.RNN(
    input_size=input_size,         # 每个时间步的输入特征维度
    hidden_size=hidden_size,       # 隐藏状态向量的维度
    num_layers=num_layers          # 层数
)

# ========= 构造输入数据 =========
# 输入形状：(时间步数 seq_len, 批大小 batch_size, 输入特征 input_size)
inputs = torch.randn(seq_len, batch_size, input_size)
# 举例：形状是 (3, 1, 4)，表示：
# - 总共有 3 个时间步
# - 每个时间步 1 个样本
# - 每个样本有 4 个特征

# ========= 初始化隐藏状态 =========
# 隐藏状态形状：(层数 num_layers, 批大小 batch_size, 隐藏维度 hidden_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)
# 举例：形状是 (1, 1, 2)，表示：
# - 第 0 层的初始隐藏状态
# - 对于 1 个样本
# - 每个样本的隐藏状态维度是 2

# ========= 执行 RNN 前向传播 =========
out, hidden = cell(inputs, hidden)

# out：是每个时间步的输出结果
#     形状为 (seq_len, batch_size, hidden_size)，也就是 (3, 1, 2)
#     表示每个时间步的输出（就是最后一层的隐藏状态）
# hidden：是“最后一个时间步”的隐藏状态（适合接下来的处理）
#     形状为 (num_layers, batch_size, hidden_size)，也就是 (1, 1, 2)

# ========= 打印输出 =========
print('Output size:', out.shape)     # 输出的形状：(3, 1, 2)
print('Output:', out)                # 每个时间步的输出
print('Hidden size:', hidden.shape)  # 最后时间步的隐藏状态形状：(1, 1, 2)
print('Hidden:', hidden)             # 最后时间步的隐藏状态值
