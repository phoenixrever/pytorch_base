import torch

# 假设我们只有一个样本（batch_size = 1） 同时输入几个样本
batch_size = 1

# 输入序列的长度为3（也就是说，有3个时间步）
seq_len = 3

# 每个时间步的输入特征是4维的  每个时间步的输入特征维度
input_size = 4

# RNN 的隐藏状态设为2维 每个样本“记忆”的维度数（隐藏状态的长度）
hidden_size = 2

# 创建一个 RNNCell，它是最基本的 RNN 单元
# 它每次只处理一个时间步的输入
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# 构造一个随机的输入序列：形状是 (seq_len, batch_size, input_size)
# 也就是 3 个时间步，每个时间步是 1×4 的向量
dataset = torch.randn(seq_len, batch_size, input_size)

# 初始隐藏状态：形状是 (batch_size, hidden_size)，也就是 1×2
# 你只有一个样本（batch=1），它的隐藏状态是 [0.0, 0.0]（2维）。
# PyTorch 的 RNN 实现默认是 batch 是第一维（维度优先），这样做的好处是和大多数数据处理一致（比如 [batch_size, input_size] 是常见格式），方便向量化处理。
# for input in dataset 每次取出的是某个时间步上 所有 batch 的输入数据，也就是一个形状为 (batch_size, input_size) 的张量。
hidden = torch.zeros(batch_size, hidden_size)

# 遍历每个时间步的数据（按时间顺序处理）
for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)

    # 每一个 input 是一个时间步的输入，形状是 (batch_size, input_size)，即 1×4
    print('Input size: ', input.shape)

    # 把当前输入和上一个隐藏状态传入 RNNCell，得到当前的隐藏状态（也就是当前的输出）
    hidden = cell(input, hidden)

    # 输出当前隐藏状态的形状，应该是 (batch_size, hidden_size)，即 1×2
    print('outputs size: ', hidden.shape)

    # 输出当前时间步的隐藏状态（输出）
    print(hidden)

'''
==================== 0 ====================
Input size:  torch.Size([1, 4])
outputs size:  torch.Size([1, 2])
tensor([[-0.0293, -0.2649]], grad_fn=<TanhBackward0>)
==================== 1 ====================
Input size:  torch.Size([1, 4])
outputs size:  torch.Size([1, 2])
tensor([[0.9556, 0.9806]], grad_fn=<TanhBackward0>)
==================== 2 ====================
Input size:  torch.Size([1, 4])
outputs size:  torch.Size([1, 2])
tensor([[0.9747, 0.8884]], grad_fn=<TanhBackward0>)


# grad_fn 全称叫做 Gradient Function（梯度函数） 
这个张量是通过某个操作计算出来的结果，而这个 grad_fn 就记录了它是怎么来的，这样在反向传播时 PyTorch 就知道怎么计算它的梯度了。

 grad_fn=<TanhBackward0> 这个输出是通过一个 tanh 函数变换出来的
 
可以执行 .detach()（断开梯度链）或者 .item()（变成纯数值）

例子
import torch
x = torch.randn(2, 2, requires_grad=True)  # 让它能求梯度
y = x * 2
z = y.sum()

print(y.grad_fn)   # <MulBackward0> 表示 y 是通过乘法得到的
print(z.grad_fn)   # <SumBackward0> 表示 z 是通过 sum 得到的

'''
