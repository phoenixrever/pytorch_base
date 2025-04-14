import os
import torch
import time
import csv
import gzip
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 模型超参数设置
# ==========================================
N_CHARS = 128  # 所有可能的字符种类（使用 ASCII 编码，因此最大为 128）
HIDDEN_SIZE = 100  # RNN 中隐藏状态的向量维度（即记忆容量）
N_LAYER = 1  # RNN 层数，这里是单层 GRU
USE_GPU = torch.cuda.is_available()  # 判断是否可以使用 GPU
BATCH_SIZE = 256  # 每次训练处理的数据条数
N_EPOCHS = 20  # 训练的轮数（每轮都看完整个训练集）


# ===============================
# 自定义数据集 NameDataset
# ===============================
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
data_dir = os.path.join(script_dir, "data")


class NameDataset(Dataset):
    """
    自定义的数据集类，继承自 PyTorch 的 Dataset。
    每一条数据由（人名, 所属国家）构成，用于训练 RNN 进行分类。
    """

    def __init__(self, is_train_set=True):
        # 根据是训练集还是测试集加载不同的数据文件
        filename = data_dir + '/names_train.csv.gz' if is_train_set else data_dir + '/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 提取人名和国家信息
        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.len = len(self.names)

        # 所有国家的列表（去重排序后）
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()  # 国家名称 -> 数字ID 的映射
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        # 返回一个数据点：(name, country_id)
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        # 返回国家名称到索引的映射
        return {country_name: idx for idx, country_name in enumerate(self.country_list)}

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num


# ===============================
# RNN 模型定义
# ===============================
class RNNClassifier(torch.nn.Module):
    """
    RNN 分类器模型，使用嵌入层 + 双向 GRU + 全连接层实现。
    输入是人名的字符序列，输出是预测的国家类别。
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        # 嵌入层：将字符序列（例如 [65, 66, 67]）映射为稠密向量
        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        # GRU（门控循环单元）是 RNN 的一种变体，适用于处理序列数据
        '''
         每个参数详细解释
          参数名	        含义	                                                       举例
          input_size	   输入特征的维度（每个时间步的输入长度）	                          如果你输入一个字符编码向量，大小就是 embedding 的维度
          hidden_size	   隐藏状态的维度（也是输出维度）	                                 例如设置为 128，就表示 GRU 内部状态和输出是 128 维
          num_layers	   堆叠多少层 GRU	                                               默认为 1，设置为 2 表示输出作为下一层 GRU 的输入
          bias	         是否使用偏置	                                                  通常保持 True 就行
          batch_first	   输入的维度是否是 (batch, seq_len, input_size)	                如果为 True，则输入格式需要这样排列
          dropout	       多层 GRU 间的 dropout（注意：仅在 num_layers > 1 时生效）	     可用于防止过拟合
          bidirectional	 是否使用双向 GRU	True                                          表示从前向后、从后向前(过去与未来)两个方向处理序列，然后拼接输出
       
       
       ❓GRU 为什么叫“门控”？
            GRU 内部结构主要有两个门：
              更新门（Update Gate）：决定当前时刻该保留多少前一时刻的信息
              重置门（Reset Gate）：决定当前时刻是否忽略过去的信息（适合短期记忆）
              它的结构比 LSTM 更简单，但性能也很接近甚至更快。
        '''
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)

        # 全连接层（分类器），用于输出分类结果
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        # 初始化隐藏状态为全 0 张量（根据层数和方向数扩展维度）
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        """
        input: [batch_size, max_seq_len]，表示字符序列的索引张量
        seq_lengths: [batch_size]，表示每个序列的实际长度
        """
        # 进行嵌入：将字符索引转换为向量
        embedded = self.embedding(input).transpose(0, 1)  # 变为 [seq_len, batch, hidden]
        hidden = self._init_hidden(input.size(0))

        # 使用打包函数加速计算（忽略填充部分）
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths)

        # 进入 GRU 网络处理
        output, hidden = self.gru(packed, hidden)

        # 如果是双向 GRU，拼接两个方向的最后输出作为特征
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]

        # 经过全连接层输出预测结果
        output = self.fc(hidden_cat)
        return output


# ===============================
# 辅助函数：张量处理
# ===============================
def create_tensor(tensor):
    # 判断是否需要将张量转移到 GPU
    if USE_GPU:
        return tensor.to(torch.device("cuda:0"))
    return tensor


def name2list(name):
    # 将字符串转换为 ASCII 编码列表，例如 'Bob' -> [66, 111, 98]
    arr = [ord(c) for c in name]
    return arr, len(arr)


def make_tensors(names, countries):
    """
    将一批名字和国家标签转换为张量：
    - 填充 name 序列
    - 按序列长度降序排列（适配 RNN）
    - 输出最终输入张量、长度张量、标签张量
    """
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 对序列进行降序排序（pack_padded_sequence 要求）
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


# ===============================
# 模型训练函数
# ===============================
def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        # 准备训练数据
        inputs, seq_lengths, target = make_tensors(names, countries)

        # 正向传播
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)

        # 反向传播 + 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 每 10 个 batch 打印一次训练信息
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} '
                  f'[{i * len(inputs)}/{len(trainset)}] '
                  f'loss={total_loss / (i * len(inputs))}')

    return total_loss


# ==========================================
# 模型测试函数
# ==========================================
def testModel():
    correct = 0
    total = len(testset)
    with torch.no_grad():
        for names, countries in testloader:
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / total * 100
    print(f"\n[Test Accuracy] {correct} / {total} = {acc:.2f}%\n")
    return acc


# ===============================
# 时间辅助函数
# ===============================
def time_since(since):
    s = time.time() - since
    m = int(s // 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # 初始化分类模型
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, 1, N_LAYER)

    # 将模型移动到 GPU（如果可用）
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    # 定义损失函数为交叉熵（常用于分类任务）
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器使用 Adam（自适应学习率）
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # 加载训练数据和测试数据（自定义的数据集）
    trainset = NameDataset(is_train_set=True)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = NameDataset(is_train_set=False)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # 获取国家总数（即输出类别总数）
    N_COUNTRY = trainset.getCountriesNum()
    # 重新定义最后一层输出层，根据国家数量调整输出维度
    classifier.fc = torch.nn.Linear(HIDDEN_SIZE * classifier.n_directions, N_COUNTRY)

    # 开始计时
    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []

    # 开始训练过程
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()  # 执行一轮训练
        acc = testModel()  # 在测试集上评估模型准确率
        acc_list.append(acc)
