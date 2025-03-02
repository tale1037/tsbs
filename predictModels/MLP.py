import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, pre_len, hidden_size, n_layers, dropout=0.05, device='cuda'):
        super(MLP, self).__init__()
        self.pre_len = pre_len  # 预测长度
        self.n_layers = n_layers  # 隐藏层数量
        self.hidden_size = hidden_size  # 隐藏层大小
        self.device = device

        # 定义多层感知机的结构
        self.hidden = nn.Linear(input_size, hidden_size)  # 输入到第一个隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.drop = nn.Dropout(dropout)  # Dropout 防止过拟合

        # 中间的隐藏层
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))  # 隐藏层到隐藏层
            self.layers.append(nn.ReLU())  # ReLU 激活
            self.layers.append(nn.Dropout(dropout))  # Dropout

        # 输出层
        self.linear = nn.Linear(hidden_size, output_size)  # 从隐藏层到输出层

    def forward(self, x):
        batch_size, obs_len, features_size = x.shape  # 获取输入数据的形状 (batch_size, obs_len, features_size)

        # 首先将输入通过第一个隐藏层
        x = self.hidden(x)  # (batch_size, obs_len, hidden_size)
        x = self.relu(x)
        x = self.drop(x)

        # 遍历并通过所有隐藏层
        for layer in self.layers:
            x = layer(x)  # (batch_size, obs_len, hidden_size)

        # 对每个时间步的输出应用 ReLU 激活
        x = self.relu(x)  # (batch_size, obs_len, hidden_size)

        # 输出预测
        x = self.linear(x)  # (batch_size, obs_len, output_size)

        # 只保留最后的 `pre_len` 个时间步的输出
        return x[:, -self.pre_len:, :]
