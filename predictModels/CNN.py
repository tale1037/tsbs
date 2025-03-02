import torch
from pyod.utils import precision_n_scores
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_size, output_size, pre_len, hidden_size, n_layers, seq_len, dropout=0.05, device='cuda'):
        super(CNN, self).__init__()
        self.pre_len = pre_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_len = seq_len  # 输入序列的长度
        self.device = device

        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=3, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算卷积和池化后的输出尺寸
        self.last_dim = self.get_last_dim(input_size)

        # 定义全连接层
        self.fc1 = nn.Linear(self.last_dim, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, output_size)

    def forward(self, x):
        # 输入形状: (batch_size, input_size, seq_len)
        print(f"Input shape: {x.shape}")
        x = x.transpose(1, 2)  # 转置，使得形状变为 (batch_size, seq_len, input_size)

        # 卷积层 + 池化层
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积 + 池化
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积 + 池化

        # 展平数据
        x = x.view(x.size(0), -1)  # 展平

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # 最终输出

        return x

    def get_last_dim(self, input_size):
        with torch.no_grad():
            # 假设输入为一个形状为 (1, input_size, seq_len) 的张量
            x = torch.randn(1, input_size, self.seq_len)

            # 通过卷积层和池化层计算输出的特征维度
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            return x.shape[2]  # 返回 seq_len 维度的大小


from torch import nn


class CNN_lstm(nn.Module):
    def __init__(self, input_size, output_size, pre_len, hidden_size, n_layers,seq_len ,dropout=0.05, device='cuda'):
        super(CNN_lstm, self).__init__()
        self.pre_len = pre_len
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len-pre_len
        self.layers = []
        self.layers.append(nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3,stride=2, padding=1))
        #inputsize-3+2*1  //2
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(kernel_size=2))
        #self.layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*self.layers)
        self.last_dim = self.get_last_dim()
        self.lstm = nn.LSTM(self.last_dim, self.hidden_size, n_layers, bias=True,
                           batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.device = device

    def forward(self, x):
        #print(x.shape)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        #print(x.shape)
        out, ht = self.lstm(x)
        #print(out.shape)
        last_output = out[:, -self.pre_len:, :]
        H = self.linear(last_output)
        # print(len(H))
        return H

    def get_last_dim(self):
        with torch.no_grad():
            x = torch.randn(1, self.output_size, self.seq_len)
            for conv in self.layers:
                x = conv(x)
            return x.shape[2]

