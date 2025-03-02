import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, input_size, output_size, pre_len, hidden_size, n_layers, dropout=0.05, device='cuda'):
        super(GRU, self).__init__()
        self.pre_len = pre_len
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, self.hidden_size, n_layers, bias=True, batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.device = device

    def forward(self, x):

        out, ht = self.gru(x)
        #print(out.shape)
        last_output = out[:, -self.pre_len:, :]
        H = self.linear(last_output)
        #print(len(H))
        return H
