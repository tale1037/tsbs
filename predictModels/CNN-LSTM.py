from torch import nn


class CNN_lstm(nn.Module):
    def __init__(self, input_size, output_size, pre_len, hidden_size, n_layers, dropout=0.05, device='cuda'):
        super(CNN_lstm, self).__init__()
        self.pre_len = pre_len
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.layers = []
        self.layers.append(nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(kernel_size=2))
        self.cnn = nn.Sequential(*self.layers)
        self.lstm = nn.LSTM(hidden_size, self.hidden_size, n_layers, bias=True,
                           batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.device = device

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        out, ht = self.lstm(x)
        # print(out.shape)
        last_output = out[:, -self.pre_len:, :]
        H = self.linear(last_output)
        # print(len(H))
        return H

