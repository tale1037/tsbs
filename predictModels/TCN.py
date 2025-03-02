class TCN(nn.Module):
    def __init__(self, input_shape, output_size=1, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8, 16], tcn_dropout=0.0, return_sequences=True, activation="linear", padding="causal", use_skip_connections=True, use_batch_norm=False, dense_layers=[], dense_dropout=0.0):
        super(TCN, self).__init__()
        self.tcn_layers = nn.ModuleList()
        input_dim = input_shape[-1]

        for dilation in dilations:
            self.tcn_layers.append(nn.Conv1d(input_dim, nb_filters, kernel_size=kernel_size, dilation=dilation, padding=padding))
            input_dim = nb_filters

        self.dense_layers = []
        if return_sequences:
            self.dense_layers.append(nn.Flatten())
        for units in dense_layers:
            self.dense_layers.append(nn.Linear(input_dim, units))
            self.dense_layers.append(nn.ReLU())
            if dense_dropout > 0:
                self.dense_layers.append(nn.Dropout(dense_dropout))
            input_dim = units

        self.dense_layers.append(nn.Linear(input_dim, output_size))
        self.dense_layers = nn.Sequential(*self.dense_layers)

    def forward(self, x):
        for tcn in self.tcn_layers:
            x = tcn(x)

        x = self.dense_layers(x)
        return x

    def compile(self, optimizer="adam", loss="mae"):
        self.optimizer = getattr(optim, optimizer)(self.parameters())
        self.loss_fn = nn.L1Loss() if loss == "mae" else nn.MSELoss()
