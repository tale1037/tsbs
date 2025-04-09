import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange



def random_generator(batch_size, z_dim, max_seq_len):
    Z_mb = list()
    for i in range(batch_size):
        temp = np.random.uniform(0., 1, [max_seq_len, z_dim])
        Z_mb.append(temp)
    return Z_mb

def batch_generator(data, batch_size):
    """Mini-batch generator.
    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch
    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)

    return X_mb

def gen_batch(ori_data,seq_len,batch_size,device,input_dim):

    # Set training batch
    X= batch_generator(ori_data,batch_size)
    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    # Random vector generation
    Z = random_generator(batch_size, input_dim, seq_len)
    Z = torch.tensor(np.array(Z), dtype=torch.float32).to(device)
    return X,Z

class TimeGAN(nn.Module):
    def __init__(
            self,
            seq_len,
            module_name,
            feat_dim,
            latent_dim,
            num_layers,
            gamma,
            device,
            learning_rate,
            batch_size,
            networks_dir
            ):
        super(TimeGAN, self).__init__()
        self.seq_len = seq_len
        self.module_name = module_name
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.gamma = gamma
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.networks_dir = networks_dir

        self.encoder = Embedder("gru", self.feat_dim, self.seq_len,self.latent_dim,self.num_layers).to(self.device)
        self.decoder = Recovery("gru", self.feat_dim, self.seq_len,self.latent_dim,self.num_layers).to(self.device)
        self.supervisor = Supervisor("gru", self.feat_dim, self.seq_len,self.latent_dim,self.num_layers).to(self.device)
        self.discriminator = Discriminator("gru", self.feat_dim, self.seq_len,self.latent_dim,self.num_layers).to(self.device)
        self.generator = Generator("gru", self.feat_dim, self.seq_len,self.latent_dim,self.num_layers).to(self.device)

        self.MSELoss = torch.nn.MSELoss()
        self.BCELoss = torch.nn.BCELoss()


        self.optim_embedder = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.optim_recovery = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.optim_generator = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=self.learning_rate)
        self.optim_supervisor = torch.optim.Adam(self.supervisor.parameters(),
                                                 lr=self.learning_rate)

    def embedding_forward(self, X):
        H = self.encoder(X)
        X_tilde = self.decoder(H)
        H_hat_supervise = self.supervisor(H)
        return H,X_tilde,H_hat_supervise
    def generator_forward(self, Z):
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        X_hat = self.decoder(H_hat)
        return E_hat,H_hat,X_hat
    def discriminator_forward(self, H,H_hat,E_hat):
        Y_real = self.discriminator(H)
        Y_fake = self.discriminator(H_hat)
        Y_fake_e = self.discriminator(E_hat)
        return Y_real,Y_fake,Y_fake_e

    def train_embedder(self,batch, join_train=False):
        self.encoder.train()
        self.decoder.train()
        self.optim_embedder.zero_grad()
        self.optim_recovery.zero_grad()
        X = batch[0].to(self.device)
        H,X_tilde,H_hat_supervise = self.embedding_forward(X)

        E_loss0 = self.MSELoss(X, X_tilde)
        #E_loss0 = 10 * torch.sqrt(E_loss_T0 + 1e-6)

        if (join_train):
            G_loss_S = self.MSELoss(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            E_loss = E_loss0 + 0.1 * G_loss_S
            E_loss.backward()
        else:
            E_loss0.backward()
        self.optim_embedder.step()
        self.optim_recovery.step()
        return E_loss0
    def train_supervisor(self, batch, join_train=False):
        self.supervisor.train()
        self.generator.train()
        self.optim_supervisor.zero_grad()
        self.optim_generator.zero_grad()
        X = batch[0].to(self.device)
        H,X_tilde,H_hat_supervise = self.embedding_forward(X)
        G_loss_S = self.MSELoss(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        G_loss_S.backward()
        self.optim_supervisor.step()
        return G_loss_S

    def train_generator(self, batch, join_train=False):
        self.optim_generator.zero_grad()
        self.optim_supervisor.zero_grad()
        X = batch[0].to(self.device)
        H,X_tilde,H_hat_supervise = self.embedding_forward(X)
        Z = random_generator(self.batch_size, self.feat_dim, self.seq_len)
        Z = torch.tensor(np.array(Z), dtype=torch.float32).to(self.device)
        E_hat,H_hat,X_hat = self.generator_forward(Z)
        Y_real,Y_fake,Y_fake_e = self.discriminator_forward(H,H_hat,E_hat)
        G_loss_U = self.BCELoss(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = self.BCELoss(Y_fake_e, torch.ones_like(Y_fake_e))
        mean_X_hat = torch.mean(X_hat, dim=0)
        mean_X = torch.mean(X, dim=0)
        std_X_hat = torch.std(X_hat, dim=0, unbiased=False)  # `unbiased=False` 以匹配 TensorFlow
        std_X = torch.std(X, dim=0, unbiased=False)
        #G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        G_loss_S = self.MSELoss(H_hat_supervise[:, :-1, :], H[:, 1:, :])
        G_loss = G_loss_U + \
                      self.gamma * G_loss_U_e + \
                      torch.sqrt(G_loss_S) * 100 + \
                      G_loss_V * 100
        if not join_train:
            G_loss.backward()
        else:
            G_loss.backward(retain_graph=True)

        self.optim_generator.step()
        self.optim_supervisor.step()
        return G_loss

    def train_discriminator(self, batch, join_train=False):
        self.discriminator.train()
        self.optim_discriminator.zero_grad()
        X = batch[0].to(self.device)
        H,X_tilde,H_hat_supervise = self.embedding_forward(X)
        Z = random_generator(self.batch_size, self.feat_dim, self.seq_len)
        Z = torch.tensor(np.array(Z), dtype=torch.float32).to(self.device)
        E_hat,H_hat,X_hat = self.generator_forward(Z)
        Y_real,Y_fake,Y_fake_e = self.discriminator_forward(H,H_hat,E_hat)
        D_loss_real = self.BCELoss(Y_real, torch.ones_like(Y_real))
        D_loss_fake = self.BCELoss(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = self.BCELoss(Y_fake_e, torch.zeros_like(Y_fake_e))
        D_loss = D_loss_real + \
                      D_loss_fake + \
                      self.gamma * D_loss_fake_e
        # Train discriminator (only when the discriminator does not work well)
        if D_loss > 0.15:
            D_loss.backward()
            self.optim_discriminator.step()
        return D_loss
    def train_gan(self, train_data, data_name,seq_len ,max_epochs=1000,verbose=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        #数据转换，先将数据转换成tensor
        #然后创造一个dataset&dataloader方便训练
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        #选用的是adam的优化器
        #optimizer = optim.Adam(self.parameters())

        #开始训练
        logger = trange(max_epochs, desc=f"Epoch: 0, Loss: 0")
        print("Start Embedding Train")
        E_loss_T0 = None
        G_loss_S = None
        G_loss = None
        D_loss = None

        # Embedder Training Loop
        for epoch in logger:
            for batch in train_loader:
                # batch = gen_batch(train_data, self.seq_len, self.batch_size, device, self.feat_dim)
                E_loss_T0 = self.train_embedder(batch)
                # 输出 Embedder 损失
                logger.set_description(f'Epoch {epoch}/{max_epochs} | E_loss: {np.round(np.sqrt(E_loss_T0.item()), 4)}')
        print('Finish Embedding Network Training')

        logger1 = trange(max_epochs, desc=f"Epoch: 0, Loss: 0")
        print("Start Supervisor Train")
        # Supervisor Training Loop
        for epoch in logger1:
            for batch in train_loader:
                # batch = gen_batch(train_data, self.seq_len, self.batch_size, device, self.feat_dim)
                G_loss_S = self.train_supervisor(batch)
                # 输出 Supervisor 损失
            logger1.set_description(
                    f'Epoch {epoch}/{max_epochs} | G_loss_S: {np.round(np.sqrt(G_loss_S.item()), 4)}')
        print('Finish Supervisor Training')

        logger2 = trange(max_epochs, desc=f"Epoch: 0, Loss: 0")
        print("Start JOINT Train")
        # Joint Training Loop
        for epoch in logger2:
            count = 0
            for batch in train_loader:
                # for i in range(2):
                #     batch = gen_batch(train_data, self.seq_len, self.batch_size, device, self.feat_dim)
                #     G_loss = self.train_generator(batch)
                #     E_loss_T0 = self.train_embedder(batch)
                # batch = gen_batch(train_data, self.seq_len, self.batch_size, device, self.feat_dim)
                # D_loss = self.train_discriminator(batch)
                if count != 2:
                    G_loss = self.train_generator(batch)
                    count += 1
                    E_loss_T0 = self.train_embedder(batch)
                else:
                    D_loss = self.train_discriminator(batch)
                    count = 0

                # 输出 JOINT 训练阶段损失
            logger2.set_description(f'Epoch {epoch}/{max_epochs} | D_loss: {np.round(D_loss.item(), 4)} | '
                                        f'G_loss: {np.round(G_loss.item(), 4)} | G_loss_S: {np.round(np.sqrt(G_loss_S.item()), 4)} | '
                                        f'E_loss_T0: {np.round(np.sqrt(E_loss_T0.item()), 4)}')

        print('Finish Joint Training')
        self.save_trained_networks(data_name,seq_len)


    def gen_synth_data(self, batch_size,data):
        Z = random_generator(batch_size, self.feat_dim, self.seq_len)
        #self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        Z = torch.tensor(np.array(Z), dtype=torch.float32).to(self.device)
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        X_hat = self.decoder(H_hat)

        return X_hat

    def save_trained_networks(self,data_name,seq_len):
        print("Saving trained networks")
        torch.save(self.encoder.state_dict(), os.path.join(self.networks_dir, f'{data_name}-embedder-{seq_len}.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(self.networks_dir, f'{data_name}-recovery-{seq_len}.pth'))
        torch.save(self.generator.state_dict(), os.path.join(self.networks_dir, f'{data_name}-generator-{seq_len}.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.networks_dir, f'{data_name}-discriminator-{seq_len}.pth'))
        torch.save(self.supervisor.state_dict(), os.path.join(self.networks_dir, f'{data_name}-supervisor-{seq_len}.pth'))
        print("Done.")

    def load_trained_networks(self,data_name,seq_len):
        print("Loading trained networks")
        self.encoder.load_state_dict(torch.load(os.path.join(self.networks_dir, f'{data_name}-embedder-{seq_len}.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(self.networks_dir, f'{data_name}-recovery-{seq_len}.pth')))
        self.generator.load_state_dict(torch.load(os.path.join(self.networks_dir, f'{data_name}-generator-{seq_len}.pth')))
        self.discriminator.load_state_dict(torch.load(os.path.join(self.networks_dir, f'{data_name}-discriminator-{seq_len}.pth')))
        self.supervisor.load_state_dict(torch.load(os.path.join(self.networks_dir, f'{data_name}-supervisor-{seq_len}.pth')))
        print("Done.")


def get_rnn_cell(module_name):
    """Basic RNN Cell.
      Args:
        - module_name: gru, lstm
      Returns:
        - rnn_cell: RNN Cell
    """
    assert module_name in ['gru', 'lstm']
    rnn_cell = None
    # GRU
    if module_name == 'gru':
        rnn_cell = nn.GRU
    # LSTM
    elif module_name == 'lstm':
        rnn_cell = nn.LSTM
    return rnn_cell


class Embedder(nn.Module):

    def __init__(self, module_name, feat_dim, seq_len,latent_dim,num_layers):
        super(Embedder, self).__init__()
        rnn_cell = get_rnn_cell(module_name)
        self.rnn = rnn_cell(input_size=feat_dim, hidden_size=latent_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        e_outputs, _ = self.rnn(X)
        H = self.fc(e_outputs)
        H = self.sigmoid(H)
        return H

class Recovery(nn.Module):

    def __init__(self, module_name, feat_dim, seq_len,latent_dim,num_layers):
        super(Recovery, self).__init__()
        rnn_cell = get_rnn_cell(module_name)
        self.rnn = rnn_cell(input_size=latent_dim, hidden_size=feat_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(feat_dim, feat_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        r_outputs, _ = self.rnn(H)
        X_tilde = self.fc(r_outputs)
        X_tilde = self.sigmoid(X_tilde)
        return X_tilde

class Generator(nn.Module):

    def __init__(self, module_name, feat_dim, seq_len,latent_dim,num_layers):
        super(Generator, self).__init__()
        rnn_cell = get_rnn_cell(module_name)
        self.rnn = rnn_cell(input_size=feat_dim, hidden_size=latent_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Z):
        g_outputs, _ = self.rnn(Z)
        E = self.fc(g_outputs)
        E = self.sigmoid(E)
        return E

class Supervisor(nn.Module):

    def __init__(self, module_name, feat_dim, seq_len,latent_dim,num_layers):
        super(Supervisor, self).__init__()
        rnn_cell = get_rnn_cell(module_name)
        self.rnn = rnn_cell(input_size=latent_dim, hidden_size=latent_dim, num_layers=num_layers - 1,
                            batch_first=True)
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        s_outputs, _ = self.rnn(H)
        S = self.fc(s_outputs)
        S = self.sigmoid(S)
        return S

class Discriminator(nn.Module):

    def __init__(self, module_name, feat_dim, seq_len,latent_dim,num_layers):
        super(Discriminator, self).__init__()
        rnn_cell = get_rnn_cell(module_name)
        self.rnn = rnn_cell(input_size=latent_dim, hidden_size=latent_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        d_outputs, _ = self.rnn(H)
        Y = self.fc(d_outputs)
        Y = self.sigmoid(Y)
        return Y
