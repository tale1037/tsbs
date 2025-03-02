import torch
from torch.utils.data import Dataset

class PredictionDataset(Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """
    def __init__(self, data ,pre_len):
        #print(data.shape)
        self.X = torch.FloatTensor(data[:, :-pre_len, :])
        #self.T = torch.LongTensor([t-1 if t == 100 else t for t in time])
        self.Y = torch.FloatTensor(data[:, -pre_len:, -1:])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



class TimeGANDataset(Dataset):
    """TimeGAN Dataset for sampling data with their respective time

    Args:
        - data (numpy.ndarray): the padded dataset to be fitted (D x S x F)
        - time (numpy.ndarray): the length of each data (D)
    Parameters:
        - x (torch.FloatTensor): the real value features of the data
        - t (torch.LongTensor): the temporal feature of the data
    """

    def __init__(self, data, time=None, padding_value=None):
        # sanity check
        if len(data) != len(time):
            raise ValueError(
                f"len(data) `{len(data)}` != len(time) {len(time)}"
            )

        if isinstance(time, type(None)):
            time = [len(x) for x in data]

        self.X = torch.FloatTensor(data)
        self.T = torch.LongTensor(time)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]

        # The actual length of each data
        T_mb = [T for T in batch[1]]

        return X_mb, T_mb
