import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def MinMaxScaler(data):
    """Min Max normalizer.
    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        """计算数据的最小值和最大值"""
        self.min = np.min(data, 0)
        self.max = np.max(data, 0)

    def transform(self, data):
        """归一化数据"""
        numerator = data - self.min
        denominator = self.max - self.min
        return numerator / (denominator + 1e-7)

    def inverse_transform(self, norm_data):
        """反归一化数据"""
        return norm_data * (self.max - self.min) + self.min

def real_data_loading(args,isaug):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: stock or energy
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    data_dir = args.data_dir
    data_name = args.data_name
    seq_len = args.seq_len
    if isaug:
        seq_len = seq_len*args.augargs.seq_len_times
    data_len_limit = args.data_len_limit
    assert data_name in ['stock', 'energy','ETTm1','ETTh1','ETTh2','ETTm2']

    ori_data = []
    if data_name == 'stock':
        ori_data = np.loadtxt(os.path.join(data_dir, 'stock_data.csv'), delimiter=",", skiprows=1)
        ori_data = ori_data[:data_len_limit]
        print(ori_data.shape)
    elif data_name == 'energy':
        ori_data = np.loadtxt(os.path.join(data_dir, 'energy_data.csv'), delimiter=",", skiprows=1)
        ori_data = ori_data[:data_len_limit]
    elif data_name =="ETTm1" or data_name =="ETTh1" or data_name =="ETTh2" or data_name =="ETTm2":
        #ori_data ==np.loadtxt(os.path.join(data_dir, 'Ettm1.csv'), delimiter=",", skiprows=1)
        data = pd.read_csv(os.path.join(data_dir, f"{data_name}.csv"))
        #print(data)
        data = data.drop(columns=['date'])
        data = data.iloc[1:data_len_limit]
        ori_data = data.values
    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    #print(ori_data)
    # Normalize the data
    scaler = MinMaxScaler()
    scaler.fit(ori_data)
    #scaler.fit(ori_data)
    ori_data = scaler.transform(ori_data)
    #ori_data = MinMaxScaler(ori_data)
    #print(ori_data)
    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    data = np.array(data)
    return data,scaler

def load_data(args,isaug):
    data , scaler = real_data_loading(args,isaug)
    train_data, test_data, train_time, test_time = train_test_split(
        data, data, test_size=args.split_ratio, random_state=args.seed
    )
    return data,train_data,test_data,scaler
def load_data_from_gen(gen_datas,model_names,seq_len):
    for model_name in model_names:
        gen_datas[model_name] = set_data_seq(gen_datas[model_name],seq_len)
    return gen_datas

def set_data_seq(ori_data,seq_len):
    print(ori_data.shape)
    temp_data = []
    for seq in range(0,len(ori_data)):
        for i in range(0, len(ori_data[0]) - seq_len):
            _x = ori_data[seq][i:i + seq_len]
            temp_data.append(_x)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    data = np.array(data)
    return data

def createlabels(args,data):
    pre_len = args.predictargs.pre_len

    x = data[:, :-pre_len,-1 :]
    y = data[:, -pre_len:,-1 :]
    return np.squeeze(x),np.squeeze(y)
