import argparse
import os

import numpy as np
import pandas as pd
import torch

from augMethod.aug_utils import train_aug_models, generate_aug_data
from data_utils.dataloader import load_data
from experment.exp_utils import trainaugmodel, get_gen_datas
from metrics.metric_utils import predict_test_metric
from params.augparams import get_augargs
from params.predictparams import get_predict_params

expparser = argparse.ArgumentParser(description='Simple experiment runner')
expparser.add_argument("-data_dir",type=str,default="./data",help="数据集")
#expparser.add_argument("-")
expparser.add_argument("-data_name",type=str,default="ETTh1",choices=['ettm1',"stock"],help="数据集")
expparser.add_argument("-seq_len",type=int,default=24,help="数据步长")
expparser.add_argument("-data_len_limit",type=int,default=500000,help="限制数据长度")
expparser.add_argument("-split_ratio",type=float,default=0.7,help="数据划分比例")
expparser.add_argument("-seed",type=int,default=154,help="全局种子")
expparser.add_argument("-feat_dim",type=int,default=7)
expparser.add_argument("-save_dir",type=str,default="./savedModels",help="数据集")
expparser.add_argument("-out_dir",type=str,default="./output",help="数据集")



if __name__ == '__main__':
    args = expparser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.predictargs= get_predict_params()
    args.augargs = get_augargs()
    origin_data,train_data,test_data,scaler = load_data(args,True)
    train_data_half = train_data[:int(0.5 * len(train_data))]
    args.feat_dim = origin_data.shape[2]

    print(train_data.shape)
    augmodels = ['timeGAN']
        #,'timeVAE','jitter','scaling','rotation','permutation','magnitude_warp','time_warp']
    predict_methods = ['XGBOOST','MLP', 'CNN-LSTM', 'GRU', 'LSTM', 'RNN']
    trainaugmodel(args,augmodels,origin_data)
    gen_datas = get_gen_datas(args,augmodels,train_data_half)
    it = 1
    for augmodel in augmodels:
        data_dict = {
            "train_data_half":train_data_half,
            "gen_data":gen_datas[augmodel],
            "test_data":test_data,
            "train_data":train_data,
        }
        zongloss = []
        for method in predict_methods:
            args.predictargs.model = method
            losss = []
            for i in range(it):
                loss = predict_test_metric(args,data_dict,i)
                losss.append(loss)
            #print(losss)
            np_losss = np.array(losss)
            column_means = np.mean(np_losss, axis=0)
            zongloss.append(column_means)
        zongloss = np.array(zongloss)
        df = pd.DataFrame(zongloss,columns=['train_data_half', 'gen_data', 'mix_data','train_data',"2timesoffirst"])
        df.insert(0, args.data_name+"_"+augmodel, predict_methods)
        outpath = os.path.join((args.out_dir+"/result"), f"{args.data_name}_{augmodel}_result.csv")
        df.to_csv(outpath, index=False)  # 不保存索引列

        print(zongloss)

