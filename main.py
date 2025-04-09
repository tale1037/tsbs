import argparse
import torch
from data_utils.dataloader import load_data, load_data_from_gen
from experment.exp_utils import trainaugmodel, get_gen_datas, exp_predict
from params.augparams import get_augargs
from params.predictparams import get_predict_params

expparser = argparse.ArgumentParser(description='Simple experiment runner')
expparser.add_argument("-data_dir",type=str,default="./data",help="数据集")
#expparser.add_argument("-")
expparser.add_argument("-data_name",type=str,default="energy",choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'stock', 'energy'],help="数据集")
expparser.add_argument("-seq_len",type=int,default=24,help="数据步长")
expparser.add_argument("-data_preprocess_len",type=int,default=64,help="数据预处理进生成模型的长度")
expparser.add_argument("-data_len_limit",type=int,default=100000,help="限制数据长度")
expparser.add_argument("-split_ratio",type=float,default=0.7,help="数据划分比例")
expparser.add_argument("-seed",type=int,default=154,help="全局种子")
expparser.add_argument("-feat_dim",type=int,default=7)
expparser.add_argument("-save_dir",type=str,default="./savedModels",help="数据集")
expparser.add_argument("-out_dir",type=str,default="./output",help="数据集")
expparser.add_argument("-iftrain",type=bool,default=True,help="是否训练")

def main():
    args = expparser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.predictargs = get_predict_params()
    args.augargs = get_augargs()
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'stock', 'energy']
    origin_data, train_data, test_data, scaler = load_data(args, False)
    origin_data_aug, train_data_aug,test_data_aug, scaler_aug = load_data(args,True)
    train_data_half = train_data[:int(0.5 * len(train_data))]
    train_data_half_aug = train_data_aug[:int(0.5 * len(train_data_aug))]
    args.feat_dim = origin_data.shape[2]
    print(train_data.shape)
    # augmodels = ['timeGAN','TimeVAE','timeGANmyimple',
    #     'jitter','scaling','rotation','permutation','magnitude_warp','time_warp']
    augmodels = ['timeGANmyimple']
    predict_methods = ['MLP', 'CNN-LSTM', 'GRU', 'LSTM', 'RNN']
    args.seq_len = args.seq_len * args.augargs.seq_len_times
    if (args.iftrain):
        trainaugmodel(args, augmodels, origin_data_aug)
    else:
        gen_datas = get_gen_datas(args, augmodels, train_data_half_aug)
        if (args.augargs.seq_len_times > 1):
            gen_datas = load_data_from_gen(gen_datas, augmodels, args.predictargs.seq_len)
        exp_predict(args, augmodels,predict_methods, origin_data, test_data, train_data_half,gen_datas)


if __name__ == '__main__':
    main()