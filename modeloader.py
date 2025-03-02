from augMethod.gan.timegan import TimeGAN
from augMethod.vae.vae_utils import instantiate_vae_model
from predictModels.CNN import CNN, CNN_lstm
from predictModels.GRU import GRU
from predictModels.LSTM import LSTM
from predictModels.LinearRegress import LinearRegress
from predictModels.MLP import MLP
from predictModels.RNN import RNN
from predictModels.SVR import Svr
from predictModels.XGBoost import XGBoost

def get_model(model_name,args,ori_data = None):
    if model_name =="timeVAE":
        vae_type = "timeVAE"
        return instantiate_vae_model(
            vae_type=vae_type,
            sequence_length=args.seq_len,
            feature_dim=args.feat_dim,
            hidden_layer_sizes=args.augargs.vaeargs.hidden_layer_sizes,
            trend_poly=args.augargs.vaeargs.trend_poly,
            custom_seas=args.augargs.vaeargs.custom_seas,
            use_residual_conn=args.augargs.vaeargs.use_residual_conn,
            batch_size = args.augargs.vaeargs.batch_size,
            latent_dim=args.augargs.vaeargs.latent_dim,
        ),True
    elif model_name =="timeGAN":
        return TimeGAN(args, ori_data),True
    elif model_name == "CNN-LSTM":
        return CNN_lstm(args.feat_dim, args.feat_dim, args.predictargs.pre_len, args.predictargs.hiddenSize, args.predictargs.laryerNum,args.seq_len,
                    args.predictargs.dropout).to(args.device),True
    elif model_name == "LSTM":
        return LSTM(args.feat_dim, args.feat_dim, args.predictargs.pre_len, args.predictargs.hiddenSize, args.predictargs.laryerNum,
                    args.predictargs.dropout).to(args.device),True
    elif model_name == "GRU":
        return GRU(args.feat_dim, args.feat_dim, args.predictargs.pre_len, args.predictargs.hiddenSize, args.predictargs.laryerNum,
                    args.predictargs.dropout).to(args.device),True
    elif model_name == "RNN":
        return RNN(args.feat_dim, args.feat_dim, args.predictargs.pre_len, args.predictargs.hiddenSize, args.predictargs.laryerNum,
                    args.predictargs.dropout).to(args.device),True
    elif model_name == "MLP":
        return MLP(args.feat_dim, args.predictargs.outputSize, args.predictargs.pre_len, args.predictargs.hiddenSize, args.predictargs.laryerNum,
                    args.predictargs.dropout).to(args.device),True
    elif model_name == "CNN":
        return CNN(args.feat_dim, args.predictargs.outputSize, args.predictargs.pre_len, args.predictargs.hiddenSize, args.predictargs.laryerNum,args.seq_len,
                    args.predictargs.dropout).to(args.device),True
    elif model_name == "XGBOOST":
        return XGBoost(args),False
    elif model_name == "SVR":
        return Svr(),False
    elif model_name == "LinearRegress":
        return LinearRegress(),False