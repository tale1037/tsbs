import os

import numpy as np
import torch

from augMethod.DiffusionTS.Data.build_dataloader import build_dataloader
from augMethod.DiffusionTS.Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from augMethod.DiffusionTS.Utils.io_utils import load_yaml_config, instantiate_from_config
from augMethod.DiffusionTS.engine.solver import Trainer
from augMethod.augmothedNonDeep import jitter, scaling, permutation, magnitude_warp, time_warp,rotation
from augMethod.gan.run import timegantrain, timegantest
from augMethod.vae.vae_utils import load_vae_model
from metrics.visualization_metrics import visualization
from modeloader import get_model


def printplot(args,train_data,gen_data,model_name,data_name):
    out_dir = args.out_dir + "/dataaug_metrics"
    visualization(train_data, gen_data, "pca", out_dir,model_name,data_name)
    visualization(train_data, gen_data, "tsne", out_dir,model_name,data_name)

def train_aug_models(args,model_name,train_data):
    if model_name=="timeVAE":
        trainVAE(args,train_data,model_name)
    if model_name=="timeGAN":
        trainGAN(args,train_data,model_name)
    if model_name=="timeGANmyimple":
        trainMYGAN(args,train_data,model_name)
    if model_name=="diffusionTS":
        trainDiff(args,train_data,model_name)


def generate_aug_data(args,ori_data,model_name):
    if model_name=="timeVAE":
        return generateDatafromVAE(args,ori_data,model_name)
    elif model_name=="timeGAN":
        return generateDatafromGAN(args,ori_data,model_name)
    elif model_name=="diffusionTS":
        return generateDatafromDiff(args,ori_data,model_name)
    if model_name=="timeGANmyimple":
        return generateDatafromMYGAN(args,ori_data,model_name)
    elif model_name=="jitter":
        return jitter(ori_data)
    elif model_name=="scaling":
        return scaling(ori_data)
    elif model_name=="rotation":
        return rotation(ori_data)
    elif model_name=="permutation":
        return permutation(ori_data)
    elif model_name=="magnitude_warp":
        return magnitude_warp(ori_data)
    elif model_name=="time_warp":
        return time_warp(ori_data)

def trainVAE(args,train_data,model_name):
    model,_ = get_model(model_name, args)
    model.fit_on_data(train_data, args.augargs.vaeargs.vae_epochs)
    model.save(args.save_dir + args.augargs.vaeargs.save_dir,args.data_name)
    #gen_data = generateDatafromVAE(args,train_data,model_name)
    gen_data = model.get_prior_samples(len(train_data))

    printplot(args,train_data,gen_data,model_name,args.data_name)

def generateDatafromVAE(args,ori_data,model_name):
    #model = get_model("timevae", args)
    model = load_vae_model(model_name,args.save_dir + args.augargs.vaeargs.save_dir,args.data_name)
    gen_data = model.get_prior_samples(len(ori_data))
    return gen_data

def trainGAN(args, train_data, model_name):
    model,_ = get_model(model_name,args,train_data)
    timegantrain(args,train_data)
    timegantest(args,train_data)

def generateDatafromGAN(args,ori_data,model_name):
    model,_ = get_model(model_name, args, ori_data)
    model.load_trained_networks(args.data_name)
    if args.augargs.ganargs.synth_size != 0:
        synth_size = args.augargs.ganargs.synth_size
    else:
        synth_size = int(len(ori_data))
    generated_data = model.gen_synth_data(synth_size, ori_data)
    generated_data = generated_data.cpu().detach().numpy()
    return generated_data

def trainMYGAN(args, train_data, model_name):
    model,_ = get_model(model_name,args,train_data)
    model.train_gan(train_data,args.data_name,args.seq_len,args.augargs.ganargs.emb_epochs,1)
    gen_data = model.gen_synth_data(len(train_data), train_data).cpu().detach().numpy()
    print(f"{model_name} ori_data's shape:{train_data.shape}")
    print(f"{model_name} gen_data's shape:{gen_data.shape}")
    printplot(args, train_data, gen_data, "mygan", args.data_name)
    #timegantest(args,train_data)

def generateDatafromMYGAN(args,ori_data,model_name):
    model,_ = get_model(model_name,args,ori_data)
    model.load_trained_networks(args.data_name,args.seq_len)
    gen_data = model.gen_synth_data(len(ori_data), ori_data).cpu().detach().numpy()
    return gen_data


def trainDiff(args, train_data, model_name):
    class Args_Example:
        def __init__(self) -> None:
            self.config_path = './augMethod/DiffusionTS/Config/'+f"{args.data_name}.yaml"
            self.save_dir = f"./experment/{args.data_name}_test_exp"
            self.gpu = 0
            os.makedirs(self.save_dir, exist_ok=True)


    args = Args_Example()
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dl_info = build_dataloader(configs, args)
    model = instantiate_from_config(configs['model']).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)
    #trainer.load(10)
    trainer.train()

def generateDatafromDiff(args, ori_data,model_name):
    class Args_Example:
        def __init__(self) -> None:
            self.config_path = './augMethod/DiffusionTS/Config/'+f"{args.data_name}.yaml"
            self.save_dir = f"./{args.data_name}_test_exp"
            self.gpu = 0
            os.makedirs(self.save_dir, exist_ok=True)


    args = Args_Example()
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dl_info = build_dataloader(configs, args)
    model = instantiate_from_config(configs['model']).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)
    trainer.load(10)
    dataset = dl_info['dataset']
    seq_length, feature_dim = dataset.window, dataset.var_num
    # ori_data = np.load(os.path.join(dataset.dir, f"sine_ground_truth_{seq_length}_train.npy"))
    fake_data = trainer.sample(num=len(ori_data), size_every=2001, shape=[seq_length, feature_dim])
    if dataset.auto_norm:
        fake_data = unnormalize_to_zero_to_one(fake_data)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_sines.npy'), fake_data)
    return fake_data
