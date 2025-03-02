

from augMethod.augmothedNonDeep import jitter, scaling, permutation, magnitude_warp, time_warp,rotation
from augMethod.gan.run import timegantrain, timegantest
from augMethod.vae.vae_utils import load_vae_model
from metrics.visualization_metrics import visualization
from modeloader import get_model


def printplot(args,train_data,gen_data,model_name):
    visualization(train_data, gen_data, "pca", args.out_dir,model_name)
    visualization(train_data, gen_data, "tsne", args.out_dir,model_name)


def train_aug_models(args,model_name,train_data):
    if model_name=="timeVAE":
        trainVAE(args,train_data,model_name)
    if model_name=="timeGAN":
        trainGAN(args,train_data,model_name)


def generate_aug_data(args,ori_data,model_name):
    if model_name=="timeVAE":
        return generateDatafromVAE(args,ori_data,model_name)
    elif model_name=="timeGAN":
        return generateDatafromGAN(args,ori_data,model_name)
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
    printplot(args,train_data,gen_data,model_name)

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
    model.load_trained_networks()
    if args.augargs.ganargs.synth_size != 0:
        synth_size = args.augargs.ganargs.synth_size
    else:
        synth_size = int(len(ori_data))
    generated_data = model.gen_synth_data(synth_size, ori_data)
    generated_data = generated_data.cpu().detach().numpy()
    return generated_data