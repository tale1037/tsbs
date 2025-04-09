import os

import numpy as np
import pandas as pd
from tqdm import trange

from augMethod.aug_utils import train_aug_models, generate_aug_data
from data_utils.dataloader import load_data_from_gen
from metrics.metric_utils import predict_test_metric
from metrics.visualization_metrics import visualization
from metrics.visualize import plot_samples


def trainaugmodel(args,model_names,train_data):
    for model_name in model_names:
        train_aug_models(args,model_name,train_data)

def get_gen_datas(args,model_names,train_data):
    gen_datas = {}
    for model_name in model_names:
        gen_data = generate_aug_data(args,train_data,model_name)
        gen_datas[model_name] = gen_data
        plot_samples(
            samples1=train_data,
            samples1_name="Original Train",
            samples2=gen_data,
            samples2_name="Reconstructed Train",
            num_samples=5,
            model_name= model_name
        )
        print(f"{model_name} ori_data's shape:{train_data.shape}")
        print(f"{model_name} gen_data's shape:{gen_data.shape}")
        visualization(train_data,gen_data,"tsne",args.out_dir,model_name,args.data_name)
    return gen_datas

def exp_predict(args,augmodels,predict_methods,train_data,test_data,train_data_half,gen_datas):

    it = 1
    columns = []
    zongloss = []
    for augmodel in augmodels:
        data_dict = {
            "train_data_half": train_data_half,
            "gen_data": gen_datas[augmodel],
            "test_data": test_data,
            "train_data": train_data,
        }
        for method in predict_methods:
            columns.append(f"{method}_{augmodel}")
            args.predictargs.model = method
            losss = []
            logger = trange(it, desc=f"{method}_Epoch: 0", disable=False)
            for i in logger:
                loss = predict_test_metric(args, data_dict, i)
                # loss = (1,1)
                losss.append(loss)
                logger.set_description(f"{method}_Epoch: {i + 1}/{it}")
            print(f"{method}:{losss}")
            np_losss = np.array(losss)
            column_means = np.mean(np_losss, axis=0)
            zongloss.append(column_means)

    zongloss = np.array(zongloss)
    print(len(columns))
    df = pd.DataFrame(zongloss, columns=['train_data', 'mix_data'])
    df.insert(0, args.data_name + "_" + augmodel, columns)
    outpath = os.path.join((args.out_dir + "/result"), f"{args.data_name}_result.csv")
    df.to_csv(outpath, index=False)  # 不保存索引列

    print(zongloss)



