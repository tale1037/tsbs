import os

import numpy as np
from torch.utils.data import DataLoader

from data_utils.dataloader import createlabels
from data_utils.datasets import PredictionDataset
from modeloader import get_model
import time
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm, trange


def predict_test(args,train_data,test_data,train_data_name,no):

    model,isdeep = get_model(args.predictargs.model,args)

    if isdeep:

        train_datasets = PredictionDataset(train_data,args.predictargs.pre_len)

        train_loader = DataLoader(train_datasets, batch_size=args.predictargs.batch_size, shuffle=False)

        test_datasets = PredictionDataset(test_data,args.predictargs.pre_len)

        test_loader = DataLoader(test_datasets, batch_size=args.predictargs.batch_size, shuffle=False)

        modeltrain(model, args, train_loader)

        return metric_predict(model, args, train_loader,test_loader,train_data_name,no)

    else:
        valid_data = test_data[:int(0.5*len(test_data))]
        x_train,y_train = createlabels(args,train_data)
        x_test,y_test = createlabels(args,test_data)
        x_valid,y_valid = createlabels(args,valid_data)
        data_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "x_valid": x_valid,
            "y_valid": y_valid
        }
        model.modelfit(data_dict)
        y_pred = model.model.predict(x_test)
        return np.mean(np.abs(y_test - y_pred))

def predict_test_metric(args,data_dict,no):

    train_data_half = data_dict['train_data_half']
    train_data = data_dict['train_data']
    test_data = data_dict['test_data']
    gen_data = data_dict['gen_data']
    train_data_hfdouble = np.concatenate((train_data_half,train_data_half),axis=0)
    print(train_data_half.shape,gen_data.shape)
    mix_data = np.concatenate((train_data_half, gen_data), axis=0)
    loss = []
    #loss.append(predict_test(args,train_data_half,test_data,"train_data_half",no))
    #loss.append(predict_test(args, gen_data, test_data,"gen_data",no))

    loss.append(predict_test(args, mix_data, test_data,"mix_data",no))
    loss.append(predict_test(args, train_data, test_data,"train_data",no))
    #loss.append(predict_test(args, train_data_hfdouble, test_data,"train_data_hfdouble",no))
    #print(f"{args.predictargs.model}:{loss}")metric_predict
    return loss

def modeltrain(model,args,train_loder):
    start_time = time.time()
    model = model.to(args.device)
    loss_function = nn.L1Loss()
    learning_rate = args.predictargs.learningRate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    epochs = args.predictargs.epochs
    model.train()
    logger = trange(epochs, desc=f"Epoch: 0, Loss: 0",disable=True)
    for i in logger:
        model = model.to(args.device)
        running_loss = 0.0
        for seq, labels in train_loder:
            seq, labels = seq.to(args.device).float(), labels.to(args.device).float()
            #print(labels.shape,seq.shape)
            optimizer.zero_grad()
            y_pred = model(seq)
            #print(y_pred.shape, seq.shape)
            single_loss = loss_function(y_pred[...,-1:], labels)

            single_loss.backward()
            #print(12213)
            optimizer.step()

            running_loss += single_loss.item()

        # 计算 epoch 的平均损失
        logger.set_description(f"Epoch {i + 1}/{epochs}, Loss: {running_loss}")

        # 保存模型
        torch.save(model.state_dict(), os.path.join((args.save_dir + "/") ,f"{args.predictargs.model}.pth"))
        # time.sleep(0.1)


def modeltest(model,args,test_loder,data_name,no):
    losss = 0.0
    criterion = torch.nn.MSELoss()
    model = model
    model.load_state_dict(torch.load(os.path.join((args.save_dir + "/") ,f"{args.predictargs.model}.pth")))
    # test_loader = loaders_dict['test']
    # scaler = loaders_dict['scaler']
    model.eval()  # 评估模式
    loss_function = nn.L1Loss()
    results = []
    labels = []
    for seq, label in test_loder:

        seq, label = seq.to(args.device).float(), label.to(args.device).float()
        # print(seq)

        pred = model(seq)
        # print(f"seq:{seq.shape}")
        # print(f"pred:{pred.shape}")
        # print(f"label:{label.shape}")
        # print(pred[0][-1].cpu().detach())
        single_loss = loss_function(pred[..., -1:], label)
        losss += single_loss
        # pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        # label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1][-1].cpu().detach())
            labels.append(label[i][-1][-1].cpu().detach())

#    print(labels.shape,results.shape)
    # 绘制历史数据
    plt.plot(labels[:100], label='TrueValue')

    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(results[:100], label='Prediction')

    filename = "/pictures"+"/"+args.data_name
    # 添加标题和图例
    plt.title(f"{args.predictargs.model}_{data_name}")
    plt.legend()
    plt.savefig(os.path.join((args.out_dir + filename) ,f"{args.predictargs.model}-{args.data_name}_{data_name}_{no}.png"))
    plt.close()
    return losss



def metric_predict(model,args,train_loder,test_loder,data_name,no):
    start_time = time.time()
    model = model.to(args.device)
    loss_function = nn.L1Loss()
    learning_rate = args.predictargs.learningRate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    epochs = args.predictargs.epochs
    model.train()
    logger = trange(epochs, desc=f"Epoch: 0, Loss: 0",disable=True)
    for i in logger:
        model = model.to(args.device)
        running_loss = 0.0
        for seq, labels in train_loder:
            seq, labels = seq.to(args.device).float(), labels.to(args.device).float()
            #print(labels.shape,seq.shape)
            optimizer.zero_grad()
            y_pred = model(seq)
            #print(y_pred.shape, seq.shape)
            single_loss = loss_function(y_pred[...,-1:], labels)

            single_loss.backward()
            #print(12213)
            optimizer.step()

            running_loss += single_loss.item()

        # 计算 epoch 的平均损失
        logger.set_description(f"Epoch {i + 1}/{epochs}, Loss: {running_loss}")

    losss = 0.0
    model.eval()  # 评估模式
    loss_function = nn.L1Loss()
    results = []
    labels = []
    for seq, label in test_loder:

        seq, label = seq.to(args.device).float(), label.to(args.device).float()
        # print(seq)

        pred = model(seq)
        # print(f"seq:{seq.shape}")
        # print(f"pred:{pred.shape}")
        # print(f"label:{label.shape}")
        # print(pred[0][-1].cpu().detach())
        single_loss = loss_function(pred[..., -1:], label)
        losss += single_loss
        # pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        # label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1][-1].cpu().detach())
            labels.append(label[i][-1][-1].cpu().detach())

#    print(labels.shape,results.shape)
    # 绘制历史数据
    plt.plot(labels[:1000], label='TrueValue')

    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(results[:1000], label='Prediction')

    filename = "/pictures"+"/"+args.data_name
    # 添加标题和图例
    plt.title(f"{args.predictargs.model}_{data_name}")
    plt.legend()
    plt.savefig(os.path.join((args.out_dir + filename) ,f"{args.predictargs.model}-{args.data_name}_{data_name}_{no}.png"))
    plt.close()
    return losss.cpu().detach()