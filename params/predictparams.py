import argparse


def get_predict_params():
    predictmodelparser = argparse.ArgumentParser(description='is dataaug really useful in ts forecasting?')
    predictmodelparser.add_argument("-model", type=str, default="GRU", help="name")
    predictmodelparser.add_argument("-outputSize", type=int, default=1, help="输出规模（横向）")
    predictmodelparser.add_argument("-learningRate", type=float, default=0.001, help="模型学习率")
    predictmodelparser.add_argument("-epochs", type=int, default=50, help="学习轮数")
    predictmodelparser.add_argument("-inputSize", type=int, default=7, help="输入大小(feat_dim)")
    predictmodelparser.add_argument("-save_dir", type=str, default="/savedmodels", help="模型存储路径")
    predictmodelparser.add_argument("-hiddenSize", type=int, default=64)
    predictmodelparser.add_argument('-kernelSizes', type=int, default=3)
    predictmodelparser.add_argument('-laryerNum', type=int, default=2)
    predictmodelparser.add_argument("-batch_size", type=int, default=256)
    predictmodelparser.add_argument('-dropout', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    predictmodelparser.add_argument('-forecasting', type=bool, default=False, help="目前是否是预测任务")
    predictmodelparser.add_argument("-feat_pred_no", type=int, default=2)
    predictmodelparser.add_argument("-pre_len", type=int, default=5)

    return predictmodelparser.parse_args()