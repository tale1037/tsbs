import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import xgboost as xgb
from sklearn.metrics import mean_squared_error


class XGBoost:
    def __init__(self,args):
        self.model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=3,
                           learning_rate=args.predictargs.learningRate,)

    def modelfit(self, data_dict):
        x_train = data_dict['x_train']
        y_train = data_dict['y_train']
        x_valid = data_dict['x_valid']
        y_valid = data_dict['y_valid']
        #print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        self.model.fit(x_train, y_train,
                 eval_set=[(x_train, y_train), (x_valid, y_valid)],
                 verbose=False)

