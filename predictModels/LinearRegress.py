import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class LinearRegress:
    def __init__(self):
        self.model = LinearRegression()
    def modelfit(self, data_dict):
        x_train = data_dict['x_train']
        y_train = data_dict['y_train']
        x_test = data_dict['x_test']
        self.model.fit(x_train,y_train)

        # 使用训练好的模型进行预测
        test_pred = self.model.predict(x_test)
        train_pred = self.model.predict(x_train)
        return train_pred, test_pred


