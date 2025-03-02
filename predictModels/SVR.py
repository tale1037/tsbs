import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


class Svr:
    def __init__(self):
        self.model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
    def modelfit(self,data_dict):
        x_train = data_dict['x_train']
        y_train = data_dict['y_train']
        x_test = data_dict['x_test']
        self.model.fit(x_train, y_train)

        y_train_pred = self.model.predict(x_train).reshape(-1, 1)
        y_test_pred = self.model.predict(x_test).reshape(-1, 1)

        return y_train_pred, y_test_pred



