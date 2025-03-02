import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
import itertools
import numpy as np
import seaborn as sns


#数据预处理

ChinaBank = pd.read_csv('//ChinaBank.csv', index_col ='Date', parse_dates=['Date'])
#print(ChinaBank.head())
ChinaBank.index = pd.to_datetime(ChinaBank.index)
sub = ChinaBank.loc['2014-01':'2014-06','Close']
print(sub.head())

train = sub.loc['2014-01':'2014-03']
test = sub.loc['2014-04':'2014-06']

#.diff(1)做一个时间间隔
ChinaBank['diff_1'] = ChinaBank['Close'].diff(1) #1阶差分

#对一阶差分数据在划分时间间隔
ChinaBank['diff_2'] = ChinaBank['diff_1'].diff(1) #2阶差分

fig = plt.figure(figsize=(12,10))
#原数据
ax1 = fig.add_subplot(311)
ax1.plot(ChinaBank['Close'])
#1阶差分
ax2 = fig.add_subplot(312)
ax2.plot(ChinaBank['diff_1'])
#2阶差分
ax3 = fig.add_subplot(313)
ax3.plot(ChinaBank['diff_2'])
plt.show()


#ADF检验

# 计算原始序列、一阶差分序列、二阶差分序列的单位根检验结果
ChinaBank['diff_1'] = ChinaBank['diff_1'].fillna(0)
ChinaBank['diff_2'] = ChinaBank['diff_2'].fillna(0)

timeseries_adf = ADF(ChinaBank['Close'].tolist())
timeseries_diff1_adf = ADF(ChinaBank['diff_1'].tolist())
timeseries_diff2_adf = ADF(ChinaBank['diff_2'].tolist())


# 打印单位根检验结果
print('timeseries_adf : ', timeseries_adf)
print('timeseries_diff1_adf : ', timeseries_diff1_adf)
print('timeseries_diff2_adf : ', timeseries_diff2_adf)

#绘制
fig = plt.figure(figsize=(12,7))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom') # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部
#fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
#fig.tight_layout()
plt.show()

#确定pq的取值范围
p_min = 0
d_min = 0
q_min = 0
p_max = 5
d_max = 0
q_max = 5

#Initialize a DataFrame to store the results,，以BIC准则
train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

#根据以上求得
p = 1
d = 0
q = 0
model = sm.tsa.ARIMA(train, order=(p,d,q))
results = model.fit()
resid = results.resid #获取残差
#绘制
#查看测试集的时间序列与数据(只包含测试集)
fig, ax = plt.subplots(figsize=(12, 5))

ax = sm.graphics.tsa.plot_acf(resid, lags=40,ax=ax)

plt.show()

predict_sunspots = results.forecast(steps=120)
print(predict_sunspots)

#查看测试集的时间序列与数据(只包含测试集)
plt.figure(figsize=(12,6))
#plt.plot(test)
#plt.xticks(rotation=45) #旋转45度
plt.plot(predict_sunspots)
plt.show()





