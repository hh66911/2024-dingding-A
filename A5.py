# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import model_predict
import random

# 指定支持中文的字体，例如SimHei或者Microsoft YaHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

random.seed(123)

# %%
series, scaler = model_predict.read_data_series(
    scale=True, filter_early=False, file_index=5)
series = series[series.index <= '2018-09']
features, targets = model_predict.gen_xgboost_data(series, feature_length=16)

# %%
# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=12/len(features), shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2)

# 创建XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, gamma=0, subsample=0.8,
                         colsample_bytree=1, max_depth=7)

# 训练模型
results = model.fit(X_train, y_train)

# %%
results, _ = model_predict.predict_to_future_xgboost(
    model, series, scaler, months=64, last_months=12)

# %%
results = results[~results.index.isin(series.index)]

series = model_predict.read_data_series(
    scale=False, filter_early=False, file_index=5)
series_new = pd.concat([series[series.index <= '2018-09'], results], axis=0)
series_new = pd.concat([series_new, series[series.index > '2018-09']], axis=0)

df_new = pd.DataFrame(series_new, columns=['销量（箱）'])
month_str = df_new.index.strftime('%Y%m')
df_new['月份'] = month_str

df_new.to_excel('series_new.xlsx')
