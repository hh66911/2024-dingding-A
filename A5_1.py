# %%
import random
import xgboost as xgb
from matplotlib import pyplot as plt
from importlib import reload
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import pandas as pd

import model_predict
from sklearn.model_selection import train_test_split

# %%
series, scaler = model_predict.read_data_series5()


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, length, 3]
        x, _ = self.gru(x)
        x = self.fc(x[:, -1:, :])
        # log_var, mean = x.split(1, dim=-1)
        # x = torch.randn_like(mean) * torch.exp(log_var / 2) + mean
        return x


model_parameters = {
    "input_size": 3, "hidden_size": 12, "num_layers": 1, "output_size": 1
}

# %%
reload(model_predict)
model = model_predict.load_rnn_model_best(GRUModel, model_parameters)
results, _ = model_predict.predict_to_future_rnn(
    model, series, scaler, feature_length=7, last_months=16, months=68)

results = results[~results.index.isin(series.index)]

series = model_predict.read_data_series(
    scale=False, filter_early=False, file_index=5)
series_new = pd.concat([series[series.index <= '2018-09'], results], axis=0)
series_new = pd.concat([series_new, series[series.index > '2018-09']], axis=0)

df_new = pd.DataFrame(series_new, columns=['销量（箱）'])
month_str = df_new.index.strftime('%Y%m')
df_new['月份'] = month_str.astype(int)

df_new.to_excel('A5_1.xlsx')


# %%
series, scaler = model_predict.read_data_series(
    scale=True, filter_early=True, file_index='5_1')
features, targets = model_predict.gen_xgboost_data(series, feature_length=16)

# %%
# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=12/len(features), shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=9999)

# 创建XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, gamma=0, subsample=0.8,
                         colsample_bytree=1, max_depth=7)

# 训练模型
results = model.fit(X_train, y_train)

results, _ = model_predict.predict_to_future_xgboost(
    model, series, scaler, months=64, last_months=32)

results = pd.DataFrame(results, columns=['销量（箱）'])
results['金额（元）'] = results['销量（箱）'] * 30740
results.to_excel('results.xlsx')
