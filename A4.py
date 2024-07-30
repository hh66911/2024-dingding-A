# %%
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
series, scaler = model_predict.read_data_series(
    scale=True, filter_early=True, file_index=4)

# %%


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, length, 3]
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        # log_var, mean = x.split(1, dim=-1)
        # x = torch.randn_like(mean) * torch.exp(log_var / 2) + mean
        return x


model_parameters = {
    "input_size": 3, "hidden_size": 12, "num_layers": 1, "output_size": 1
}

# %%
model = model_predict.load_rnn_model_best(GRUModel, model_parameters)
results, _ = model_predict.predict_to_future_rnn(
    model, series, scaler, feature_length=16, last_months=12, months=34)

results = results * 11925
results.to_excel("results.xlsx")
