import torch
from torch import nn

import model_predict

series, scaler = model_predict.read_data_series(scale=True, file_index=3)

# 4.1 LSTM 模型


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
        return out

    def init_weights(self):
        # 遍历 LSTM 层的参数，对参数进行初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.1)  # 使用正态分布初始化权重
            elif 'bias' in name:
                nn.init.constant_(param, 0)  # 将偏置项初始化为零


model_parameters = {
    "input_size": 3, "hidden_size": 24, "num_layers": 1, "output_size": 1
}

model = model_predict.load_rnn_model_best(LSTMModel, model_parameters)
results, _ = model_predict.predict_to_future_rnn(
    model, series, scaler, last_months=12)

results = results * 11925

results.to_excel('results.xlsx')
