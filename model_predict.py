import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)


def read_data_series():
    # 读取Excel文件
    df = pd.read_excel('A1.xlsx', usecols=['销量（箱）', '金额（元）'])
    # 去掉无数值的行
    df.dropna(inplace=True)
    # 将数据转换为NumPy数组
    data_array = df.values

    data = data_array[:, 0]  # 选择销量数据
    index = pd.date_range(start='2011-01', periods=len(data), freq='M')
    series = pd.Series(data, index=index)

    return series


def time_transform(year=None, month=None, ):
    year = (year - 2014) / 5
    month = (month - 1) / 6
    if year is None:
        return month
    if month is None:
        return year
    return np.array((year, month), dtype=np.float32)


def prepare_data(series, last_months=12):
    last_date = series.index[:-last_months]
    data = series.values[:-last_months]
    y_true = series.values[-last_months:]
    last_months = last_date.month
    last_years = last_date.year
    return data, last_years, last_months, y_true


def evaluate_model(series_origin, series_pred):
    series_true = series_origin[series_pred.index[0]:]
    series_pred_cover = series_pred[:len(series_true)]

    mse = mean_squared_error(series_true, series_pred_cover)
    mae = mean_absolute_error(series_true, series_pred_cover)
    r2 = r2_score(series_true, series_pred_cover)
    mape = mean_absolute_percentage_error(series_true, series_pred_cover)

    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R^2: {r2:.2f}')
    print(f'MAPE: {mape:.2f}')

    plt.figure(figsize=(10, 6))
    plt.plot(series_true, label='True')
    plt.plot(series_pred, label='Predicted')
    plt.legend()
    plt.show()


def predict_to_future_lstm(model, series, months, scaler, feature_length=12, last_months=12):
    # 准备初始输入
    data, data_years, data_months, y_true = prepare_data(series, last_months)
    year_next = data_years[-1]
    month_next = data_months[-1]

    # 拼接 data, data_years, data_months
    data = scaler.transform(data.reshape(-1, 1)).reshape(-1)
    input_tensor = torch.from_numpy(data).unsqueeze(0)
    input_tensor = torch.cat(
        (input_tensor, torch.tensor(time_transform(data_years, data_months))))
    input_tensor = input_tensor.T[-feature_length:].unsqueeze(0).float()

    print(f'从{year_next}年{month_next}月开始后续数据推理')

    # 初始化一个列表来保存预测结果
    result = []

    # 进行months月的预测
    for _ in range(months):
        # 预测下一天的数据
        predicted = model(input_tensor)

        predicted = predicted.item()
        # 将预测结果添加到结果列表中
        result.append(predicted)

        # 创建一个新的输入张量，用于下一次预测
        input_next = torch.tensor(
            [[[predicted, *time_transform(year_next, month_next + 1).tolist()]]], dtype=torch.float32)
        # print(input_next)

        year_next += 1 if month_next == 11 else 0
        month_next = (month_next + 1) % 12

        # 更新input_tensor为最新的预测结果，用于下一次预测
        # 注意：这里我们假设模型输出的是下一天的数据，形状为(1, 3)
        # 我们需要将其添加到输入序列的末尾，并移除最旧的一天数据
        input_tensor = torch.cat((input_tensor[:, 1:], input_next), dim=1)

    # 将结果转换为numpy数组
    result = np.array(result)

    result = scaler.inverse_transform(result.reshape(-1, 1)).reshape(-1)
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)

    pred_index = pd.date_range(
        start=series.index[-last_months], periods=months, freq='M')
    series_pred = pd.Series(result, index=pred_index)

    evaluate_model(series, series_pred)

    return result
