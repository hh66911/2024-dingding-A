import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)


def read_data_series(filter_early=True, scale=True):
    # 读取Excel文件
    df = pd.read_excel('A1.xlsx', usecols=['月份', '销量（箱）', '金额（元）'])
    # 去掉无数值的行
    df.dropna(inplace=True)

    # 月份的格式为 yyyymm
    df.set_index('月份', inplace=True)

    if filter_early:
        df = df[df.index >= 201401]
        index = pd.date_range(start='2014-01', periods=len(df), freq='M')
    else:
        index = pd.date_range(start='2011-01', periods=len(df), freq='M')

    # 将数据转换为NumPy数组
    data_array = df.values

    data = data_array[:, 0]  # 选择销量数据

    if scale:
        from sklearn.preprocessing import RobustScaler
        # 创建 MinMaxScaler 的实例
        scaler = RobustScaler()
        # 使用 fit_transform 方法来拟合数据并转换它
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
        series = pd.Series(normalized_data, index=index)
        return series, scaler
    else:
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


def evaluate_model_prophet(series_test, df_pred):
    df_pred = df_pred[['ds', 'yhat']].set_index('ds')
    series_pred = df_pred['yhat']
    series_pred = series_pred[series_test.index[0]:]
    evaluate_model(series_test, series_pred)


def predict_to_future_lstm(model, series, months, scaler=None, feature_length=12, last_months=12):
    import torch
    # 准备初始输入
    data, data_years, data_months, y_true = prepare_data(series, last_months)
    year_next = data_years[-1]
    month_next = data_months[-1]

    # 拼接 data, data_years, data_months
    if scaler is not None:
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

    if scaler is not None:
        result = scaler.inverse_transform(result.reshape(-1, 1)).reshape(-1)
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)

    pred_index = pd.date_range(
        start=series.index[-last_months], periods=months, freq='M')
    series_pred = pd.Series(result, index=pred_index)

    evaluate_model(series, series_pred)

    return result


def predict_to_future_arima(model, series, scaler=None, months=24, last_months=12):
    forecast_start = series.index[-last_months]
    forecast_end = forecast_start + pd.DateOffset(months=months)

    forecast = model.predict(start=forecast_start, end=forecast_end)

    if scaler is not None:
        forecast = scaler.inverse_transform(
            forecast.values.reshape(-1, 1)).reshape(-1)
        series = scaler.inverse_transform(
            series.values.reshape(-1, 1)).reshape(-1)

    evaluate_model(series, forecast)

    return forecast


def predict_to_future_prophet(model, series, scaler=None, months=24):
    future = model.make_future_dataframe(periods=months, freq='M')
    forecast = model.predict(future)
    model.plot(forecast, include_legend=True)

    y_pred = pd.Series(forecast['yhat'].values, index=forecast['ds'])

    model_history = model.history
    series_test = series[model_history.index[-1]:][1:]
    series_test_index = series_test.index
    y_pred = y_pred[model_history.index[-1]:][1:]
    y_pred_index = y_pred.index

    if scaler is not None:
        y_pred = scaler.inverse_transform(
            y_pred.values.reshape(-1, 1)).reshape(-1)
        series_test = scaler.inverse_transform(
            series_test.values.reshape(-1, 1)).reshape(-1)
        y_pred = pd.Series(y_pred, index=y_pred_index)
        series_test = pd.Series(series_test, index=series_test_index)

    evaluate_model(series_test, y_pred)


if __name__ != '__main__':
    print('小助手已加载')
