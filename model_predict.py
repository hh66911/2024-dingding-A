import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)


class MyScaler:
    def __init__(self):
        import sklearn.preprocessing
        self.sklearn_scaler = sklearn.preprocessing.RobustScaler()

    def fit_transform(self, data):
        data = np.log(1 + data)
        data = self.sklearn_scaler.fit_transform(data)
        return data

    def transform(self, data):
        data = np.log(1 + data)
        data = self.sklearn_scaler.transform(data)
        return data

    def inverse_transform(self, data):
        data = self.sklearn_scaler.inverse_transform(data)
        data = np.exp(data) - 1
        return data


def fill_series_full(series):
    # 计算每个月份的平均值（忽略NaN），并存储在一个Series中
    monthly_averages = series.groupby(series.index.month).mean()
    print('月份平均：')
    print(monthly_averages)

    # 创建一个新的DataFrame来存储完整的时间序列
    start_month = series.index.min()
    end_month = series.index.max()
    full_range = pd.date_range(
        start=start_month, end=end_month, freq='M')
    series_full = pd.Series(index=full_range)

    # 使用原始 series 中的数据填充新 series 的对应位置
    series_full.update(series)

    # 填充缺失值：对于每个月份，使用该月份的平均值填充缺失值
    for month in range(1, 13):
        month_mask = series_full.index.month == month
        missing_values = series_full.loc[month_mask].isna()
        # 从monthly_averages中获取对应月份的平均值
        month_avg = monthly_averages.loc[month]
        # 使用对应月份的平均值填充缺失值
        if missing_values.any():
            series_full[missing_values.index[missing_values]] = month_avg

    return series_full


def fix_series_index(series):
    # 确保月份是字符串类型，并且是 yyyymm 格式
    series.index = series.index.astype(str)
    # 将月份转换为 pandas 的 Period 类型
    series.index = pd.to_datetime(
        series.index, format='%Y%m') + pd.offsets.MonthEnd(0)
    return series


def contiguous_month_index(indices):
    # 生成一个完整的月份范围
    start_month = indices.min()
    end_month = indices.max()
    full_range = pd.date_range(start=start_month, end=end_month, freq='M')
    # 检查是否有缺失的月份
    missing_months = full_range.difference(indices)
    return missing_months


def read_data_series(filter_early=True, scale=True, file_index=1):
    # 读取Excel文件
    df = pd.read_excel(f'A{file_index}.xlsx', usecols=['月份', '销量（箱）'])
    # 去掉无数值的行
    df.dropna(inplace=True)

    # 月份的格式为 yyyymm
    df.set_index('月份', inplace=True)

    series = pd.Series(df['销量（箱）'], index=df.index)
    series = fix_series_index(series)

    missing_months = contiguous_month_index(series.index)

    if not missing_months.empty:
        print("时间序列不连贯，缺失的月份：",
              [x.strftime('%Y-%m') for x in missing_months.tolist()])

        if (len(missing_months) > 4):
            print("缺失的月份太多，无法填充")
        elif len(missing_months == 3) and contiguous_month_index(missing_months).empty:
            print("缺失的月份为连续3个，不建议填充")
        else:
            print('即将填充缺失的月份')
            series = fill_series_full(series)
    else:
        print("时间序列连贯")

    if filter_early:
        series = series[series.index >= '2014-01']

    if scale:
        # 创建 Scaler 的实例
        scaler = MyScaler()
        # 使用 fit_transform 方法来拟合数据并转换它
        normalized_data = scaler.fit_transform(
            series.values.reshape(-1, 1)).reshape(-1)
        series = pd.Series(normalized_data, index=series.index)
        return series, scaler
    else:
        return series


def plot_series_info(series, scaler=None):
    # 指定支持中文的字体，例如SimHei或者Microsoft YaHei
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    if scaler is not None:
        series = pd.Series(scaler.inverse_transform(
            series.values.reshape(-1, 1)).reshape(-1), index=series.index)

    # 绘制时间序列的折线图
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title('销量时间序列')
    plt.xlabel('时间')
    plt.ylabel('销量')
    plt.show()

    # 绘制时间序列的自相关性和偏自相关性图
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, ax=ax[0], lags=24)
    plot_pacf(series, ax=ax[1], lags=24)
    plt.show()


def time_transform(year=None, month=None):
    year = (year - 2014) / 5
    month = (month - 1) / 6
    if year is None:
        return month
    if month is None:
        return year
    return np.array((year, month), dtype=np.float32)


def series_year_month(series):
    year_month = np.array([time_transform(year, month)
                          for year, month in zip(series.index.year, series.index.month)])
    df = pd.DataFrame(
        {'y': series.values, 'year': year_month[:, 0], 'month': year_month[:, 1]}, index=series.index)
    return df


def gen_xgboost_data(series, feature_length=12):
    sales_df = series_year_month(series)
    # 使用前 feature_length 个月的销量作为特征，预测下一个月的销量
    features = []
    targets = []

    for i in range(feature_length, len(sales_df)):
        # 获取销量特征
        sales_features = sales_df['y'].iloc[i-feature_length:i].tolist()

        # 获取年份和月份特征
        year_feature = sales_df['year'].iloc[i]
        month_feature = sales_df['month'].iloc[i]

        # 将年份和月份添加到特征向量中
        feature_vector = sales_features + [year_feature, month_feature]

        # 添加到特征集中
        features.append(feature_vector)

        # 添加目标销量
        targets.append(sales_df['y'].iloc[i])

    # 将特征和目标转换为numpy数组
    features = np.array(features)
    targets = np.array(targets)

    return features, targets


def gen_rnn_dataset(series, feature_length=12):
    import torch
    sales_df = series_year_month(series)
    # 使用前 feature_length 个月的销量作为特征，预测下一个月的销量
    features = []
    targets = []

    for i in range(feature_length, len(sales_df)):
        # 获取销量特征
        sales_features = sales_df['y'].iloc[i-feature_length:i].tolist()

        # 获取年份和月份特征
        year_features = sales_df['year'].iloc[i-feature_length:i].tolist()
        month_features = sales_df['month'].iloc[i-feature_length:i].tolist()

        # 将年份和月份添加到特征向量中
        feature_vector = [sales_features, year_features, month_features]

        # 添加到特征集中
        features.append(feature_vector)

        # 添加目标销量
        targets.append(sales_df['y'].iloc[i])

    # 将特征和目标转换为numpy数组
    features = torch.tensor(features, dtype=torch.float32)
    features = features.permute(0, 2, 1)
    targets = torch.tensor(targets, dtype=torch.float32)

    return features, targets


def train_rnn_model(model_type, model_params, series, epochs=1000,
                    feature_length=12, last_months=12, device='cuda'):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.model_selection import train_test_split

    features, targets = gen_rnn_dataset(series, feature_length)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=last_months/features.shape[0], shuffle=False)
    X_train, X_test, y_train, y_test = map(
        lambda x: x.to(device), (X_train, X_test, y_train, y_test))

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=12)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=12)

    model_name = str(model_type).split('.')[-1].split('\'')[0]
    save_path = model_name + "_best.pth"
    print("训练", model_name, "模型")
    print("结果保存到：", save_path)

    model = model_type(**model_params)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5)

    # 初始化最佳模型参数和最佳验证损失
    best_model_params = model.state_dict()
    best_val_loss = float('inf')

    model.to(device)
    # 5.2 训练模型
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        for batch_input, batch_target in train_dataloader:
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output.squeeze(), batch_target)
            train_loss += loss
            loss.backward()
            optimizer.step()
        train_loss = train_loss/len(train_dataloader)
        train_loss_list.append(train_loss.item())

        scheduler.step(train_loss)

        if (epoch + 1) % 50 == 0:
            # 在验证集上计算损失并保存最佳模型
            with torch.no_grad():
                val_losses = []
                for val_batch_input, val_batch_target in test_dataloader:
                    val_output = model(val_batch_input)
                    val_loss = criterion(
                        val_output.squeeze(), val_batch_target)
                    val_losses.append(val_loss.item())
                avg_val_loss = np.mean(val_losses)
                val_loss_list.append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_params = model.state_dict()

            print(f'Epoch [{epoch+1}/{epochs}], Learn Rate: {scheduler.get_last_lr()[0]:.4e}, ' +
                  f'Training Loss: {train_loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}')

    print('Best Validation Loss:', best_val_loss)
    torch.save(best_model_params, save_path,
               _use_new_zipfile_serialization=False)


def load_rnn_model_best(model_type, model_params):
    import torch
    model = model_type(**model_params)
    model.load_state_dict(torch.load(
        str(model_type).split('.')[-1].split('\'')[0] + "_best.pth"))
    return model


def _prepare_data_rnn(series, last_months=12):
    last_date = series.index[:-last_months]
    data = series.values[:-last_months]
    y_true = series.values[-last_months:]
    last_months = last_date.month
    last_years = last_date.year
    return data, last_years, last_months, y_true


def evaluate_model(series_origin, series_pred):
    # 指定支持中文的字体，例如SimHei或者Microsoft YaHei
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    if not series_origin.index.isin(series_pred.index).any():
        print("预测的数据与已知无重合，不进行评估")
        print(series_pred, series_origin)
        plt.figure(figsize=(10, 6))
        plt.plot(series_pred, label='预测值')
        plt.show()
        return

    series_true = series_origin[series_origin.index.isin(series_pred.index)]
    series_pred_cover = series_pred[series_pred.index.isin(
        series_origin.index)]

    mse = mean_squared_error(series_true, series_pred_cover)
    mae = mean_absolute_error(series_true, series_pred_cover)
    r2 = r2_score(series_true, series_pred_cover)
    mape = mean_absolute_percentage_error(series_true, series_pred_cover)

    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R^2: {r2:.2f}')
    print(f'MAPE: {mape:.2f}')

    plt.figure(figsize=(10, 6))
    plt.plot(series_true, label='真实值')
    plt.plot(series_pred, label='预测值')
    plt.legend()
    plt.show()


def evaluate_model_xgboost(series_origin, series_pred, model):
    from xgboost import plot_importance
    # 绘制特征重要性
    plt.figure(figsize=(12, 6))
    plot_importance(model, ax=plt.gca())
    plt.show()

    evaluate_model(series_origin, series_pred)


def evaluate_model_prophet(series_test, df_pred):
    df_pred = df_pred[['ds', 'yhat']].set_index('ds')
    series_pred = df_pred['yhat']
    series_pred = series_pred[series_test.index[0]:]
    evaluate_model(series_test, series_pred)


def predict_to_future_rnn(model, series, scaler=None, months=24, last_months=12, feature_length=12):
    import torch
    # 准备初始输入
    data, data_years, data_months, _ = _prepare_data_rnn(series, last_months)
    year_next = data_years[-1]
    month_next = data_months[-1]

    # 拼接 data, data_years, data_months
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
        scaled_series = scaler.inverse_transform(
            series.values.reshape(-1, 1)).reshape(-1)
        series = pd.Series(scaled_series, index=series.index)

    pred_index = pd.date_range(
        start=series.index[-last_months], periods=months, freq='M')
    series_pred = pd.Series(result, index=pred_index)

    evaluate_model(series, series_pred)

    return result


def predict_to_future_arima(model, series, scaler=None, months=24, last_months=12):
    forecast_start = series.index[-last_months]
    forecast_end = forecast_start + pd.DateOffset(months=months)

    forecast: pd.Series = model.predict(start=forecast_start, end=forecast_end)

    if scaler is not None:
        scaled_forecast = scaler.inverse_transform(
            forecast.values.reshape(-1, 1)).reshape(-1)
        scaled_series = scaler.inverse_transform(
            series.values.reshape(-1, 1)).reshape(-1)
        forecast = pd.Series(scaled_forecast, index=forecast.index)
        series = pd.Series(scaled_series, index=series.index)

    evaluate_model(series, forecast)

    return forecast


def predict_to_future_es(model, series, scaler=None, months=24, last_months=12):
    return predict_to_future_arima(model, series, scaler, months, last_months)


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

    return forecast


def predict_to_future_xgboost(model, series, scaler=None, months=24, last_months=12):
    feature_length = len(model.feature_importances_) - 2
    features, _ = gen_xgboost_data(series, feature_length=feature_length)

    input_data = features[-last_months]
    year_next = series.index[-last_months].year
    month_next = series.index[-last_months].month - 1

    results = np.array([])
    # 预测和评估
    for _ in range(months):
        # 预测下一个月的销量
        y_pred = model.predict(input_data.reshape(1, -1)).item()

        # 将预测结果添加到结果列表中
        results = np.append(results, y_pred)

        # 更新输入数据
        input_data[:feature_length] = np.append(
            input_data[1:feature_length], y_pred)
        input_data[feature_length:] = time_transform(year_next, month_next + 1)

    year_next += 1 if month_next == 11 else 0
    month_next = (month_next + 1) % 12

    if scaler is not None:
        results = scaler.inverse_transform(
            results.reshape(-1, 1)).reshape(-1)
        scaled_series = scaler.inverse_transform(
            series.values.reshape(-1, 1)).reshape(-1)
        results = pd.Series(results, index=pd.date_range(
            series.index[-last_months], periods=months, freq='M'))
        series = pd.Series(scaled_series, index=series.index)

    evaluate_model_xgboost(series, results, model)


def _find_best_param_worker(series, order, seasonal_order):
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings('ignore')
    try:
        model_arima = SARIMAX(series, order=order,
                              seasonal_order=seasonal_order)  # 创建 SARIMA 模型对象
        model_arima_fit = model_arima.fit(disp=False)
    except:
        return order, seasonal_order, np.inf
    return order, seasonal_order, model_arima_fit.aic


def find_best_param(series, combined_param, process_num=8):
    import warnings
    from tqdm import tqdm
    import multiprocessing

    results = []

    with multiprocessing.Pool(processes=process_num) as pool:
        tasks = [pool.apply_async(_find_best_param_worker, args=(series, order, seasonal_order))
                 for order, seasonal_order in combined_param]

        # 等待所有任务完成，并实时打印进度
        # 使用tqdm显示进度
        for index, task in enumerate(tqdm(tasks, total=len(tasks), desc='Progress')):
            task.wait()  # 等待任务完成以更新进度条

        for task in tasks:
            results.append(task.get())

    warnings.filterwarnings('default')

    return results
