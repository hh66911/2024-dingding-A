# %%
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import model_predict

# %%
series, scaler = model_predict.read_data_series_filter(
    filter_early='2016-01', file_index=2, scale=True)
features, targets = model_predict.gen_xgboost_data(series, feature_length=12)

# %%
# 假设X和y已经被定义
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2)

# 对于回归任务
reg = LGBMRegressor(
    objective='mse',
    n_estimators=50,
    learning_rate=.1,
    max_depth=7,
    num_leaves=15,
    subsample=.8,
    colsample_bytree=.8,
    force_row_wise=True,
    boosting_type='gbdt',
)
reg.fit(features, targets, eval_set=[(X_test, y_test)], eval_metric='mse')

results, _ = model_predict.predict_to_future_lgbm(
    reg, series, scaler, last_months=12)

results.to_excel('results.xlsx')
