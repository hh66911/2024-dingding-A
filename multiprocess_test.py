import warnings
import time
import multiprocessing
import itertools
import model_predict
from statsmodels.tsa.statespace.sarimax import SARIMAX

series = model_predict.read_data_series(scale=False)

d = range(0, 2)
p = q = range(1, 3)  # 减小范围以便更快看到结果
pdq = list(itertools.product(p, d, q))

# 季节性部分：季节 AR 阶数为 1，差分阶数为 0，季节 MA 阶数为 1，季节周期为 12 (假设数据是按月季节性的)
SEASONAL_ORDER = (1, 0, 1, 12)


if __name__ == "__main__":
    model_predict.find_best_param(series, pdq, SEASONAL_ORDER)
