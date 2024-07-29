import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import model_predict
import sys
import itertools
import pickle

series = model_predict.read_data_series(scale=False)

d = range(0, 2)
p = q = range(1, 12)
pdq = list(itertools.product(p, d, q))

sd = range(0, 2)
sp = sq = range(1, 3)
seasonal_pdq = list(itertools.product(sp, sd, sq, [12]))

combined_param = list(itertools.product(pdq, seasonal_pdq))

d = range(0, 3)
p = q = [1, 2, 3] + list(range(12, 16))
pdq = list(itertools.product(p, d, q))

sd = range(0, 3)
sp = sq = [1, 2, 3]
seasonal_pdq = list(itertools.product(sp, sd, sq, [12]))

combined_param = list(itertools.product(pdq, seasonal_pdq))

if __name__ == '__main__':
    results = model_predict.find_best_param(series, combined_param, 12)
    pickle.dump(results, open('results.pkl', 'wb'))
