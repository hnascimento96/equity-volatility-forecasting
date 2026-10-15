import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, norm, skew, kurtosis, jarque_bera, chi2, t
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from pathlib import Path
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, StudentsT
from src.io_utils import save_table, save_forecast
from src.config import TICKERS, B3_ASSET_SECTOR_INDEX
from src.forecast import garch_forecast

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

TRAIN_DATE_START = "2021-01-01"
TRAIN_DATE_END = "2024-12-31"

if __name__ == "__main__":
    

    #print("-"*120)
    #print("GARCH(1,1)-normal forecast analysis")

    #summary_garch_n = garch_forecast(data_rst, count_train_days, dist="normal", name="garch_n")

    print("-"*120)
    print("GARCH(1,1)-t forecast analysis")
    summary_garch_t = garch_forecast(dist="t", name="garch_t")
    

    
    
