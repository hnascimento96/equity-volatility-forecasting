import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, norm, skew, kurtosis, jarque_bera, chi2
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from pathlib import Path
from arch import arch_model
from src.config import TICKERS, B3_ASSET_SECTOR_INDEX

PROCESSED_DIR = Path("data/processed")

TRAIN_DATE_START = "2021-01-01"
TRAIN_DATE_END = "2024-12-31"
NLAGS = 20
ALPHA_SIGNIFICANCE = 0.05

def acf_correlations(returns, std_error):
    
    ret_corr = acf(returns, nlags=NLAGS)

    ret_corr = [float(element) for element in ret_corr]
    upper_lim = float(1.96*std_error)
    lower_lim = -float(1.96*std_error)

    acf_dict = {}
    acf_dict["lag"] = []
    acf_dict["acf_value"] = []
    acf_dict["ci_lower"] = []
    acf_dict["ci_upper"] = []
    acf_dict["out of range"] = []

    for i in range(1, len(ret_corr)):
        acf_dict["lag"].append(i)
        acf_dict["acf_value"].append(ret_corr[i])
        acf_dict["ci_lower"].append(lower_lim)
        acf_dict["ci_upper"].append(upper_lim)

        out_range = False if (lower_lim <= ret_corr[i] <= upper_lim) else True
        acf_dict["out of range"].append(out_range)

    acf_df = pd.DataFrame(acf_dict).set_index("lag")

    return acf_df


def lb_test(returns):
    lb = acorr_ljungbox(returns, lags=NLAGS)

    lb_df = lb.reset_index().loc[:,["index", "lb_pvalue"]]
    lb_df.columns = ["lag", "lb_pvalue"]
    lb_df = lb_df.set_index("lag")

    lb_df["Reject H0?"] = np.where(lb_df["lb_pvalue"] < ALPHA_SIGNIFICANCE, "Yes", "No")
    print(lb_df)

def get_correlations(returns, asset):
    se = 1/np.sqrt(len(returns))
    
    #Returns correlations
    ret_corr_df = acf_correlations(returns, se)

    ret_corr_rlv_df = ret_corr_df[ret_corr_df["out of range"] == True]
    print("-"*120)
    print("ACF (Return)")
    print(f"Asset: {asset}")
    print(ret_corr_rlv_df)


    print("-"*120)
    print("LB (Returns)")
    print(f"Asset: {asset}")
    lb_test(returns)

    #Returns^2 correlations
    ret2_corr_df = acf_correlations(returns**2, se)
    
    ret2_corr_rlv_df = ret2_corr_df[ret2_corr_df["out of range"] == True]
    print("-"*120)
    print("ACF (Return^2)")
    print(f"Asset: {asset}")
    print(ret2_corr_rlv_df)


    print("-"*120)
    print("LB (Returns^2)")
    print(f"Asset: {asset}")
    lb_test(returns**2)

def garch_training(returns):
    



if __name__== "__main__":

    asset = "PETR4.SA"

    data = pd.read_csv(PROCESSED_DIR / f"{asset}_daily.csv", index_col="Date", parse_dates=True)
    train = data["Daily log-return"].loc[TRAIN_DATE_START:TRAIN_DATE_END].dropna()

    get_correlations(train, asset)

    #print(f"Mean (train): {train.mean()}")
    #print(lb_test(train))

