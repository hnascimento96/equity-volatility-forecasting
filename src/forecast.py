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

import warnings
from arch.utility.exceptions import StartingValueWarning

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

TRAIN_DATE_START = "2021-01-01"
TRAIN_DATE_END = "2024-12-31"

TOL = 1e-6


def garch_forecast(data, count_train_days, dist, name):

    forecast_vol = {}
    forecast_vol["date"] = []
    forecast_vol["return"] = []
    forecast_vol["sigma2"] = []
    forecast_vol["mu"] = []
    forecast_vol["omega"] = []
    forecast_vol["alpha"]= []
    forecast_vol["beta"] = []
    forecast_vol["persistence"] = []
    forecast_vol["valid_params"] = []
    forecast_vol["converged"] = []

    if dist == "t": forecast_vol["nu"] = []

    prev_params = None

    for today in range(count_train_days, len(data)):

        # Gets GARCH params from day 0 to yesterday
        # Due to convergence concerns, we multiply log-returns*100
        window_data = data.iloc[: today]
        
        model = arch_model(100*window_data["Daily log-return"],
                           mean="Constant",
                           vol="GARCH",
                           p=1,
                           q=1,
                           dist=dist
        )


        res = model.fit(disp="off", starting_values=prev_params)
        prev_params = res.params

        # Estimates today's volatility
        omega = res.params["omega"]
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]

        residual_yesterday = res.resid.iloc[-1]
        vol_yesterday = res.conditional_volatility.iloc[-1]
        variance_today = (omega + alpha*residual_yesterday**2 + beta*vol_yesterday**2)/10**4

        # Stores data in a dictionary
        forecast_vol["date"].append(data.loc[today,"Date"])
        forecast_vol["return"].append(data.loc[today,"Daily log-return"])
        forecast_vol["sigma2"].append(variance_today)
        forecast_vol["mu"].append(res.params["mu"]/100)
        forecast_vol["alpha"].append(alpha)
        forecast_vol["beta"].append(beta)
        forecast_vol["omega"].append(omega)
        
        forecast_vol["converged"].append(not(res.convergence_flag))

        if dist == "t": forecast_vol["nu"].append(res.params["nu"])

        forecast_vol["persistence"].append(res.params["alpha[1]"] + res.params["beta[1]"])
        #forecast_vol["start_ignored"].append(start_ignored)


        # Implements "valid_params" flag
        conditions = [alpha >= TOL,
                      beta >= TOL,
                      alpha + beta < 1 - TOL,
                      omega > TOL
        ]

        forecast_vol["valid_params"].append(all(conditions))

    forecast_df = pd.DataFrame(forecast_vol).set_index("date")

    save_forecast(forecast_df, "garch_t")

    # Summary of GARCH(1,1)-t convergence
    summary = pd.Series({
        "n_forecasts": len(forecast_df),
        "n_not_converged": int((~forecast_df["converged"]).sum()),
        "first_date": forecast_df.index.min().strftime("%Y-%m-%d"),
        "last_date": forecast_df.index.max().strftime("%Y-%m-%d"),
        }, name=name)

    # Prints last 10 forecast volatilities
    print(f"Forecast volatilities (last 10):")
    print(forecast_df.tail(10))

    # Prints summary of convergence
    print("\n")
    print("Convergence summary:")
    print(summary)

    # Checking for invalid params
    print("\n")
    print("Invalid params:")
    print(forecast_df[forecast_df["valid_params"] == 0]
          if len(forecast_df[forecast_df["valid_params"] == 0]) > 0
          else "No invalid params detected")

    return summary

if __name__ == "__main__":
    asset = "PETR4.SA"
    
    data = pd.read_csv(PROCESSED_DIR / f"{asset}_daily.csv", index_col="Date", parse_dates=True)

    data_returns = data["Daily log-return"].dropna()
    train = data_returns.loc[TRAIN_DATE_START:TRAIN_DATE_END]

    count_train_days = len(train)

    data_rst = data_returns.reset_index()

    print("-"*120)
    print("GARCH(1,1)-normal forecast analysis")

    summary_garch_n = garch_forecast(data_rst, count_train_days, dist="normal", name="garch_n")

    print("-"*120)
    print("GARCH(1,1)-t forecast analysis")
    summary_garch_t = garch_forecast(data_rst, count_train_days, dist="t", name="garch_t")
    

    
    
