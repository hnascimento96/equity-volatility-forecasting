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
from src.config import TICKERS, B3_ASSET_SECTOR_INDEX, DECAY_FACTOR_EWMA, ROLLING_WINDOW_SIZE

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

TRAIN_DATE_START = "2021-01-01"
TRAIN_DATE_END = "2024-12-31"

TOL = 1e-6

# Parameters alpha, beta, 
def garch_forecast(dist, name):

    asset = "PETR4.SA"
    data = pd.read_csv(PROCESSED_DIR / f"{asset}_daily.csv", index_col="Date", parse_dates=True)
    data_returns = data["Daily log-return"].dropna()
    train = data_returns.loc[TRAIN_DATE_START:TRAIN_DATE_END]
    count_train_days = len(train)
    data_rst = data_returns.reset_index()
    data = data_rst

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

    save_forecast(forecast_df, name)

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

def ewma_forecast(asset="PETR4.SA"):

    # Take last date values
    diag_data = pd.read_csv(MODELS_DIR / "ewma" / "ewma_PETR4.SA_diagnostics.csv",
                            parse_dates=True
                            )
    yesterday_date = diag_data.iloc[-1]["Date"]
    yesterday_return = diag_data.iloc[-1]["returns"]
    yesterday_var = diag_data.iloc[-1]["conditional_volatility"]**2

    test_data = pd.read_csv(PROCESSED_DIR / f"PETR4.SA_daily.csv", index_col="Date", parse_dates=True)
    test_data = test_data[test_data.index > yesterday_date]

    forecast_ewma = {}
    forecast_ewma["date"] = []
    forecast_ewma["return"] = []
    forecast_ewma["sigma2"] = []

    for date in test_data.index.to_list():
        variance = DECAY_FACTOR_EWMA*yesterday_var + (1 - DECAY_FACTOR_EWMA)*yesterday_return**2

        forecast_ewma["date"].append(date)
        forecast_ewma["return"].append(test_data.loc[date, "Daily log-return"])
        forecast_ewma["sigma2"].append(variance)
        
        yesterday_date = date
        yesterday_return = test_data.loc[date, "Daily log-return"]
        yesterday_var = variance

    forecast_df = pd.DataFrame(forecast_ewma).set_index("date")
    save_forecast(forecast_df, name="ewma")

    print("Forecast ewma (first 10 days)")
    print(forecast_df.head(10))

def rolling_forecast():

    data = pd.read_csv(PROCESSED_DIR / f"PETR4.SA_daily.csv", index_col="Date", parse_dates=True)

    # Take last date values
    diag_data = pd.read_csv(MODELS_DIR / "rolling_volatility" / "rolling_vol_PETR4.SA_diagnostics.csv",
                            parse_dates=True
                            )
    yesterday_date = diag_data.iloc[-1]["Date"]

    forecast_vol = {}
    forecast_vol["date"] = data[data.index > yesterday_date].index.to_list()
    forecast_vol["return"] = data[data.index > yesterday_date]["Daily log-return"].to_list()
    ahead_var = data["Daily log-return"].rolling(window=ROLLING_WINDOW_SIZE).var()
    forecast_vol["sigma2"] = ahead_var.shift(1)[ahead_var.index > yesterday_date].to_list()

    forecast_df = pd.DataFrame(forecast_vol).set_index("date")
    save_forecast(forecast_df, name="rolling_vol")

    print("Forecast rolling volatility (first 10 days)")
    print(forecast_df.head(10))


    

    
    
