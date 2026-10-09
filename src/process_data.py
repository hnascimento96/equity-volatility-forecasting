import pandas as pd
import numpy as np
from pathlib import Path
from src.config import TICKERS, ROLLING_WINDOW_SIZE, B3_SECTOR_INDEXES

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = Path("data/raw")

def get_basic_metrics(df, name):

    #daily returns
    df["Daily return"] = df['Close'].pct_change()
    df["Daily log-return"] = np.log(df['Close'] / df['Close'].shift(1))
    total_return = df["Daily log-return"].sum()

    
    #monthly returns
    monthly_prices = df['Close'].resample('ME').last()
    monthly_returns = pd.DataFrame(index=monthly_prices.index, 
                                   columns=["Monthly returns", "Monthly log-returns"])
    monthly_returns["Monthly returns"] = monthly_prices.pct_change()
    monthly_returns["Monthly log-returns"] = np.log(monthly_prices / monthly_prices.shift(1))
    mean_monthly_return_arith = monthly_returns["Monthly returns"].mean()
    mean_monthly_return_log = monthly_returns["Monthly log-returns"].mean()

    #volatility
    df[f"Volatility_{ROLLING_WINDOW_SIZE}"] = df['Daily log-return'].rolling(ROLLING_WINDOW_SIZE).std()*np.sqrt(252)
    volatility_mean = df[f"Volatility_{ROLLING_WINDOW_SIZE}"].mean()

    #drawdown
    peak = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] - peak)/peak
    max_drawdown = df["Drawdown"].min()

    df.to_csv(PROCESSED_DIR / f"{name}_daily.csv")
    monthly_returns.to_csv(PROCESSED_DIR / f"{name}_monthly.csv")

    return max_drawdown, volatility_mean, total_return, mean_monthly_return_arith, mean_monthly_return_log

def process_data():

    summary_df = pd.DataFrame(columns=["asset", "max_drawdown", "volatility_mean",
                                    "total_return", "mean_monthly_return_arith", "mean_monthly_return_log"])

    for ticker in TICKERS:
        df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col=0, parse_dates=True, skiprows=[1,2], dtype={
            "Close": float,
            "High": float,
            "Low": float,
            "Open": float,
            "Volume": float
        })

        df.index.name = "Date"

        max_drawdown, volatility_mean, total_return, mean_monthly_return_arith, mean_monthly_return_log = get_basic_metrics(df, ticker)
        summary_df.loc[len(summary_df)] = [ticker, max_drawdown, volatility_mean,
                                        total_return, mean_monthly_return_arith, mean_monthly_return_log]

    summary_df.to_csv(PROCESSED_DIR / f"summary.csv", index=False)
    print("Summary")
    print(summary_df)

def add_log_returns(prices):
    """Keep dates where every index has a value and append log returns."""
    complete = prices.dropna()
    dropped = len(prices) - len(complete)
    if dropped:
        print(f"WARNING: dropped {dropped} dates with a missing value in some index.")
 
    returns = np.log(complete / complete.shift(1))
    returns.columns = [f"{name} (log) return" for name in complete.columns]
 
    table = pd.concat([complete, returns], axis=1)
    table.index.name = "Date"
    return table

def process_sector_indexes():
    prices = pd.read_csv(RAW_DIR / "sector_indexes_raw.csv", index_col="Date", parse_dates=True)
    table = add_log_returns(prices)
    table.to_csv(PROCESSED_DIR / "sector_indexes_daily.csv", date_format="%Y-%m-%d")
    print(f"Saved {len(table)} rows to {PROCESSED_DIR / "sector_indexes_daily.csv"} "
          f"({table.index[0]:%Y-%m-%d} to {table.index[-1]:%Y-%m-%d})")



if __name__ == "__main__":
    process_sector_indexes()
    process_data()
    