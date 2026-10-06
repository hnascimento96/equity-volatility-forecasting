import yfinance as yf
import pandas as pd
import json
import time
import base64
import requests

from pathlib import Path
from src.config import DATE_END, DATE_START, TICKERS, B3_SECTOR_INDEXES

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

B3_SECTOR_INDEXES_OUTPUT_FILE = Path(RAW_DIR / "sector_indexes_raw.csv")
 
REQUEST_PAUSE_SECONDS = 1
B3_URL = "https://sistemaswebb3-listados.b3.com.br/indexStatisticsProxy/IndexCall/GetPortfolioDay/"

def parse_response(results, index_name, year):
    """Turn B3's day-by-month table into a date-indexed series."""
    values = {}
    for row in results or []:
        for month in range(1, 13):
            raw = row.get(f"rateValue{month}")
            if raw:  # non-trading days come empty
                date = pd.Timestamp(year, month, row["day"])
                values[date] = float(raw.replace(".", "").replace(",", "."))
    return pd.Series(values, name=index_name, dtype=float).sort_index()
 
 
def fetch_index_year(index_name, year):
    """Daily closing values of one index for one calendar year."""
    params = json.dumps({"language": "pt-br", "index": index_name, "year": str(year)})
    token = base64.b64encode(params.encode()).decode()
    response = requests.get(B3_URL + token, timeout=30)
    response.raise_for_status()
    return parse_response(response.json().get("results"), index_name, year)
 
 
def download_prices(indexes, date_start, date_end):
    """Closing values of all indexes between date_start and date_end (inclusive)."""
    start, end = pd.Timestamp(date_start), pd.Timestamp(date_end)
    if start > end:
        raise ValueError(f"DATE_START ({date_start}) is after DATE_END ({date_end})")
 
    columns = []
    for index_name in indexes:
        parts = []
        for year in range(start.year, end.year + 1):
            parts.append(fetch_index_year(index_name, year))
            time.sleep(REQUEST_PAUSE_SECONDS)
        columns.append(pd.concat(parts))
 
    prices = pd.concat(columns, axis=1).sort_index().loc[start:end]
    prices.index.name = "Date"
    return prices
 
 
def print_coverage(prices):
    """Show the first and last available date of each index."""
    print("Coverage:")
    for index_name in prices.columns:
        series = prices[index_name].dropna()
        if series.empty:
            print(f"  {index_name}: no data")
        else:
            print(f"  {index_name}: {series.index[0]:%Y-%m-%d} to {series.index[-1]:%Y-%m-%d} "
                  f"({len(series)} trading days)")

def download(start=DATE_START, end=DATE_END):
    for ticker in TICKERS:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df.to_csv(RAW_DIR / f"{ticker}.csv")

    prices = download_prices(B3_SECTOR_INDEXES, DATE_START, DATE_END)
    print_coverage(prices)
    prices.to_csv(B3_SECTOR_INDEXES_OUTPUT_FILE, date_format="%Y-%m-%d")
    print(f"Saved {len(prices)} rows to {B3_SECTOR_INDEXES_OUTPUT_FILE}")

if __name__ == "__main__":
    download()
