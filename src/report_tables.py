"""
Tables for the technical report.

Reads the processed data and the diagnostics CSVs produced by the model
scripts, and writes report-ready tables to results/tables/.
Nothing is re-estimated here.

Run from the repository root:
    python -m src.report_tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.io_utils import save_table

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

ASSET = "PETR4.SA"
TRAIN_DATE_START = "2021-01-01"
TRAIN_DATE_END = "2024-12-31"
TRADING_DAYS = 252

LB_LAGS_RETURNS = [5, 10, 20]
LB_LAGS_RESIDUALS = [10, 20]
ARCH_LM_LAGS = 10

# Model name -> prefix of its diagnostics CSV inside models/
MODELS = {
    "Rolling": "rolling_volatility/rolling_vol",
    "EWMA-N": "ewma/ewma",
    "GARCH-N": "normal_garch/normal_garch",
    "GARCH-t": "garch/garch",
}


def load_train_returns(asset):
    data = pd.read_csv(PROCESSED_DIR / f"{asset}_daily.csv",
                       index_col="Date", parse_dates=True)
    return data["Daily log-return"].loc[TRAIN_DATE_START:TRAIN_DATE_END].dropna()


def arch_lm_pvalue(z, nlags):
    # result_object=False keeps the current tuple output and silences the
    # statsmodels FutureWarning; older versions don't accept the argument.
    try:
        out = het_arch(z, nlags=nlags, result_object=False)
    except TypeError:
        out = het_arch(z, nlags=nlags)
    return out[1]


def descriptive_stats(asset):
    r = load_train_returns(asset)
    jb = jarque_bera(r)

    stats = pd.Series({
        "start": r.index.min().date().isoformat(),
        "end": r.index.max().date().isoformat(),
        "nobs": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "annualized_vol": r.std() * np.sqrt(TRADING_DAYS),
        "min": r.min(),
        "max": r.max(),
        "skew": skew(r),
        "excess_kurt": kurtosis(r),
        "jb_stat": jb.statistic,
        "jb_p": jb.pvalue,
    })

    save_table(stats.to_frame("value"), "descriptive_stats")


def lb_returns(asset):
    r = load_train_returns(asset)

    rows = []
    for series_name, series in [("r", r), ("r2", r**2)]:
        lb = acorr_ljungbox(series, lags=LB_LAGS_RETURNS)
        for lag in LB_LAGS_RETURNS:
            rows.append({
                "series": series_name,
                "lag": lag,
                "lb_stat": lb.loc[lag, "lb_stat"],
                "lb_p": lb.loc[lag, "lb_pvalue"],
            })

    save_table(pd.DataFrame(rows).set_index(["series", "lag"]), "lb_returns")


def residual_diagnostics(asset):
    rows = {}
    for name, prefix in MODELS.items():
        z = pd.read_csv(MODELS_DIR / f"{prefix}_{asset}_diagnostics.csv",
                        index_col="Date", parse_dates=True)["standardized_residuals"].dropna()

        jb = jarque_bera(z)
        lb = acorr_ljungbox(z, lags=LB_LAGS_RESIDUALS)
        lb2 = acorr_ljungbox(z**2, lags=LB_LAGS_RESIDUALS)

        row = {
            "nobs": len(z),
            "mean": z.mean(),
            "std": z.std(),
            "skew": skew(z),
            "excess_kurt": kurtosis(z),
            "jb_stat": jb.statistic,
            "jb_p": jb.pvalue,
        }
        for lag in LB_LAGS_RESIDUALS:
            row[f"lb_z_{lag}_p"] = lb.loc[lag, "lb_pvalue"]
        for lag in LB_LAGS_RESIDUALS:
            row[f"lb_z2_{lag}_p"] = lb2.loc[lag, "lb_pvalue"]
        row[f"arch_lm_{ARCH_LM_LAGS}_p"] = arch_lm_pvalue(z, ARCH_LM_LAGS)

        rows[name] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    save_table(df, "residual_diagnostics")


if __name__ == "__main__":
    descriptive_stats(ASSET)
    lb_returns(ASSET)
    residual_diagnostics(ASSET)
    print("Tables saved to results/tables/")