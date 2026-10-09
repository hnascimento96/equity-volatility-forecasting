import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, norm, skew, kurtosis, jarque_bera, chi2
from statsmodels.tsa.stattools import acf
from pathlib import Path
from src.config import TICKERS, B3_ASSET_SECTOR_INDEX

PROCESSED_DIR = Path("data/processed")

def concat_returns():
    dfs = []
    for name in TICKERS:
        df = pd.read_csv(PROCESSED_DIR / f"{name}_daily.csv")
        dfs.append(df["Daily log-return"])

    df_returns = pd.concat(dfs, axis=1)
    df_returns.columns = TICKERS.keys()

    return df_returns.dropna()

def normality_tests(daily_returns):
    jb_values = []
    for asset in daily_returns.columns:
        jb = {}
        res = jarque_bera(daily_returns[asset])
        jb["p-value"] = res.pvalue
        jb["asset"] = asset
        jb_values.append(jb)
    return jb_values

def symmetry_analysis(daily_returns):
    assets = []

    for asset in daily_returns.columns:
        data = {}
        data["asset"] = asset
        data["skew"] = skew(daily_returns[asset])
        data["kurtosis"] = kurtosis(daily_returns[asset])
        assets.append(data)
    return assets
    

def normality_analysis(daily_returns):

    #name = "Itau"

    for name in daily_returns.columns:
        plt.figure(figsize=(10, 6))

        plt.hist(
            daily_returns[name],
            bins=50,
            density=True,
            alpha=0.6,
            color="steelblue",
            label="Observed returns"
        )

        mu = daily_returns[name].mean()
        sigma = daily_returns[name].std()
        
        x = np.linspace(daily_returns[name].min(), daily_returns[name].max(), 500)
        normal = norm.pdf(x, loc=mu, scale=sigma)

        plt.plot(
            x,
            normal,
            color="red",
            linewidth=2,
            label="Theoretical normal"
        )

        plt.xlabel("Daily log-return")
        plt.ylabel("Density")
        plt.title(f"{name} returns")
        plt.legend()

def correlation_analysis(daily_returns):

    pearson_r, spearman_r, pearson_p, spearman_p = [np.zeros((len(TICKERS), len(TICKERS))) for _ in range(4)]

    #print(daily_returns.head())
    for i, asset_i in enumerate(daily_returns.columns):
        for j, asset_j in enumerate(daily_returns.columns):
            pearson_r[i, j], pearson_p[i, j] = pearsonr(daily_returns[asset_i], daily_returns[asset_j])
            spearman_r[i, j], spearman_p[i, j] = spearmanr(daily_returns[asset_i], daily_returns[asset_j])
    return pearson_r, spearman_r, pearson_p, spearman_p

def ibov_regression(daily_returns):

    assets_list = daily_returns.columns.to_list()
    assets_list.remove("^BVSP")
    regression_params = {}
    for asset in assets_list:
        y = daily_returns[asset]
        X = daily_returns["^BVSP"]

        # adds constant to linear regression
        X = sm.add_constant(X)

        model = sm.OLS(y,X)
        regression = model.fit()
        res = regression.resid

        regression_params[asset] = {}

        regression_params[asset]["const"] = float(regression.params["const"])
        regression_params[asset]["^BVSP"] = float(regression.params["^BVSP"])

        p_value = regression.pvalues
        regression_params[asset]["p-value"] = {
            "const" : float(p_value["const"]),
            "^BVSP" : float(p_value["^BVSP"])
        }

        regression_params[asset]["r-squared"] = float(regression.rsquared)
        regression_params[asset]["r-squared-adj"] = float(regression.rsquared_adj)

        #significance level: 0,05
        conf_int = regression.conf_int()
        regression_params[asset]["conf_int"] = {
        "const": [
            float(conf_int.loc["const", 0]),
            float(conf_int.loc["const", 1])
        ],
        "IBOV": [
            float(conf_int.loc["^BVSP", 0]),
            float(conf_int.loc["^BVSP", 1])
        ]
        }

    print("-"*120)
    print("Linear Regression: ^BVSP")

    rows = {}
    for asset, r in regression_params.items():
        rows[asset] = {
            "alfa": r["const"],
            "beta": r["^BVSP"],
            "p_alfa": r["p-value"]["const"],
            "p_beta": r["p-value"]["^BVSP"],
            "r2": r["r-squared"],
            "r2_adj": r["r-squared-adj"],
            "alfa_ic_inf": r["conf_int"]["const"][0],
            "alfa_ic_sup": r["conf_int"]["const"][1],
            "beta_ic_inf": r["conf_int"]["IBOV"][0],
            "beta_ic_sup": r["conf_int"]["IBOV"][1],
        }

    df_regression = pd.DataFrame.from_dict(rows, orient="index")
    df_regression.index.name = "asset"

    print(df_regression)

    return df_regression

def sector_regression(daily_returns):

    sector_returns = pd.read_csv(PROCESSED_DIR / "sector_indexes_daily.csv")

    assets_list = daily_returns.columns.to_list()

    print(assets_list)
    
    assets_list.remove("^BVSP")

    daily_returns = daily_returns.join(sector_returns, how="inner")

    rows = {}

    rows["alpha"] = {}
    rows["beta_bvsp"] = {}
    rows["beta_sector"] = {}
    for asset in assets_list:

        sector = B3_ASSET_SECTOR_INDEX[asset]

        y = daily_returns[asset]
        X = daily_returns[["^BVSP", sector]]

        # adds constant to linear regression
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        regression = model.fit()

        conf_int = regression.conf_int()

        rows["alpha"][asset] = {
            "param" : "const",
            "alpha": float(regression.params["const"]),
            "p_value": float(regression.pvalues["const"]),
            "r2": float(regression.rsquared),
            "r2_adj": float(regression.rsquared_adj),
            "const_ic_inf": float(conf_int.loc["const", 0]),
            "const_ic_sup": float(conf_int.loc["const", 1]),
        }

        rows["beta_bvsp"][asset] = {
            "param" : "^BVSP",
            "beta": float(regression.params["^BVSP"]),
            "p_value": float(regression.pvalues["^BVSP"]),
            "r2": float(regression.rsquared),
            "r2_adj": float(regression.rsquared_adj),
            "beta_ci_lower": float(conf_int.loc["^BVSP", 0]),
            "beta_ci_upper": float(conf_int.loc["^BVSP", 1]),
        }

        rows["beta_sector"][asset] = {
            "param": sector,
            "sector_beta": float(regression.params[sector]),
            "p_value": float(regression.pvalues[sector]),
            "r2": float(regression.rsquared),
            "r2_adj": float(regression.rsquared_adj),
            "sector_beta_ci_lower": float(conf_int.loc[sector, 0]),
            "sector_beta_ci_upper": float(conf_int.loc[sector, 1]),
        }

    df_alpha = pd.DataFrame.from_dict(rows["alpha"], orient="index")
    df_alpha.index.name = "asset"

    df_beta_bvsp = pd.DataFrame.from_dict(rows["beta_bvsp"], orient="index")
    df_beta_bvsp.index.name = "asset"

    df_sector = pd.DataFrame.from_dict(rows["beta_sector"], orient="index")
    df_sector.index.name = "asset"

    print("-"*120)
    print("Multiple Regression: Asset vs (^BVSP, Sector Index)")
    print("\n")
    print("const params")
    print(df_alpha)
    print("\n")
    print("^BVSP params")
    print(df_beta_bvsp)
    print("\n")
    print("sector index params")
    print(df_sector)

if __name__ == "__main__":
    daily_returns = concat_returns()
    pearson_r, spearman_r, pearson_p, spearman_p = correlation_analysis(daily_returns)
    normality_analysis(daily_returns)
    assets_symmetry = symmetry_analysis(daily_returns)
    normality_tests(daily_returns)

    ibov_regression(daily_returns)

    sector_regression(daily_returns)

    #plt.show()

    #print(pearson_r)
    #print("-"*50)
    #print(pearson_p)
    #print("-"*50)
    #print(spearman_r)
    #print("-"*50)
    #print(spearman_p)




