import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, norm, skew, kurtosis, jarque_bera, chi2
from statsmodels.tsa.stattools import acf
from pathlib import Path
from src.config import TICKERS, B3_ASSET_SECTOR_INDEX

PROCESSED_DIR = Path("data/processed")

def returns_correlations(daily_returns):
    NLAGS = 20
    asset = "ITUB4.SA"
    ret_corr = acf(daily_returns[asset], nlags=NLAGS)

    acf_data = []
    for corr in ret_corr:
        value = []
        value.append(corr)
        value.append(1.96/np.sqrt(len(daily_returns)))
        value.append(1.96/np.sqrt(len(daily_returns)))
        acf_data.append(value)

    acf_df = pd.DataFrame(acf_data, columns=["ACF", "Lower Lim", "Upper Lim"])
    print(f"ACF:\n{acf_df}")


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

    print(daily_returns.head())
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

    print(regression_params)

def sector_regression(daily_returns):

    sector_returns = pd.read_csv(PROCESSED_DIR / "sector_indexes_daily.csv")

    assets_list = daily_returns.columns.to_list()

    print(assets_list)
    
    assets_list.remove("^BVSP")

    daily_returns = daily_returns.join(sector_returns, how="inner")

    regression_params = {}
    for asset in assets_list:

        sector = B3_ASSET_SECTOR_INDEX[asset]

        y = daily_returns[asset]
        X = daily_returns[["^BVSP", sector]]

        # adds constant to linear regression
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        regression = model.fit()

        regression_params[asset] = {}

        regression_params[asset]["const"] = float(regression.params["const"])
        regression_params[asset]["^BVSP"] = float(regression.params["^BVSP"])
        regression_params[asset][sector] = float(regression.params[sector])

        p_value = regression.pvalues
        regression_params[asset]["p-value"] = {
            "const": float(p_value["const"]),
            "^BVSP": float(p_value["^BVSP"]),
            sector: float(p_value[sector])
        }

        regression_params[asset]["r-squared"] = float(regression.rsquared)
        regression_params[asset]["r-squared-adj"] = float(regression.rsquared_adj)

        # significance level: 0,05
        conf_int = regression.conf_int()
        regression_params[asset]["conf_int"] = {
            "const": [
                float(conf_int.loc["const", 0]),
                float(conf_int.loc["const", 1])
            ],
            "IBOV": [
                float(conf_int.loc["^BVSP", 0]),
                float(conf_int.loc["^BVSP", 1])
            ],
            sector: [
                float(conf_int.loc[sector, 0]),
                float(conf_int.loc[sector, 1])
            ]
        }

    print(regression_params)


if __name__ == "__main__":
    daily_returns = concat_returns()
    pearson_r, spearman_r, pearson_p, spearman_p = correlation_analysis(daily_returns)
    normality_analysis(daily_returns)
    assets_symmetry = symmetry_analysis(daily_returns)
    normality_tests(daily_returns)

    ibov_regression(daily_returns)

    sector_regression(daily_returns)

    returns_correlations(daily_returns)

    #plt.show()

    #print(pearson_r)
    #print("-"*50)
    #print(pearson_p)
    #print("-"*50)
    #print(spearman_r)
    #print("-"*50)
    #print(spearman_p)




