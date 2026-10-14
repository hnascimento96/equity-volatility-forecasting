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
from src.config import TICKERS, B3_ASSET_SECTOR_INDEX


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

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

def garch_training(returns, asset_name):

    model = arch_model(100*returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = model.fit(disp="off")

    #Save GARCH params to .csv file
    params_df = res.params.to_frame(name="value")
    params_df.index.name = "param"
    #params_df = params_df.reset_index()
    params_df.to_csv(MODELS_DIR / "garch" / f"garch_{asset_name}_params.csv")
    
    std_res = np.asarray(res.std_resid)

    diagnostics = pd.DataFrame({
        "returns" : returns,
        "conditional_volatility" : res.conditional_volatility,
        "standardized_residuals" : res.std_resid
    })

    diagnostics.to_csv(MODELS_DIR / "garch" / f"garch_{asset_name}_diagnostics.csv")

    print(f"-"*100)
    print(f"GARCH params:")
    print(res.summary())

    

    plt.figure(figsize=(6.4, 8))
    plt.plot(returns.index, std_res)
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(f"Standardized residuals ")
    plt.title(f"Garch(1,1) Model: {asset_name}")
    plt.tight_layout()

    return std_res

def residual_correlation(res, asset_name):
    print("-"*120)
    print("\n")
    print("Residual analysis")

    #Standardized residual analysis

    #ACF(res)
    std_res_acf_df = acf_correlations(res,
                     std_error=1/np.sqrt(len(res)))
    print("ACF Test")
    print(f"Asset:{asset_name}")
    print(std_res_acf_df)

    #LB(res)
    print("-"*120)
    print("LB Test")
    print(f"Asset:{asset_name}")
    lb_test(res)

    ##########################
    #Residual^2 Analysis
    print("-"*120)
    print("\n")
    print("Residual analysis^2")
    print(f"Asset: {asset_name}")

    #Standardized residual analysis

    #ACF(res^2)
    std_res_acf_df = acf_correlations(res**2,
                     std_error=1/np.sqrt(len(res)))
    print("ACF Test")
    print(f"Asset:{asset_name}")
    print(std_res_acf_df)

    #LB(res^2)
    print("-"*120)
    print("LB Test")
    print(f"Asset:{asset_name}")
    lb_test(res**2)

def qq_plot_garch_std_res(res, res_params, asset_name):
    nu = res_params.loc["nu", "value"]
    scale = np.sqrt((nu-2)/nu)


    sm.qqplot(
        res["standardized_residuals"],
        dist=t,
        distargs=(nu,),
        loc=0,
        scale=scale,
        line="45"
    )

def ad_test_tstudent(res_std, nu):

    sorted_res = np.sort(res_std)
    scale = np.sqrt((nu-2)/nu)

    # Generates a CDF for t-student with res_params
    F = t.cdf(
        sorted_res,
        df = nu,
        loc = 0,
        scale=scale
    )

    n = len(sorted_res)
    A2_sum = 0

    for i in range(1, n + 1):
        A2_sum += (2 * i - 1) * (
            np.log(F[i - 1]) +
            np.log(1 - F[n - i])
        )

    A2 = -n - A2_sum/n
    return A2

def garch_adherence_test(res, res_params, seed=42):
    N_bootstrap = 999
    rng = np.random.default_rng(seed)

    A2_obs = ad_test_tstudent(res["standardized_residuals"], res_params.loc["nu", "value"])
    print(f"A2 obs: {A2_obs}")

    count_above = 0

    sim_model = ConstantMean(
        None,
        volatility=GARCH(p=1, q=1),
        distribution=StudentsT(seed=rng),
    )
            

    for _ in range(0, N_bootstrap):

        # Generates unconditional volatility observations \sigma_t satisfying 
        # \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2,
        # with \epsilon_t innovations from a t-student distribution with Var(\epsilon_t) = 1
        # 
        # burn: drops out first 'burn' observations
        # nobs: number of valid observations after burn
        # 

        sim = sim_model.simulate(
            params=res_params["value"],
            nobs=700,
            burn=30
        )

        ret = sim["data"]

        # GARCH Model
        model = arch_model(ret, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        res = model.fit(disp="off")

        # Anderson-Darling (AD) Test

        A2 = ad_test_tstudent(res.std_resid, res.params["nu"])
        if A2 > A2_obs: count_above += 1

    p_value = (1 + count_above)/(N_bootstrap + 1)

    print(f"p-value for AD: {p_value}")
    
def residual_analysis(res_diag, asset_name):
    residual_correlation(res_diag["standardized_residuals"], asset_name)
    

if __name__== "__main__":

    asset = "PETR4.SA"

    data = pd.read_csv(PROCESSED_DIR / f"{asset}_daily.csv", index_col="Date", parse_dates=True)
    train = data["Daily log-return"].loc[TRAIN_DATE_START:TRAIN_DATE_END].dropna()

    get_correlations(train, asset)

    garch_training(train, asset)

    garch_residual_diag = pd.read_csv(MODELS_DIR / "garch" / f"garch_{asset}_diagnostics.csv")
    residual_analysis(garch_residual_diag, asset)

    garch_params = pd.read_csv(MODELS_DIR / "garch" / f"garch_{asset}_params.csv", index_col="param" )
    print(f"GARCH DF params:")
    print(garch_params)

    qq_plot_garch_std_res(garch_residual_diag, garch_params, asset)

    print("-"*120)
    print("GARCH-t adherence test")
    garch_adherence_test(garch_residual_diag, garch_params)

    #ad_test_tstudent(garch_residual_diag["standardized_residuals"], garch_params)

    plt.show()

    #print(f"Mean (train): {train.mean()}")
    #print(lb_test(train))

