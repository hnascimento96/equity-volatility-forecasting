import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from src.io_utils import save_table

MODELS_DIR = Path("models")
MODELS = {"Rolling": "rolling_volatility/rolling_vol", "EWMA-N": "ewma/ewma",
          "GARCH-N": "normal_garch/normal_garch", "GARCH-t": "garch/garch"}

def residual_diagnostics(asset):
    rows = {}
    for name, prefix in MODELS.items():
        z = pd.read_csv(MODELS_DIR / f"{prefix}_{asset}_diagnostics.csv",
                        index_col="Date", parse_dates=True)["standardized_residuals"].dropna()
        jb = jarque_bera(z)
        lb = acorr_ljungbox(z, lags=[10, 20])
        lb2 = acorr_ljungbox(z**2, lags=[10, 20])
        _, arch_lm_p, _, _ = het_arch(z, nlags=10)
        rows[name] = {"nobs": len(z), "mean": z.mean(), "std": z.std(),
                      "skew": skew(z), "excess_kurt": kurtosis(z),
                      "jb_stat": jb.statistic, "jb_p": jb.pvalue,
                      "lb_z_10_p": lb.loc[10, "lb_pvalue"], "lb_z_20_p": lb.loc[20, "lb_pvalue"],
                      "lb_z2_10_p": lb2.loc[10, "lb_pvalue"], "lb_z2_20_p": lb2.loc[20, "lb_pvalue"],
                      "arch_lm_10_p": arch_lm_p}
    save_table(pd.DataFrame.from_dict(rows, orient="index"), "residual_diagnostics")

if __name__ == "__main__":
    residual_diagnostics("PETR4.SA")