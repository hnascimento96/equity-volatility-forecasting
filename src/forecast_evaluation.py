import pandas as pd
import numpy as np
from pathlib import Path

FORECASTS_DIR = Path("results/forecasts")

def mse(model_name):
    data = pd.read_csv(FORECASTS_DIR / f"forecast_{model_name}.csv")
    
    data["diff2"] = (data["return"]**2 - data["sigma2"])**2
    mse = data["diff2"].mean()

    return mse

def qlike(model_name):
     data = pd.read_csv(FORECASTS_DIR / f"forecast_{model_name}.csv")

     data["qlike_inst"] = np.log(data["sigma2"]) + data["return"]**2/data["sigma2"]
     qlike = data["qlike_inst"].mean()
     
     return qlike

def qlike_evaluation():
    values = []
    values.append(qlike("rolling_vol"))
    values.append(qlike("ewma"))
    values.append(qlike("garch_n"))
    values.append(qlike("garch_t"))
    
    return values


def mse_evaluation():

    values = []
    values.append(mse("rolling_vol"))
    values.append(mse("ewma"))
    values.append(mse("garch_n"))
    values.append(mse("garch_t"))

    return values
    


if __name__ == "__main__":

    evaluation = {}
    evaluation["model"] = ["Rolling volatility",
                           "EWMA",
                           "GARCH(1,1)-n",
                           "GARCH(1,1)-t"
                            ]
    evaluation["MSE"] = mse_evaluation()
    evaluation["QLIKE"] = qlike_evaluation()

    evaluation_df = pd.DataFrame(evaluation).set_index("model")

    print("Evaluation")
    print(evaluation_df)

