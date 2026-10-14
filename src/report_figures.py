"""
Figures for the technical report.

Reads the processed data and the diagnostics CSVs produced by the model
scripts, and writes vector PDFs to results/figures/.
Figures have no titles on purpose: titles go in the LaTeX captions.

Run from the repository root:
    python -m src.report_figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, t, probplot

from src.io_utils import save_fig

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

ASSET = "PETR4.SA"
TRAIN_DATE_START = "2021-01-01"
TRAIN_DATE_END = "2024-12-31"
TRADING_DAYS = 252
NLAGS = 20

FIG_WIDTH = 6.5  # inches: text width of the report page

# Fixed colour per model (same colour = same model in every figure)
COLORS = {
    "Rolling": "#2a78d6",
    "EWMA-N": "#eb6834",
    "GARCH-N": "#1baf7a",
    "GARCH-t": "#eda100",
}
MAIN_COLOR = "#2a78d6"
REF_COLOR = "#52514e"   # reference lines (normal density, 45-degree line, bands)
SHADE_COLOR = "#e9e8e4"

# Model name -> (prefix of diagnostics CSV, volatility in percent?)
# GARCH models were fitted on 100*returns, so their volatility is in %.
MODELS = {
    "Rolling": ("rolling_volatility/rolling_vol", False),
    "EWMA-N": ("ewma/ewma", False),
    "GARCH-N": ("normal_garch/normal_garch", True),
    "GARCH-t": ("garch/garch", True),
}


def load_prices(asset):
    return pd.read_csv(PROCESSED_DIR / f"{asset}_daily.csv",
                       index_col="Date", parse_dates=True)


def load_train_returns(asset):
    data = load_prices(asset)
    return data["Daily log-return"].loc[TRAIN_DATE_START:TRAIN_DATE_END].dropna()


def load_diagnostics(asset, model):
    prefix, _ = MODELS[model]
    return pd.read_csv(MODELS_DIR / f"{prefix}_{asset}_diagnostics.csv",
                       index_col="Date", parse_dates=True)


def sample_acf(x, nlags):
    # Same estimator as statsmodels.tsa.stattools.acf (adjusted=False)
    x = np.asarray(x) - np.mean(x)
    denom = np.sum(x**2)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def qq_panel(ax, z, dist, label):
    (theoretical, ordered), _ = probplot(z, dist=dist)
    ax.scatter(theoretical, ordered, s=6, color=MAIN_COLOR, alpha=0.7, linewidths=0)
    lims = [min(theoretical.min(), ordered.min()), max(theoretical.max(), ordered.max())]
    ax.plot(lims, lims, color=REF_COLOR, lw=1, ls="--")
    ax.set_xlabel(f"Theoretical quantiles ({label})")
    ax.set_ylabel("Sample quantiles")


def fig_price_returns(asset):
    data = load_prices(asset)
    train_end = pd.Timestamp(TRAIN_DATE_END)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4.2), sharex=True)

    ax1.plot(data.index, data["Close"], color=MAIN_COLOR, lw=1)
    ax1.set_ylabel("Adjusted close (BRL)")

    ax2.plot(data.index, data["Daily log-return"], color=MAIN_COLOR, lw=0.6)
    ax2.set_ylabel("Daily log-return")

    for ax in (ax1, ax2):
        ax.axvspan(train_end, data.index.max(), color=SHADE_COLOR, zorder=0)
    ax1.text(train_end, ax1.get_ylim()[1], "  out-of-sample", va="top",
             fontsize=8, color=REF_COLOR)

    fig.tight_layout()
    save_fig(fig, "fig_price_returns")
    plt.close(fig)


def fig_acf_returns(asset):
    r = load_train_returns(asset)
    band = 1.96 / np.sqrt(len(r))
    lags = np.arange(1, NLAGS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.6), sharey=True)
    for ax, series, label in [(axes[0], r, r"$r_t$"), (axes[1], r**2, r"$r_t^2$")]:
        ax.vlines(lags, 0, sample_acf(series, NLAGS), color=MAIN_COLOR, lw=2)
        ax.axhline(0, color=REF_COLOR, lw=0.8)
        ax.axhline(band, color=REF_COLOR, lw=0.8, ls="--")
        ax.axhline(-band, color=REF_COLOR, lw=0.8, ls="--")
        ax.set_xlabel("Lag")
        ax.set_title(f"ACF of {label}", fontsize=9)
    axes[0].set_ylabel("Autocorrelation")

    fig.tight_layout()
    save_fig(fig, "fig_acf_returns")
    plt.close(fig)


def fig_returns_distribution(asset):
    r = load_train_returns(asset)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.8))

    ax1.hist(r, bins=60, density=True, color=MAIN_COLOR, alpha=0.6,
             edgecolor="white", linewidth=0.3, label="Observed")
    x = np.linspace(r.min(), r.max(), 500)
    ax1.plot(x, norm.pdf(x, r.mean(), r.std()), color=REF_COLOR, lw=1.2,
             label="Normal (same mean and std)")
    ax1.set_xlabel("Daily log-return")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False, fontsize=7)

    z = (r - r.mean()) / r.std()
    qq_panel(ax2, z, norm, "normal")

    fig.tight_layout()
    save_fig(fig, "fig_returns_distribution")
    plt.close(fig)


def fig_conditional_vol(asset):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3))

    for model, (_, in_percent) in MODELS.items():
        vol = load_diagnostics(asset, model)["conditional_volatility"].dropna()
        if in_percent:
            vol = vol / 100
        ax.plot(vol.index, vol * np.sqrt(TRADING_DAYS), lw=0.9,
                color=COLORS[model], label=model)

    ax.set_ylabel("Annualized volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")

    fig.tight_layout()
    save_fig(fig, "fig_conditional_vol")
    plt.close(fig)


def fig_qq_residuals(asset):
    params = pd.read_csv(MODELS_DIR / "garch" / f"garch_{asset}_params.csv",
                         index_col="param")
    nu = params.loc["nu", "value"]
    std_t = t(nu, scale=np.sqrt((nu - 2) / nu))   # Student-t with unit variance

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 5.6))

    for ax, model in zip(axes.flat, MODELS):
        z = load_diagnostics(asset, model)["standardized_residuals"].dropna()
        if model == "GARCH-t":
            qq_panel(ax, z, std_t, rf"Student-t, $\nu$={nu:.2f}")
        else:
            qq_panel(ax, z, norm, "normal")
        ax.set_title(model, fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig_qq_residuals")
    plt.close(fig)


if __name__ == "__main__":
    fig_price_returns(ASSET)
    fig_acf_returns(ASSET)
    fig_returns_distribution(ASSET)
    fig_conditional_vol(ASSET)
    fig_qq_residuals(ASSET)
    print("Figures saved to results/figures/")