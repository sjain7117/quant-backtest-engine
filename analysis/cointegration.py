"""Cointegration analysis: which pairs are genuinely tradeable?

Correlation says two prices MOVE together day to day. Cointegration says
something stronger and more useful: the GAP between them (the 'spread') stays
tethered over the long run -- it wanders off but reliably gets pulled back.
Only a cointegrated pair has a spread you can bet on mean-reverting.

We use the Engle-Granger two-step method:
  1. regress one price on the other (OLS) -> the slope is the HEDGE RATIO (beta),
     i.e. how many units of B offset one unit of A to form the spread.
  2. test whether the regression residuals (the spread) are STATIONARY using the
     Augmented Dickey-Fuller (ADF) test. Stationary residuals = cointegrated.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def hedge_ratio(y, x):
    """OLS of y on x (with intercept). Returns (beta, alpha, residuals).

    beta is the hedge ratio; residuals are the spread we test for stationarity.
    """
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    alpha, beta = model.params.iloc[0], model.params.iloc[1]
    residuals = y - (alpha + beta * x)
    return beta, alpha, residuals


def adf_pvalue(series):
    """Augmented Dickey-Fuller test. Null hypothesis = 'has a unit root'
    (non-stationary). A LOW p-value lets us reject that -> stationary."""
    stat, pvalue, *_ = adfuller(series.dropna(), autolag="AIC")
    return stat, pvalue


def engle_granger_pvalue(y, x):
    """statsmodels' direct Engle-Granger cointegration test on the two prices."""
    stat, pvalue, _ = coint(y, x)
    return stat, pvalue


def half_life(spread):
    """Speed of mean reversion, expressed as a half-life in days.

    Fit the discretized Ornstein-Uhlenbeck / AR(1) relationship:
        delta_spread_t = a + b * spread_{t-1} + noise
    A negative b means the spread is pulled back toward its mean; the half-life
    (time to close half the gap) is -ln(2)/b. b >= 0 means no mean reversion.
    """
    spread = pd.Series(spread).dropna()
    lag = spread.shift(1)
    delta = spread - lag
    df = pd.DataFrame({"delta": delta, "lag": lag}).dropna()
    model = sm.OLS(df["delta"], sm.add_constant(df["lag"])).fit()
    b = model.params.iloc[1]
    if b >= 0:
        return np.inf
    return -np.log(2) / b


def analyze_pair(prices, sym_a, sym_b):
    """Full diagnostic for one pair. Returns a dict of all the numbers."""
    df = pd.concat([prices[sym_a], prices[sym_b]], axis=1).dropna()
    y, x = df[sym_a], df[sym_b]

    corr = y.corr(x)
    beta, alpha, spread = hedge_ratio(y, x)
    adf_stat, adf_p = adf_pvalue(spread)
    eg_stat, eg_p = engle_granger_pvalue(y, x)
    hl = half_life(spread)

    z = (spread - spread.mean()) / spread.std()

    return {
        "pair": f"{sym_a}/{sym_b}",
        "corr": corr,
        "beta": beta,
        "eg_pvalue": eg_p,
        "adf_pvalue": adf_p,
        "half_life_days": hl,
        "current_z": z.iloc[-1],
        "n_obs": len(spread),
    }


def screen_all_pairs(prices, tickers):
    """Test every ticker combination; return a DataFrame ranked by EG p-value.

    Shows that economically-linked pairs cointegrate while unrelated ones don't.
    """
    rows = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            try:
                rows.append(analyze_pair(prices, a, b))
            except Exception:
                rows.append({"pair": f"{a}/{b}", "eg_pvalue": np.nan})
    return pd.DataFrame(rows).sort_values("eg_pvalue").reset_index(drop=True)
