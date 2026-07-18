"""Regime features: the signals a detector can watch to tell 'calm' from 'turbulent'.

Every feature here is TRAILING — its value on day t uses only data up to and
including t — so nothing in this file can peek at the future. (Standardizing
these features for the HMM in Phase 3 is where lookahead can sneak in; we handle
that separately, with an expanding window, when we get there.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.loader import download_prices, CANDIDATE_UNIVERSE

# A broad market proxy for the volatility and trend features. SPY tracks the
# S&P 500 (the 500 largest US companies) — a cleaner "what is the whole market
# doing" gauge than any single name in our small, quirky universe. It is only
# ever OBSERVED to read the market's state; it is never traded by the strategies.
MARKET_PROXY = "SPY"

# Sensible defaults, in trading days (~21 = 1 month, ~63 = 1 quarter).
VOL_WINDOW = 21
TREND_WINDOW = 63
DISPERSION_SMOOTH = 21
AUTOCORR_WINDOW = 63
TRADING_DAYS = 252  # ~number of trading days in a year, used to annualize


def log_returns(prices):
    """Daily LOG returns: ln(P_t / P_{t-1}).

    Log returns are the convention for volatility work because they add up
    cleanly across time and are symmetric around zero. The first row is NaN
    (no prior day to compare against).
    """
    return np.log(prices).diff()


def load_market_proxy(start="2015-01-01", end=None):
    """Adjusted daily prices for the market proxy (SPY), as a Series."""
    px = download_prices([MARKET_PROXY], start=start, end=end)
    return px[MARKET_PROXY].dropna()


def load_universe(start="2015-01-01", end=None):
    """Adjusted daily prices for the full candidate universe (one col per ticker)."""
    return download_prices(list(CANDIDATE_UNIVERSE), start=start, end=end)


def realized_volatility(market_px, window=VOL_WINDOW):
    """Trailing annualized volatility of the market proxy.

    Realized volatility = the standard deviation of recent daily returns, scaled
    to a yearly figure (times sqrt(252)). It is the primary calm-vs-turbulent
    axis, and it works BECAUSE volatility clusters: turbulent days bunch together
    rather than scattering at random, which is exactly what makes a 'regime'
    persistent enough to be worth detecting.
    """
    daily = log_returns(market_px)
    return daily.rolling(window).std() * np.sqrt(TRADING_DAYS)


def trend_strength(market_px, window=TREND_WINDOW):
    """How far the market sits above/below its own moving average.

    A moving average is the rolling mean of recent prices — a smoothed trend
    line. (price / moving_average) - 1 is positive in an uptrend, negative in a
    downtrend, and near zero in a flat, choppy market. Trending markets tend to
    favor momentum; choppy ones favor mean-reversion.
    """
    ma = market_px.rolling(window).mean()
    return market_px / ma - 1.0


def cross_sectional_dispersion(universe_px, smooth=DISPERSION_SMOOTH):
    """How far apart the universe's assets move on each day.

    Each day we take the standard deviation of that day's returns ACROSS the
    assets (a cross-section). High dispersion = names diverging (a rotation /
    momentum environment); low dispersion = names moving together (often calm).
    Raw daily dispersion is jumpy, so we smooth it with a short trailing average.
    """
    daily = log_returns(universe_px)
    raw = daily.std(axis=1)  # std across columns (assets) within each row (day)
    return raw.rolling(smooth).mean()


def return_autocorrelation(market_px, window=AUTOCORR_WINDOW, lag=1):
    """Trailing autocorrelation of daily returns (a bonus, thematic feature).

    Autocorrelation measures whether a day's return tends to be followed by a
    same-signed one. Positive => the market is trending (momentum-friendly);
    negative => it is mean-reverting. This directly measures which of our two
    strategies the market currently rewards — but it is noisy, so treat it as a
    supporting signal, not a primary one.
    """
    daily = log_returns(market_px).dropna()
    return daily.rolling(window).apply(lambda w: w.autocorr(lag=lag), raw=False)


def build_feature_frame(start="2015-01-01", end=None):
    """Assemble every regime feature into one date-indexed DataFrame.

    Columns: realized_vol, trend, dispersion, autocorr. Early rows (before the
    longest look-back window has enough history) are NaN and are dropped, so the
    frame begins a few months after the start date.
    """
    market_px = load_market_proxy(start=start, end=end)
    universe_px = load_universe(start=start, end=end)

    features = pd.DataFrame({
        "realized_vol": realized_volatility(market_px),
        "trend": trend_strength(market_px),
        "dispersion": cross_sectional_dispersion(universe_px),
        "autocorr": return_autocorrelation(market_px),
    })
    return features.dropna()


if __name__ == "__main__":
    f = build_feature_frame()
    print(f"Built {len(f)} rows of regime features "
          f"({f.index.min().date()} -> {f.index.max().date()})")
    print(f.describe().round(4))
