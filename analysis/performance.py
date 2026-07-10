"""Performance metrics computed from an equity curve.

All ratio metrics use SIMPLE (arithmetic) daily returns -- the standard
convention for Sharpe / Sortino / volatility -- annualized with 252 trading
days (the rough number of days markets are open per year).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve_to_series(equity_curve):
    """[(timestamp, value), ...]  ->  pandas Series indexed by date."""
    if not equity_curve:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([t for t, _ in equity_curve])
    return pd.Series([v for _, v in equity_curve], index=idx)


def compute_metrics(equity, risk_free_rate=0.0, periods_per_year=TRADING_DAYS):
    """Return a dict of performance stats from an equity Series.

    risk_free_rate is an ANNUAL rate (e.g. 0.04 for 4%); default 0 keeps the
    numbers simple and is a common, clearly-stated simplification.
    """
    equity = equity.dropna()
    returns = equity.pct_change().dropna()
    n = len(returns)
    if n == 0:
        return {}

    rf_period = risk_free_rate / periods_per_year   # de-annualize the risk-free rate
    excess = returns - rf_period
    std = returns.std(ddof=1)                        # sample standard deviation

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = n / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = std * np.sqrt(periods_per_year)

    # Sharpe: excess reward per unit of TOTAL volatility
    sharpe = (excess.mean() / std) * np.sqrt(periods_per_year) if std > 0 else np.nan

    # Sortino: excess reward per unit of DOWNSIDE volatility only
    downside = np.minimum(returns - rf_period, 0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    sortino = (excess.mean() / downside_dev) * np.sqrt(periods_per_year) if downside_dev > 0 else np.nan

    # Max drawdown: worst peak-to-trough drop along the equity curve
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    # Calmar: annualized return per unit of worst drawdown
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # Longest underwater stretch: most consecutive bars below a prior peak
    longest, current = 0, 0
    for at_peak in (equity >= running_max):
        current = 0 if at_peak else current + 1
        longest = max(longest, current)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "max_dd_duration_days": longest,
        "n_periods": n,
    }


def print_report(metrics, title="Performance"):
    if not metrics:
        print("No data to report.")
        return
    print(title)
    print("-" * len(title))
    print(f"Total return        : {metrics['total_return']:>10.2%}")
    print(f"CAGR (annualized)   : {metrics['cagr']:>10.2%}")
    print(f"Annual volatility   : {metrics['ann_volatility']:>10.2%}")
    print(f"Sharpe ratio        : {metrics['sharpe']:>10.2f}")
    print(f"Sortino ratio       : {metrics['sortino']:>10.2f}")
    print(f"Max drawdown        : {metrics['max_drawdown']:>10.2%}")
    print(f"Calmar ratio        : {metrics['calmar']:>10.2f}")
    print(f"Longest underwater  : {metrics['max_dd_duration_days']:>7} days")
    print(f"Observations        : {metrics['n_periods']:>7} bars")
