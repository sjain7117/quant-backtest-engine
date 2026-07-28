"""Phase 4 sleeves — run each strategy standalone and return its daily returns.

Robustness update: the pairs sleeve is now an equal-weight BASKET of five
economically-motivated pairs, not a single fragile relationship. Each pair's beta
is frozen from pre-2022. Momentum's lookback/skip are fixed hyperparameters.
"""
import pandas as pd
from data.loader import download_prices, CANDIDATE_UNIVERSE
from analysis.cointegration import hedge_ratio
from analysis.performance import equity_curve_to_series
from engine.data_handler import PairDataHandler, MultiDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from strategies.momentum import CrossSectionalMomentum

CAPITAL = 100_000

# Five economically-motivated pairs from your universe (basket = robust sleeve).
CANDIDATE_PAIRS = [("KO", "PEP"), ("GLD", "GDX"), ("XOM", "CVX"),
                   ("V", "MA"), ("EWA", "EWC")]


def _exec():
    return SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))


def _run_to_returns(data, strat):
    port = Portfolio(data, initial_capital=CAPITAL)
    Backtest(data, strat, port, _exec()).run()
    eq = equity_curve_to_series(port.equity_curve)
    return eq.pct_change().dropna()


def _single_pair_returns(sym_a, sym_b, train_end="2021-12-31", base_units=100):
    prices = download_prices([sym_a, sym_b], start="2015-01-01").dropna()
    beta, _, _ = hedge_ratio(prices.loc[:train_end, sym_a], prices.loc[:train_end, sym_b])
    data = PairDataHandler(sym_a, sym_b, prices=prices)
    strat = PairsTradingStrategy(sym_a, sym_b, beta=beta, lookback=30,
                                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=base_units)
    return _run_to_returns(data, strat), beta


def pairs_basket_returns(pairs=None, train_end="2021-12-31", base_units=100):
    """Equal-weight basket of several pairs -> a robust pairs sleeve.

    Each pair runs standalone (beta frozen from train); we average their daily
    returns. Averaging largely-uncorrelated pair P&Ls both raises and stabilizes
    the sleeve's volatility, so the vol-target no longer has to lever one fragile
    pair to the moon.
    """
    pairs = pairs or CANDIDATE_PAIRS
    streams, betas = {}, {}
    for a, b in pairs:
        r, beta = _single_pair_returns(a, b, train_end, base_units)
        streams[f"{a}/{b}"] = r
        betas[f"{a}/{b}"] = round(float(beta), 4)
    basket = pd.DataFrame(streams).mean(axis=1)   # equal weight across available pairs
    basket.name = "pairs"
    return basket, betas


def momentum_returns(symbols=None, lookback=126, skip=21, n_long=3, n_short=3,
                     gross_per_side=50_000):
    symbols = symbols or list(CANDIDATE_UNIVERSE)
    prices = download_prices(symbols, start="2015-01-01").dropna()
    data = MultiDataHandler(symbols, prices=prices)
    strat = CrossSectionalMomentum(symbols, lookback=lookback, skip=skip,
                                   n_long=n_long, n_short=n_short, gross_per_side=gross_per_side)
    r = _run_to_returns(data, strat); r.name = "momentum"
    return r
