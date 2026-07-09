"""Data layer: download, cache, and load adjusted daily price data."""
from pathlib import Path
import pandas as pd
import yfinance as yf

# Cached downloads live here so we don't re-hit the internet every run.
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Candidate pairs: tickers grouped by a REAL economic reason to move together.
# (A "ticker" is the short symbol that identifies a tradeable asset, e.g. KO = Coca-Cola.)
CANDIDATE_UNIVERSE = {
    "KO":  "Coca-Cola",              # KO / PEP: two beverage giants, same industry forces
    "PEP": "PepsiCo",
    "GLD": "Gold ETF",               # GLD / GDX: the price of gold vs. the companies that mine it
    "GDX": "Gold-miners ETF",
    "XOM": "ExxonMobil",             # XOM / CVX: two oil majors driven by the same oil price
    "CVX": "Chevron",
    "V":   "Visa",                   # V / MA: the two dominant payment networks
    "MA":  "Mastercard",
    "EWA": "Australia country ETF",  # EWA / EWC: two commodity-driven economies (textbook pair)
    "EWC": "Canada country ETF",
}


def _cache_path(ticker):
    return CACHE_DIR / f"{ticker}.parquet"


def download_prices(tickers, start="2015-01-01", end=None, force=False):
    """Download adjusted daily closing prices, caching each ticker to disk.

    We use ADJUSTED prices (auto_adjust=True). Adjusted prices already bake in
    dividends and stock splits, so the series reflects true economic return
    with no fake price jumps. Using raw/unadjusted prices is a common beginner
    bug that makes a strategy 'see' events that never really happened.

    Returns a DataFrame: one column per ticker, indexed by date.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    frames = {}
    for ticker in tickers:
        path = _cache_path(ticker)
        if path.exists() and not force:
            series = pd.read_parquet(path)["close"]
        else:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  [warn] no data returned for {ticker}")
                continue
            series = raw["Close"]
            # yfinance sometimes hands back a 1-column frame instead of a series.
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series.name = "close"
            series.to_frame().to_parquet(path)
        frames[ticker] = series

    return pd.DataFrame(frames)


def load_pair(ticker_a, ticker_b, start="2015-01-01", end=None):
    """Return aligned prices for two tickers, keeping only dates BOTH traded.

    This is what the pairs strategy will consume in Phase 4.
    """
    prices = download_prices([ticker_a, ticker_b], start=start, end=end)
    return prices.dropna()  # drop any date where one of the two is missing
