"""Download the full candidate universe and print a quick health check."""
from data.loader import CANDIDATE_UNIVERSE, download_prices


def main():
    tickers = list(CANDIDATE_UNIVERSE)
    print(f"Downloading {len(tickers)} tickers...")
    prices = download_prices(tickers, start="2015-01-01")

    print("\nHealth check")
    print("-" * 50)
    print(f"Date range : {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Rows       : {len(prices)}")
    print(f"Columns    : {list(prices.columns)}")

    missing = prices.isna().sum()
    if missing.any():
        print("\nMissing values per ticker (expected if an ETF launched later):")
        print(missing[missing > 0])
    else:
        print("No missing values.")


if __name__ == "__main__":
    main()
