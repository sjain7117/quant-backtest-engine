"""Phase 4 foundation: which candidate pairs are statistically tradeable?

Analyzed on a TRAINING window only (2015-2021); 2022+ is held out for the
out-of-sample reckoning in Phase 5. Estimating the hedge ratio and spread on the
same data we later trade would be lookahead -- so we split now, up front.
"""
import numpy as np
import pandas as pd

from data.loader import download_prices, CANDIDATE_UNIVERSE
from analysis.cointegration import analyze_pair, screen_all_pairs

TRAIN_END = "2021-12-31"
DEFINED_PAIRS = [("KO", "PEP"), ("GLD", "GDX"), ("XOM", "CVX"),
                 ("V", "MA"), ("EWA", "EWC")]


def main():
    tickers = list(CANDIDATE_UNIVERSE)
    prices = download_prices(tickers, start="2015-01-01")
    train = prices.loc[:TRAIN_END].dropna()
    print(f"Training window: {train.index.min().date()} -> "
          f"{train.index.max().date()}  ({len(train)} bars)\n")

    print("=" * 72)
    print("DETAILED ANALYSIS -- the five economically-motivated pairs")
    print("=" * 72)
    print(f"{'pair':<9}{'corr':>7}{'beta':>9}{'EG p':>9}{'ADF p':>9}{'half-life':>11}{'z-now':>8}")
    print("-" * 72)
    for a, b in DEFINED_PAIRS:
        r = analyze_pair(train, a, b)
        hl = r["half_life_days"]
        hl_str = f"{hl:.0f}d" if np.isfinite(hl) else "inf"
        flag = "  <-- cointegrated" if r["eg_pvalue"] < 0.05 else ""
        print(f"{r['pair']:<9}{r['corr']:>7.2f}{r['beta']:>9.3f}"
              f"{r['eg_pvalue']:>9.4f}{r['adf_pvalue']:>9.4f}{hl_str:>11}"
              f"{r['current_z']:>8.2f}{flag}")

    print("\nHow to read this:")
    print("  EG p < 0.05  -> cointegrated: a mean-reverting edge genuinely exists")
    print("  beta         -> hedge ratio: units of the 2nd name per unit of the 1st")
    print("  half-life    -> typical number of days for a round-trip trade")
    print("  z-now        -> how stretched the spread is right now, in std devs")

    print("\n" + "=" * 72)
    print("FULL SCREEN -- all combinations, ranked by cointegration p-value")
    print("=" * 72)
    screen = screen_all_pairs(train, tickers)
    show = screen[["pair", "corr", "eg_pvalue", "adf_pvalue", "half_life_days"]].head(12).copy()
    show["half_life_days"] = show["half_life_days"].map(
        lambda v: f"{v:.0f}" if np.isfinite(v) else "inf")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(show.to_string(index=False))

    print("\nEconomically-linked pairs should cluster near the top (low p-values);")
    print("unrelated combinations should sink to high p-values. That contrast is")
    print("the point: a real edge needs a real economic reason behind it.")


if __name__ == "__main__":
    main()
