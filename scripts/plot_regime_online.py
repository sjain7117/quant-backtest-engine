"""Phase 3 validation — the honest, past-only detector, and what honesty costs.

Run:   python -m scripts.plot_regime_online
Saves: regime/regime_online.png

Compares the online (past-only) labels against the Phase 2 in-sample labels that
peeked at the future. The gap is the price of honesty: how much of the pretty
in-sample regime story survives when you cannot see ahead.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from regime.features import build_feature_frame, realized_volatility, load_market_proxy
from regime.detector_online import online_regimes, CALM, TURBULENT
from regime.detector_hmm import fit_hmm, label_states
from regime.detector_rule import label_regimes as rule_labels

SPLIT = pd.Timestamp("2022-01-01")
OUT = Path(__file__).resolve().parent.parent / "regime" / "regime_online.png"


def main():
    features = build_feature_frame()

    labels, p = online_regimes(features)        # honest, past-only
    turbulent = labels == TURBULENT
    vol = features["realized_vol"]

    model, X = fit_hmm(features)                 # in-sample (peeks), for comparison
    insample, _ = label_states(model, X, features.index)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(vol.index, vol.values, linewidth=0.8, color="steelblue")
    a1.fill_between(vol.index, 0, float(vol.max()), where=turbulent.values,
                    color="crimson", alpha=0.12, step="mid")
    a1.axvline(SPLIT, color="black", linestyle=":", linewidth=1)
    a1.set_ylabel("Realized vol"); a1.margins(x=0)
    a1.set_title("Online (past-only) HMM regimes  (shaded = turbulent)")
    a2.plot(p.index, p.values, linewidth=0.8, color="darkorange")
    a2.axhline(0.5, color="0.6", linewidth=0.6)
    a2.axvline(SPLIT, color="black", linestyle=":", linewidth=1)
    a2.set_ylabel("P(turbulent), filtered"); a2.set_xlabel("Date"); a2.margins(x=0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT}")

    live = p.notna()
    print(f"\nSignal starts: {p[live].index.min().date()} (after ~2y burn-in)")
    tt = turbulent[live]
    print(f"Turbulent fraction (online)  ->  overall: {tt.mean():.1%} | "
          f"pre-2022: {tt[tt.index < SPLIT].mean():.1%} | "
          f"2022+: {tt[tt.index >= SPLIT].mean():.1%}")

    covid = labels.loc["2020-03-15":"2020-04-15"]
    cf = (covid == TURBULENT).mean()
    print(f"COVID window flagged turbulent (online): {cf:.0%} "
          f"-> {'PASS' if cf > 0.8 else 'CHECK'}")
    print(f"Regime switches (online): {(labels[live] != labels[live].shift()).sum() - 1}")

    agree_is = (labels[live] == insample[live]).mean()
    print(f"\nAgreement online vs in-sample (Phase 2): {agree_is:.1%}")

    rule = rule_labels(realized_volatility(load_market_proxy()).dropna())
    common = labels[live].index.intersection(rule.index)
    print(f"Agreement online vs rule (Phase 1): "
          f"{(labels.loc[common] == rule.loc[common]).mean():.1%}")


if __name__ == "__main__":
    main()
