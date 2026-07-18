"""Phase 1 validation — does the rule-based detector label days sensibly?

Run:   python -m scripts.plot_regime_labels
Saves: regime/regime_labels.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from regime.features import realized_volatility, load_market_proxy
from regime.detector_rule import (
    label_regimes, TURBULENT, ENTER_TURBULENT, EXIT_TURBULENT,
)

SPLIT = pd.Timestamp("2022-01-01")
OUT = Path(__file__).resolve().parent.parent / "regime" / "regime_labels.png"


def main():
    vol = realized_volatility(load_market_proxy()).dropna()
    labels = label_regimes(vol)
    turbulent = labels == TURBULENT

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(vol.index, vol.values, linewidth=0.8, color="steelblue")
    ax.axhline(ENTER_TURBULENT, color="crimson", linestyle="--", linewidth=0.8,
               label=f"enter turbulent ({ENTER_TURBULENT:.0%})")
    ax.axhline(EXIT_TURBULENT, color="seagreen", linestyle="--", linewidth=0.8,
               label=f"exit turbulent ({EXIT_TURBULENT:.0%})")
    ax.fill_between(vol.index, 0, float(vol.max()), where=turbulent.values,
                    color="crimson", alpha=0.12, step="mid")
    ax.axvline(SPLIT, color="black", linestyle=":", linewidth=1)
    ax.set_title("Rule-based regime labels  (shaded = turbulent)")
    ax.set_ylabel("Annualized realized vol")
    ax.set_xlabel("Date")
    ax.margins(x=0)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT}")

    frac_all = turbulent.mean()
    frac_pre = turbulent[turbulent.index < SPLIT].mean()
    frac_post = turbulent[turbulent.index >= SPLIT].mean()
    print(f"\nTurbulent fraction  ->  overall: {frac_all:.1%} | "
          f"2015-21: {frac_pre:.1%} | 2022+: {frac_post:.1%}")

    covid = labels.loc["2020-03-15":"2020-04-15"]
    covid_frac = (covid == TURBULENT).mean()
    print(f"COVID window (mid-Mar to mid-Apr 2020) flagged turbulent: "
          f"{covid_frac:.0%}  -> {'PASS' if covid_frac > 0.8 else 'CHECK'}")

    switches = (labels != labels.shift()).sum() - 1
    print(f"Total regime switches over the whole history: {switches} "
          f"(fewer = less whipsaw)")


if __name__ == "__main__":
    main()
