"""Plot the regime features across time, marking the 2022 regime boundary.

Run:   python -m scripts.plot_regime_features
Saves: regime/regime_features.png   (and prints a quick numeric sanity check)

The point of this plot is a Phase-0 gut check: BEFORE building any detector, look
at whether the features visibly shift between the calm 2015-21 regime and the
turbulent 2022+ regime the earlier project already found. If they do, the premise
holds and it is worth building a detector. If they do NOT, that is itself an
important early finding.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from regime.features import build_feature_frame

SPLIT = pd.Timestamp("2022-01-01")  # the regime boundary from the prior project
OUT = Path(__file__).resolve().parent.parent / "regime" / "regime_features.png"

LABELS = {
    "realized_vol": "Realized volatility (annualized)",
    "trend": "Trend strength  (price / MA - 1)",
    "dispersion": "Cross-sectional dispersion",
    "autocorr": "Return autocorrelation (lag 1)",
}


def main():
    f = build_feature_frame()

    fig, axes = plt.subplots(len(LABELS), 1, figsize=(11, 9), sharex=True)
    for ax, (col, label) in zip(axes, LABELS.items()):
        ax.plot(f.index, f[col], linewidth=0.9)
        ax.axvline(SPLIT, color="crimson", linestyle="--", linewidth=1)
        ax.axhline(0, color="0.7", linewidth=0.6)
        ax.set_ylabel(label, fontsize=9)
        ax.margins(x=0)
    axes[0].set_title(
        "Regime features over time  (dashed line = 2015-21 | 2022+ boundary)",
        fontsize=11,
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT}")

    # Numeric gut check: do the feature averages actually differ across regimes?
    before = f[f.index < SPLIT].mean()
    after = f[f.index >= SPLIT].mean()
    summary = pd.DataFrame({"2015-21 mean": before, "2022+ mean": after}).round(4)
    print("\nFeature means by regime:")
    print(summary)


if __name__ == "__main__":
    main()
