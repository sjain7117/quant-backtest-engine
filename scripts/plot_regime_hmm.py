"""Phase 2 validation — fit the HMM, interpret its states, compare to the baseline.

Run:   python -m scripts.plot_regime_hmm
Saves: regime/regime_hmm.png

IN-SAMPLE ONLY (see the warning in regime/detector_hmm.py). This tells us WHAT
regimes the model finds and whether it meaningfully differs from the dumb rule —
not whether it makes money. That verdict waits for Phase 3+.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from regime.features import build_feature_frame, realized_volatility, load_market_proxy
from regime.detector_hmm import fit_hmm, label_states, HMM_FEATURES, CALM, TURBULENT
from regime.detector_rule import label_regimes as rule_labels

SPLIT = pd.Timestamp("2022-01-01")
OUT = Path(__file__).resolve().parent.parent / "regime" / "regime_hmm.png"


def main():
    features = build_feature_frame()
    model, X = fit_hmm(features)
    labels, p_turb = label_states(model, X, features.index)
    turbulent = labels == TURBULENT
    vol = features["realized_vol"]

    # ---- interpret the learned model ----
    print("Per-state feature means (standardized units):")
    print(pd.DataFrame(model.means_, columns=HMM_FEATURES).round(2))
    stay = np.diag(model.transmat_)
    dur = 1.0 / (1.0 - stay)
    print("\nExpected regime persistence (days):",
          {f"state{i}": round(d, 1) for i, d in enumerate(dur)})

    # ---- two-panel plot: vol + shading, and the soft P(turbulent) ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(vol.index, vol.values, linewidth=0.8, color="steelblue")
    a1.fill_between(vol.index, 0, float(vol.max()), where=turbulent.values,
                    color="crimson", alpha=0.12, step="mid")
    a1.axvline(SPLIT, color="black", linestyle=":", linewidth=1)
    a1.set_ylabel("Realized vol"); a1.margins(x=0)
    a1.set_title("HMM regimes  (shaded = turbulent)")
    a2.plot(p_turb.index, p_turb.values, linewidth=0.8, color="darkorange")
    a2.axhline(0.5, color="0.6", linewidth=0.6)
    a2.axvline(SPLIT, color="black", linestyle=":", linewidth=1)
    a2.set_ylabel("P(turbulent)"); a2.set_xlabel("Date"); a2.margins(x=0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nSaved {OUT}")

    # ---- same metrics as Phase 1, for a direct comparison ----
    print(f"\nTurbulent fraction  ->  overall: {turbulent.mean():.1%} | "
          f"2015-21: {turbulent[turbulent.index < SPLIT].mean():.1%} | "
          f"2022+: {turbulent[turbulent.index >= SPLIT].mean():.1%}")
    covid = labels.loc["2020-03-15":"2020-04-15"]
    cf = (covid == TURBULENT).mean()
    print(f"COVID window flagged turbulent: {cf:.0%} -> {'PASS' if cf > 0.8 else 'CHECK'}")
    print(f"Regime switches: {(labels != labels.shift()).sum() - 1}")

    # ---- agreement with the rule-based baseline ----
    rule = rule_labels(realized_volatility(load_market_proxy()).dropna())
    common = labels.index.intersection(rule.index)
    agree = (labels.loc[common] == rule.loc[common]).mean()
    print(f"\nAgreement with Phase 1 rule-based labels: {agree:.1%}")


if __name__ == "__main__":
    main()
