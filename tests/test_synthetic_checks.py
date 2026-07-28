"""Synthetic-data correctness checks for the regime layer.

Four independent checks, each on planted or hand-built data where the right answer
is known in advance:

  1. HMM regime recovery   -- the in-sample HMM finds two planted regimes and maps
                              the high-volatility state to 'turbulent'.
  2. Hysteresis behaviour  -- the rule detector STAYS turbulent through a dip that
                              never clears the lower exit threshold, and only flips
                              back to calm once vol drops below it.
  3. Overlay weights sum   -- for any P(turbulent), the [pairs, momentum, cash]
                              allocation is non-negative and sums to exactly 1.
  4. Equity / buy-and-hold -- to_equity() compounds a constant daily return exactly
     accounting              like buy-and-hold (day-over-day factor (1+r), cumulative
                              (1+r)^(n-1)), and a flat series stays flat (no phantom P&L).

Run:  python -m pytest tests/test_synthetic_checks.py -q
  or: python tests/test_synthetic_checks.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from regime.detector_hmm import fit_hmm, label_states, HMM_FEATURES, CALM, TURBULENT
from regime.detector_rule import label_regimes, ENTER_TURBULENT, EXIT_TURBULENT
from regime.overlay import regime_weights, to_equity


def test_hmm_recovers_planted_regimes():
    rng = np.random.default_rng(1)
    n = 600
    idx = pd.bdate_range("2015-01-01", periods=2 * n)
    calm = pd.DataFrame({
        "realized_vol": rng.normal(0.12, 0.02, n),
        "dispersion":   rng.normal(0.008, 0.001, n),
        "autocorr":     rng.normal(-0.10, 0.05, n)})
    turb = pd.DataFrame({
        "realized_vol": rng.normal(0.35, 0.05, n),
        "dispersion":   rng.normal(0.020, 0.003, n),
        "autocorr":     rng.normal(0.02, 0.05, n)})
    features = pd.concat([calm, turb], ignore_index=True)[HMM_FEATURES]
    features.index = idx

    model, X = fit_hmm(features, n_fits=5)
    labels, p = label_states(model, X, features.index)

    calm_acc = (labels.iloc[:n] == CALM).mean()
    turb_acc = (labels.iloc[n:] == TURBULENT).mean()
    assert calm_acc > 0.9 and turb_acc > 0.9, (
        f"HMM failed to recover planted regimes (calm {calm_acc:.0%}, turb {turb_acc:.0%})")
    assert p.between(0, 1).all(), "P(turbulent) escaped [0, 1]"


def test_hysteresis_holds_through_dip():
    idx = pd.bdate_range("2020-01-01", periods=8)
    vol = pd.Series([0.10, 0.18, 0.21, 0.16, 0.19, 0.14, 0.13, 0.22], index=idx)
    got = list(label_regimes(vol))
    exp = [CALM, CALM, TURBULENT, TURBULENT, TURBULENT, CALM, CALM, TURBULENT]
    assert got == exp, f"hysteresis broke\n got {got}\n exp {exp}"
    assert ENTER_TURBULENT > EXIT_TURBULENT, "dead band must be positive"


def test_overlay_weights_sum_to_one():
    p = pd.Series(np.linspace(0.0, 1.0, 501))
    w = regime_weights(p)
    total = w[["pairs", "momentum", "cash"]].sum(axis=1)
    assert np.allclose(total.values, 1.0, atol=1e-9), "weights do not sum to 1"
    assert (w.values >= -1e-12).all(), "a weight went negative"
    assert (w.values <= 1 + 1e-9).all(), "a weight exceeded 1"


def test_equity_accounting_matches_buy_and_hold():
    # A constant daily return r must compound geometrically: the day-over-day growth
    # factor is exactly (1+r) everywhere, and cumulative growth over the window is
    # (1+r)^(n-1). This is the defining property of correct buy-and-hold accounting,
    # and it holds whatever base capital the equity curve is seeded with.
    r = 0.001
    eq = to_equity(pd.Series([r] * 252))
    ratios = (eq / eq.shift(1)).dropna()
    assert np.allclose(ratios.values, 1 + r), "equity does not compound at (1+r) per day"
    assert np.isclose(eq.iloc[-1] / eq.iloc[0], (1 + r) ** (len(eq) - 1)), \
        "cumulative growth != (1+r)^(n-1)"
    flat = to_equity(pd.Series([0.0] * 50))
    assert np.allclose(flat.values, flat.iloc[0]), "flat returns produced non-flat equity"


if __name__ == "__main__":
    test_hmm_recovers_planted_regimes()
    print("PASS 1/4: HMM recovers the two planted regimes")
    test_hysteresis_holds_through_dip()
    print("PASS 2/4: hysteresis holds through the dip, exits only below the lower band")
    test_overlay_weights_sum_to_one()
    print("PASS 3/4: overlay weights are non-negative and sum to 1 across all p")
    test_equity_accounting_matches_buy_and_hold()
    print("PASS 4/4: equity compounds like buy-and-hold; flat returns stay flat")
    print("ALL SYNTHETIC CHECKS PASS")
