"""No-lookahead guarantee for the online regime detector (the headline claim).

The online detector must produce, for any past day, a P(turbulent) that depends
ONLY on data up to that day. We prove it with an *append-the-future* test: run the
detector on a series, then append genuine future rows and run it again. Every
past-day probability must be byte-for-byte identical. If any future information
could leak backwards, an earlier day's number would move; it does not.

Run:  python -m pytest tests/test_no_lookahead.py -q
  or: python tests/test_no_lookahead.py
"""
import warnings

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # hmmlearn non-convergence on tiny synthetic data is cosmetic

from regime.detector_online import online_regimes
from regime.detector_hmm import HMM_FEATURES


def _synthetic_features(n_days, seed=0):
    """Two planted regimes so the HMM has real structure to lock onto.
    Small burn-in so the walk-forward loop actually produces signal in a test.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    half = n_days // 2
    calm = pd.DataFrame({
        "realized_vol": rng.normal(0.12, 0.02, half),
        "dispersion":   rng.normal(0.008, 0.001, half),
        "autocorr":     rng.normal(-0.10, 0.05, half)})
    turb = pd.DataFrame({
        "realized_vol": rng.normal(0.35, 0.05, n_days - half),
        "dispersion":   rng.normal(0.020, 0.003, n_days - half),
        "autocorr":     rng.normal(0.02, 0.05, n_days - half)})
    f = pd.concat([calm, turb], ignore_index=True)
    f = f[HMM_FEATURES]           # exact column order the detector expects
    f.index = idx
    return f


def test_append_the_future_is_byte_identical():
    burn_in = 60                  # small burn-in for a fast, self-contained test
    full = _synthetic_features(600, seed=1)
    past = full.iloc[:400]        # the "present": what we can see today
    future = full                 # the same series with genuine future rows appended

    _, p_past = online_regimes(past, burn_in=burn_in)
    _, p_future = online_regimes(future, burn_in=burn_in)

    common = p_past.index.intersection(future.index[:len(past)])
    a = p_past.loc[common].dropna()
    b = p_future.loc[common].reindex(a.index)

    # every past-day probability must be *identical* with or without future data
    assert np.array_equal(a.values, b.values), (
        "LOOKAHEAD DETECTED: a past P(turbulent) changed when future data was appended")


if __name__ == "__main__":
    test_append_the_future_is_byte_identical()
    print("PASS: online detector is lookahead-free (append-the-future, byte-identical)")
