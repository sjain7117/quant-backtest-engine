"""Phase 3 — the ONLINE (past-only) regime detector. The crux of the project.

Phases 1-2 peeked: the HMM was fit on all 11 years and standardized with
full-sample stats, so its labels used 2026 data to describe 2016. This module
removes every peek. At the first trading day of each month it refits the HMM on
history up to that day, and freezes standardization stats from that same past.
Then for EVERY day it reads the FILTERED probability of turbulence: predict_proba
over the sequence up to and including that day, taking the LAST row — whose
posterior depends on no data after the day.

Guarantee (verified with an append-the-future test): a past estimate cannot move
when future data changes. This is your engine's no-lookahead principle, one level
up — applied to the regime signal itself.

The honest cost is a LAG: the model must accumulate enough surprising days before
it becomes confident a regime has changed. Measuring that lag, and whether the
signal still helps once it exists, is the whole point of Phases 4-5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from regime.detector_hmm import HMM_FEATURES, CALM, TURBULENT
from regime.features import build_feature_frame

BURN_IN = 504   # ~2 trading years of history required before the first signal
N_FITS = 3      # we refit ~130 times, so keep each refit lighter than in-sample
N_STATES = 2
SEED0 = 42


def _fit_best(X, n_states, n_fits, seed0):
    """Fit the HMM a few times and keep the highest-likelihood solution."""
    best, best_score = None, -np.inf
    for k in range(n_fits):
        m = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, random_state=seed0 + k)
        m.fit(X)
        s = m.score(X)
        if s > best_score:
            best, best_score = m, s
    return best


def online_regimes(features, burn_in=BURN_IN, n_fits=N_FITS,
                   n_states=N_STATES, seed0=SEED0):
    """Walk-forward, PAST-ONLY regime labels and P(turbulent).

    Refits monthly on the expanding past; standardizes with frozen past stats;
    reads the filtered (last-row) posterior for each day so nothing sees ahead.
    Days inside the burn-in get no signal (label defaults to calm, prob = NaN).
    """
    raw = features[HMM_FEATURES].values
    idx = features.index
    n = len(idx)
    vc = HMM_FEATURES.index("realized_vol")

    # refit at the first row of each new month, once past the burn-in
    periods = idx.to_period("M")
    refit_set = {i for i in range(burn_in, n)
                 if i == burn_in or periods[i] != periods[i - 1]}

    p_turb = np.full(n, np.nan)
    model = turb = mu = sd = None
    for i in range(burn_in, n):
        if i in refit_set:
            train = raw[:i]                          # strictly past data
            mu, sd = train.mean(axis=0), train.std(axis=0)
            Xtr = (train - mu) / sd
            model = _fit_best(Xtr, n_states, n_fits, seed0)
            st = model.predict(Xtr)
            means = [Xtr[st == s, vc].mean() if (st == s).any() else -np.inf
                     for s in range(n_states)]
            turb = int(np.argmax(means))             # re-map high-vol state each refit
        Xpre = (raw[:i + 1] - mu) / sd               # frozen past stats
        p_turb[i] = model.predict_proba(Xpre)[-1][turb]  # filtered: last row only

    p = pd.Series(p_turb, index=idx, name="p_turbulent")
    labels = pd.Series(np.where(p >= 0.5, TURBULENT, CALM), index=idx, name="regime")
    labels[p.isna()] = CALM                          # burn-in default
    return labels, p


def detect(start="2015-01-01", end=None, **kw):
    """End-to-end: features -> online labels + filtered P(turbulent)."""
    features = build_feature_frame(start=start, end=end)
    return online_regimes(features, **kw)
