"""Phase 2 — a Gaussian Hidden Markov Model regime detector.

An HMM assumes the market is always in one of K HIDDEN states (here 2: calm,
turbulent). You never see the state directly; you see FEATURES it emits — realized
vol, dispersion, autocorr — each drawn from that state's own Gaussian
distribution. From the data alone the model learns each state's feature
distribution and a TRANSITION MATRIX (the probabilities of staying in / switching
between states), then infers the most likely hidden-state sequence.

Its edge over the Phase 1 rule: it reads SEVERAL features jointly and returns a
PROBABILITY of turbulence, not a hard yes/no line on one feature.

*** IN-SAMPLE WARNING ***
This module fits on the WHOLE history and standardizes features with FULL-SAMPLE
statistics, so it PEEKS at the future. That is fine for Phase 2's only job —
seeing WHAT regimes the model finds and whether it differs from the dumb rule —
but these labels are NOT tradeable. Phase 3 rebuilds this to run online, using
only past data. Do not judge the strategy on the labels produced here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from regime.features import build_feature_frame

CALM, TURBULENT = "calm", "turbulent"

# Features fed to the HMM. Unlike the rule (vol only), the HMM can use several at
# once: vol + dispersion measure turbulence; autocorr measures revert-vs-trend.
HMM_FEATURES = ["realized_vol", "dispersion", "autocorr"]
N_STATES = 2       # two regimes, to stay comparable with the rule-based baseline
N_FITS = 10        # EM finds LOCAL optima; fit several times and keep the best
RANDOM_STATE = 42  # reproducibility


def _standardize(df):
    """Z-score each feature (full-sample — see the in-sample warning up top).

    The HMM's Gaussian emissions are scale-sensitive: vol (~0.15), dispersion
    (~0.01) and autocorr (~-0.1) live on wildly different scales, so without this
    the largest-scale feature would dominate. Standardizing puts them on equal
    footing.
    """
    return (df - df.mean()) / df.std()


def fit_hmm(features, n_states=N_STATES, n_fits=N_FITS, random_state=RANDOM_STATE):
    """Fit a Gaussian HMM, returning the best model (highest log-likelihood).

    EM (the fitting algorithm) can settle into different solutions depending on
    its random start, so we fit n_fits times and keep the one that explains the
    data best. Returns (model, X) where X is the standardized feature matrix.
    """
    X = _standardize(features[HMM_FEATURES]).values
    best_model, best_score = None, -np.inf
    for seed in range(n_fits):
        model = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=200, random_state=random_state + seed)
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_model, best_score = model, score
    return best_model, X


def label_states(model, X, index):
    """Decode the hidden states and map them to calm/turbulent by vol level.

    hmmlearn numbers its states arbitrarily (0/1) — the 'label switching' problem
    — so we identify which state has the higher MEAN realized volatility and call
    that one 'turbulent'. We also return the posterior PROBABILITY of the
    turbulent state, which is the HMM's soft read that the hard rule can't give.
    """
    states = model.predict(X)          # most-likely hard state per day (0/1)
    proba = model.predict_proba(X)     # posterior probability of each state

    vc = HMM_FEATURES.index("realized_vol")
    mean_vol = [X[states == s, vc].mean() if (states == s).any() else -np.inf
                for s in range(model.n_components)]
    turbulent_state = int(np.argmax(mean_vol))

    labels = pd.Series(np.where(states == turbulent_state, TURBULENT, CALM),
                       index=index, name="regime")
    p_turbulent = pd.Series(proba[:, turbulent_state], index=index,
                            name="p_turbulent")
    return labels, p_turbulent


def detect(start="2015-01-01", end=None):
    """End-to-end: features -> fit HMM -> calm/turbulent labels + P(turbulent)."""
    features = build_feature_frame(start=start, end=end)
    model, X = fit_hmm(features)
    labels, p_turbulent = label_states(model, X, features.index)
    return labels, p_turbulent, model
