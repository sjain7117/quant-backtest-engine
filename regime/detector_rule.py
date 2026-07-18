"""Phase 1 — a simple, checkable rule-based regime detector (the HONEST BASELINE).

It labels each day 'calm' or 'turbulent' from one transparent rule on realized
volatility. It is deliberately dumb: if the Hidden Markov Model in Phase 2 can't
beat this, we've learned the fancy method buys us nothing — exactly the role
buy-and-hold played as the baseline in the last project.

What the labels mean downstream (wired up in Phase 4):
    calm      -> the mean-reversion (pairs) world
    turbulent -> the momentum (trend) world

HYSTERESIS: a single threshold makes the label flicker every time vol wobbles
across the line. Instead we use TWO levels — a high one to switch INTO turbulent
and a lower one to switch back to calm — so once a regime starts it persists
until vol clearly leaves the zone. The gap between them is a 'dead band'.

Lookahead status: the two thresholds are ROUND, DOMAIN-CHOSEN numbers (long-run
US equity vol is ~15-16%), NOT values fitted to this sample, and the labeling is
a one-pass, past-only walk — so this baseline already does not peek at the future.
Phase 3 goes further, letting the 'normal' vol level be *learned online* from past
data instead of hard-coded.
"""
from __future__ import annotations

import pandas as pd

from regime.features import realized_volatility, load_market_proxy

CALM, TURBULENT = "calm", "turbulent"

# Hysteresis thresholds on ANNUALIZED realized volatility.
ENTER_TURBULENT = 0.20   # cross ABOVE 20% -> switch into 'turbulent'
EXIT_TURBULENT = 0.15    # fall BELOW 15%  -> switch back to 'calm'


def label_regimes(vol, enter_th=ENTER_TURBULENT, exit_th=EXIT_TURBULENT):
    """Turn a realized-volatility series into calm/turbulent labels via hysteresis.

    We walk forward day by day. Once turbulent, we STAY turbulent until vol drops
    below the lower (exit) threshold; once calm, we STAY calm until vol rises above
    the upper (enter) threshold. The label on day t depends only on vol up to t.
    """
    labels, state = [], CALM  # assume calm before the first trigger
    for v in vol:
        if state == CALM and v >= enter_th:
            state = TURBULENT
        elif state == TURBULENT and v <= exit_th:
            state = CALM
        labels.append(state)
    return pd.Series(labels, index=vol.index, name="regime")


def detect(start="2015-01-01", end=None, enter_th=ENTER_TURBULENT, exit_th=EXIT_TURBULENT):
    """Convenience: load the market proxy, compute vol, return regime labels."""
    vol = realized_volatility(load_market_proxy(start=start, end=end)).dropna()
    return label_regimes(vol, enter_th, exit_th)
