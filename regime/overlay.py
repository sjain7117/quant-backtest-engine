"""Phase 4 core — the regime overlay: allocate between strategy 'sleeves' by regime.

Each strategy (pairs, momentum) is a SLEEVE returning a daily-return series; the
online detector is a CAPITAL ALLOCATOR on top. This is how real multi-strategy
books run, and it reuses the per-strategy backtests you already trust instead of
re-plumbing the event-driven engine.

Allocation (soft blend + cash overlay), using P(turbulent) that is (a) smoothed
over a few trailing days to damp churn and (b) LAGGED one day, so today's trade
can never use a regime computed from today's close. Both operations are past-only.

    momentum weight = p          pairs weight = 1 - p            (base split)
    cash = clip((p - CASH_ENTER) / (1 - CASH_ENTER), 0, 1)       (de-risk in extremes)
    invested = 1 - cash
    daily return = invested * (w_pairs * r_pairs + w_mom * r_mom)   [cash earns 0]

A turnover cost (bps on how far the sleeve weights move day to day) is charged and
also returned as a diagnostic. NOTE: a soft blend does NOT automatically trade less
than a hard switch -- it nudges every day -- which is exactly why we smooth p and
watch the turnover number.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CASH_ENTER = 0.85   # above this P(turbulent), start moving to cash
COST_BPS = 10.0     # cost per unit of sleeve-weight turnover (conservative)
LAG = 1             # act on YESTERDAY's regime (no same-day lookahead)
SMOOTH_P = 5        # trailing days to smooth P(turbulent); past-only


def _smooth(p, smooth_p):
    if smooth_p and smooth_p > 1:
        return p.rolling(smooth_p, min_periods=1).mean()   # trailing = past-only
    return p


def _cash_fraction(p, cash_enter=CASH_ENTER):
    return ((p - cash_enter) / (1.0 - cash_enter)).clip(0.0, 1.0)


def regime_weights(p, cash_enter=CASH_ENTER, lag=LAG, smooth_p=SMOOTH_P):
    """Daily weights [pairs, momentum, cash] from smoothed, lagged P(turbulent)."""
    pl = _smooth(p, smooth_p).shift(lag).clip(0.0, 1.0)
    cash = _cash_fraction(pl, cash_enter)
    invested = 1.0 - cash
    w = pd.DataFrame({
        "pairs": invested * (1.0 - pl),
        "momentum": invested * pl,
        "cash": cash,
    }, index=p.index)
    return w.dropna()


def _apply_cost(gross, risky_w, cost_bps):
    turnover = risky_w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - cost_bps / 1e4 * turnover, turnover


def regime_blend(pairs, momentum, p, cash_enter=CASH_ENTER, cost_bps=COST_BPS,
                 lag=LAG, smooth_p=SMOOTH_P):
    """Net daily return of the regime-allocated book, plus weights and turnover."""
    w = regime_weights(p, cash_enter, lag, smooth_p)
    idx = w.index.intersection(pairs.index).intersection(momentum.index)
    w = w.loc[idx]
    gross = w["pairs"] * pairs.loc[idx] + w["momentum"] * momentum.loc[idx]
    net, turnover = _apply_cost(gross, w[["pairs", "momentum"]], cost_bps)
    net.name = "regime_blend"
    return net, w, turnover


def four_books(pairs, momentum, p, **kw):
    """The four comparison books as daily-return series, plus weights/turnover.

    pairs_only and momentum_only are the standalone sleeves; static_5050 is the
    honest no-information baseline (equal weight, no regime signal); regime_blend
    is the detector-driven book. If regime_blend can't beat static_5050, the
    detector added nothing.
    """
    idx = pairs.index.intersection(momentum.index)
    pairs, momentum = pairs.loc[idx], momentum.loc[idx]
    blend, w, turnover = regime_blend(pairs, momentum, p, **kw)
    common = blend.index
    books = {
        "pairs_only": pairs.loc[common],
        "momentum_only": momentum.loc[common],
        "static_5050": 0.5 * pairs.loc[common] + 0.5 * momentum.loc[common],
        "regime_blend": blend,
    }
    return books, w, turnover


def to_equity(returns, capital=100_000):
    """Turn a daily-return series into an equity curve for your compute_metrics()."""
    return capital * (1.0 + returns).cumprod()
