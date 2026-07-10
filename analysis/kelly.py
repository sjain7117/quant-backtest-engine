"""Kelly position sizing.

Poker origin (discrete bet):  f = p - q/b
  p = win prob, q = 1-p, b = net odds. Bet a fraction f of bankroll.

Investment analog (a stream of returns), the continuous Kelly / Merton fraction:
  f* = mu / sigma^2       (expected excess return over variance)

f* is the LEVERAGE multiple on the strategy's base exposure that maximizes
long-run (log) growth. We then apply FRACTIONAL Kelly (half by default) and a
leverage cap, because full Kelly is very aggressive and extremely sensitive to
estimation error -- practitioners never bet full Kelly.

Key honesty point: Kelly does not create edge or raise the Sharpe ratio. It only
sizes the bet. A marginal edge -> a small f*. A negative edge -> f* <= 0, i.e.
'do not trade'.
"""


def continuous_kelly(returns, risk_free=0.0, periods_per_year=252):
    """Full Kelly leverage f* = mean(excess) / variance, from a return series."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    var = r.var(ddof=1)
    if var == 0:
        return 0.0
    return excess.mean() / var


def kelly_sized_units(base_units, returns, fraction=0.5, max_leverage=3.0):
    """Scale base_units by fractional Kelly, clamped to [0, max_leverage].

    Returns (sized_units, f_star, f_applied) so we can report all three.
    """
    f_star = continuous_kelly(returns)
    f_applied = fraction * f_star
    f_applied = max(0.0, min(f_applied, max_leverage))   # no shorting the edge; cap leverage
    return int(round(base_units * f_applied)), f_star, f_applied
