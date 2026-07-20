"""Phase 4 — regime overlay: four-book comparison, in-sample vs 2022+.

Run:   python -m scripts.regime_overlay
Saves: regime/regime_overlay.png  (+ caches the online probability to parquet)

Sleeves are VOL-TARGETED to a common 10% annual vol, with the scaling constant
frozen from the pre-2022 train period (honest), so the blend is a fair risk-
balanced contest rather than being dominated by the higher-vol strategy.

Books:
  pairs_only, momentum_only  -- the standalone (vol-targeted) sleeves
  static_5050                -- HONEST baseline: equal RISK weight, NO regime info
  regime_blend               -- detector-driven soft blend + cash overlay

The question: does regime_blend beat static_5050 OUT-OF-SAMPLE (2022+)?
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from regime.sleeves import pairs_returns, momentum_returns
from regime.features import build_feature_frame
from regime.detector_online import online_regimes
from regime.overlay import four_books, to_equity
from analysis.performance import compute_metrics

SPLIT = pd.Timestamp("2022-01-01")
VOL_TARGET = 0.10                      # 10% annualized vol per sleeve
BASE = Path(__file__).resolve().parent.parent / "regime"
OUT = BASE / "regime_overlay.png"
P_CACHE = BASE / "online_p.parquet"


def _online_p():
    if P_CACHE.exists():
        return pd.read_parquet(P_CACHE)["p_turbulent"]
    _, p = online_regimes(build_feature_frame())
    BASE.mkdir(parents=True, exist_ok=True)
    p.to_frame().to_parquet(P_CACHE)
    return p


def _vol_scalar(returns, target=VOL_TARGET):
    """Fixed leverage to hit `target` annual vol, measured on PRE-2022 data only."""
    train = returns[returns.index < SPLIT]
    v = train.std() * (252 ** 0.5)
    return target / v if v > 0 else 1.0


def _metrics(returns):
    return compute_metrics(to_equity(returns)) if len(returns) >= 5 else None


def main():
    pairs, beta = pairs_returns()
    mom = momentum_returns()
    print(f"pairs beta (train) = {beta:.4f}")

    sp, sm = _vol_scalar(pairs), _vol_scalar(mom)
    print(f"vol-target leverage (from pre-2022): pairs x{sp:.1f}, momentum x{sm:.1f}")
    pairs, mom = pairs * sp, mom * sm    # risk-balanced sleeves

    p = _online_p()
    books, w, turnover = four_books(pairs, mom, p)
    live = books["regime_blend"].index
    print(f"live window: {live.min().date()} -> {live.max().date()}")
    print(f"regime_blend avg daily turnover: {turnover.mean():.4f} "
          f"(~{turnover.mean() * 252:.1f}x/yr)")

    hdr = (f"{'book':<15}{'IS Sharpe':>10}{'IS CAGR':>9}{'IS DD':>8}   "
           f"{'OOS Sharpe':>11}{'OOS CAGR':>10}{'OOS DD':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for name, r in books.items():
        im, om = _metrics(r[r.index < SPLIT]), _metrics(r[r.index >= SPLIT])
        if im and om:
            print(f"{name:<15}{im['sharpe']:>10.2f}{im['cagr']:>9.2%}{im['max_drawdown']:>8.1%}   "
                  f"{om['sharpe']:>11.2f}{om['cagr']:>10.2%}{om['max_drawdown']:>8.1%}")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in books.items():
        eq = to_equity(r)
        ax.plot(eq.index, eq.values, linewidth=1.1, label=name)
    ax.axvline(SPLIT, color="black", linestyle=":", linewidth=1)
    ax.set_title("Regime overlay vs baselines  (dotted = in-sample | 2022+)")
    ax.set_ylabel("Equity ($100k start)"); ax.set_xlabel("Date")
    ax.legend(fontsize=8); ax.margins(x=0)
    fig.tight_layout(); BASE.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
