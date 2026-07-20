"""Phase 4/5 — regime overlay: four-book comparison, in-sample vs 2022+.

Run:   python -m scripts.regime_overlay
Saves: regime/regime_overlay.png  (+ caches the online probability to parquet)

Robustness build: pairs sleeve is a 5-pair BASKET, and vol-target leverage is
CAPPED at 10x, so the verdict can't rest on one fragile pair levered to the moon.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from regime.sleeves import pairs_basket_returns, momentum_returns
from regime.features import build_feature_frame
from regime.detector_online import online_regimes
from regime.overlay import four_books, to_equity
from analysis.performance import compute_metrics

SPLIT = pd.Timestamp("2022-01-01")
VOL_TARGET = 0.10       # 10% annualized vol per sleeve
MAX_LEV = 10.0          # cap leverage so no fragile sleeve gets levered 29x
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


def _vol_scalar(returns, target=VOL_TARGET, max_lev=MAX_LEV):
    """Leverage to hit `target` annual vol (from PRE-2022 data), capped at max_lev."""
    train = returns[returns.index < SPLIT]
    v = train.std() * (252 ** 0.5)
    return min(target / v, max_lev) if v > 0 else 1.0


def _metrics(returns):
    return compute_metrics(to_equity(returns)) if len(returns) >= 5 else None


def main():
    pairs, betas = pairs_basket_returns()
    mom = momentum_returns()
    print(f"pair betas (train): {betas}")

    sp, sm = _vol_scalar(pairs), _vol_scalar(mom)
    print(f"vol-target leverage (capped at {MAX_LEV:.0f}x): pairs x{sp:.1f}, momentum x{sm:.1f}")
    pairs, mom = pairs * sp, mom * sm

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
    ax.set_title("Regime overlay vs baselines  (5-pair basket, capped lev)  dotted = IS | 2022+")
    ax.set_ylabel("Equity ($100k start)"); ax.set_xlabel("Date")
    ax.legend(fontsize=8); ax.margins(x=0)
    fig.tight_layout(); BASE.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
