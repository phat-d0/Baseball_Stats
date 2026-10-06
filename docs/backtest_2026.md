# Backtest on historical DraftKings prices (Aug 1 – Sep 27, 2026)

Prices: `scripts/fetch_history.py` (Fetch historical prices workflow), each game's
strikeout and H+R+RBI props one hour before first pitch, 777 games, 15,519 credits.
Model: the live models trained walk-forward (`scripts/backtest.py`): hitters on the
game-level model, strikeouts on pa_simple, each month trained only on earlier games.
15,198 lines; one bet per player and prop (best side), $10 flat. Full numbers:
`backtest_2026.json`.

## Model vs DraftKings

| | Lines | Log loss, model | Log loss, DraftKings (no vig) |
|---|---|---|---|
| Hitters H+R+RBI | 13,763 | 0.6897 | **0.6857** |
| Pitcher strikeouts | 1,435 | 0.6901 | **0.6843** |

DraftKings' price an hour before first pitch is more accurate than the model on both.
Regressing the result on both chances (log-odds): hitters model 0.12 / DraftKings 0.92,
strikeouts 0.28 / 0.95. The model carries a little information the price doesn't, but
its edges are several times too big.

## Return by edge (both props)

| Edge | Bets | Won | Model said | Break-even | Return | 90% range |
|---|---|---|---|---|---|---|
| 0–2% | 1,615 | 51.5% | 54.3% | 53.8% | -4.6% | -8 to -1% |
| 2–5% | 1,997 | 50.0% | 56.3% | 54.4% | -8.1% | -12 to -5% |
| 5–8% | 1,452 | 55.2% | 59.6% | 56.0% | -1.1% | -5 to +3% |
| **8–12%** | **1,249** | **54.0%** | **62.3%** | **56.8%** | **-4.9%** | **-9 to -1%** |
| 12–20% | 898 | 56.3% | 63.5% | 55.3% | +1.6% | -3 to +7% |
| 20%+ | 252 | 50.0% | 65.0% | 51.4% | -2.8% | -13 to +7% |

| Strategy ($10 flat) | Bets | Profit | Return | Worst drawdown |
|---|---|---|---|---|
| 12%+ | 1,150 | +$72 | +0.6% | $364 |
| 8–12% | 1,249 | -$612 | -4.9% | $718 |
| 8%+ | 2,399 | -$540 | -2.3% | $799 |
| 5%+ | 3,851 | -$693 | -1.8% | $1,114 |

## Sizing

A model/DraftKings blend fitted on August (hitters 8% model, strikeouts 67%) and bet in
September: flat $10 on every positive blended edge -6.1% (377 bets); quarter Kelly
from $1,000 ended at $787, half Kelly at $586. No sizing rule makes money while the
underlying bets don't.

## Caveats

One snapshot per game, an hour before first pitch (close to the closing price, which is
the hardest to beat; live trading takes earlier prices). Lineups are the ones actually
used. Two months, one season.
