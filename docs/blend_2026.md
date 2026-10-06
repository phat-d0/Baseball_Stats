# Model/DraftKings blend (edges from Oct 6, 2026)

The backtest (`backtest_2026.md`) showed DraftKings' price is more accurate than the
model, and that the model's edges were several times too big. Since Oct 6, 2026 the app
works out edges from a blend instead (`baseball_stats/blend.py`):

    logit(p) = a * logit(model chance) + b * logit(DraftKings no-vig chance) + c

## Fit: June-September 2025 (1,561 games, 30,188 lines; `scripts/fit_blend.py`)

| | Model weight | DraftKings weight | Intercept | Lines |
|---|---|---|---|---|
| Hitters H+R+RBI | 0.17 | 0.91 | -0.04 | 27,228 |
| Pitcher strikeouts | 0.26 | 0.73 | +0.01 | 2,960 |

Model weight by month: hitters 0.23 / 0.24 / 0.10 / 0.15, strikeouts 0.32 / 0.08 / 0.30 /
0.37 (June to September). Model chances are walk-forward, as in the backtest.

## Test: August-September 2026 (777 games, 15,198 lines, not used in the fit)

| Log loss | Model | DraftKings | Blend |
|---|---|---|---|
| Hitters | 0.6897 | **0.6857** | 0.6860 |
| Strikeouts | 0.6901 | 0.6843 | **0.6832** |

| Blended edge | Bets | Won | Blend said | Return | 90% range |
|---|---|---|---|---|---|
| 0%+ | 221 | 52.9% | 52.1% | +3.7% | -8% to +15% |
| 1%+ | 150 | 52.7% | 52.4% | +3.7% | -10% to +17% |
| 2%+ | 94 | 48.9% | 52.9% | -3.3% | -20% to +13% |
| 3%+ | 68 | 50.0% | 54.0% | -2.0% | -21% to +17% |
| 5%+ | 32 | 43.8% | 56.5% | -17.5% | -45% to +9% |

218 of the 221 were strikeouts: for hitters the model's tilt almost never beats
DraftKings' margin. The blend's chances are honest (won 52.9% vs said 52.1%, where the
model's own edges said 59% and won 53.6%), but two months can't tell +3.7% from zero.

## In the app

- Edges, value picks, highlights and paper trades use the blended chance; the minimum
  edge control is 1 / 2 / 3 / 5%. The player sheet shows model, DraftKings and blend.
- The price log keeps the model's own chance (`p_model`), so Record → vs DraftKings
  still scores the model, and adds `p_blend`.
- Paper portfolio: "Blended 1%+" ($10 on the first price with a 1%+ blended edge). The
  model-edge strategies (12%+, 8-12%) keep their trades from before Oct 6 but take no
  new ones.
- Refit with `scripts/fit_blend.py --weights baseball_stats/blend.json` as more prices
  are logged.
