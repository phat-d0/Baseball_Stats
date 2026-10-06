# Phase 2 gate: hitter H+R+RBI (whole-game simulation)

Run: `python scripts/evaluate.py --kind batter` (one process per month, run in parallel) on
2024-03-20 .. 2026-10-04 data (557,212 plate appearances). Five walk-forward months, 35,568
hitter games, 10,000 simulations per game. Full numbers: `gate_phase2_batter.json`.

| Variant | Log loss (lines) | RPS | MAE | Cover 50 / 80 | Calib error 0.5 / 1.5 / 2.5 (pp) |
|---|---|---|---|---|---|
| season_avg | 0.6637 | 0.1258 | 1.505 | 0.75 / 0.91 | 3.2 / 5.4 / 5.0 |
| **current** | **0.6341** | 0.1234 | **1.501** | 0.72 / 0.92 | 3.8 / 1.9 / 2.6 |
| components | 0.6338 | 0.1233 | 1.502 | 0.73 / 0.92 | 3.8 / 1.7 / 2.4 |
| pa_sim | 0.6346 | 0.1235 | 1.519 | 0.77 / 0.93 | **2.4** / 2.8 / **2.1** |

Log loss by month (current / components / pa_sim):
2025-06 0.6314 / 0.6307 / 0.6319 · 2025-08 0.6404 / 0.6393 / 0.6380 ·
2026-05 0.6297 / 0.6299 / 0.6314 · 2026-07 0.6360 / 0.6361 / 0.6383 ·
2026-09 0.6331 / 0.6328 / 0.6333

By batting order (log loss, current / components / pa_sim): slots 1-3 0.6382 / 0.6381 / 0.6395 ·
slots 4-6 0.6399 / 0.6393 / 0.6389 · slots 7-9 0.6243 / 0.6240 / 0.6253

## Gate

| Check | components | pa_sim |
|---|---|---|
| Mean log loss below current | yes (-0.0003) | **no** (+0.0005) |
| Below current in >= 4 of 5 months | **no (3)** | **no (1)** |
| Mean RPS below current | yes | **no** |
| Calibration within 0.5 pp of current at every line | yes | **no** (1.5 line: +0.9 pp) |
| **Result** | NOT PASSED | NOT PASSED |

Hitters stay on `current`. `pa_sim` runs in shadow: it is the best calibrated model at the
0.5 line (2.4 pp vs 3.8), DraftKings' main H+R+RBI line, so its closing-line log loss
against the market is worth collecting. The differences in this table are all under 0.1%
of log loss; one hitter's H+R+RBI is mostly noise no model removes.
