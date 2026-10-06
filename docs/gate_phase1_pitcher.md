# Phase 1 gate: pitcher strikeouts (plate appearance models)

Run: `python scripts/evaluate.py --kind pitcher` on 2024-03-20 .. 2026-10-04 data
(557,212 plate appearances, 7,393 games; PA/box score reconciliation: 7,392 of 7,393
games match on K and hits, RBIs within 1.5%). Five walk-forward months, 3,952 starts,
10,000 simulations per start. Full numbers: `gate_phase1_pitcher.json`.

| Variant | Log loss (lines) | RPS | MAE | Cover 50 / 80 | Calib error 3.5 / 4.5 / 5.5 / 6.5 / 7.5 (pp) |
|---|---|---|---|---|---|
| season_avg | 0.5440 | 0.0865 | 1.847 | 0.61 / 0.86 | 5.8 / 6.1 / 5.1 / 4.2 / 3.7 |
| current | 0.5125 | 0.0808 | 1.732 | 0.63 / 0.86 | 4.2 / 4.7 / 4.4 / 3.0 / 3.3 |
| **pa_simple** | **0.5102** | **0.0804** | **1.727** | 0.63 / 0.86 | **3.5 / 4.1 / 3.9 / 3.0 / 3.0** |
| pa_seq | 0.5119 | 0.0808 | 1.737 | 0.60 / 0.84 | 4.5 / 4.7 / 4.4 / 3.3 / 3.0 |

Log loss by month (current / pa_simple / pa_seq):
2025-06 0.5083 / 0.5103 / 0.5105 · 2025-08 0.5385 / 0.5278 / 0.5296 ·
2026-05 0.5094 / 0.5093 / 0.5080 · 2026-07 0.5011 / 0.5007 / 0.5049 ·
2026-09 0.5051 / 0.5027 / 0.5063

By workload (log loss, current / pa_simple / pa_seq): short leash 0.4656 / 0.4567 / 0.4612 ·
middle 0.5325 / 0.5370 / 0.5384 · workhorse 0.5526 / 0.5512 / 0.5511

## Gate (steps 1-4)

| Check | pa_simple | pa_seq |
|---|---|---|
| Mean log loss below current | yes | yes |
| Below current in >= 4 of 5 months | yes (4) | **no (2)** |
| Mean RPS below current | yes | **no (equal)** |
| Calibration within 0.5 pp of current at every line | yes (better at all) | yes |
| Reconciliation >= 99% games, RBI within 2% | yes | yes |
| Speed (simulation ~0.17 s per start; training ~30 s) | yes | yes |
| **Result** | **PASSED** | NOT PASSED |

Step 5 (market check on 300 graded closing lines with shadow rows) needs the price log
to grow; with the season ending, it is reviewed in the spring.

pa_simple's gain over current is small (log loss -0.45%) and comes mostly from short-leash
starters. The stay-or-go simulation (pa_seq) did not earn its complexity: the batters-faced
model alone is as good.
