# Strikeouts: five ideas tested, none pass (Oct 2026)

All five ideas in the Strikeouts brief were tested against the live model `pa_simple`.
**None passed the walk-forward gate, so nothing goes live.** One finding matters for
betting now: **the model only adds information beyond DraftKings when the opposing
lineup is confirmed**, but the live log and the paper strategy treat every strikeout price
as if it were.

## Method

- Data: the `data-snapshot` tables (2024-03-20 .. 2026-10-05), plus pitch-level
  Statcast downloaded from Baseball Savant for every game day (pitch type, location,
  strike zone; ~2.1M pitches) for the umpire, catcher and pitch-mix features.
- Gate: the five walk-forward months of `scripts/evaluate.py` (2025-06, 2025-08,
  2026-05, 2026-07, 2026-09; 3,952 starts), against `pa_simple` rebuilt the same way.
  Rules: mean log loss lower, better in ≥4 of 5 months, RPS lower, calibration within
  0.5 pp at every line.
- Market test: walk-forward predictions for Jun–Sep 2025 and Aug–Sep 2026, matched to
  the `odds-history` prices (strikeouts only: 2,960 lines in 2025, 1,435 in 2026). The
  blend was fitted on 2025 and tested on 2026, as `scripts/fit_blend.py` does.
- My `pa_simple` baseline: log loss 0.5110, RPS 0.0806. The market baseline gives the
  model a blend weight of 0.20 and a blend log loss of 0.6835 vs DraftKings 0.6843. The
  blend doc says 0.26 and 0.6832; the gap comes from the newer data snapshot. All
  comparisons below are against this rebuilt baseline.

## Results

Δ log loss is the candidate minus `pa_simple` (negative is better).

| Idea | Variant | Δ log loss | Months better | Δ RPS | Calibration | Gate | Blend weight (0.20) | Blend log loss (0.6835) |
|---|---|---|---|---|---|---|---|---|
| 5 Calibration / spread | empirical batters-faced + K-rate random effect | +0.0001 | 2/5 | 0.0000 | ok | fail | 0.20 | 0.6836 |
| 3 Leash / workload | bullpen pitches last 1-3 days, team starter pitch counts, day of season | +0.0001 | 2/5 | -0.0000 | 6.5 line +0.8 pp | fail | 0.23 | 0.6838 |
| 1 Umpire | umpire called strikes above expected (outcome-model input) | +0.0002 | 1/5 | -0.0000 | ok | fail | 0.24 | 0.6835 |
| 1 Umpire + catcher | + catcher framing (outcome-model input) | -0.0003 | 4/5 | -0.0001 | 6.5 line +0.53 pp | fail | 0.26 | 0.6830 |
| 1 Umpire + catcher | same two as a fitted logit offset | +0.0006 | 3/5 | +0.0001 | 4.5 line +1.0 pp | fail | 0.20 | 0.6838 |
| 2 Pitch mix | hitter whiff rate vs this pitcher's mix (4 pitch groups) | +0.0005 | 3/5 | +0.0001 | ok | fail | 0.21 | 0.6829 |
| 4 Lineups | projected lineup (the app's rule: last starting nine) | +0.0026 | 1/5 | +0.0004 | worse | (diagnostic) | **0.09** | 0.6840 |
| 4 Lineups | projected: last nine vs a same-handed starter | +0.0033 | 1/5 | +0.0005 | worse | (diagnostic) | **0.10** | 0.6844 |
| — | oracle: actual batters faced known | -0.0241 | 5/5 | -0.0046 | — | (ceiling) | — | — |

### 5. Calibration by line: not a calibration problem

- Over all starts, `pa_simple` is calibrated at every line. Predicted vs actual
  over-rate: 3.5 → 66.6% / 66.3%, 4.5 → 50.0 / 50.5, 5.5 → 34.5 / 35.1, 6.5 → 22.0 /
  22.7, 7.5 → 13.0 / 13.6. The mean is right in every fifth of projections.
- On the lines DraftKings actually priced, the model sits further from 50% than the
  results. At the 7.5 line: model 41%, DraftKings 48%, actual 46%. At 3.5: model 54%,
  DraftKings 51%, actual 51%. Where the model and DraftKings differ most, the actual
  rate moves only about a third of the way toward the model. That is the market knowing
  more, and the blend already corrects it.
- Two errors cancel in the spread:
  - Batters faced is modelled as Poisson (variance 21.7), but the real variance is
    13.3, skewed by early exits.
  - Strikeouts given batters faced vary more than binomial (4.27 vs 3.63), because the
    per-game strikeout rate varies.
  - Fixing both (empirical batters-faced distribution, logit-normal random effect,
    σ = 0.1–0.3 on non-gate months) changed log loss by under 0.0005.

### 3. Leash / workload: real but too small

- Batters faced is only part of the problem. Even knowing the actual batters faced
  would cut log loss just from 0.511 to 0.487.
- The batters-faced misses do follow two patterns:
  - Starters behind a tired bullpen (most relief pitches in the last 3 days) face about
    0.5 more batters.
  - September starts face 0.7 fewer batters than predicted.
- The new features shrank the September bias from −0.69 to −0.43 and cut the
  batters-faced error a little (RMSE 3.644 → 3.613). That was too small to show up in
  strikeout log loss.

### 1. Umpire (and catcher framing): fails

- Called strikes above expected came from a location zone model fitted on 2024 pitches
  only, then summed per umpire over prior games and shrunk.
- The signal is real but small. The top and bottom fifth of umpires differ by about 2%
  in strikeout rate, about 0.1 K per start. Catcher framing is similar.
- Added to the outcome model, umpire plus catcher looked like a narrow miss on
  calibration. Applied as a separate fitted adjustment that leaves the outcome model
  unchanged, it was worse (+0.0006). So the earlier small gain was noise from
  retraining the boosted model.
- The fitted umpire effect falls from 0.024 (2025 folds) to about 0 in 2026, plausibly
  because of the 2026 ABS challenge system. Don't revisit unless the zone rules change
  again.

### 2. Pitch mix vs lineup whiffs: fails

Used four pitch groups: fastball, cutter, breaking, offspeed.

- At the plate-appearance level, the mix-weighted log5 whiff score improves strikeout
  log loss by only 0.05% beyond the existing strikeout matchup.
- The hitter-specific part (this hitter's whiff rate against this pitcher's mix) adds
  nothing (0.51339 → 0.51337). Hitters' pitch-type splits are mostly noise at these
  sample sizes.

### 4. Lineup certainty: the important result

- The app projects a lineup as the team's last starting nine. Only 76% of those players
  actually start.
- With projected lineups, `pa_simple` loses its whole edge over `current`. Its log loss
  is 0.5136, the same as `current`'s 0.5136 and +0.0026 vs confirmed lineups (worse in
  4 of 5 months).
- In the market test its blend weight falls from **0.20 to 0.09**. For three of the four
  2025 months the weight was about 0 (−0.01 to 0.15).
- Better guessing doesn't fix it. "Last lineup vs a same-handed starter" gets 81% of
  players right but scores no better (weight 0.10).
- The model's information beyond DraftKings is in the confirmed lineup.

## Recommendations (for Lead / Edge; these touch files I don't own)

1. **Log whether the opposing lineup is confirmed on strikeout rows.**
   - `tracking.py` hard-codes `lineup_confirmed = True` for pitcher rows. The flag is
     only computed for hitters (`slate.py`).
   - So the price log can't separate projected-lineup strikeout edges from confirmed
     ones.
   - Fix: set it from whether the opponent's lineup was posted. This is an append-only
     column change; old rows stay as they are.
2. **Don't bet strikeout edges on projected lineups.**
   - `paper_trades` lets pitchers through unconfirmed, and it takes the *first* price
     with an edge. That is often the 24 h / 4 h pull, before lineups post.
   - Those edges use the confirmed-lineup blend weight (0.26) for a model that is worth
     about 0.09 at that point. Most of them are probably not real.
   - Either require a confirmed opposing lineup, or use a separate blend weight for
     unconfirmed rows (fitted value about 0.09).
3. When Edge pulls the 6-hours-before prices, compare them with the confirmed-lineup
   (1 hour before) prices. Earlier prices can only be "softer" for us if we can wait for
   lineups.

## What not to repeat

- More dispersion tuning of `pa_simple`.
- Umpire, catcher framing, bullpen-fatigue, or pitch-type-split features in this form.
- All are within ±0.0006 log loss of baseline. Retraining the boosted outcome model with
  any extra column moves single months by 0.003–0.005, which is bigger than these effects.
  Any future small feature should be tested as a fixed offset first, as was done for
  the umpire here.
