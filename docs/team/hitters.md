# Brief: Hitters

You improve the batter H+R+RBI model. Right now the market barely needs it: blend weight
0.17, and only 3 blended edges in two months of backtest. Either find information
DraftKings misses, or show convincingly that there isn't any.
Branch: `team/hitters`. Read `CLAUDE.md`, `docs/WORK_PLAN.md`,
`docs/gate_phase2_batter.md` and `docs/blend_2026.md` first.

## You own

- `baseball_stats/game_sim.py`, `GameSimH` and batter parts of
  `baseball_stats/sim_models.py`
- Batter features in `baseball_stats/features.py`. That file is shared with Strikeouts:
  touch only batter columns and keep the diff small.
- Your write-ups: `docs/hitters_*.md`

## How your work is judged

Two steps, both required before a change goes live:

1. **Walk-forward gate** (`scripts/evaluate.py --kind batter`, five folds) against
   `current`:
   - mean log loss lower,
   - better in ≥4 of 5 folds,
   - RPS lower,
   - calibration within 0.5 pp at every line.

   Whole-game simulations are slow (~30 min per fold at 10,000 sims). Develop with fewer
   sims, then run the final gate at full size.
2. **Market test** on the historical prices (`odds-history` branch) with
   `scripts/backtest.py` / `scripts/fit_blend.py`. The model's blend weight for hitters
   should rise, or the blend's out-of-sample log loss should beat DraftKings' (currently
   0.6860 vs 0.6857).

You may not spend Odds API credits. Live shadow model: `pa_sim` (logged as `p_shadow`).

## Ideas to try

1. **The 0.5 line.** It's DraftKings' main hitter line, and `pa_sim` was best calibrated
   there (2.4 pp vs 3.8). Consider a model aimed at P(H+R+RBI ≥ 1) directly, or check
   whether `pa_sim` beats `current` at 0.5 in the market test.
2. **Plate appearances.** Lineup slot and team run environment drive how many plate
   appearances a hitter gets, which drives most of the total. Check the projected-PA error.
3. **Teammates.** Runs and RBIs depend on who bats around the hitter (on-base ahead,
   power behind).
4. **Opposing bullpen quality and park/weather** for runs.
5. **Confirmed vs projected lineups.** Quantify the cost of projected lineups.

Open a PR per idea that passes, with numbers. Write up failures briefly so nobody repeats
them.
