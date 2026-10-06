# Brief: Strikeouts

You improve the starting-pitcher strikeout model. It is the one place the model adds
information beyond DraftKings (blend weight 0.26), so gains here matter most.
Branch: `team/strikeouts`. Read `CLAUDE.md`, `docs/WORK_PLAN.md`,
`docs/gate_phase1_pitcher.md` and `docs/blend_2026.md` first.

## You own

- `baseball_stats/pa_model.py`, `baseball_stats/simulate.py`, the pitcher parts of
  `baseball_stats/sim_models.py` (`PASimpleK`, `PASeqK`)
- Pitcher features in `baseball_stats/features.py`. That file is shared with Hitters:
  touch only pitcher columns and keep the diff small.
- Your write-ups: `docs/strikeouts_*.md`

## How your work is judged

Two steps, both required before a change goes live:

1. **Walk-forward gate** (`scripts/evaluate.py --kind pitcher`, five folds) against the
   live model `pa_simple`:
   - mean log loss lower,
   - better in ≥4 of 5 folds,
   - RPS lower,
   - calibration within 0.5 pp at every line.
2. **Market test.** On the historical prices (`odds-history` branch, extract with
   `git archive origin/odds-history | tar -x -C <dir>`), run `scripts/backtest.py` and
   `scripts/fit_blend.py` with your model in place of `pa_simple`. The model's weight in
   the blend should rise, or the blend's out-of-sample log loss on strikeouts should fall
   (currently 0.6832 vs DraftKings 0.6843).

If step 1 passes but step 2 doesn't, say so plainly: the change can go live as a shadow
model only. You may not spend Odds API credits; use the prices already downloaded.

## Ideas to try (pick by expected value)

1. **Umpire strike-zone tendency.** Rolling, leakage-free called-strike rate above
   expected by home-plate umpire (Statcast has `umpire`; games have `hp_umpire_id`).
2. **Pitch mix vs lineup whiffs.** Each pitch type's whiff rate for the pitcher, weighted
   against the opposing hitters' whiff rates by pitch type.
3. **Leash / workload.** Recent pitch counts, days rest, bullpen fatigue, manager tendency.
   Batters faced is half of the strikeout total.
4. **Lineup certainty.** Projected vs confirmed opposing lineups. Measure how much error
   comes from projected lineups.
5. **Calibration by line.** The backtest showed overs at 4.5 and 5.5 losing badly. Check
   whether the high lines are over-predicted.

Open a PR per idea that passes, with numbers. Report ideas that fail in a short write-up
too, so nobody repeats them.
