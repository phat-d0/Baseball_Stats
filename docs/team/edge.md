# Brief: Edge

You find where DraftKings' prices are beatable, and you decide which chance the app bets
on. Today that chance is the model/DraftKings logistic blend (`docs/blend_2026.md`).
Branch: `team/edge`. Read `CLAUDE.md`, `docs/WORK_PLAN.md`, `docs/backtest_2026.md` and
`docs/blend_2026.md` first.

## You own

- `baseball_stats/blend.py`, `baseball_stats/blend.json`
- `scripts/fit_blend.py`, `scripts/backtest.py`, `scripts/fetch_history.py`,
  `.github/workflows/fetch-history.yml`
- Paper strategies (`PAPER_STRATEGIES` in `tracking.py`) and anything about bet sizing
- Your write-ups: `docs/edge_*.md`

## Credits

You are the only role allowed to spend Odds API credits:
- **Budget:** 15,000 credits until the Nov 5, 2026 reset.
- **Floor:** never let the shared balance fall below 25,000. Pass `reserve` ≥ 25000 to the
  workflow.
- **Cost:** a historical game costs ~20 credits (both props, one snapshot).
- **Record:** log every spend (date, range, credits, balance after) in your write-up.

## How your work is judged

- **Out of sample only.** Fit on one period and test on a later one you didn't fit on.
  Report log loss against DraftKings and the model, plus return with a 90% bootstrap
  range, by edge band.
- **Honesty first.** In each band, the predicted win rate must match the actual win rate.
  An edge that looks bigger but is less honest is worse.
- **More bets are not a goal.** Fewer, real edges beat many fake ones.

## First tasks (in order)

1. **Earlier prices.**
   - Extend `fetch_history.py` and the workflow so a second snapshot time can be stored
     beside the existing one, e.g. `--minutes-before 360` into its own folder or file
     key.
   - Pull Aug 1–Sep 27, 2026 at 6 hours before first pitch (~15k credits; this is your
     whole budget, so check the balance first and scale the range down if needed).
   - Question to answer: are earlier prices softer, i.e. do blended edges at 6 h return
     more than at 1 h? This decides when the app should trust an edge.
2. **Blend refinements, tested out of sample on the existing data:**
   - per-line or per-side terms (overs vs unders behaved differently in the backtest);
   - a confirmed vs projected lineup term for hitters;
   - shrinking the model weight when the model and DraftKings disagree a lot.

   Keep the simplest version that wins.
3. **Refit pipeline.** Make it one command to refit `blend.json` from all historical
   prices plus the live price log (`odds-log` branch, rows with closing prices). Document
   when to rerun it: after any Strikeouts or Hitters model PR merges, because the blend
   weights depend on the model.
4. **Sizing.** Only if task 1 or 2 finds a band with a positive return whose 90% range is
   mostly above zero: backtest fractional Kelly on `p_blend`, capped at 1–2% of bankroll.
