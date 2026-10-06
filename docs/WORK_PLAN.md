# Work plan and status

Last updated: 2026-10-06 (MLB postseason under way; regular season ended Sep 28).

## Done

| # | Work | Where / result |
|---|---|---|
| 1 | Data pipeline: box scores, Statcast, leakage-free features, H+R+RBI and strikeout models (Poisson GBM + negative binomial) | `baseball_stats/`, README "Models" |
| 2 | iPhone PWA on GitHub Pages; publish every ~20 min on game days (self-chaining workflow, since GitHub throttles cron) | `web/`, `.github/workflows/publish.yml` |
| 3 | DraftKings props via The Odds API under a daily credit budget; prices from 24 h before first pitch (every 4 h, then 2 h inside 8 h, 30 min in the last 3 h, plus a closing pull) | `odds.py` |
| 4 | Log matchup scores, tuning | `features.py`, README "Matchup scores" |
| 5 | Spec "Pick Logging and Market Grading" (Claude Docs artifact Fr2Eg2WPKKftqTPYYjwXW3): append-only price log, grading, CLV, Record → vs DraftKings, per-line calibration | `tracking.py` |
| 6 | Spec "Plate Appearance Models and Full Distributions" (artifact 1cWLGEuvXu66BYH4etE6XV), steps 1–5 | see below |
| 6a | PA table + base-out transitions; reconciliation passes (≥99% of games match, RBIs within 1.5%) | `pa_data.py` |
| 6b | Phase 1 gate (strikeouts): `pa_simple` PASSED and is live; `pa_seq` failed and runs in shadow | `docs/gate_phase1_pitcher.md` |
| 6c | Phase 2 gate (hitters, whole-game simulation `pa_sim`): FAILED (log loss 0.6346 vs 0.6341, better in 1 of 5 months); runs in shadow. `components` also failed (3 of 5) | `docs/gate_phase2_batter.md` |
| 7 | Paper trading, $10 a trade, Paper Portfolio tab with a strategy switch | `tracking.PAPER_STRATEGIES` |
| 8 | Game sheet lists and highlights edges; rows show DraftKings' line with implied % | `web/app.js` |
| 9 | Backtest on historical DraftKings prices, Aug–Sep 2026 (777 games, 15,198 lines): **DraftKings is more accurate than the model**; raw model edges were overconfident; 8–12% band −4.9%, 12%+ +0.6%; Kelly sizing lost | `docs/backtest_2026.md` |
| 10 | **Edges now come from a model/DraftKings blend** fitted on Jun–Sep 2025 (hitters 0.17 model / 0.91 DK, strikeouts 0.26 / 0.73). Out of sample on Aug–Sep 2026 it beat DraftKings' log loss on strikeouts (0.6832 vs 0.6843) and matched it on hitters. 221 bets, +3.7% (90% range −8% to +15%), honest calibration. 218 of 221 were strikeouts | `blend.py`, `blend.json`, `docs/blend_2026.md` |

## Live configuration (as of the last push)

- **Models:**
  - Strikeouts: `pa_simple` live, `pa_seq` shadow.
  - Hitters: `current` live, `pa_sim` shadow (4,000 simulations a game).
- **Edges:** blended; minimum edge control 1 / 2 / 3 / 5%, default 1%.
- **Paper strategies:**
  - `Blended 1%+`: main, started Oct 6, 2026.
  - `Model 12%+` and `Model 8–12%`: retired; they keep their record from Oct 5 (12%+ went
    2–4, −$21.61; 8–12% went 1–1, −$4.05).
- **Odds API:** 100k credits/month shared with the soccer app, reset on the 5th. After
  the backtests about **39,400 were left on Oct 6**. Something else, probably the soccer
  app, spent ~13.7k in under an hour that day, so watch the balance in the publish logs.

## Next steps (in order)

1. **Watch the blend live.**
   - Record → vs DraftKings and the Blended 1%+ paper strategy accumulate from Oct 6.
   - Expect few edges in the postseason (2–4 games a day).
   - Don't judge it before a few hundred graded bets.
2. **Refit the blend with more data.**
   - Next spring, or once the live log has ~1,000+ closing lines, rerun
     `scripts/fit_blend.py` with the 2025 + 2026 historical prices plus the live log.
   - Consider downloading Apr–May 2025 and Apr–Jul 2026 (~40k credits) after the Nov 5
     reset if there's room.
3. **Improve the strikeout model.** It's the only place the model adds information beyond
   DraftKings (blend weight 0.26, out-of-sample gain). Ideas, each to be gated with
   `scripts/evaluate.py`, then checked with `scripts/backtest.py` and `fit_blend.py`:
   - Confirmed lineups at pricing time; projected lineups add noise.
   - Umpire strike-zone tendencies.
   - Pitch-mix vs lineup whiff rates.
   - Pitch-count / leash signals from the stay-or-go model.
4. **Hitters.** The blend gives them almost nothing (3 edges in 2 months). Either:
   - accept that and keep showing hitter projections without edges, or
   - test whether `pa_sim` (shadow) beats `current` on live closing lines. Its calibration
     at the 0.5 line was better in the gate.

   Review the shadow's numbers in Record → vs DraftKings once ~300 graded closing lines
   exist (spec step 5).
5. **Earlier prices.** The backtests used prices from 1 hour before first pitch. A second
   historical pull at ~6 h before would show whether earlier prices are softer (another
   ~15k credits for Aug–Sep 2026). Run `fetch_history.py` with `--minutes-before 360`
   into a separate folder; it needs a small change to keep both.
6. **Bet sizing.** Revisit only if the blended edges show a positive return with a
   reasonably tight range. Use fractional Kelly on `p_blend`, capped at 1–2% of bankroll.

## How to continue in a new session

1. Read `CLAUDE.md`, this file, and the latest `docs/*.md` results.
2. `git log --oneline | head -20` for recent changes. The working branch is
   `claude/stoic-davinci-v4p6k9`, which is also the default branch and deploys the app.
3. For live state, check the latest "Publish app" run's build log. It shows the odds
   balance, price-log rows, grades and paper results per strategy.
4. Restore data if needed (see `CLAUDE.md` → Environment gotchas), then run the tests
   before changing anything.
