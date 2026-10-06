# Work plan and status

Last updated: 2026-10-06, evening, after the team's first round (MLB postseason under way;
regular season ended Sep 28). The team setup is in `CLAUDE.md` → Team and `docs/team/`.

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
| 11 | Team round 1, Strikeouts: umpire/catcher framing, pitch mix vs lineup whiffs, leash/workload, spread fix and calibration by line all **failed** the gate. Key finding: on projected lineups the strikeout model's blend weight falls from 0.20 to 0.09 | `docs/strikeouts_2026-10.md` |
| 12 | Team round 1, Hitters: **`dist` (H+R+RBI distribution from a classifier) passed the gate 5/5** (log loss 0.6320 vs 0.6341; 0.5-line calibration 1.4 pp vs 3.8; Lead reproduced it). It is the live hitter model; blend hitter weight rose 0.17 → 0.26. Out of sample the blend still only matches DraftKings on hitters. PA, bullpen and teammate features added nothing | `docs/hitters_dist.md` |
| 13 | Team round 1, UI: each edge shows DraftKings price + implied %, our chance and the edge; audit fixes (PR #1 merged; Record/Portfolio redesign in PR #5) | `docs/ui_audit.md` |
| 14 | Team round 1, Edge: no blend refinement (per-side, per-line, shrink-on-disagreement) beat the live blend; 6-hour price tooling and one-command refit built (PR #3); no credits spent | `docs/edge_2026.md` |
| 15 | Lead: past postseasons backfilled once (CI lacked ~90 games of 2024–25 postseason; that alone moved the strikeout blend weight 0.26 → 0.20, so treat the edge estimate as fragile, ≈0); Record scores the blend against DraftKings (`logloss_blend`) and grades blend-era picks only; publish runs only on pushes to the deploy branch (team-branch pushes were cancelling deploys) | `publish.py`, `tracking.py`, `publish.yml` |
| 10 | **Edges now come from a model/DraftKings blend** fitted on Jun–Sep 2025 (hitters 0.17 model / 0.91 DK, strikeouts 0.26 / 0.73). Out of sample on Aug–Sep 2026 it beat DraftKings' log loss on strikeouts (0.6832 vs 0.6843) and matched it on hitters. 221 bets, +3.7% (90% range −8% to +15%), honest calibration. 218 of 221 were strikeouts | `blend.py`, `blend.json`, `docs/blend_2026.md` |

## Live configuration (as of the last push)

- **Models:**
  - Strikeouts: `pa_simple` live, `pa_seq` shadow.
  - Hitters: `dist` live (unless a repo variable `BASEBALL_MODEL_BATTER` overrides it),
    `pa_sim` shadow (4,000 simulations a game); `current` stays as the fallback.
- **Edges:** blended; minimum edge control 1 / 2 / 3 / 5%, default 1%.
- **Paper strategies:**
  - `Blended 1%+`: main, started Oct 6, 2026.
  - `Model 12%+` and `Model 8–12%`: retired; they keep their record from Oct 5 (12%+ went
    2–4, −$21.61; 8–12% went 1–1, −$4.05).
- **Odds API:** 100k credits/month shared with the soccer app, reset on the 5th. After
  the backtests about 39,400 were left on Oct 6, and **22,958 at 17:41 UTC**, below the
  team's 25,000 floor (something else, probably the soccer app, is spending fast). No
  historical downloads until the Nov 5 reset.

## Next steps (in order)

0. **In flight (team round 2):**
   - Edge: merge PR #3, check that the hitter blend weights reproduce, then a new PR: log
     whether the opposing lineup was confirmed on strikeout rows, take strikeout paper
     trades only once it is, and add a data.json field for the UI.
   - UI: PR #5 (Record "Are we beating DraftKings?" verdict, Portfolio), wired to
     `logloss_blend` and `picks_since`.
1. **Watch the blend live.**
   - Record → vs DraftKings and the Blended 1%+ paper strategy accumulate from Oct 6.
   - Expect few edges in the postseason (2–4 games a day).
   - Don't judge it before a few hundred graded bets.
2. **Refit the blend with more data.**
   - After any model PR merges, and next spring or once the live log has ~1,000+ closing
     lines: `scripts/refit_blend.sh` (from PR #3) on the historical prices plus the live log.
   - Consider downloading Apr–May 2025 and Apr–Jul 2026 (~40k credits) after the Nov 5
     reset if there's room.
3. **Strikeout model.** Round 1 ideas all failed (`docs/strikeouts_2026-10.md`). The
   biggest lever is lineup certainty: the model is worth ~0.20 weight on confirmed
   lineups and ~0.09 on projected ones. Better lineup projection (last lineup vs a
   same-handed starter, 81% right) didn't help. Next ideas need new data (e.g. pitch-level
   arsenal changes, injury/rest news).
4. **Hitters.** `dist` raised the model's blend weight to 0.26 (46 backtest edges in two
   months instead of 3), but the blend still only matches DraftKings. Watch live hitter
   edges; review `pa_sim` (shadow) once ~300 graded closing lines exist (spec step 5).
5. **Earlier prices.** The tooling is ready (PR #3). After the Nov 5 reset, run the Fetch
   historical prices workflow with `minutes_before=360`, `max_credits=15000`,
   `reserve=25000` (Aug–Sep 2026), to see whether earlier prices are softer.
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
