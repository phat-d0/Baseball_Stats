# Hitters: the whole H+R+RBI distribution from a classifier (`dist`)

Hitters role, Oct 6, 2026. One change passed both tests; three feature ideas failed.

## What changed

`current` predicts a hitter's expected H+R+RBI and spreads it with one negative binomial
for everyone. H+R+RBI isn't shaped like that (a home run alone is 3; a single with
nobody on is 1), so the chances at the 0.5 line were 3.8 pp off. `model.DistModel`
(`dist`) uses the same features and the same heavy smoothing, but a gradient-boosted
classifier on the count (0..7, 8+) gives the whole distribution directly.

## Gate (`scripts/evaluate.py --kind batter`, five walk-forward months, 35,568 games)

| Variant | Log loss | RPS | MAE | Cover 50 / 80 | Calib 0.5 / 1.5 / 2.5 (pp) |
|---|---|---|---|---|---|
| season_avg | 0.6637 | 0.1258 | 1.505 | 0.75 / 0.91 | 3.2 / 5.4 / 5.0 |
| current | 0.6344 | 0.1235 | 1.500 | 0.72 / 0.92 | 3.8 / 1.8 / 2.7 |
| **dist** | **0.6319** | **0.1230** | 1.502 | 0.78 / 0.93 | **1.5** / 1.9 / **1.7** |

By month (current / dist): 2025-06 0.6312 / 0.6297 · 2025-08 0.6407 / 0.6366 ·
2026-05 0.6303 / 0.6282 · 2026-07 0.6363 / 0.6341 · 2026-09 0.6334 / 0.6308.
Better in 5 of 5 months, RPS lower, calibration within 0.5 pp everywhere:
**PASSED**. The gain (-0.0025) is about 8 times what `components` or `pa_sim` moved.
Full numbers: `hitters_dist_gate.json`.

## Market test (`scripts/fit_blend.py --batter dist`, fit Jun-Sep 2025, test Aug-Sep 2026)

| Hitters, 13,763 test lines | current | dist |
|---|---|---|
| Blend model weight (fit) | 0.17 | **0.26** |
| Model log loss | 0.6897 | **0.6870** |
| DraftKings log loss | 0.6857 | 0.6857 |
| Blend log loss | 0.6860 | 0.6859 |
| Hitter bets with a blended edge | 3 | 46 |

- The model weight rises in every 2025 month (0.27 / 0.23 / 0.16 / 0.44), and also when
  fitting on 2026 and testing on 2025 (0.16 -> 0.22).
- The blend still only matches DraftKings out of sample; it doesn't beat it.
- The 46 hitter bets won 58.7% where the blend said 58.4% (honest), for +0.5%
  (90% range -21% to +21%). That is no evidence of profit either way.

Full numbers: `hitters_dist_blend.json`.

**Recommendation:** make `dist` the live hitter model, with the refitted batter blend
weights (`blend.json`). The chances shown in the app get better calibrated (the 0.5 line
was 3.8 pp off), and edges stay honest. Don't expect hitter profit from it yet.

## What failed (don't repeat without a new angle)

All tested with `dist` as the base, same folds:

- **Plate appearances:** a team run-environment index (own offense x opposing run
  prevention x park) and an expected-PA estimate from lineup slot, run environment and
  home/away.
- **Opposing bullpen:** expected times on base vs the bullpen (PAs after the starter x a
  log5 of hitter on-base vs bullpen on-base).
- **Teammates:** on-base of the three hitters ahead and home-run rate of the three
  behind, weighted by distance, and chances to drive in / be driven in.

Together: log loss 0.63193 vs 0.63189 without them, blend weight 0.25 vs 0.25.
The existing features (`slot_pa_exp`, `pa_vs_sp`, `ahead_onbase`/`behind_onbase`,
`opp_team_rp_*`, park, weather) already carry this. Tuning the classifier (15 leaves,
or capping at 6) changed log loss by under 0.0001.

## Where DraftKings may be wrong (screen, for whoever comes next)

The residual (result minus DraftKings' no-vig chance) against every feature, both years
separately (27,142 lines in 2025, 13,710 in 2026). With ~450 tests, a t of about 2 in
both years happens by chance. Nothing is strong, but these held their sign:

- **1.5 line** (DraftKings' main hitter line, 78% of lines; 0.5 is only 16%): overs win
  more for hitters with more expected PAs vs the starter (`exp_h_vs_sp` t = 2.4 / 2.8,
  `pa_vs_sp` 1.7 / 3.4) and more PAs a game (`pa_per_g_*`).
- **2.5 line:** overs win less for power hitters (`hr_per_pa_shr` t = -1.8 / -2.1) and
  more against weak bullpens (`opp_team_rp_onbase_pct` +1.5 / +2.0).

These are what the blend already uses through the model; none was strong enough to be
worth a separate rule.

## Notes

- Fitting `dist` adds about 60 s per publish run (two fits on the full history, as
  `current` does). If that matters, it can be cached daily like the plate-appearance
  models.
- `current` can shadow `dist` (`BASEBALL_SHADOW_BATTER=current`); the default shadow stays
  `pa_sim`.
- Confirmed vs projected lineups (brief idea 5) isn't measured here: the historical
  prices come with the lineups actually used, and the live log has too few hitter rows
  with projected lineups so far.
