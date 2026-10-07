# Confidence tiers (from Oct 8, 2026)

The app used to sort edges by percentage cutoffs (1 / 2 / 3 / 5% expected return). On
Aug–Sep 2026 none of those cutoffs won significantly more often than break-even, so
edges are now measured against how far our chance usually strays from DraftKings'.

## The measure

- **σ (sigma), per kind:** the standard deviation of `p_blend − p_book` on the prices the
  blend was fitted on. It's stored in `blend.json` (`sigma`) and written by
  `fit_blend.py`.
- **z for a side:** `(our blended chance of the side − 1 / decimal price) / σ`. It says
  how many σ our chance sits above that price's break-even rate, vig included.
- **Tiers** (cumulative: a Strong pick is also a Lean one):
  - **Lean:** z ≥ 1.
  - **Strong:** z ≥ 2.
- **Proven:** at least 200 graded bets, with the win rate at least 2 standard errors
  above break-even. That's `z_realized = (win − break-even) / sqrt(mean(be·(1−be)) / n)`.

σ from the Jun–Sep 2025 fit prices, with the current `blend.json` weights:

| | σ, 2025 (fit, live) | σ, Aug–Sep 2026 |
|---|---|---|
| Hitters (dist) | 0.0123 | 0.0123 |
| Strikeouts | 0.0251 | 0.0225 |

The hitter σ is small: the blend rarely moves far from DraftKings on hitters. To reach
z ≥ 1, our chance must beat break-even by 1.2 points. Break-even already sits about 2
points above DraftKings' no-vig chance, so a hitter Lean needs the blend about 3 σ off
DraftKings. Almost every tier bet is a strikeout.

## Backtest by tier

Rules:

- One bet per player and prop: the line and side with the highest z.
- $1 flat stake.
- Model chances are walk-forward.
- Weights from `blend.json`, with dist hitters.
- 90% bootstrap range on the return.

Full numbers: `edge_tiers.json`.

### Aug–Sep 2026 (out of sample), σ from the 2025 fit (what the app uses)

| Tier | Bets | Won | Break-even | z_realized | Return | 90% range |
|---|---|---|---|---|---|---|
| Lean (≥1σ) | 35 | 48.6% | 51.5% | −0.35 | −6.0% | −34% to +21% |
| Strong (≥2σ) | 7 | 57.1% | 54.1% | +0.16 | +3.9% | −49% to +57% |

34 of the 35 Lean bets were strikeouts.

### Aug–Sep 2026, σ from Aug–Sep 2026 (the Lead's setup)

| Tier | Bets | Won | Break-even | z_realized | Return | 90% range |
|---|---|---|---|---|---|---|
| Lean (≥1σ) | 41 | 53.7% | 51.6% | +0.27 | +3.7% | −23% to +29% |
| Strong (≥2σ) | 9 | 55.6% | 53.8% | +0.10 | +2.6% | −42% to +48% |

I couldn't reproduce the Lead's numbers exactly:

- **Lead:** σ of 0.0175 / 0.0222; Lean 41 bets at −7.6%, Strong 13 bets at +2.2%.
- **Here:** the bet count for Lean matches (41), but the returns don't. The hitter σ
  differs (0.0123 here vs 0.0175).
- **Likely cause:** a different choice of rows or of side. Here the best side is chosen
  by z. Choosing it by expected return first and then filtering on z gives different
  bets.

Either way, both results are noise: a few dozen bets, with z_realized between −0.4 and
+0.3.

### Jun–Sep 2025 (in sample, for reference)

| Tier | Bets | Won | Break-even | z_realized | Return | 90% range |
|---|---|---|---|---|---|---|
| Lean (≥1σ) | 90 | 46.7% | 52.4% | −1.10 | −9.3% | −26% to +8% |
| Strong (≥2σ) | 35 | 54.3% | 51.6% | +0.32 | +7.3% | −21% to +36% |

Even on the prices the blend was fitted on, Lean bets lost.

## What this means

- **The tiers measure the right thing, but they don't create an edge.** No tier, in
  any period, has a win rate meaningfully above break-even. The best is z_realized
  +0.32 on 35 bets. None is close to "proven".
- **Expect about 15–20 bets a month in the regular season, almost all strikeouts.**
  The postseason will be far fewer.
- **The paper strategy is a test, not a recommendation.** "Lean (1σ+)" starts clean on
  Oct 8 and needs a few hundred graded bets before its record means anything. The
  Record tab marks a tier "proven" only past 200 bets at z_realized ≥ 2.

## In the app (data contract)

- **`data.json`:**
  - `edge_tiers`: the tiers, as above.
  - `edge_sigma`: `{batter, pitcher}`.
  - Every DraftKings line in a player's `book` has `z_over` / `z_under`.
- **Price log:**
  - New columns `z_over` / `z_under`, logged at download time.
  - Blend-era rows logged before the columns existed get them filled from `p_blend`,
    the prices and the current σ.
- **Picks:** `tracking.picks(…, by="z")` chooses each snapshot's best line and side by
  z, then thresholds on z. Within one player all lines share a σ, so the highest z is
  the biggest gap over break-even. Picking by expected return first could pass over a
  line that clears the bar.
- **Paper:**
  - "Lean (1σ+)" (`lean`) is the main strategy and trades prices fetched from
    `TIERS_FROM` (2026-10-08 00:00 UTC). The lineup rules are unchanged.
  - "Blended 1%+" is retired: it keeps the trades it placed before then and takes no
    new ones.
  - Every strategy shows `metric`, `since` and `until`.
- **Record:**
  - `record.market.<kind>.by_tier` and `record.market.tiers` (both kinds) hold n, win,
    break-even, z_realized, return with its 90% range, and `proven`.
  - Both use blend-era rows only.
  - `by_threshold` / `by_edge` stay until the app stops using them.
