# Strikeout bets wait for the opposing lineup (Oct 2026)

## Why

Strikeouts found that the model's extra information about strikeouts comes from the
confirmed lineup (`strikeouts_2026-10.md`, section 4):

- With the app's projected lineup (the team's last starting nine), `pa_simple` is no
  better than `current`.
- Its blend weight falls from 0.20 to 0.09.

Before this change, the live system still treated every strikeout price as if the
lineup were confirmed:

- The price log set `lineup_confirmed = True` on every pitcher row.
- The Blended 1%+ paper strategy took the first strikeout edge it saw, often from the
  24-hour or 4-hour pull, before lineups post.
- Those edges used the confirmed-lineup weight, about three times what the model is
  worth at that point.

## What changed

- **Slate** (`slate.pending_rows`): each probable starter gets `opp_lineup_confirmed`,
  which says whether the team he faces has posted its lineup.
- **Price log** (`tracking.snapshot_rows`): pitcher rows get a new column,
  `opp_lineup_confirmed`. Batter rows leave it empty.
  - Old rows are not touched. They read as empty.
  - `lineup_confirmed` keeps its old meaning.
- **Paper strategies** (`tracking.paper_trades`): a strikeout trade is placed at the
  first logged price, with a qualifying edge, after the opposing lineup is posted.
  - Earlier prices against a projected lineup are skipped.
  - Rows from before the column existed count as confirmed, so trades already recorded
    stay as they are.
- **data.json**: every pitcher has `opp_lineup_confirmed` (true/false). The UI can use
  it to label strikeout edges against a projected lineup.

Value picks and Record → vs DraftKings (`tracking.picks`, `bestBet` in `app.js`) are
unchanged. The Record tab grades the model at the closing price, where lineups are
almost always posted.

## Not done

- **No separate blend weight for unconfirmed rows** (Strikeouts fitted about 0.09).
  Requiring a confirmed lineup is simpler, and the paper strategy doesn't need both
  rules.
- **No effect yet on historical numbers.** Historical prices are an hour before first
  pitch with real lineups, so the backtest and blend tests in `edge_2026.md` are
  unaffected.
- **No live data yet.** Live numbers will start once this is deployed.

Tests:

- `test_pipeline.py::test_starters_know_if_the_lineup_they_face_is_posted`
- `test_tracking.py::test_strikeout_trades_wait_for_the_opposing_lineup`, which also
  checks that old rows keep their trades.
- The end-to-end publish test, which checks the data.json field.

## First live days (Oct 6–8, 2026)

This counts strikeout prices from the live log (`odds-log`) that carry
`opp_lineup_confirmed`, starting with the first one at Oct 6, 22:28 UTC. There are 63
pitcher rows covering 10 starter-games in the postseason.

| Game day | Lineup | Snapshots | Best side ≥ 1% edge | Best side z ≥ 1 | Median hours before first pitch |
|---|---|---|---|---|---|
| Oct 6 | confirmed | 8 | 4 | – (no z yet) | 1.2 |
| Oct 7 | projected | 14 | 4 | 0 | 13.5 |
| Oct 7 | confirmed | 19 | 8 | 4 | 1.3 |
| Oct 8 | projected | 9 | 0 | 0 | 7.6 |
| Oct 8 | confirmed | 13 | 0 | 0 | 1.7 |

By starter (one starter can show up in both rows):

- **Projected lineup:** 2 of 9 starter-games showed a ≥ 1% edge while the opposing lineup
  was still projected. Neither reached z ≥ 1.
- **Once the lineup was posted:** one of those two still had its edge. The other lost
  it.
- **Edges that only appeared on the posted lineup:** two more starters.
- **Lean (z ≥ 1):** one starter (Oct 7) reached Lean, and only on the posted lineup.

Under the old rule, both projected-lineup edges would have been paper trades taken 13
hours out. One of those two was gone by the time the lineup posted. That's consistent
with the Strikeouts finding, but two cases prove nothing. Recount after the first
regular-season month, when there are hundreds of starts rather than ten.
