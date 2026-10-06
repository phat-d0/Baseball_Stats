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
