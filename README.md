# Baseball_Stats

Data pipeline for building models that predict:

- **Batters: Hits + Runs + RBIs (H+R+RBI)** per game
- **Starting pitchers: strikeouts** per game

It downloads game logs from the public MLB Stats API (and, optionally, pitch-level
Statcast data from Baseball Savant), stores them as Parquet, and builds
model-ready feature tables. Every feature on a row is computed **only from games
on earlier dates**, so the training data matches what you would know before
first pitch.

## iPhone app

A phone-first web app you add to your home screen. It opens full-screen with its own
icon, works offline, and follows your phone's dark mode. Tabs:

- **Games**: today's (and the next day's) games with both probable starters' projected
  strikeouts. Tap a game for both lineups with each hitter's H+R+RBI chances.
- **Hitters**: every batter in today's lineups ranked by chance of going over 0.5 / 1.5 / 2.5
  H+R+RBI. Tap one for the full distribution, fair odds for each line and the stats behind it.
- **Pitchers**: starters ranked by projected strikeouts, with the chance of going over
  3.5–7.5. Tap one for the distribution, fair odds and why (recent form, opposing lineup's
  strikeout rate, umpire, park).
- **Record**: how the model did over the most recent 30 days it wasn't trained on: average
  miss vs the player's season average, the daily top-10 picks' hit rate, and a calibration
  chart showing whether "70%" really happens about 70% of the time.

**DraftKings props:** player prop prices (pitcher strikeouts, batter H+R+RBI) come from
DraftKings via [The Odds API](https://the-odds-api.com) (`baseball_stats/odds.py`). The Games
tab then opens with **Value picks**: the lines where DraftKings pays more than the model's
chance is worth, filtered by a minimum edge you choose. Each player's sheet shows the
DraftKings line next to the model. Add a free API key as the repo secret `ODDS_API_KEY`
(Settings → Secrets and variables → Actions). Without a key the app just shows fair odds.

Props are priced per game (about 2 credits a game), so refreshes run on a strict budget:
- Listing the day's games is free and reports the real credit balance.
- Each day may spend `(credits left − 20 reserve) ÷ days until the monthly reset`. That's
  about 15 credits, or ~7 games, on the free 500-credit plan. A paid plan automatically
  buys more games and more refreshes.
- Games are priced soonest first pitch first, only within 8 hours of first pitch, and
  each game at most once every 2 hours. A game DraftKings hasn't posted props for yet
  costs nothing.
- The reset day is set to the 5th in the workflow and learned automatically when the
  balance goes back up.
- The key is shared with the soccer app, so this app spends at most **200 credits a month**
  (about 6 a day, ~3 games priced once). The soccer app budgets from the real balance and
  adapts to what's left. Change the cap with the repository variable `ODDS_API_MONTHLY_CAP`
  (Settings → Secrets and variables → Actions → Variables); `0` removes it, e.g. after
  upgrading the plan or giving this app its own key.

Refreshing the app on your phone never uses credits.

"Fair" odds are the model's probability written as American odds with no bookmaker margin:
a bet is only worth a look when your sportsbook pays more than that.

Until a team posts its lineup (usually 1–4 hours before first pitch), its last starting
nine stands in, marked **proj**.

A GitHub Actions job (`.github/workflows/publish.yml`) runs every hour, and every 20 minutes
from 11am to 10pm US Eastern. Each run fetches newly finished games, retrains both models,
scores the next slate and publishes to GitHub Pages.

**One-time setup**
1. On GitHub: repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. **Actions** tab → *Publish app* → **Run workflow** (or wait for the next scheduled run).
   The first run downloads two seasons of box scores and takes roughly an hour. Later runs
   take a few minutes. Statcast history fills in 45 days per run over the next day or so.
3. On your iPhone, open `https://phat-d0.github.io/Baseball_Stats/` in **Safari**, tap
   **Share → Add to Home Screen**.

Build it locally: `python -m baseball_stats publish --out _site && python -m http.server -d _site`.
The app code lives in `web/`; `baseball_stats/publish.py` writes the `data.json` it reads.
Icons are drawn by `scripts/make_icons.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Usage

```bash
# 1. Download historical games (regular season + postseason by default) (box scores, lineups, umpires, weather).
#    Raw responses are cached under data/raw/, so re-runs are cheap.
python -m baseball_stats collect --start 2023-03-30 --end 2025-09-28

#    Add --statcast for pitch-level metrics (xwOBA, barrels, whiff/CSW%, velocity).
#    Slower: one Savant request per day.
python -m baseball_stats collect --start 2023-03-30 --end 2025-09-28 --statcast

# 2. Build training tables -> data/features/{batter_hrr,pitcher_k}.parquet
python -m baseball_stats features

# 3. Or do everything the phone app needs in one go (collect, train, score, write site)
python -m baseball_stats publish --out _site --statcast

# 4. Build feature rows for today's games, to feed a trained model
#    -> data/features/slates/{batter_hrr,pitcher_k}_<date>.parquet
python -m baseball_stats slate --date 2026-04-15
```

Set `BASEBALL_DATA_DIR` to store data somewhere other than `./data`.

Batter rows in a slate are only created once a team's lineup is posted
(usually 1–4 hours before first pitch). Re-run `slate` closer to game time to
pick them up.

## Data layout

| Table (`data/processed/`) | Grain | Contents |
|---|---|---|
| `games` | game | date, venue, teams, score, probable starters, HP umpire, temp/wind |
| `batter_games` | batter × game | PA, AB, H, R, RBI, **hrr**, TB, HR, BB, SO, batting order, opposing starter |
| `pitcher_games` | pitcher × game | outs, BF, pitches, **K**, BB, H, HR, ER, starter flag |
| `lineups` | batter × game | posted lineups from the schedule feed |
| `players` | player | bat side, throwing hand, birth date |
| `statcast_batter` / `statcast_pitcher` | player × game | pitch counts, swings, whiffs, chases, BIP, hard-hit, barrels, xwOBA, FB velocity |

## Features

**`batter_hrr`** (target `target_hrr`; also `target_h`, `target_r`, `target_rbi`)

- Own form over the last 7/15/30 games, season to date and career: H+R+RBI per game and per PA,
  PA per game, H/R/RBI per PA, ISO, BB%, K%
- Regressed (shrunk) H+R+RBI/PA and K%, plus splits vs the opposing starter's hand
- Batting order slot, home/away, platoon advantage, days since the last game
- Opposing starter: K% (regressed, last 5 starts, season), BB%, H/BF, HR/BF, ERA,
  innings per start (more PAs against the bullpen), xwOBA allowed
- Opposing staff: runs allowed per game, bullpen on-base and K rates
- Own team runs per game (drives R and RBI chances)
- Park factors for runs, hits and HR; temperature, wind in/out, roof, day/night
- Statcast (with `--statcast`): xwOBA, barrel%, hard-hit%, whiff%, chase%

**`pitcher_k`** (target `target_k`; also `target_outs`, `target_bf`, `target_pitches`)

- Own form over the last 3/5/10 starts, season and career: K per start, K%, BB%, BF, outs and
  pitches per start, ERA, last start's pitch count; days of rest, age, throwing hand
- Opposing lineup: average regressed K% of the actual starting nine, overall and vs this
  pitcher's hand; number of lefties
- Opposing team K% over the last 15/30 games, season, and vs this hand
- Own team's average starter length (the manager's hook) and bullpen form
- Home-plate umpire K factor, park K factor, weather
- Statcast (with `--statcast`): whiff%, CSW%, chase%, zone%, xwOBA allowed, FB velocity and its trend

Early-season and debut rows have NaN for windows with no history. Tree models such as
LightGBM/XGBoost handle that natively; for linear models, impute or use the `_shr` columns.

## Models

`baseball_stats/model.py` fits a gradient-boosted Poisson regression (scikit-learn) for each
target to predict the average. Real outcomes are more spread out than Poisson, so each
prediction becomes a negative binomial distribution whose extra spread is fitted on recent
held-out games. That distribution gives the chance of going over any line.

## Modelling notes

- Split train/validation **by date** (e.g. train 2023–2024, validate 2025), never randomly.
- Both targets are counts: try Poisson/Tweedie objectives, then turn the predicted mean into
  over/under probabilities for a line (e.g. P(K ≥ 6)).
- Train the batter model on `is_starter == True` rows, which is what a slate contains.

## Known limitations

- Batter "vs hand" splits use the opposing **starter's** hand for the whole game, not per PA.
- Doubleheader game 2 is featurised as of the start of the day (game 1 isn't counted).
- Park, umpire and league baselines only use seasons you have collected; collect at least two
  seasons so they stabilise.

## Tests

```bash
pytest
```

The tests run the full pipeline offline against synthetic API responses (`tests/fake_mlb.py`),
including checks that no feature uses data from the same or a later day.
