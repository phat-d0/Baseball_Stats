# Baseball_Stats

Data pipeline for building models that predict:

- **Batters: Hits + Runs + RBIs (H+R+RBI)** per game
- **Starting pitchers: strikeouts** per game

It downloads game logs from the public MLB Stats API (and, optionally, pitch-level
Statcast data from Baseball Savant), stores them as Parquet, and builds
model-ready feature tables. Every feature on a row is computed **only from games
on earlier dates**, so the training data matches what you would know before
first pitch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Usage

```bash
# 1. Download historical games (box scores, lineups, umpires, weather).
#    Raw responses are cached under data/raw/, so re-runs are cheap.
python -m baseball_stats collect --start 2023-03-30 --end 2025-09-28

#    Add --statcast for pitch-level metrics (xwOBA, barrels, whiff/CSW%, velocity).
#    Slower: one Savant request per day.
python -m baseball_stats collect --start 2023-03-30 --end 2025-09-28 --statcast

# 2. Build training tables -> data/features/{batter_hrr,pitcher_k}.parquet
python -m baseball_stats features

# 3. Build feature rows for today's games, to feed a trained model
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
