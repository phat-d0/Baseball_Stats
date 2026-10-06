# CLAUDE.md

Guidance for Claude sessions working in this repo. Read `docs/WORK_PLAN.md` first: it
says what has been built, what the latest results are, and what to do next. If you
have a team role, read your brief in `docs/team/` next (see Team below).

## What this is

An MLB player-prop model with an iPhone web app (PWA):

- **Targets:** batter Hits+Runs+RBIs (H+R+RBI) and starting-pitcher strikeouts per game.
- **Data:** MLB Stats API box scores, Baseball Savant Statcast, DraftKings props via The
  Odds API.
- **App:** live at https://phat-d0.github.io/Baseball_Stats/ (add it via Safari → Share →
  Add to Home Screen). GitHub Actions rebuilds and publishes it about every 20 minutes
  on game days.

The owner is a bettor who wants honest numbers. They prefer recommendations to option
lists, and they want results reported plainly, including bad ones.

## Team (several Claude sessions at once)

Work is split across sessions, each with a brief in `docs/team/`:

| Role | Brief | Branch |
|---|---|---|
| Lead / Reviewer | `docs/team/lead.md` | the default branch |
| Edge | `docs/team/edge.md` | `team/edge` |
| Strikeouts | `docs/team/strikeouts.md` | `team/strikeouts` |
| Hitters | `docs/team/hitters.md` | `team/hitters` |
| UI Design | `docs/team/ui.md` | `team/ui` |

If you were started with a role, read its brief before anything else. Rules for every
role except Lead:

1. **Never push to `claude/stoic-davinci-v4p6k9`** (it deploys the app). Work on your own
   branch and open a pull request into it when a piece of work is finished and tested.
   One topic per PR; keep PRs small enough to review.
2. **Stay in your files.** The brief lists what you own. If you need a change in shared
   code (`features.py`, `publish.py`, `tracking.py`, `model.py`, `web/app.js`), keep it
   minimal, say so in the PR, and expect the Lead to sequence merges.
3. **Before opening a PR:**
   - `python -m pytest -q tests` passes.
   - Model changes pass the walk-forward gate (see Conventions).
   - App changes have phone-size screenshots in light and dark mode.
4. **Credits:** only Edge may call The Odds API (through the Fetch historical prices
   workflow), within a 15,000-credit budget until the Nov 5, 2026 reset, never letting
   the shared balance fall below 25,000. Everyone else reuses the `odds-history` branch.
5. **Results go in `docs/`** (your own write-up file, with numbers, including bad
   results). Don't edit `docs/WORK_PLAN.md` or this file; propose changes in your PR
   description and the Lead updates them.
6. **Bring your branch up to date by merging the default branch in, never by rebasing**,
   and before opening a PR. Keep 4-core CPU limits in mind (one heavy job at a time).

## Branches and publishing

- **`claude/stoic-davinci-v4p6k9` is the default branch, and pushing to it deploys the
  app.** Any push that touches `baseball_stats/`, `web/`, `requirements.txt` or
  `publish.yml` triggers `.github/workflows/publish.yml` (build → Pages deploy → `next`
  job that chains the 20-minute runs).
- Commit only tested work. Park unfinished changes in `git stash` instead of pushing them.
- Bot-managed branches. Never edit them by hand:
  - `odds-log`: `prop_snapshots.parquet`, the append-only DraftKings price log. It can't
    be re-downloaded.
  - `data-snapshot`: processed parquet tables, refreshed daily at 10 UTC.
  - `odds-history`: historical DraftKings props, one JSON file per day, for backtests
    (`2025-06-01..2025-09-28`, `2026-08-01..2026-09-27`).
- Repository variables or secrets used by the workflow:
  - `ODDS_API_KEY` (secret).
  - `ODDS_API_RESET_DAY` (5).
  - `ODDS_API_MONTHLY_CAP`.
  - `BASEBALL_MODEL_PITCHER` / `BASEBALL_SHADOW_PITCHER` (`pa_simple` / `pa_seq`).
  - `BASEBALL_MODEL_BATTER` / `BASEBALL_SHADOW_BATTER` (`current` / `pa_sim`).

## Environment gotchas (cloud sessions)

- **No Odds API key locally**, so any download from The Odds API must run in GitHub
  Actions:
  - Live props: the publish workflow.
  - Historical props: the "Fetch historical prices" workflow (`fetch-history.yml`),
    started with the GitHub MCP `actions_run_trigger` (`run_workflow`, `ref` = the
    default branch, inputs `start`, `end`, `max_credits`, `reserve`).
- **The key is shared with the owner's soccer app** (100k credits/month, resets on the
  5th). A historical game costs ~20 credits; live pricing costs ~2 credits per game per
  pull. Check the balance in the workflow logs and keep a reserve. Ask before large spends.
- **github.io is blocked by the egress proxy.** To see live `data.json`, rebuild it
  locally: `python -m baseball_stats publish --out <scratch>/live`. Or read the Actions
  logs with the GitHub MCP tools (`list_workflow_runs`, `list_workflow_jobs`,
  `get_job_logs`).
- **Local data:** `data/processed/*.parquet` (gitignored). If it's missing, restore it
  from the `data-snapshot` branch, and the price log from `odds-log`:
  `git show origin/odds-log:prop_snapshots.parquet > data/processed/prop_snapshots.parquet`.
- **CPU:** 4 cores. Never run two heavy jobs (evaluate/backtest/full tests) at once; they
  slow down ~50×. For parallel folds use `OMP_NUM_THREADS=1` per process.
- **`pkill -f` / `pgrep -f`** can match your own shell. Kill by a specific pattern in a
  separate command.

## Commands

```bash
python -m pytest -q tests                  # ~2 min, 70 tests; run before every push
python -m baseball_stats publish --out _site [--statcast]   # what CI runs
python scripts/evaluate.py --kind pitcher|batter [--folds 2026-09] [--variants current pa_simple] --out x.json
python scripts/backtest.py --prices <dir of odds-history json> --out x.json     # model vs DK, edge bands, sizing
python scripts/fit_blend.py --fit <2025 dir> --test <2026 dir> --weights baseball_stats/blend.json --report docs/blend_2026.json
node --check web/app.js                    # after any app edit
```

- Extract historical prices with
  `git archive origin/odds-history | tar -x -C <dir>`, then split them into year folders.
- Check the app at phone size with Playwright (`playwright-core`, Chromium at
  `/opt/pw-browsers/chromium`; set `NODE_PATH=$(npm root -g)`). Serve a copy of `web/`
  plus a `data.json`, then screenshot at 390×844 in light and dark mode. Check for page
  errors and horizontal overflow.

## Code map

| Path | What |
|---|---|
| `baseball_stats/mlb_api.py`, `collect.py`, `parse.py`, `statcast.py`, `storage.py` | Download and store data (all game types `R,F,D,L,W`; suspended games deduped) |
| `features.py` | Leakage-free rolling features, shrinkage, log5 matchups |
| `model.py` | `CountModel` (HistGradientBoosting Poisson + negative binomial), `evaluate`, `summarize_holdout` |
| `pa_data.py`, `pa_model.py`, `simulate.py`, `game_sim.py`, `sim_models.py` | Plate-appearance models: `pa_simple` (live strikeouts), `pa_seq` (shadow), `GameSimH`/`pa_sim` (hitter shadow); saved under `data/processed/models/` and retrained when a newer day of plate appearances arrives |
| `odds.py` | Odds API client, daily credit budget, `fetch_priority` (24 h window, 4 h / 2 h / 30 min refresh, closing pull) |
| `blend.py` + `blend.json` | Model/DraftKings logistic blend that all edges use |
| `tracking.py` | Price log (`prop_snapshots`), grading (`prop_grades`), CLV, `picks`, Record summary, paper strategies (`PAPER_STRATEGIES`) |
| `publish.py` | Orchestrates a run and writes `data.json` (slates, record, paper, paper_strategies) |
| `web/` | The PWA (`index.html`, `app.js`, `style.css`, `sw.js`, manifest, icons) |
| `scripts/` | `evaluate.py`, `backtest.py`, `fit_blend.py`, `fetch_history.py`, `make_icons.py` |
| `tests/` | `fake_mlb.py` simulates an MLB season (box scores + pitches); `FakeDK` in `test_publish.py` |
| `docs/` | Gate and backtest write-ups with their JSON; `WORK_PLAN.md` |

## Conventions

- **Write like the surrounding code:** short docstrings explaining *why*, sparse comments,
  plain names. Commit messages: an imperative subject, then a body explaining what changed
  and why, with numbers.
- **No leakage:** every feature and evaluation is walk-forward (train only on earlier
  dates). Any new model must beat `current` in `scripts/evaluate.py` before going live:
  - mean log loss lower,
  - better in ≥4 of 5 folds,
  - RPS lower,
  - calibration within 0.5 pp at every line.

  Failing models run in shadow (logged as `p_shadow`), never live.
- **The price log is append-only.** Grades are rebuilt from it each run, so grading rules
  can change. Add columns; never rewrite old rows.
- **Edges:** use `tracking.line_values(entry, pmf, kind)`. It returns `p_model`, `p_book`,
  `p_blend`, `ev_over`, `ev_under`, and edges come from `p_blend`. `bestBet` in `app.js`
  mirrors `tracking.picks`.
- **App:**
  - Bump `CACHE` in `web/sw.js` whenever `app.js` or `style.css` change.
  - Colors are CSS tokens with light and dark variants.
  - Keep phone width (390 px) free of horizontal overflow.
- **Pandas 3:** string dtype columns. Use `is_numeric_dtype` checks, and
  `astype(object)` before filling all-NaN columns.
- **Tests:** never let them write to the real `data/` dir (monkeypatch to `tmp_path`;
  see the existing fixtures).
