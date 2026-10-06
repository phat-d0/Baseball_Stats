"""Build the static phone app: update data, train models, write `data.json` next to the web files.

    python -m baseball_stats publish --out _site

The web app (web/) is copied as-is; everything it shows comes from data.json, so the
site needs no server and can be hosted on GitHub Pages.
"""

from __future__ import annotations

import json
import os
import logging
import math
import shutil
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import collect, config, features, mlb_api, model, odds, pa_data, sim_models, slate, storage, tracking

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
EASTERN = ZoneInfo("America/New_York")  # MLB's "game day" follows US Eastern time
BATTER_LINES = [0.5, 1.5, 2.5]
PITCHER_LINES = [3.5, 4.5, 5.5, 6.5, 7.5]
LOOKAHEAD_DAYS = 7  # with no games today (off days, off-season) look this far ahead
SLATES = 2  # how many game days to show (today + next)
STATCAST_DAYS_PER_RUN = 45


def _clean(obj):
    """Make numpy/pandas values JSON-safe (NaN -> null, rounded floats)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.bool_ | bool):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return None if math.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def _num(row: pd.Series, col: str):
    v = row.get(col)
    return None if v is None or pd.isna(v) else float(v)


def update_data(today: date, history_start: date | None) -> None:
    """Collect every finished game not yet stored, up to yesterday."""
    start = history_start or date(today.year - 2, 3, 1)
    games = storage.read("games")
    if not games.empty:
        final = games[games["status"] == "Final"]
        if not final.empty:
            # A few days of overlap picks up suspended/late-finishing games.
            last = pd.to_datetime(final["game_date"]).max().date()
            start = max(start, last - timedelta(days=3))
    end = today - timedelta(days=1)
    if start <= end:
        collect.collect_games(start, end)
    backfill_postseasons(today, history_start)


POSTSEASON_TYPES = "F,D,L,W"


def backfill_postseasons(today: date, history_start: date | None) -> None:
    """Collect past seasons' postseason games once.

    The first backfill fetched regular-season games only, and later runs only fetch
    recent days, so the 2024 and 2025 postseasons (about 90 games of training data)
    were never stored. Seasons done are remembered in ``postseason_backfill.json``.
    """
    start = history_start or date(today.year - 2, 3, 1)
    path = config.PROCESSED_DIR / "postseason_backfill.json"
    done = set(json.loads(path.read_text())) if path.exists() else set()
    for year in range(start.year, today.year):
        if year in done:
            continue
        try:
            collect.collect_games(date(year, 9, 25), date(year, 11, 10), game_type=POSTSEASON_TYPES)
        except Exception as exc:  # retried next run
            log.warning("postseason %d backfill failed (%s)", year, exc)
            continue
        done.add(year)
        path.write_text(json.dumps(sorted(done)))
        log.info("backfilled the %d postseason", year)


def statcast_days_missing(today: date, history_start: date | None) -> list[date]:
    """Days with finished games but no plate appearances yet, newest first."""
    start = history_start or date(today.year - 2, 3, 1)
    games = collect.played_games(storage.read("games"))
    if games.empty:
        return []
    have = storage.read("plate_appearances")
    if not have.empty:
        games = games[~games["game_pk"].isin(have["game_pk"])]
    days = sorted({d for d in pd.to_datetime(games["game_date"]).dt.date
                   if start <= d < today}, reverse=True)
    skip_path = config.PROCESSED_DIR / "statcast_empty_days.json"
    skip = set(json.loads(skip_path.read_text())) if skip_path.exists() else set()
    return [d for d in days if d.isoformat() not in skip]


def update_statcast(today: date, history_start: date | None) -> dict | None:
    """Fetch Statcast for days whose games have no plate appearances yet.

    Newest days first (so yesterday is always covered), at most
    ``STATCAST_DAYS_PER_RUN`` days a run since Savant takes one slow request per day.
    Days Savant has nothing for are remembered and skipped. Failures only cost the
    Statcast features. Returns the PA/box score reconciliation summary.
    """
    days = statcast_days_missing(today, history_start)[:STATCAST_DAYS_PER_RUN]
    if days:
        try:
            empty = collect.collect_statcast_days(days, game_type=mlb_api.ALL_GAME_TYPES)
        except Exception as exc:  # Savant is slow/flaky at times; keep publishing without it
            log.warning("statcast update failed (%s); continuing without new Statcast data", exc)
            empty = []
        if empty:
            skip_path = config.PROCESSED_DIR / "statcast_empty_days.json"
            skip = set(json.loads(skip_path.read_text())) if skip_path.exists() else set()
            skip |= {d.isoformat() for d in empty}
            skip_path.write_text(json.dumps(sorted(skip)))
    pa = storage.read("plate_appearances")
    if pa.empty:
        return None
    check = pa_data.reconcile(pa, storage.read("batter_games"), storage.read("pitcher_games"))
    (config.PROCESSED_DIR / "pa_reconcile.json").write_text(json.dumps(check, indent=1))
    log.info("plate appearances: %d rows, %d games; reconciliation %s (%.1f%% games match, RBI gap %.1f%%)",
             len(pa), check["games"], "passed" if check["passed"] else "FAILED",
             100 * (check["match_share"] or 0), 100 * (check["rbi_gap"] or 0))
    return check


MIN_PA_ROWS = 50_000  # plate appearances needed before the PA models are used


PA_MODELS = {"pitcher": ("pa_simple", "pa_seq"), "batter": ("pa_sim",)}
DAILY_EVAL_SIMS = 4000  # simulations per game for the daily 30-day Record check
SHADOW_SIMS = 4000  # simulations per game for a whole-game simulation run only in shadow


def pa_models_for(kind: str, now: datetime, built_hist: dict, hist_rows: pd.DataFrame,
                  current: model.CountModel, players: pd.DataFrame,
                  names: set[str], active: str | None = None) -> dict[str, sim_models._Base]:
    """Load (or, when new plate appearances arrive, train) the plate-appearance models of
    ``kind`` in ``names``. Only the ``active`` one gets the 30-day Record check: replaying
    a month of whole-game simulations takes minutes, and a shadow's record isn't shown."""
    names = {n for n in names if n in PA_MODELS[kind]}
    if not names:
        return {}
    pa = storage.read("plate_appearances")
    if len(pa) < MIN_PA_ROWS:
        log.warning("only %d plate appearances; %s models stay on 'current'", len(pa), kind)
        return {}
    latest = pd.to_datetime(pa["game_date"]).max().date().isoformat()
    bg = storage.read("batter_games") if kind == "batter" else None
    lines = PITCHER_LINES if kind == "pitcher" else BATTER_LINES
    out = {}
    for name in sorted(names):
        try:
            if sim_models.is_stale(name, now, latest):
                t0 = datetime.now(UTC)
                starts = model.training_rows(hist_rows, "pitcher") if kind == "pitcher" else None
                m = sim_models.train(name, pa, built_hist, starts, current, players, batter_games=bg)
                rec = (sim_models.holdout_record(name, pa, built_hist, hist_rows, None, players, lines,
                                                 batter_games=bg, n_sims=DAILY_EVAL_SIMS)
                       if name == active else {})
                sim_models.save(m, now, latest, rec)
                log.info("trained %s on %d plate appearances in %.0fs", name, len(pa),
                         (datetime.now(UTC) - t0).total_seconds())
            m, _ = sim_models.load(name, current, players)
            if name != active and hasattr(m, "n_sims") and kind == "batter":
                m.n_sims = SHADOW_SIMS
            out[name] = m
        except Exception as exc:  # never let a new model take the app down
            log.warning("%s model %s unavailable (%s); using 'current'", kind, name, exc)
    return out


def find_slates(today: date, inputs: dict) -> list[tuple[date, dict]]:
    """Feature rows for the next ``SLATES`` days with games; the first carries history."""
    out = []
    d = today
    while len(out) < SLATES and d <= today + timedelta(days=LOOKAHEAD_DAYS):
        s = slate.build_slate(d, inputs, include_history=not out)
        if not s["pitcher_k"].empty or not s["batter_hrr"].empty:
            out.append((d, s))
        d += timedelta(days=1)
    return out


def book_lines(entries: list[dict] | None, pmf: np.ndarray, kind: str) -> list[dict] | None:
    """DraftKings lines for one player, with the model's and blended chances and the
    expected value per side."""
    if not entries:
        return None
    return [{**e, **tracking.line_values(e, pmf, kind)} for e in entries]


def _pitcher_json(row: pd.Series, mu: float, pmf: np.ndarray, names: dict) -> dict:
    return {
        "id": int(row["player_id"]),
        "name": row.get("player_name") or names.get(int(row["player_id"]), "TBD"),
        "hand": row.get("pitch_hand"),
        "mu": mu,
        "pmf": pmf,
        "stats": {
            "k_pct_szn": _num(row, "k_pct_szn"),
            "k_per_start_l5": _num(row, "k_per_start_l5"),
            "k_per_start_szn": _num(row, "k_per_start_szn"),
            "outs_per_start_l5": _num(row, "outs_per_start_l5"),
            "pitches_per_start_l5": _num(row, "pitches_per_start_l5"),
            "starts_szn": _num(row, "starts_szn"),
            "era_szn": _num(row, "era_szn"),
            "whiff_pct_szn": _num(row, "whiff_pct_szn"),
            "days_rest": _num(row, "days_rest"),
            "lineup_k_pct": _num(row, "lineup_k_pct_vs_hand"),
            "opp_team_k_pct": _num(row, "opp_team_k_pct_szn"),
            "ump_k_factor": _num(row, "ump_k_factor"),
            "park_k_factor": _num(row, "park_so_factor"),
        },
    }


def _batter_json(row: pd.Series, mu: float, pmf: np.ndarray, names: dict) -> dict:
    return {
        "id": int(row["player_id"]),
        "name": row.get("player_name") or names.get(int(row["player_id"]), ""),
        "order": int(row["batting_order"]) if pd.notna(row.get("batting_order")) else None,
        "pos": row.get("position"),
        "bats": row.get("bat_side"),
        "vs": row.get("opp_sp_hand"),
        "confirmed": bool(row.get("lineup_confirmed", True)),
        "mu": mu,
        "pmf": pmf,
        "stats": {
            "hrr_per_g_szn": _num(row, "hrr_per_g_szn"),
            "hrr_per_g_l15": _num(row, "hrr_per_g_l15"),
            "hrr_per_pa_vs_hand": _num(row, "hrr_per_pa_vs_hand_shr"),
            "pa_per_g_l15": _num(row, "pa_per_g_l15"),
            "games_szn": _num(row, "games_szn"),
            "xwoba_szn": _num(row, "xwoba_szn"),
            "opp_sp_k_pct": _num(row, "opp_sp_k_pct_shr"),
            "opp_sp_era": _num(row, "opp_sp_era_szn"),
            "park_r_factor": _num(row, "park_r_factor"),
        },
    }


def slate_json(day: date, s: dict, models: dict[str, model.CountModel], names: dict,
               label: str, props: dict | None = None) -> dict:
    bat, pit, games = s["batter_hrr"], s["pitcher_k"], s["games"]
    by_game: dict[int, dict] = {}
    for g in games.itertuples(index=False):
        by_game[g.game_pk] = {
            "game_pk": g.game_pk,
            "time": g.game_datetime,
            "status": g.detailed_state,
            "venue": g.venue_name,
            "away": {"id": g.away_team_id, "name": g.away_team, "abbr": getattr(g, "away_abbr", None)},
            "home": {"id": g.home_team_id, "name": g.home_team, "abbr": getattr(g, "home_abbr", None)},
            "temp_f": g.temp_f, "wind": " ".join(
                str(x) for x in (f"{g.wind_mph:.0f} mph" if pd.notna(g.wind_mph) else None, g.wind_dir)
                if x) or None,
            "umpire": g.hp_umpire,
            "pitchers": {"away": None, "home": None},
            "lineups": {"away": [], "home": []},
        }
    if not pit.empty:
        mu, dist = models["pitcher"].distribution(pit)
        exp_bf = getattr(models["pitcher"], "last_exp_bf", None)
        for i, ((_, row), m, d) in enumerate(zip(pit.iterrows(), mu, dist)):
            side = "home" if row["is_home"] else "away"
            pj = _pitcher_json(row, m, d, names)
            pj["book"] = book_lines((props or {}).get((row["game_pk"], pj["id"], "pitcher")), d, "pitcher")
            if exp_bf is not None and np.isfinite(exp_bf[i]):
                pj["stats"]["exp_bf"] = float(exp_bf[i])
            by_game[row["game_pk"]]["pitchers"][side] = pj
    if not bat.empty:
        mu, dist = models["batter"].distribution(bat)
        parts = getattr(models["batter"], "last_parts", None) or {}
        for (_, row), m, d in zip(bat.iterrows(), mu, dist):
            side = "home" if row["is_home"] else "away"
            bj = _batter_json(row, m, d, names)
            bj["book"] = book_lines((props or {}).get((row["game_pk"], bj["id"], "batter")), d, "batter")
            p = parts.get((int(row["game_pk"]), bj["id"]))
            if p:  # expected hits, runs and RBIs from the game simulation
                bj["stats"].update({"exp_h": p["H"], "exp_r": p["R"], "exp_rbi": p["RBI"]})
            by_game[row["game_pk"]]["lineups"][side].append(bj)
        for g in by_game.values():
            for side in ("away", "home"):
                g["lineups"][side].sort(key=lambda b: b["order"] or 99)
    ordered = sorted(by_game.values(), key=lambda g: (g["time"] or "", g["game_pk"]))
    return {"date": day.isoformat(), "label": label, "games": ordered}


def publish(out: str | Path, *, history_start: date | None = None,
            now: datetime | None = None, statcast: bool = False) -> dict:
    now = now or datetime.now(UTC)
    today = now.astimezone(EASTERN).date()
    update_data(today, history_start)
    if statcast:
        update_statcast(today, history_start)

    inputs = features.load_inputs()
    slates = find_slates(today, inputs)
    if slates:
        hist = {"batter": slates[0][1]["history_batter_hrr"], "pitcher": slates[0][1]["history_pitcher_k"]}
    else:
        built = features.build_features(inputs)
        hist = {"batter": built["batter_hrr"], "pitcher": built["pitcher_k"]}

    models, record = {}, {}
    for kind, lines in (("batter", BATTER_LINES), ("pitcher", PITCHER_LINES)):
        summary, alpha = model.evaluate(hist[kind], kind, lines=lines)
        record[kind] = summary
        if len(model.training_rows(hist[kind], kind)) < 50:
            raise RuntimeError(f"not enough finished games to train the {kind} model; "
                               "collect more history with --history-start")
        m = model.CountModel(kind).fit(hist[kind])
        m.alpha = alpha
        models[kind] = m
        log.info("%s model: alpha=%.3f %s", kind, alpha,
                 {k: v for k, v in summary.items() if k.startswith(("mae", "n"))})

    # Game-level alternatives to 'current' (model.MODELS), switched like the PA models below.
    game_level = {}  # (kind, name) -> (model, Record summary)
    for kind, lines in (("batter", BATTER_LINES), ("pitcher", PITCHER_LINES)):
        for role in ("MODEL", "SHADOW"):
            name = os.environ.get(f"BASEBALL_{role}_{kind.upper()}", "")
            if name in model.MODELS and name != "current" and (kind, name) not in game_level:
                cls = model.MODELS[name]
                summary, alpha = model.evaluate(hist[kind], kind, lines=lines, cls=cls)
                m = cls(kind).fit(hist[kind])
                m.alpha = alpha
                game_level[kind, name] = (m, summary)

    # build_slate may have fetched bios for new players, so read the table fresh.
    players = storage.read("players")
    names = dict(zip(players.get("player_id", []), players.get("full_name", [])))

    # Model switch and shadow per kind (see sim_models and docs/gate_phase1_pitcher.md).
    if slates:
        h = slates[0][1]
        built_hist = {"batter_hrr": h["history_batter_hrr"], "pitcher_k": h["history_pitcher_k"],
                      "pitcher_rates": h["history_pitcher_rates"]}
    else:
        built_hist = built
    shadows = {}  # logged beside the active model's chance, never shown in the app
    for kind in ("pitcher", "batter"):
        active_n = os.environ.get(f"BASEBALL_MODEL_{kind.upper()}", "current")
        shadow_n = os.environ.get(f"BASEBALL_SHADOW_{kind.upper()}", "")
        loaded = pa_models_for(kind, now, built_hist, hist[kind], models[kind], players,
                               {active_n, shadow_n}, active=active_n)
        current_m = models[kind]
        if (kind, active_n) in game_level:
            models[kind], record[kind] = game_level[kind, active_n]
        elif active_n in loaded:
            models[kind] = loaded[active_n]
            rec = json.loads(sim_models._stamp_path(active_n).read_text()).get("record") or {}
            if rec:
                record[kind] = rec
        if shadow_n and shadow_n != models[kind].name:
            sh = (current_m if shadow_n == "current" else
                  game_level.get((kind, shadow_n), (None,))[0] or loaded.get(shadow_n))
            if sh is not None:
                shadows[kind] = sh
    shadow = shadows.get("pitcher")
    for kind in record:
        if isinstance(record[kind], dict) and record[kind]:
            record[kind]["model_name"] = models[kind].name

    def label(d: date) -> str:
        if d == today:
            return "Today"
        if d == today + timedelta(days=1):
            return "Tomorrow"
        return d.strftime("%a %b %-d")

    # DraftKings props for today's games only, under the credit budget (see odds.py).
    props, odds_status = {}, odds.OddsStatus(error="no games today")
    if slates and slates[0][0] == today:
        s = slates[0][1]
        roster = pd.concat([s["batter_hrr"][["game_pk", "player_id"]],
                            s["pitcher_k"][["game_pk", "player_id"]]], ignore_index=True)
        roster["name"] = roster["player_id"].map(names)
        players_by_game = {pk: list(zip(g["player_id"], g["name"].fillna("")))
                           for pk, g in roster.groupby("game_pk")}
        events, odds_status = odds.fetch_props(s["games"], now=now)
        props = odds.props_by_player(events, players_by_game)
        log.info("odds: %s", odds_status)
        added = tracking.log_snapshots(s, models, props, now, shadow=shadows)
        log.info("price log: %d new snapshot rows", len(added))

    # Grade the full log (including prices logged just now) and build the paper portfolio.
    grades = tracking.update_grades()
    market = tracking.summary(grades)
    strategies = tracking.paper_strategies(grades, names=names)
    if not grades.empty:
        log.info("price log grades: %s; paper trades: %s", grades["status"].value_counts().to_dict(),
                 {st["key"]: {k: st["summary"].get(k) for k in ("n", "open", "won", "lost", "profit")}
                  for st in strategies})

    stored = storage.read("games")
    final = stored[stored["status"] == "Final"] if not stored.empty else stored
    data = {
        "generated_at": now.isoformat(),
        "data_through": (pd.to_datetime(final["game_date"]).max().date().isoformat()
                         if not final.empty else None),
        "lines": {"batter": BATTER_LINES, "pitcher": PITCHER_LINES},
        "model": {k: {"name": m.name, "alpha": m.alpha,
                      "shadow": shadows[k].name if k in shadows else None,
                      "trained_rows": int(len(model.training_rows(hist[k], k)))}
                  for k, m in models.items()},
        "slates": [slate_json(d, s, {k: m.for_slate(s, players) for k, m in models.items()},
                              names, label(d), props if d == today else None)
                   for d, s in slates],
        "odds_source": asdict(odds_status),
        "record": {**record, "market": market},
        "paper": strategies[0],  # the main strategy
        "paper_strategies": strategies,
    }
    data = _clean(data)

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB_DIR, out)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    log.info("wrote %s (%d slates)", out / "data.json", len(data["slates"]))
    return data
