"""Leakage-free feature engineering for the two prediction targets.

* Batters:  ``hrr`` = hits + runs + RBIs in a game.
* Pitchers: ``k``   = strikeouts by the starting pitcher.

Every feature on a row is computed only from games played on *earlier dates*
than that row's game (doubleheader game 2 does not see game 1), so features
match what you would know before first pitch on game day.

Upcoming games ("pending" rows with no stats yet) go through exactly the same
code path: they are appended to the history, featurised, and returned with
empty targets. See :mod:`baseball_stats.slate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("game_date", "game_datetime", "game_pk") if c in df.columns]
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def prior_sums(df: pd.DataFrame, by: list[str], cols: list[str], window: int | None,
               *, suffix: str) -> pd.DataFrame:
    """Sum of ``cols`` over the previous ``window`` rows of each ``by`` group.

    ``window=None`` means everything before. ``df`` must already be sorted
    chronologically. The result is then pinned to the state at the start of
    each game day, so same-day games never leak into each other.
    """
    vals = df[cols].fillna(0).astype(float)
    keys = [df[b] for b in by]
    grp = vals.groupby(keys, sort=False)
    if window is None:
        incl = grp.cumsum()
    else:
        incl = (grp.rolling(window + 1, min_periods=1).sum()
                .droplevel(list(range(len(by)))).reindex(df.index))
    prior = incl - vals
    prior = _as_of_day_start(df, prior, by)
    prior.columns = [f"{c}_{suffix}" for c in cols]
    return prior


def _as_of_day_start(df: pd.DataFrame, feat: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Give every row the value of the first row of its (group, game_date)."""
    day_group = df.groupby(by + ["game_date"], sort=False).ngroup()
    pos = pd.Series(np.arange(len(df)), index=df.index)
    first = pos.groupby(day_group.values).transform("min").to_numpy()
    out = feat.iloc[first].copy()
    out.index = df.index
    return out


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).astype(float)


def shrink(num: pd.Series, den: pd.Series, prior: pd.Series | float, k: float) -> pd.Series:
    """Empirical-Bayes style rate: regress toward ``prior`` with ``k`` pseudo-trials."""
    return (num + prior * k) / (den + k)


def log5(batter: pd.Series, pitcher: pd.Series, league: pd.Series | float) -> pd.Series:
    """Rate for this batter vs this pitcher (odds-ratio / "log5" method).

    Combines the batter's rate, the pitcher's rate allowed and the league rate:
    a .300 strikeout batter vs a .300 strikeout pitcher in a .220 league -> ~.39.
    """
    def odds(x):
        x = np.clip(x, 1e-3, 1 - 1e-3)
        return x / (1 - x)
    o = odds(batter) * odds(pitcher) / odds(league)
    return o / (1 + o)


def pa_vs_starter(exp_bf: pd.Series, slot: pd.Series) -> pd.Series:
    """Plate appearances lineup slot ``slot`` gets against a starter who faces ``exp_bf``."""
    n = exp_bf.round()
    return (((n - slot) // 9) + 1).clip(lower=0).where(slot.notna() & n.notna())


def league_prior_rate(df: pd.DataFrame, num: str, den: str, default: float) -> pd.Series:
    """League-wide num/den using only dates before each row (leak-free)."""
    daily = df.groupby("game_date")[[num, den]].sum().sort_index()
    cum = daily.cumsum().shift(1)
    rate = (cum[num] / cum[den].where(cum[den] > 0)).fillna(default)
    return df["game_date"].map(rate).astype(float)


def _windowed(df: pd.DataFrame, by: list[str], cols: list[str],
              windows: dict[str, int | None], season: bool = True) -> pd.DataFrame:
    parts = [prior_sums(df, by, cols, w, suffix=name) for name, w in windows.items()]
    if season:
        parts.append(prior_sums(df, by + ["season"], cols, None, suffix="szn"))
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_inputs(base=None) -> dict[str, pd.DataFrame]:
    names = ["games", "batter_games", "pitcher_games", "players",
             "statcast_batter", "statcast_pitcher", "lineups"]
    return {n: storage.read(n, base) for n in names}


def _prep_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    if "game_datetime" in g:
        g["game_datetime"] = pd.to_datetime(g["game_datetime"], utc=True, errors="coerce")
    return _sort(g)


def _attach_game_cols(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_pk", "game_date", "game_datetime", "season", "venue_id"]
    df = df.drop(columns=[c for c in cols if c != "game_pk" and c in df.columns])
    return _sort(df.merge(games[cols], on="game_pk", how="inner"))


# --------------------------------------------------------------------------- #
# Game-level context: park, umpire, team offense/bullpen
# --------------------------------------------------------------------------- #


def game_context(games: pd.DataFrame, bat: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Per-game park and umpire factors built from prior games only."""
    tot = bat.groupby("game_pk")[["pa", "h", "r", "so", "hr"]].sum()
    g = games[["game_pk", "game_date", "game_datetime", "season", "venue_id",
               "hp_umpire_id"]].merge(tot, on="game_pk", how="left").fillna(
        {"pa": 0, "h": 0, "r": 0, "so": 0, "hr": 0})
    g = _sort(g)

    out = g[["game_pk"]].copy()
    for stat, default in (("r", 0.12), ("h", 0.22), ("so", 0.22), ("hr", 0.03)):
        lg = league_prior_rate(g, stat, "pa", default)
        v = prior_sums(g, ["venue_id"], [stat, "pa"], None, suffix="v")
        out[f"park_{stat}_factor"] = shrink(v[f"{stat}_v"], v["pa_v"], lg, 3000) / lg

    lg_k = league_prior_rate(g, "so", "pa", 0.22)
    u = prior_sums(g.assign(hp_umpire_id=g["hp_umpire_id"].fillna(-1)),
                   ["hp_umpire_id"], ["so", "pa"], None, suffix="u")
    ump = shrink(u["so_u"], u["pa_u"], lg_k, 1500) / lg_k
    out["ump_k_factor"] = ump.where(g["hp_umpire_id"].notna())
    out["ump_games"] = (u["pa_u"] / 76).round().where(g["hp_umpire_id"].notna())
    out["league_k_pct"] = lg_k
    out["league_r_per_pa"] = league_prior_rate(g, "r", "pa", 0.12)
    return out


def team_context(games: pd.DataFrame, bat: pd.DataFrame, pit: pd.DataFrame,
                 players: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team) offense and pitching-staff form from prior games."""
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        s = games[["game_pk", "game_date", "game_datetime", "season"]].copy()
        s["team_id"] = games[f"{side}_team_id"]
        s["opp_team_id"] = games[f"{opp}_team_id"]
        sides.append(s)
    tg = pd.concat(sides, ignore_index=True)

    off = bat.groupby(["game_pk", "team_id"])[["pa", "h", "r", "so", "bb", "hrr"]].sum()
    off.columns = [f"off_{c}" for c in off.columns]
    sp = pit[pit["is_starter"]].groupby(["game_pk", "team_id"]).agg(
        sp_outs=("outs", "sum"), sp_pitches=("pitches", "sum"), sp_id=("player_id", "first"))
    rp = pit[~pit["is_starter"]].groupby(["game_pk", "team_id"])[
        ["bf", "k", "h", "bb", "er", "outs"]].sum()
    rp.columns = [f"rp_{c}" for c in rp.columns]
    runs_allowed = pit.groupby(["game_pk", "team_id"])[["r"]].sum().rename(columns={"r": "ra"})

    tg = (tg.merge(off, on=["game_pk", "team_id"], how="left")
            .merge(sp, on=["game_pk", "team_id"], how="left")
            .merge(rp, on=["game_pk", "team_id"], how="left")
            .merge(runs_allowed, on=["game_pk", "team_id"], how="left"))
    tg["played"] = tg["off_pa"].fillna(0).gt(0).astype(int)

    # Hand of the starter this team's hitters faced.
    hand = players.set_index("player_id")["pitch_hand"]
    opp_sp = sp.reset_index()[["game_pk", "team_id", "sp_id"]].rename(
        columns={"team_id": "opp_team_id", "sp_id": "opp_sp_id"})
    tg = tg.merge(opp_sp, on=["game_pk", "opp_team_id"], how="left")
    tg["opp_sp_hand"] = tg["opp_sp_id"].map(hand).fillna("R")
    tg = _sort(tg)

    off_cols = ["played", "off_pa", "off_r", "off_so", "off_hrr"]
    pit_cols = ["played", "ra", "sp_outs", "sp_pitches", "rp_bf", "rp_k", "rp_h", "rp_bb", "rp_er"]
    w = _windowed(tg, ["team_id"], sorted(set(off_cols + pit_cols)), {"l15": 15, "l30": 30})
    vh = prior_sums(tg, ["team_id", "opp_sp_hand", "season"], ["off_so", "off_pa"], None,
                    suffix="vh")

    out = tg[["game_pk", "team_id", "opp_sp_hand"]].copy()
    for s in ("l15", "l30", "szn"):
        out[f"team_r_per_g_{s}"] = ratio(w[f"off_r_{s}"], w[f"played_{s}"])
        out[f"team_k_pct_{s}"] = ratio(w[f"off_so_{s}"], w[f"off_pa_{s}"])
        out[f"team_ra_per_g_{s}"] = ratio(w[f"ra_{s}"], w[f"played_{s}"])
        out[f"team_sp_outs_{s}"] = ratio(w[f"sp_outs_{s}"], w[f"played_{s}"])
        out[f"team_rp_k_pct_{s}"] = ratio(w[f"rp_k_{s}"], w[f"rp_bf_{s}"])
        out[f"team_rp_onbase_pct_{s}"] = ratio(w[f"rp_h_{s}"] + w[f"rp_bb_{s}"], w[f"rp_bf_{s}"])
    # K% vs the hand of *today's* opposing starter, this season.
    out["team_k_pct_vs_hand_szn"] = ratio(vh["off_so_vh"], vh["off_pa_vh"])
    return out


# --------------------------------------------------------------------------- #
# Batters
# --------------------------------------------------------------------------- #

BAT_COLS = ["g", "pa", "ab", "h", "r", "rbi", "hrr", "tb", "hr", "bb", "so"]
BAT_WINDOWS = {"l7": 7, "l15": 15, "l30": 30, "car": None}
SC_COLS = ["pitch", "swing", "whiff", "out_zone", "chase", "bip", "hard_hit", "barrel",
           "xwoba_num", "xwoba_den"]


def batter_base(bat: pd.DataFrame, players: pd.DataFrame, sc_bat: pd.DataFrame,
                league_k: pd.Series) -> pd.DataFrame:
    """Batter's own form. ``bat`` must be sorted and have game columns attached."""
    b = bat.copy()
    b["g"] = b["pa"].notna().astype(int) * b["_played"]
    hand = players.set_index("player_id")
    b["bat_side"] = b["player_id"].map(hand["bat_side"])
    b["opp_sp_hand"] = b["opp_starter_id"].map(hand["pitch_hand"])
    b["platoon_adv"] = np.where(
        b["bat_side"] == "S", 1,
        (b["bat_side"].notna() & b["opp_sp_hand"].notna()
         & (b["bat_side"] != b["opp_sp_hand"])).astype(int))

    w = _windowed(b, ["player_id"], BAT_COLS, BAT_WINDOWS)
    feats = pd.DataFrame(index=b.index)
    for s in [*BAT_WINDOWS, "szn"]:
        feats[f"games_{s}"] = w[f"g_{s}"]
        feats[f"hrr_per_g_{s}"] = ratio(w[f"hrr_{s}"], w[f"g_{s}"])
        feats[f"pa_per_g_{s}"] = ratio(w[f"pa_{s}"], w[f"g_{s}"])
        feats[f"hrr_per_pa_{s}"] = ratio(w[f"hrr_{s}"], w[f"pa_{s}"])
        feats[f"h_per_pa_{s}"] = ratio(w[f"h_{s}"], w[f"pa_{s}"])
        feats[f"r_per_pa_{s}"] = ratio(w[f"r_{s}"], w[f"pa_{s}"])
        feats[f"rbi_per_pa_{s}"] = ratio(w[f"rbi_{s}"], w[f"pa_{s}"])
        feats[f"iso_{s}"] = ratio(w[f"tb_{s}"] - w[f"h_{s}"], w[f"ab_{s}"])
        feats[f"bb_pct_{s}"] = ratio(w[f"bb_{s}"], w[f"pa_{s}"])
        feats[f"k_pct_{s}"] = ratio(w[f"so_{s}"], w[f"pa_{s}"])

    # Regressed rates (stable for small samples) — also used for lineup K%.
    lg_hrr = league_prior_rate(b.assign(_pa=b["pa"].fillna(0), _hrr=b["hrr"].fillna(0)),
                               "_hrr", "_pa", 0.33)
    feats["hrr_per_pa_shr"] = shrink(w["hrr_car"], w["pa_car"], lg_hrr, 200)
    feats["k_pct_shr"] = shrink(w["so_car"], w["pa_car"], league_k, 150)
    played = b.assign(_pa=b["pa"].fillna(0), _h=b["h"].fillna(0), _bb=b["bb"].fillna(0),
                      _hr=b["hr"].fillna(0))
    feats["lg_k_pa"] = league_k
    feats["lg_h_pa"] = league_prior_rate(played, "_h", "_pa", 0.22)
    feats["lg_bb_pa"] = league_prior_rate(played, "_bb", "_pa", 0.08)
    feats["lg_hr_pa"] = league_prior_rate(played, "_hr", "_pa", 0.03)
    feats["h_per_pa_shr"] = shrink(w["h_car"], w["pa_car"], feats["lg_h_pa"], 250)
    feats["bb_pct_shr"] = shrink(w["bb_car"], w["pa_car"], feats["lg_bb_pa"], 150)
    feats["hr_per_pa_shr"] = shrink(w["hr_car"], w["pa_car"], feats["lg_hr_pa"], 300)
    feats["onbase_shr"] = shrink(w["h_car"] + w["bb_car"], w["pa_car"],
                                 feats["lg_h_pa"] + feats["lg_bb_pa"], 200)

    # Typical plate appearances for this lineup slot (league-wide, prior days only).
    slot = b.assign(batting_order=b["batting_order"].fillna(0))
    ps = prior_sums(slot, ["batting_order"], ["pa", "g"], None, suffix="slot")
    feats["slot_pa_exp"] = ratio(ps["pa_slot"], ps["g_slot"]).where(b["batting_order"].notna())

    # Splits vs the hand of today's opposing starter (career to date).
    side = b.assign(opp_sp_hand=b["opp_sp_hand"].fillna("R"))
    vh = prior_sums(side, ["player_id", "opp_sp_hand"], ["pa", "hrr", "so", "h"], None, suffix="vh")
    feats["pa_vs_hand"] = vh["pa_vh"]
    feats["hrr_per_pa_vs_hand_shr"] = shrink(vh["hrr_vh"], vh["pa_vh"], feats["hrr_per_pa_shr"], 150)
    feats["k_pct_vs_hand_shr"] = shrink(vh["so_vh"], vh["pa_vh"], feats["k_pct_shr"], 100)
    feats["h_per_pa_vs_hand_shr"] = shrink(vh["h_vh"], vh["pa_vh"], feats["h_per_pa_shr"], 150)

    # Rest / recent usage.
    prev = b.groupby("player_id")["game_date"].shift(1)
    feats["days_since_last_game"] = (b["game_date"] - prev).dt.days
    feats["days_since_last_game"] = _as_of_day_start(
        b, feats[["days_since_last_game"]], ["player_id"])["days_since_last_game"]

    if not sc_bat.empty:
        sc = b[["game_pk", "player_id"]].merge(
            sc_bat.rename(columns={"batter": "player_id"}), on=["game_pk", "player_id"], how="left")
        sc.index = b.index
        sc = pd.concat([b[["player_id", "season", "game_date"]], sc[SC_COLS]], axis=1)
        sw = _windowed(sc, ["player_id"], SC_COLS, {"l15": 15, "car": None})
        for s in ("l15", "szn", "car"):
            feats[f"xwoba_{s}"] = ratio(sw[f"xwoba_num_{s}"], sw[f"xwoba_den_{s}"])
            feats[f"barrel_pct_{s}"] = ratio(sw[f"barrel_{s}"], sw[f"bip_{s}"])
            feats[f"hard_hit_pct_{s}"] = ratio(sw[f"hard_hit_{s}"], sw[f"bip_{s}"])
            feats[f"whiff_pct_{s}"] = ratio(sw[f"whiff_{s}"], sw[f"swing_{s}"])
            feats[f"chase_pct_{s}"] = ratio(sw[f"chase_{s}"], sw[f"out_zone_{s}"])

    id_cols = ["game_pk", "game_date", "season", "player_id", "player_name", "team_id",
               "opp_team_id", "is_home", "position", "batting_order", "is_starter",
               "opp_starter_id", "venue_id", "bat_side", "opp_sp_hand", "platoon_adv"]
    targets = ["h", "r", "rbi", "hrr", "pa"]
    t = b[targets].where(b["_played"] == 1)
    t.columns = [f"target_{c}" for c in targets]
    return pd.concat([b[id_cols], feats, t], axis=1)


# --------------------------------------------------------------------------- #
# Pitchers
# --------------------------------------------------------------------------- #

PIT_COLS = ["g", "outs", "bf", "pitches", "k", "bb", "h", "hr", "er"]
PIT_WINDOWS = {"l3": 3, "l5": 5, "l10": 10, "car": None}
SC_PIT_COLS = ["pitch", "swing", "whiff", "called_strike", "in_zone", "out_zone", "chase",
               "bip", "hard_hit", "barrel", "xwoba_num", "xwoba_den"]


def pitcher_base(pit: pd.DataFrame, players: pd.DataFrame, sc_pit: pd.DataFrame,
                 league_k: pd.Series) -> pd.DataFrame:
    """Starter's own form over prior *starts* (plus rest and career K% over all games)."""
    allp = pit.copy()
    allp["g"] = allp["_played"]

    # Career K% over every appearance and days of rest since the last one.
    car = prior_sums(allp, ["player_id"], ["k", "bf", "h", "bb", "hr"], None, suffix="all")
    allp["k_pct_all_car"] = ratio(car["k_all"], car["bf_all"])
    allp["k_pct_shr"] = shrink(car["k_all"], car["bf_all"], league_k, 200)
    # Rates allowed, regressed to the league (used by the log5 matchup scores).
    for stat, default, k in (("h", 0.22, 300), ("bb", 0.08, 200), ("hr", 0.03, 400)):
        lg = league_prior_rate(allp, stat, "bf", default)
        allp[f"{stat}_per_bf_shr"] = shrink(car[f"{stat}_all"], car["bf_all"], lg, k)
    prev = allp.groupby("player_id")["game_date"].shift(1)
    allp["days_rest"] = _as_of_day_start(
        allp, pd.DataFrame({"d": (allp["game_date"] - prev).dt.days}), ["player_id"])["d"]

    s = allp[allp["is_starter"]].copy()
    w = _windowed(s, ["player_id"], PIT_COLS, PIT_WINDOWS)
    feats = pd.DataFrame(index=s.index)
    for k in [*PIT_WINDOWS, "szn"]:
        feats[f"starts_{k}"] = w[f"g_{k}"]
        feats[f"k_per_start_{k}"] = ratio(w[f"k_{k}"], w[f"g_{k}"])
        feats[f"bf_per_start_{k}"] = ratio(w[f"bf_{k}"], w[f"g_{k}"])
        feats[f"outs_per_start_{k}"] = ratio(w[f"outs_{k}"], w[f"g_{k}"])
        feats[f"pitches_per_start_{k}"] = ratio(w[f"pitches_{k}"], w[f"g_{k}"])
        feats[f"k_pct_{k}"] = ratio(w[f"k_{k}"], w[f"bf_{k}"])
        feats[f"bb_pct_{k}"] = ratio(w[f"bb_{k}"], w[f"bf_{k}"])
        feats[f"h_per_bf_{k}"] = ratio(w[f"h_{k}"], w[f"bf_{k}"])
        feats[f"hr_per_bf_{k}"] = ratio(w[f"hr_{k}"], w[f"bf_{k}"])
        feats[f"era_{k}"] = ratio(w[f"er_{k}"] * 27, w[f"outs_{k}"])
    # Last start, raw (pitch-count carryover, short outings).
    last = s.groupby("player_id")[["pitches", "outs", "k"]].shift(1)
    last = _as_of_day_start(s, last, ["player_id"])
    feats[["last_pitches", "last_outs", "last_k"]] = last.to_numpy()
    # Batters he's expected to face: recent workload, else season/career, else typical.
    feats["exp_bf"] = (feats["bf_per_start_l10"].fillna(feats["bf_per_start_szn"])
                       .fillna(feats["bf_per_start_car"]).fillna(22.0))

    if not sc_pit.empty:
        sc = s[["game_pk", "player_id"]].merge(
            sc_pit.rename(columns={"pitcher": "player_id"}), on=["game_pk", "player_id"], how="left")
        sc.index = s.index
        velo = sc["fb_velo"] if "fb_velo" in sc else pd.Series(np.nan, index=s.index)
        sc = pd.concat([s[["player_id", "season", "game_date"]], sc[SC_PIT_COLS]], axis=1)
        sc["velo_sum"] = velo.fillna(0)
        sc["velo_n"] = velo.notna().astype(int)
        cols = SC_PIT_COLS + ["velo_sum", "velo_n"]
        sw = _windowed(sc, ["player_id"], cols, {"l3": 3, "l10": 10, "car": None})
        for k in ("l3", "l10", "szn", "car"):
            feats[f"whiff_pct_{k}"] = ratio(sw[f"whiff_{k}"], sw[f"swing_{k}"])
            feats[f"csw_pct_{k}"] = ratio(sw[f"whiff_{k}"] + sw[f"called_strike_{k}"], sw[f"pitch_{k}"])
            feats[f"chase_pct_{k}"] = ratio(sw[f"chase_{k}"], sw[f"out_zone_{k}"])
            feats[f"zone_pct_{k}"] = ratio(sw[f"in_zone_{k}"], sw[f"pitch_{k}"])
            feats[f"xwoba_allowed_{k}"] = ratio(sw[f"xwoba_num_{k}"], sw[f"xwoba_den_{k}"])
            feats[f"fb_velo_{k}"] = ratio(sw[f"velo_sum_{k}"], sw[f"velo_n_{k}"])
        feats["fb_velo_trend"] = feats["fb_velo_l3"] - feats["fb_velo_szn"]

    hand = players.set_index("player_id")
    s["pitch_hand"] = s["player_id"].map(hand["pitch_hand"])
    s["age"] = ((s["game_date"] - pd.to_datetime(s["player_id"].map(hand["birth_date"]))).dt.days
                / 365.25)
    id_cols = ["game_pk", "game_date", "season", "player_id", "player_name", "team_id",
               "opp_team_id", "is_home", "venue_id", "pitch_hand", "age",
               "k_pct_all_car", "k_pct_shr", "h_per_bf_shr", "bb_per_bf_shr", "hr_per_bf_shr",
               "days_rest"]
    targets = ["k", "outs", "bf", "pitches"]
    t = s[targets].where(s["_played"] == 1)
    t.columns = [f"target_{c}" for c in targets]
    return pd.concat([s[id_cols], feats, t], axis=1)


MATCHUP_RATES = {  # score: (batter rate, starter's rate allowed, league rate)
    "k": ("k_pct_vs_hand_shr", "opp_sp_k_pct_shr", "lg_k_pa"),
    "h": ("h_per_pa_vs_hand_shr", "opp_sp_h_per_bf_shr", "lg_h_pa"),
    "bb": ("bb_pct_shr", "opp_sp_bb_per_bf_shr", "lg_bb_pa"),
    "hr": ("hr_per_pa_shr", "opp_sp_hr_per_bf_shr", "lg_hr_pa"),
}


def matchups(bf: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Batter-vs-today's-starter scores, expected PAs against him, and lineup neighbours.

    * ``m_k``/``m_h``/``m_bb``/``m_hr``: log5 per-PA rates for this batter vs this starter.
    * ``pa_vs_sp``: PAs his lineup slot gets before the starter's expected exit.
    * ``exp_*_vs_sp``: those two multiplied (expected hits / times on base vs the starter).
    * ``ahead_onbase`` / ``behind_onbase``: on-base rates of the two hitters before him
      (runners for his RBIs) and after him (who drive him in for runs).
    """
    sp = pf[["game_pk", "player_id", "k_pct_shr", "h_per_bf_shr", "bb_per_bf_shr",
             "hr_per_bf_shr", "exp_bf"]]
    sp = sp.rename(columns={"player_id": "opp_starter_id",
                            **{c: f"opp_sp_{c}" for c in sp.columns[2:]}})
    b = bf.merge(sp, on=["game_pk", "opp_starter_id"], how="left")
    for name, (bat, pit, lg) in MATCHUP_RATES.items():
        b[f"m_{name}"] = log5(b[bat], b[pit], b[lg])
    b["m_onbase"] = b["m_h"] + b["m_bb"]
    b["pa_vs_sp"] = pa_vs_starter(b["opp_sp_exp_bf"], b["batting_order"])
    b["exp_h_vs_sp"] = b["m_h"] * b["pa_vs_sp"]
    b["exp_onbase_vs_sp"] = b["m_onbase"] * b["pa_vs_sp"]

    # Lineup neighbours (starters only; slots wrap around 9 -> 1).
    st = b[b["is_starter"].astype(bool) & b["batting_order"].between(1, 9)]
    ob = st.set_index(["game_pk", "team_id", "batting_order"])["onbase_shr"]
    ob = ob[~ob.index.duplicated()]
    def neighbours(offsets):
        vals = []
        for off in offsets:
            slot = ((b["batting_order"] - 1 + off) % 9) + 1
            idx = pd.MultiIndex.from_arrays([b["game_pk"], b["team_id"], slot])
            vals.append(ob.reindex(idx).to_numpy())
        return np.nanmean(np.vstack(vals), axis=0)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        b["ahead_onbase"] = neighbours((-1, -2))
        b["behind_onbase"] = neighbours((1, 2))
    starter = b["is_starter"].astype(bool)
    b.loc[~starter, ["ahead_onbase", "behind_onbase"]] = np.nan
    return b


def lineup_matchup(bf: pd.DataFrame) -> pd.DataFrame:
    """The opposing starting lineup scored against each starter (from :func:`matchups`).

    ``exp_k_matchup`` = sum over the nine hitters of (log5 strikeout chance vs him) x
    (PAs that lineup slot gets before his expected exit): the strikeouts this lineup
    should give him. ``lineup_k_log5`` is the average per-PA strikeout chance.
    """
    st = bf[bf["is_starter"].astype(bool) & bf["opp_starter_id"].notna()]
    st = st.assign(_k=st["m_k"] * st["pa_vs_sp"], _ob=st["m_onbase"] * st["pa_vs_sp"],
                   _hr=st["m_hr"] * st["pa_vs_sp"])
    agg = st.groupby(["game_pk", "opp_starter_id"]).agg(
        exp_k_matchup=("_k", "sum"),
        lineup_k_log5=("m_k", "mean"),
        exp_onbase_matchup=("_ob", "sum"),
        exp_hr_matchup=("_hr", "sum"),
        matchup_batters=("m_k", "count"),
    ).reset_index().rename(columns={"opp_starter_id": "player_id"})
    agg["player_id"] = agg["player_id"].astype(int)
    return agg


def lineup_strength(batter_feats: pd.DataFrame) -> pd.DataFrame:
    """Starting-lineup averages per (game, batting team), as faced by the opposing starter."""
    st = batter_feats[batter_feats["is_starter"]]
    agg = st.groupby(["game_pk", "team_id"]).agg(
        lineup_size=("player_id", "size"),
        lineup_k_pct=("k_pct_shr", "mean"),
        lineup_k_pct_vs_hand=("k_pct_vs_hand_shr", "mean"),
        lineup_hrr_per_pa=("hrr_per_pa_shr", "mean"),
        lineup_lefties=("bat_side", lambda x: (x == "L").sum()),
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

WEATHER_COLS = ["game_pk", "temp_f", "wind_mph", "wind_dir", "day_night", "weather_condition"]


def _weather(games: pd.DataFrame) -> pd.DataFrame:
    w = games[[c for c in WEATHER_COLS if c in games.columns]].copy()
    d = w.get("wind_dir", pd.Series(index=w.index, dtype=object)).fillna("").str.lower()
    w["wind_out"] = d.str.startswith("out").astype(int)
    w["wind_in"] = d.str.startswith("in").astype(int)
    cond = w.get("weather_condition", pd.Series(index=w.index, dtype=object)).fillna("").str.lower()
    w["roof_closed"] = cond.isin(["dome", "roof closed"]).astype(int)
    w["is_night"] = (w.get("day_night") == "night").astype(int)
    return w.drop(columns=[c for c in ("wind_dir", "day_night", "weather_condition") if c in w])


def build_features(inputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build ``batter_hrr`` and ``pitcher_k`` feature tables from processed inputs.

    Rows whose stats are missing (``_played == 0``, i.e. upcoming games) keep
    NaN targets and are featurised from history only.
    """
    games = _prep_games(inputs["games"])
    players = inputs["players"]
    sc_bat = inputs.get("statcast_batter", pd.DataFrame())
    sc_pit = inputs.get("statcast_pitcher", pd.DataFrame())

    bat = _attach_game_cols(inputs["batter_games"], games)
    pit = _attach_game_cols(inputs["pitcher_games"], games)
    for df in (bat, pit):
        if "_played" not in df:
            df["_played"] = 1
        df["_played"] = df["_played"].fillna(1).astype(int)

    ctx = game_context(games, bat, pit)
    team = team_context(games, bat, pit, players)
    league_k_bat = bat["game_pk"].map(ctx.set_index("game_pk")["league_k_pct"])
    league_k_pit = pit["game_pk"].map(ctx.set_index("game_pk")["league_k_pct"])
    weather = _weather(games)

    bf = batter_base(bat, players, sc_bat, league_k_bat)
    pf = pitcher_base(pit, players, sc_pit, league_k_pit)
    bf = matchups(bf, pf)

    # ---- pitcher table: + opposing lineup / team, park, ump, weather, own team leash
    lineups = lineup_strength(bf).rename(columns={"team_id": "opp_team_id"})
    opp_team = team.drop(columns=["opp_sp_hand"]).rename(columns={"team_id": "opp_team_id"})
    opp_team = opp_team[["game_pk", "opp_team_id"] + [c for c in opp_team if c.startswith("team_k_pct")]]
    opp_team.columns = ["game_pk", "opp_team_id"] + [f"opp_{c}" for c in opp_team.columns[2:]]
    own_team = team[["game_pk", "team_id"] + [c for c in team if c.startswith(("team_sp_outs", "team_rp"))]]
    pitcher_k = (pf.merge(lineups, on=["game_pk", "opp_team_id"], how="left")
                   .merge(lineup_matchup(bf), on=["game_pk", "player_id"], how="left")
                   .merge(opp_team, on=["game_pk", "opp_team_id"], how="left")
                   .merge(own_team, on=["game_pk", "team_id"], how="left")
                   .merge(ctx[["game_pk", "park_so_factor", "ump_k_factor", "ump_games",
                               "league_k_pct"]], on="game_pk", how="left")
                   .merge(weather, on="game_pk", how="left"))

    # ---- batter table: + opposing starter, opposing staff, own offense, park, weather
    sp_cols = ["k_pct_l5", "k_pct_szn", "bb_pct_szn", "h_per_bf_szn",
               "hr_per_bf_szn", "era_szn", "outs_per_start_l5", "outs_per_start_szn",
               "xwoba_allowed_szn", "xwoba_allowed_car", "whiff_pct_szn", "starts_car"]
    sp = pitcher_k[["game_pk", "player_id"] + [c for c in sp_cols if c in pitcher_k]]
    sp = sp.rename(columns={"player_id": "opp_starter_id",
                            **{c: f"opp_sp_{c}" for c in sp_cols}})
    opp_staff = team[["game_pk", "team_id"] + [c for c in team if c.startswith(
        ("team_ra_per_g", "team_rp_onbase", "team_rp_k"))]]
    opp_staff = opp_staff.rename(columns={"team_id": "opp_team_id",
                                          **{c: f"opp_{c}" for c in opp_staff.columns[2:]}})
    own_off = team[["game_pk", "team_id"] + [c for c in team if c.startswith("team_r_per_g")]]
    batter_hrr = (bf.merge(sp, on=["game_pk", "opp_starter_id"], how="left")
                    .merge(opp_staff, on=["game_pk", "opp_team_id"], how="left")
                    .merge(own_off, on=["game_pk", "team_id"], how="left")
                    .merge(ctx[["game_pk", "park_r_factor", "park_h_factor", "park_hr_factor",
                                "league_r_per_pa"]], on="game_pk", how="left")
                    .merge(weather, on="game_pk", how="left"))

    return {"batter_hrr": _sort(batter_hrr), "pitcher_k": _sort(pitcher_k)}
