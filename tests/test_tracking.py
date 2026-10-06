"""Pick logging and market grading (spec: "Pick Logging and Market Grading")."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from baseball_stats import odds, publish, storage, tracking
from test_publish import START, FakeDK, fake  # noqa: F401  (fixture)

RUN1 = datetime(2025, 6, 15, 16, 0, tzinfo=UTC)  # 7 h before the 23:05 UTC first pitch


@pytest.fixture
def dk(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("ODDS_API_RESET_DAY", "1")
    monkeypatch.delenv("ODDS_API_MONTHLY_CAP", raising=False)
    api = FakeDK()
    monkeypatch.setattr(odds.requests, "get", api.get)
    return api


def _publish(tmp_path, now):
    return publish.publish(tmp_path / "site", history_start=START, now=now)


def test_snapshots_match_data_json(fake, dk, tmp_path):
    data = _publish(tmp_path, RUN1)
    snaps = storage.read("prop_snapshots")
    # One row per player and line: P10151 (K 4.5) and P10101 (H+R+RBI 1.5).
    assert len(snaps) == 2
    assert not snaps.duplicated(["game_pk", "player_id", "kind", "line"]).any()
    g = data["slates"][0]["games"][0]
    shown = {("pitcher", g["pitchers"]["home"]["id"]): g["pitchers"]["home"]["book"][0]}
    hitter = next(b for b in g["lineups"]["home"] if b["id"] == 10101)
    shown[("batter", 10101)] = hitter["book"][0]
    for r in snaps.itertuples():
        b = shown[(r.kind, r.player_id)]
        for col in ("p_model", "p_book", "ev_over", "ev_under"):
            assert getattr(r, col) == pytest.approx(b[col], abs=1e-4), col
        assert r.fetched_at == pd.Timestamp(RUN1)
        assert r.game_start == pd.Timestamp("2025-06-15T23:05:00Z")


def test_rerun_on_same_prices_adds_nothing(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    before = storage.read("prop_snapshots")
    _publish(tmp_path, RUN1 + timedelta(minutes=20))  # inside the 2 h refresh gap
    after = storage.read("prop_snapshots")
    assert len(after) == len(before)


def test_second_download_adds_rows_and_keeps_old(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    first = storage.read("prop_snapshots")
    _publish(tmp_path, RUN1 + timedelta(hours=2, minutes=30))
    after = storage.read("prop_snapshots")
    assert len(after) == 2 * len(first)
    old = after[after["fetched_at"] == pd.Timestamp(RUN1)].sort_values(["kind"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(old, first.sort_values(["kind"]).reset_index(drop=True),
                                  check_dtype=False)


def test_no_key_still_publishes(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    data = _publish(tmp_path, RUN1)
    assert data["slates"] and storage.read("prop_snapshots").empty


# ---- pricing schedule -------------------------------------------------------

START_TS = pd.Timestamp("2026-06-10T23:00:00Z")


@pytest.mark.parametrize("minutes_before,last_before,expected", [
    (1500, None, None),                      # outside the 24 h window
    (300, None, odds.UNPRICED),
    (300, 360, None),                        # 1 h since last pull, 2 h gap
    (300, 430, odds.REFRESH),                # 2 h 10 m since last
    (150, 170, None),                        # final 3 h: 20 min since last
    (150, 185, odds.REFRESH),                # final 3 h: 35 min since last
    (40, 60, odds.CLOSING),                  # inside 45 min, last pull before it
    (30, 40, None),                          # closing pull already taken
])
def test_fetch_priority(minutes_before, last_before, expected):
    now = START_TS - pd.Timedelta(minutes=minutes_before)
    last = None if last_before is None else START_TS - pd.Timedelta(minutes=last_before)
    assert odds.fetch_priority(START_TS, last, now) == expected


# ---- grading ----------------------------------------------------------------

def _finish_game(k=7, hrr=2, pitcher_starts=True, batter_starts=True, state="Final"):
    """Make the fake upcoming game (9999) final with box score lines for both props."""
    storage.upsert("games", pd.DataFrame([{
        "game_pk": 9999, "game_date": "2025-06-15", "status": "Final", "detailed_state": state,
        "home_team_id": 101, "away_team_id": 104, "season": 2025, "venue_id": 1}]))
    storage.upsert("pitcher_games", pd.DataFrame([{
        "game_pk": 9999, "player_id": 10151, "team_id": 101, "is_starter": pitcher_starts, "k": k}]))
    storage.upsert("batter_games", pd.DataFrame([{
        "game_pk": 9999, "player_id": 10101, "team_id": 101, "is_starter": batter_starts, "hrr": hrr}]))


def test_graded_after_final(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    pending = tracking.update_grades()
    assert set(pending["status"]) == {"pending"}
    _finish_game(k=7, hrr=1)
    g = tracking.update_grades().set_index("kind")
    assert g.loc["pitcher", "status"] == "graded" and g.loc["pitcher", "actual"] == 7
    assert bool(g.loc["pitcher", "over_won"]) is True  # 7 > 4.5
    assert g.loc["batter", "actual"] == 1 and bool(g.loc["batter", "over_won"]) is False  # 1 < 1.5
    assert storage.read("prop_grades").shape[0] == 2


def test_scratched_starter_and_bench_batter_void(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    _finish_game(pitcher_starts=False, batter_starts=False)
    g = tracking.update_grades()
    assert set(g["status"]) == {"void"} and g["actual"].isna().all()


def _snap(minutes_before, line, over, under, player=1, kind="pitcher", start="2026-06-10T23:00:00Z"):
    start = pd.Timestamp(start)
    p = odds.no_vig_over(over, under)
    return {"game_pk": 1, "player_id": player, "kind": kind, "line": line, "over": over,
            "under": under, "p_book": p, "p_model": 0.5, "ev_over": 0.0, "ev_under": 0.0,
            "fetched_at": start - pd.Timedelta(minutes=minutes_before), "game_start": start,
            "game_date": pd.Timestamp("2026-06-10"), "lineup_confirmed": True}


def _grade(rows, k=6, starter=True, state="Final"):
    games = pd.DataFrame([{"game_pk": 1, "game_date": "2026-06-10", "status": "Final",
                           "detailed_state": state}])
    pit = pd.DataFrame([{"game_pk": 1, "player_id": 1, "is_starter": starter, "k": k}])
    return tracking.grade(pd.DataFrame(rows), games, pd.DataFrame(), pit).sort_values("fetched_at")


def test_clv_between_two_snapshots_at_same_line():
    g = _grade([_snap(300, 5.5, -110, -110), _snap(30, 5.5, -140, 115)])
    early, late = g.iloc[0], g.iloc[1]
    assert bool(late["is_close"]) and not bool(early["is_close"])
    assert early["clv_over"] == pytest.approx(odds.no_vig_over(-140, 115) - 0.5)
    assert np.isnan(late["clv_over"])  # never compared with itself


def test_lone_early_snapshot_has_no_close():
    g = _grade([_snap(150, 5.5, -110, -110)])
    assert bool(g.iloc[0]["is_close"]) and np.isnan(g.iloc[0]["p_book_close"])


def test_close_at_a_different_line():
    g = _grade([_snap(300, 5.5, -110, -110), _snap(20, 6.5, -110, -110)])
    early = g.iloc[0]
    assert np.isnan(early["p_book_close"]) and early["close_line"] == 6.5


def test_postponed_and_push_are_void():
    assert _grade([_snap(30, 5.5, -110, -110)], state="Postponed")["status"].iloc[0] == "void"
    assert _grade([_snap(30, 6.0, -110, -110)], k=6)["void_reason"].iloc[0] == "push"
    assert _grade([_snap(30, 6.0, -110, -110)], k=7)["status"].iloc[0] == "graded"


def test_closing_pull_paid_before_unpriced_games(tmp_path, monkeypatch):
    from baseball_stats import config
    from test_odds import FakeOddsAPI, NOW
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    api = FakeOddsAPI(n_games=2, credits=400)
    # Game 0 starts in 2 h; game 1 in 6 h. First run prices both.
    api.events[0]["commence_time"] = (NOW + timedelta(hours=2)).isoformat()
    api.events[1]["commence_time"] = (NOW + timedelta(hours=6)).isoformat()
    games = pd.DataFrame([{"game_pk": 100 + i, "home_team": ev["home_team"], "away_team": ev["away_team"],
                           "game_datetime": ev["commence_time"]} for i, ev in enumerate(api.events)])
    odds.fetch_props(games.iloc[[0]], now=NOW, api_key="k", session=api)  # prices game 0 only
    assert api.paid_calls == 1
    # 1 h 20 m later: game 0 is inside 45 min -> closing pull; game 1 is unpriced.
    # Leave room in today's allowance for exactly one game.
    ledger = odds.Ledger.load()
    ledger.spent = ledger.allowance - 2
    ledger.save()
    later = NOW + timedelta(hours=1, minutes=20)
    odds.fetch_props(games, now=later, api_key="k", session=api)
    assert api.paid_calls == 2
    fetched = odds.Ledger.load().fetched
    assert fetched["ev0"] == later.isoformat(timespec="seconds") and "ev1" not in fetched
    # And the closing pull happens only once.
    ledger = odds.Ledger.load(); ledger.spent = 0; ledger.save()
    odds.fetch_props(games.iloc[[0]], now=later + timedelta(minutes=20), api_key="k", session=api)
    assert api.paid_calls == 2


# ---- summary ----------------------------------------------------------------

def test_five_hand_built_picks():
    # +100 W, -110 L, +150 W, -200 W, +120 L  -> profits 1, -1, 1.5, 0.5, -1
    p = pd.DataFrame({
        "price": [100, -110, 150, -200, 120], "won": [True, False, True, True, False],
        "p": [0.55, 0.6, 0.45, 0.7, 0.5], "status": ["graded"] * 5,
        "clv": [0.02, -0.01, None, 0.03, 0.0], "close_line": [None] * 5,
        "line": [5.5] * 5, "side": ["over"] * 5,
    })
    m = tracking.pick_metrics(p)
    assert m["n"] == 5 and m["n_void"] == 0
    assert m["win"] == pytest.approx(0.6, abs=1e-4)
    assert m["breakeven"] == pytest.approx((0.5 + 110 / 210 + 0.4 + 2 / 3 + 100 / 220) / 5, abs=1e-4)
    assert m["roi"] == pytest.approx(0.2, abs=1e-4)
    assert m["expected"] == pytest.approx(0.56, abs=1e-4)
    assert m["roi_lo"] < m["roi"] < m["roi_hi"]
    assert m["clv"] == pytest.approx(1.0, abs=1e-4)  # mean of +2, -1, +3, 0 points
    assert m["moves"] == {"toward": 0.5, "away": 0.25, "stayed": 0.25}


def test_picks_one_per_player_and_threshold():
    rows = [_snap(300, 5.5, 120, -140), _snap(30, 5.5, 105, -125)]
    g = _grade(rows, k=7)
    g["p_model"] = 0.55  # over at +120 -> EV 0.21; at +105 -> 0.1275
    g["ev_over"] = g["p_model"] * g["over"].map(odds.american_to_decimal) - 1
    g["ev_under"] = -0.5
    p5 = tracking.picks(g, 0.05)
    assert len(p5) == 1 and p5.iloc[0]["price"] == 120 and p5.iloc[0]["won"]
    assert p5.iloc[0]["clv"] == pytest.approx(g.iloc[0]["clv_over"])
    assert tracking.picks(g, 0.25).empty


def test_market_summary_end_to_end(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    _finish_game(k=7, hrr=1)
    data = _publish(tmp_path, datetime(2025, 6, 16, 16, 0, tzinfo=UTC))
    market = data["record"]["market"]
    assert market["first_snapshot"] == "2025-06-15"
    assert market["pitcher"]["n_lines"] == 1
    assert set(market["pitcher"]["by_threshold"]) == {"0.01", "0.02", "0.03", "0.05"}


def test_no_key_market_empty(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert _publish(tmp_path, RUN1)["record"]["market"] == {}


# ---- paper portfolio ----------------------------------------------------------

def _paper_grades():
    rows = []
    def add(player, kind, over, ev_over, status, over_won, confirmed=True, minutes=120):
        r = _snap(minutes, 5.5, over, -120, player=player, kind=kind)
        r.update(ev_over=ev_over, ev_under=-0.3, p_model=0.6, status=status, over_won=over_won,
                 lineup_confirmed=confirmed, clv_over=np.nan, close_line=np.nan, player_name=f"P{player}")
        rows.append(r)
    add(1, "pitcher", 150, 0.20, "graded", True)     # trade, won: +$15
    add(2, "pitcher", -110, 0.13, "graded", False)   # trade, lost: -$10
    add(3, "pitcher", 120, 0.10, "graded", True)     # below 12%: no trade
    add(4, "batter", 140, 0.30, "graded", True, confirmed=False)  # projected lineup: skipped
    add(5, "batter", 100, 0.12, "pending", None)     # trade, still open
    add(6, "pitcher", 105, 0.25, "void", None)       # trade, void: $0
    return pd.DataFrame(rows)


def test_paper_trades_rules():
    t = tracking.paper_trades(_paper_grades())
    assert sorted(t["player_id"]) == [1, 2, 5, 6]
    by = t.set_index("player_id")
    assert by.loc[1, "profit"] == pytest.approx(15.0) and by.loc[1, "result"] == "won"
    assert by.loc[2, "profit"] == pytest.approx(-10.0) and by.loc[2, "result"] == "lost"
    assert np.isnan(by.loc[5, "profit"]) and by.loc[5, "result"] == "open"
    assert by.loc[6, "profit"] == 0 and by.loc[6, "result"] == "void"
    s = tracking.paper_portfolio(_paper_grades())["summary"]
    assert (s["n"], s["won"], s["lost"], s["open"], s["void"]) == (4, 1, 1, 1, 1)
    assert s["staked"] == 20 and s["profit"] == pytest.approx(5.0) and s["roi"] == pytest.approx(0.25)
    assert s["at_risk"] == 10


def test_paper_band_strategy():
    # 8-12%: only player 3 (10%); player 5 sits exactly at 12%, the ceiling, so it is left out.
    t = tracking.paper_trades(_paper_grades(), threshold=0.08, ceiling=0.12)
    assert sorted(t["player_id"]) == [3]
    assert t.iloc[0]["profit"] == pytest.approx(12.0)
    st = {s["key"]: s for s in tracking.paper_strategies(_paper_grades())}
    assert list(st) == ["blend1", "edge12", "edge8_12"]
    # These rows predate the blend (no p_blend): only the retired model strategies trade them.
    assert st["blend1"]["summary"]["n"] == 0
    assert st["edge12"]["summary"]["n"] == 4 and st["edge8_12"]["summary"]["n"] == 1
    assert st["edge8_12"]["ceiling"] == 0.12 and st["edge8_12"]["label"]
    blended = _paper_grades().assign(p_blend=0.6)
    st = {s["key"]: s for s in tracking.paper_strategies(blended)}
    assert st["blend1"]["summary"]["n"] == 5 and st["edge12"]["summary"]["n"] == 0


def test_paper_portfolio_end_to_end(fake, dk, tmp_path):
    data = _publish(tmp_path, RUN1)
    assert data["paper_strategies"][0] == data["paper"]
    for st in data["paper_strategies"][1:]:
        assert not st["trades"]  # retired: new prices carry blended edges
    paper = data["paper"]
    assert paper["stake"] == 10 and paper["threshold"] == 0.01 and paper["source"] == "blend"
    assert all(t["ev"] >= 0.01 and t["result"] == "open" for t in paper["trades"])
    _finish_game(k=7, hrr=1)
    paper = _publish(tmp_path, datetime(2025, 6, 16, 16, 0, tzinfo=UTC))["paper"]
    for t in paper["trades"]:
        dec = odds.american_to_decimal(t["price"])
        assert t["result"] in ("won", "lost")
        assert t["profit"] == pytest.approx(10 * (dec - 1) if t["result"] == "won" else -10)
    if paper["trades"]:
        assert paper["summary"]["profit"] == pytest.approx(sum(t["profit"] for t in paper["trades"]))
        assert paper["curve"][-1]["cum"] == pytest.approx(paper["summary"]["profit"])


def test_line_values_use_the_blend(monkeypatch):
    from baseball_stats import blend
    pmf = np.zeros(15)
    pmf[7] = 1.0  # always 7: model says over 5.5 for sure
    entry = {"line": 5.5, "over": 100, "under": -120}
    monkeypatch.setattr(blend, "weights", lambda: {})
    raw = tracking.line_values(entry, pmf, "pitcher")
    assert raw["p_blend"] == raw["p_model"] and raw["ev_over"] == pytest.approx(raw["p_model"] * 2 - 1)
    monkeypatch.setattr(blend, "weights", lambda: {"pitcher": {"model": 0.0, "book": 1.0, "intercept": 0.0}})
    v = tracking.line_values(entry, pmf, "pitcher")
    assert v["p_blend"] == pytest.approx(v["p_book"], abs=1e-6)  # all book: no edge left
    assert v["ev_over"] < 0 and v["ev_under"] < 0
    assert tracking.line_values(entry, pmf, "batter")["p_blend"] == v["p_model"]  # no weights for hitters


def test_summary_scores_the_blend_on_its_own_rows():
    rows = []
    for i, (pm, pb, pbl, won) in enumerate([(0.7, 0.5, 0.55, True), (0.4, 0.5, 0.47, False),
                                           (0.6, 0.5, None, True)]):
        rows.append({"kind": "pitcher", "game_pk": i, "player_id": i, "line": 5.5,
                     "fetched_at": pd.Timestamp("2026-10-06T20:00Z"), "game_date": pd.Timestamp("2026-10-06"),
                     "is_close": True, "status": "graded", "p_model": pm, "p_book": pb, "p_blend": pbl,
                     "over_won": won, "over": -110, "under": -110, "ev_over": 0.0, "ev_under": 0.0,
                     "clv_over": np.nan, "close_line": np.nan, "lineup_confirmed": True})
    res = tracking.summary(pd.DataFrame(rows))["pitcher"]
    assert res["n_lines"] == 3 and res["n_lines_blend"] == 2
    assert res["logloss_blend"] == pytest.approx(-(np.log(0.55) + np.log(0.53)) / 2)
    assert res["logloss_book_blend"] == pytest.approx(np.log(2))
