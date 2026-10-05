from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from baseball_stats import config, odds

NOW = datetime(2026, 6, 10, 16, 0, tzinfo=UTC)  # noon Eastern


class Resp:
    def __init__(self, body, left, last, status=200):
        self._body, self.status_code, self.ok = body, status, status == 200
        self.headers = {"x-requests-remaining": str(left), "x-requests-last": str(last)}
        import json
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeOddsAPI:
    """Stands in for requests: /events is free, each event's props cost 2 credits."""

    def __init__(self, n_games: int, credits: int, cost: int = 2):
        self.credits, self.cost, self.paid_calls = credits, cost, 0
        self.events = [{
            "id": f"ev{i}", "home_team": f"Home {i} Sox", "away_team": f"Away {i} Jays",
            "commence_time": (NOW + timedelta(hours=1 + i * 0.5)).isoformat(),
        } for i in range(n_games)]

    def get(self, url, params=None, timeout=None):
        assert params["apiKey"] == "k"
        if url.endswith("/events"):
            return Resp(self.events, self.credits, 0)
        self.paid_calls += 1
        self.credits -= self.cost
        ev_id = url.split("/events/")[1].split("/")[0]
        i = int(ev_id[2:])
        body = {"id": ev_id, "bookmakers": [{"key": "draftkings", "markets": [
            {"key": "pitcher_strikeouts", "outcomes": [
                {"name": "Over", "description": f"Pitcher {i}", "price": -120, "point": 5.5},
                {"name": "Under", "description": f"Pitcher {i}", "price": 100, "point": 5.5}]},
            {"key": "batter_hits_runs_rbis", "outcomes": [
                {"name": "Over", "description": "José Ramírez Jr.", "price": 105, "point": 1.5},
                {"name": "Under", "description": "José Ramírez Jr.", "price": -135, "point": 1.5}]},
        ]}]}
        return Resp(body, self.credits, self.cost)


def _games(api):
    return pd.DataFrame([{
        "game_pk": 100 + i, "home_team": f"Home {i} Sox", "away_team": f"Away {i} Jays",
        "game_datetime": ev["commence_time"],
    } for i, ev in enumerate(api.events)])


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.delenv("ODDS_API_MONTHLY_CAP", raising=False)
    monkeypatch.setenv("ODDS_API_RESET_DAY", "12")  # resets in 2 days


def test_budget_maths():
    assert odds.next_reset(NOW, 12).isoformat() == "2026-06-12"
    assert odds.next_reset(NOW, 5).isoformat() == "2026-07-05"
    assert odds.period_start(NOW, 5).isoformat() == "2026-06-05"
    assert odds.daily_allowance(500, NOW, 12) == (500 - odds.RESERVE_CREDITS) // 2
    assert odds.daily_allowance(15, NOW, 12) == 0
    assert odds.daily_allowance(500, NOW, 12, cap_left=40) == 20


def test_names():
    assert odds.same_team("Oakland Athletics", "Athletics")
    assert odds.same_team("Boston Red Sox", "Boston Red Sox")
    assert not odds.same_team("Boston Red Sox", "Chicago White Sox")
    idx = odds.PlayerIndex([(1, "José Ramírez"), (2, "Luis García Jr."), (3, "Luis Gil")])
    assert idx.find("Jose Ramirez") == 1
    assert idx.find("Luis Garcia") == 2
    assert idx.find("L. Gil") == 3


def test_no_key():
    out, status = odds.fetch_props(pd.DataFrame([{"game_pk": 1}]), now=NOW, api_key="")
    assert out == {} and "ODDS_API_KEY" in status.error


def test_spends_only_the_daily_allowance():
    api = FakeOddsAPI(n_games=12, credits=60)  # (60 - 20) / 2 days = 20 credits today
    events, status = odds.fetch_props(_games(api), now=NOW, api_key="k", session=api)
    assert status.daily_allowance == 20
    assert api.paid_calls == 10 and status.spent_today == 20
    assert len(events) == 10 and status.events_priced == 10
    # Soonest games first; games 10 and 11 (latest starts) wait.
    assert 110 not in events and 111 not in events
    # A run an hour later the same day spends nothing more.
    _, status = odds.fetch_props(_games(api), now=NOW + timedelta(hours=1), api_key="k", session=api)
    assert api.paid_calls == 10 and status.events_priced == 10


def test_never_below_reserve():
    api = FakeOddsAPI(n_games=5, credits=odds.RESERVE_CREDITS + 1)
    _, status = odds.fetch_props(_games(api), now=NOW, api_key="k", session=api)
    assert api.paid_calls == 0 and status.daily_allowance == 0


def test_monthly_cap(monkeypatch):
    monkeypatch.setenv("ODDS_API_MONTHLY_CAP", "8")  # shared key: this app may spend 8
    api = FakeOddsAPI(n_games=12, credits=400)
    odds.fetch_props(_games(api), now=NOW, api_key="k", session=api)
    assert api.paid_calls == 2  # 8 credits / 2 days = 4 today = 2 games at 2 credits


def test_props_matched_to_players():
    api = FakeOddsAPI(n_games=2, credits=200)
    events, _ = odds.fetch_props(_games(api), now=NOW, api_key="k", session=api)
    props = odds.props_by_player(events, {100: [(7, "Pitcher 0"), (8, "José Ramírez")]})
    assert props[(100, 7, "pitcher")] == [
        {"line": 5.5, "over": -120, "under": 100, "updated": None}]
    assert props[(100, 8, "batter")][0]["over"] == 105
    assert odds.no_vig_over(-120, 100) == pytest.approx((1 / (1 + 100 / 120)) / (1 / (1 + 100 / 120) + 0.5))
