import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_history  # noqa: E402


class Resp:
    def __init__(self, body, left):
        self._body, self.status_code = body, 200
        self.headers = {"x-requests-remaining": str(left), "x-requests-last": "1"}

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class FakeHistory:
    """One game a day; every call costs a credit."""

    def __init__(self, credits):
        self.credits, self.dates = credits, []

    def get(self, url, params=None, timeout=None):
        self.credits -= 1
        self.dates.append(params["date"])
        if url.endswith("/events"):
            day = params["date"][:10]
            return Resp({"data": [{"id": f"ev{day}", "commence_time": f"{day}T23:05:00Z"}]}, self.credits)
        return Resp({"timestamp": params["date"], "data": {"id": "x", "bookmakers": []}}, self.credits)


@pytest.fixture
def fake(monkeypatch):
    def make(credits):
        f = FakeHistory(credits)
        monkeypatch.setenv("ODDS_API_KEY", "k")
        monkeypatch.setattr(fetch_history.requests, "Session", lambda: f)
        return f
    return make


def test_six_hour_snapshot_goes_in_its_own_folder(fake, tmp_path):
    fake(100_000)
    (tmp_path / "2026-08-01.json").write_text("{}")  # the one-hour file must not stop the 6 h pull
    fetch_history.main(["--start", "2026-08-01", "--end", "2026-08-02", "--minutes-before", "360",
                        "--out", str(tmp_path)])
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json"))
    assert files == ["2026-08-01.json", "m360/2026-08-01.json", "m360/2026-08-02.json"]
    day = json.loads((tmp_path / "m360/2026-08-01.json").read_text())
    assert day["minutes_before"] == 360
    assert day["snapshots"][0]["snapshot"] == "2026-08-01T17:05:00Z"  # 23:05 minus 6 h


def test_default_snapshot_stays_in_the_root(fake, tmp_path):
    fake(100_000)
    fetch_history.main(["--start", "2026-08-01", "--end", "2026-08-01", "--out", str(tmp_path)])
    assert [p.name for p in tmp_path.iterdir()] == ["2026-08-01.json"]


def test_stops_at_the_reserve(fake, tmp_path):
    f = fake(24_000)  # already under the 25,000 default reserve
    fetch_history.main(["--start", "2026-08-01", "--end", "2026-08-05", "--minutes-before", "360",
                        "--out", str(tmp_path)])
    assert len(f.dates) == 1  # the day's event list, then it stops before any priced call
    assert not list(tmp_path.rglob("*.json"))
