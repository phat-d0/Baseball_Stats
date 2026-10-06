import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_blend  # noqa: E402


def graded_lines(months, n=400, seed=0):
    """Lines where the truth sits between the model and DraftKings, so the blend should
    weigh both."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        truth = rng.uniform(0.3, 0.7, n)
        lt = np.log(truth / (1 - truth))
        p_book = 1 / (1 + np.exp(-(lt + rng.normal(0, 0.3, n))))
        p_model = 1 / (1 + np.exp(-(lt + rng.normal(0, 0.5, n))))
        for i in range(n):
            for kind in ("batter", "pitcher"):
                rows.append({"game_pk": hash((m, i)) % 10**6, "player_id": i, "kind": kind, "line": 1.5,
                             "over": -110, "under": -110, "p_model": p_model[i], "p_book": p_book[i],
                             "over_won": bool(rng.random() < truth[i]), "status": "graded",
                             "fetched_at": pd.Timestamp(f"{m}-15", tz="UTC"), "clv_over": np.nan,
                             "game_date": f"{m}-15", "month": m})
    return pd.DataFrame(rows)


def test_refit_without_test_set_scores_walk_forward(monkeypatch, tmp_path):
    g = graded_lines(["2025-06", "2025-07", "2025-08", "2025-09", "2026-08"])
    monkeypatch.setattr(fit_blend, "load", lambda folder, cache=None, batter="current": g)
    w, rep = tmp_path / "blend.json", tmp_path / "report.json"
    fit_blend.main(["--fit", "prices", "--weights", str(w), "--report", str(rep)])
    weights = json.loads(w.read_text())
    assert weights["fit_prices"] == "historical prices 2025-06-15..2026-08-15"
    for kind in ("batter", "pitcher"):
        assert 0 < weights["weights"][kind]["model"] < weights["weights"][kind]["book"]
    wf = json.loads(rep.read_text())["walk_forward"]["pitcher"]
    assert list(wf) == ["2025-09", "2026-08"]  # the first three months only fit
    assert all(r["logloss_blend"] < r["logloss_model"] for r in wf.values())
