"""The chance the app bets on: DraftKings' no-vig chance, tilted by the model.

On historical prices DraftKings' own chance is more accurate than the model's, but the
model still carries a little it doesn't (docs/backtest_2026.md). The blend is a
logistic stack fitted per kind by scripts/fit_blend.py:

    logit(p) = model_w * logit(p_model) + book_w * logit(p_book) + intercept

Edges, value picks and paper trades use it; the model's own chance is still logged so
the Record tab can keep scoring it against DraftKings. With no weights file, or no
DraftKings price, the model's chance is used unchanged.

``sigma`` per kind is how far the blend usually strays from DraftKings (the sd of
``p_blend - p_book`` on the fit prices). Edges are graded in those units (confidence
tiers, docs/edge_tiers.md): a 2-point gap means more on a prop where the blend rarely
moves than on one where it often does.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

WEIGHTS_PATH = Path(__file__).with_name("blend.json")


@lru_cache(maxsize=1)
def _file() -> dict:
    try:
        return json.loads(WEIGHTS_PATH.read_text())
    except FileNotFoundError:
        return {}


def weights() -> dict:
    return _file().get("weights", {})


def sigma() -> dict:
    """{kind: sd of p_blend - p_book} from the fit; empty without one."""
    return _file().get("sigma", {})


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def blended(p_model: float, p_book: float | None, kind: str | None) -> float:
    w = weights().get(kind or "")
    if w is None or p_book is None or (isinstance(p_book, float) and math.isnan(p_book)):
        return p_model
    z = w["model"] * _logit(p_model) + w["book"] * _logit(p_book) + w.get("intercept", 0.0)
    return 1 / (1 + math.exp(-z))
