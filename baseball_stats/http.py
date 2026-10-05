"""HTTP session with retries and an on-disk response cache.

Responses for finished games never change, so caching them makes re-runs of
the collector cheap and lets you rebuild processed tables without re-downloading.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "baseball-stats-pipeline/0.1"
    return s


_SESSION = _session()


def _cache_path(url: str, params: dict | None, suffix: str) -> Path:
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()
    return config.RAW_DIR / "http" / digest[:2] / f"{digest}{suffix}"


def get(url: str, params: dict | None = None, *, cache: bool = True,
        suffix: str = ".json", timeout: float = 60) -> bytes:
    """GET ``url`` and return the body, using the disk cache when allowed."""
    path = _cache_path(url, params, suffix)
    if cache and path.exists():
        return path.read_bytes()
    resp = _SESSION.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    body = resp.content
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    time.sleep(config.REQUEST_DELAY)
    return body


def get_json(url: str, params: dict | None = None, *, cache: bool = True) -> dict:
    return json.loads(get(url, params, cache=cache, suffix=".json"))
