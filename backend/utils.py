from __future__ import annotations

import os
import re
import time
import threading
import logging
from dataclasses import dataclass
from typing import List, Tuple, Union, Iterable
from urllib.parse import urlparse

import requests

logger = logging.getLogger("utils")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

@dataclass
class Tweet:
    url: str
    handle: str
    author_name: str
    text: str
    lang: str

class CrownTALKError(Exception):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code

_MIN_GAP_SECONDS = float(os.environ.get("UPSTREAM_MIN_GAP_SECONDS", "0.5"))
_DEFAULT_TIMEOUT  = float(os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "12"))

_last_call_ts = 0.0
_rl_lock = threading.Lock()

def _rate_limit_yield():
    global _last_call_ts
    with _rl_lock:
        now = time.time()
        wait = _MIN_GAP_SECONDS - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.time()

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "CrownTALK/1.0 (+https://crowndex.app)"})

_X_DOMAINS = {"x.com", "twitter.com", "mobile.twitter.com", "m.twitter.com"}

def _flatten_to_string(text_or_list: Union[str, Iterable]) -> str:
    """
    Helper: accepts a string or an iterable of strings (possibly nested)
    and returns a single whitespace-joined string.
    """
    if isinstance(text_or_list, (list, tuple, set)):
        parts: List[str] = []
        stack = list(text_or_list)
        while stack:
            item = stack.pop(0)
            if item is None:
                continue
            if isinstance(item, (list, tuple, set)):
                stack[:0] = list(item)
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(text_or_list or "")

def clean_and_normalize_urls(text_or_list: Union[str, List[str]]) -> List[str]:
    """
    Accepts either:
      - a string with one or more URLs (any separators), or
      - a list (or nested lists) of URL strings.
    Returns a de-duplicated list of normalized X/Twitter status URLs.
    """
    raw = _flatten_to_string(text_or_list)
    if not raw:
        return []

    candidates = re.split(r"[\s,]+", raw.strip())
    out, seen = [], set()

    for u in candidates:
        if not u:
            continue
        if not u.startswith("http"):
            u = "https://" + u
        try:
            p = urlparse(u)
            host = (p.netloc or "").lower().split(":")[0]
            if host not in _X_DOMAINS:
                continue
            clean = f"https://{host}{p.path}"
            if clean not in seen:
                seen.add(clean)
                out.append(clean)
        except Exception:
            continue
    return out

def _extract_handle_and_id(url: str) -> Tuple[str, str]:
    u = url.strip()
    if not u: raise CrownTALKError("invalid_url", code="invalid_url")
    if not u.startswith("http"): u = "https://" + u
    p = urlparse(u); host = p.netloc.lower().split(":")[0]
    if host not in _X_DOMAINS: raise CrownTALKError("invalid_url", code="invalid_url")
    parts = [seg for seg in p.path.split("/") if seg]
    handle, status_id = "", ""
    if len(parts) >= 3 and parts[1] == "status":
        handle, status_id = parts[0], parts[2]
    elif len(parts) >= 4 and parts[-2] == "status":
        handle, status_id = parts[-4], parts[-1]
    if not handle or not status_id: raise CrownTALKError("invalid_url", code="invalid_url")
    status_id = re.sub(r"[^\d]", "", status_id)
    return handle, status_id

def _fetch_syndication(tweet_id: str) -> dict:
    url = f"https://cdn.syndication.twimg.com/tweet?id={tweet_id}"
    r = _SESSION.get(url, timeout=_DEFAULT_TIMEOUT)
    if r.status_code != 200:
        raise requests.HTTPError(f"syndication status {r.status_code}", response=r)
    return r.json()

def _parse_syndication(j: dict, url: str, handle: str) -> Tweet:
    text = j.get("text") or ""
    user = j.get("user") or {}
    screen_name = user.get("screen_name") or handle
    name = user.get("name") or ""
    lang = j.get("lang") or ""
    return Tweet(url=url, handle=screen_name, author_name=name, text=text, lang=lang)

def fetch_tweet_data(url: str) -> Tweet:
    handle, tweet_id = _extract_handle_and_id(url)

    def _backoff_sleep(attempt: int):
        time.sleep(min(2.0, 0.25 * (attempt + 1)))

    _rate_limit_yield()

    for attempt in range(3):
        try:
            j = _fetch_syndication(tweet_id)
            if j and isinstance(j, dict):
                return _parse_syndication(j, url, handle)
        except requests.RequestException:
            logger.warning("syndication request error (attempt %s)", attempt + 1)
            _backoff_sleep(attempt)
        except Exception:
            logger.exception("syndication parse error")
            _backoff_sleep(attempt)

    try:
        proxy = f"https://r.jina.ai/http://x.com/{handle}/status/{tweet_id}"
        r = _SESSION.get(proxy, timeout=_DEFAULT_TIMEOUT)
        if r.status_code == 200:
            txt = r.text
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            lines = [ln for ln in lines if len(ln) > 3 and not ln.startswith("@")]
            best = max(lines, key=len) if lines else ""
            return Tweet(url=url, handle=handle, author_name="", text=best[:280], lang="")
    except Exception:
        logger.warning("fallback proxy parse failed")

    raise CrownTALKError("upstream_unavailable_or_rate_limited", code="upstream_rate_limited")
