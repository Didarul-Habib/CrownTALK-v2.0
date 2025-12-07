from __future__ import annotations

import logging
import re
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple
import os
import requests
from urllib.parse import urlparse

logger = logging.getLogger("utils")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class CrownTALKError(Exception):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code

# Global upstream pacing (VX/FX Twitter scrapers)
_MIN_GAP_SECONDS = float(os.environ.get("UPSTREAM_MIN_GAP_SECONDS", "0.5"))
_last_call_ts = 0.0
_rl_lock = threading.Lock()

def _rate_limit_yield():
    """Global process-level pacing for upstream calls."""
    global _last_call_ts
    with _rl_lock:
        now = time.time()
        wait = _MIN_GAP_SECONDS - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.time()

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "CrownTALK/1.0 (+https://crowndex.app)",
    "Accept": "application/json,text/json,application/*+json;q=0.9,*/*;q=0.5",
})

_DEFAULT_TIMEOUT = 12

_X_DOMAINS = {"x.com", "twitter.com", "mobile.twitter.com", "m.twitter.com"}

# --------------------------------------------------------------------------
# URL parsing / cleaning
# --------------------------------------------------------------------------
def _extract_handle_and_id(url: str) -> Tuple[str, str]:
    """
    Extract handle and status_id from a Twitter/X URL.

    Supports:
      - https://x.com/handle/status/1234567890
      - https://twitter.com/handle/status/1234567890
      - mobile / m. variants
      - With or without query params (?s=20 etc.)
      - Raw "x.com/handle/status/..." without scheme (handled upstream)
    """
    try:
        u = url.strip()
        if not u:
            raise ValueError("empty url")
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        p = urlparse(u)
        host = p.netloc.lower().split(":")[0]
        if host not in _X_DOMAINS:
            raise ValueError("not an x.com/twitter.com URL")

        parts = [seg for seg in p.path.split("/") if seg]
        # /{handle}/status/{id}
        if len(parts) >= 3 and parts[1] == "status":
            handle = parts[0]
            status_id = parts[2]
        # sometimes there are extra segments, be defensive
        elif len(parts) >= 4 and parts[-2] == "status":
            handle = parts[-4]
            status_id = parts[-1]
        else:
            raise ValueError("couldn't parse status path")

        # strip non-digits from status id (?s=20 or trailing stuff)
        status_id = re.sub(r"[^\d]", "", status_id)
        if not handle or not status_id:
            raise ValueError("missing handle or id")
        return handle, status_id
    except Exception as e:
        raise CrownTALKError(f"Bad tweet URL: {url}", code="bad_tweet_url") from e

def _normalize_x_url(url: str) -> str:
    """
    Normalize any valid X/Twitter status URL into:
      https://x.com/{handle}/status/{status_id}
    """
    handle, status_id = _extract_handle_and_id(url)
    return f"https://x.com/{handle}/status/{status_id}"

def clean_and_normalize_urls(urls: List[str]) -> List[str]:
    """
    Take a mixed list of strings (possibly multi-line, messy, without scheme),
    extract valid X/Twitter status URLs, normalize them, and dedupe.

    - Accepts:
        "https://x.com/handle/status/123..."
        "x.com/handle/status/123..."
        "twitter.com/handle/status/123..."
    - Ignores non-status URLs and non-X/Twitter domains.
    """
    out: List[str] = []
    seen = set()

    for item in urls:
        if not item:
            continue
        # each item might contain multiple URLs line by line
        for raw in str(item).splitlines():
            raw = raw.strip()
            if not raw:
                continue
            # Try to normalize even if scheme is missing; _normalize_x_url
            # will add https:// where needed and validate the structure.
            try:
                canon = _normalize_x_url(raw)
            except CrownTALKError:
                continue
            if canon not in seen:
                seen.add(canon)
                out.append(canon)

    if not out:
        raise CrownTALKError("No valid X/Twitter links found", code="no_valid_urls")
    return out

# --------------------------------------------------------------------------
# Tweet payload model + upstream helpers
# --------------------------------------------------------------------------
@dataclass
class TweetData:
    text: str
    author_name: str | None
    lang: str | None

_VX_FMT = "https://api.vxtwitter.com/{handle}/status/{status_id}"
_FX_FMT = "https://api.fxtwitter.com/{handle}/status/{status_id}"

def _backoff_sleep(attempt: int, base: float = 0.35, cap: float = 6.0):
    """Exponential backoff with tiny jitter."""
    sleep = min(cap, base * (2 ** (attempt - 1)))
    sleep *= (0.9 + 0.2 * (attempt % 3))  # tiny jitter
    time.sleep(sleep)

def _do_get_json(url: str) -> requests.Response:
    _rate_limit_yield()
    return _SESSION.get(url, timeout=_DEFAULT_TIMEOUT)

def _read_json_payload(resp: requests.Response) -> dict:
    """
    Safely parse JSON, guarding against HTML error pages or text.
    """
    ctype = (resp.headers.get("Content-Type") or "").lower()
    # If upstream sends HTML (Cloudflare / error page), treat as shape error
    if "text/html" in ctype:
        raise CrownTALKError("HTML page from upstream, not JSON", code="upstream_bad_json")
    try:
        return resp.json()
    except Exception:
        raise CrownTALKError("Bad JSON from upstream", code="upstream_bad_json")

def _parse_payload(payload: dict) -> TweetData:
    """
    Extract tweet text, author name, and language from the VX/FX-style payload.
    We accept both top-level and nested 'tweet' structures.
    """
    lang = payload.get("lang") or payload.get("tweet", {}).get("lang")
    text = (
        payload.get("text")
        or payload.get("full_text")
        or payload.get("tweet", {}).get("text")
        or payload.get("tweet", {}).get("full_text")
    )
    user_name = (
        payload.get("user_name")
        or payload.get("user", {}).get("name")
        or payload.get("tweet", {}).get("user", {}).get("name")
    )
    if not text:
        raise CrownTALKError("Tweet text missing in upstream payload", code="upstream_shape_changed")
    # Normalize whitespace a bit to avoid weird spacing
    text = re.sub(r"\s+", " ", text).strip()
    return TweetData(text=text, author_name=user_name, lang=lang)

# optional small in-process cache to avoid hitting upstream multiple times
# for the exact same status in a short window
_TWEET_CACHE_TTL = float(os.environ.get("TWEET_CACHE_TTL_SECONDS", "120"))
_TWEET_CACHE: dict[tuple[str, str], tuple[float, TweetData]] = {}
_TWEET_CACHE_LOCK = threading.Lock()

def _cache_get(handle: str, status_id: str) -> TweetData | None:
    key = (handle, status_id)
    now = time.time()
    with _TWEET_CACHE_LOCK:
        val = _TWEET_CACHE.get(key)
        if not val:
            return None
        ts, data = val
        if now - ts > _TWEET_CACHE_TTL:
            _TWEET_CACHE.pop(key, None)
            return None
        return data

def _cache_set(handle: str, status_id: str, data: TweetData) -> None:
    key = (handle, status_id)
    with _TWEET_CACHE_LOCK:
        _TWEET_CACHE[key] = (time.time(), data)

def fetch_tweet_data(x_url: str) -> TweetData:
    """
    Fetch TweetData for a given X/Twitter URL.

    - Normalizes / parses the URL to {handle, status_id}.
    - First tries VXTwitter, then FXTwitter as fallback.
    - Uses basic exponential backoff on transient errors (429/5xx).
    - Raises CrownTALKError with a structured .code for the caller.
    """
    handle, status_id = _extract_handle_and_id(x_url)

    # Check small in-process cache first
    cached = _cache_get(handle, status_id)
    if cached is not None:
        return cached

    vx_url = _VX_FMT.format(handle=handle, status_id=status_id)
    last_status = None

    for attempt in range(1, 4):
        try:
            logger.info("Fetching VXTwitter data for %s -> %s", x_url, vx_url)
            r = _do_get_json(vx_url)
            last_status = r.status_code
            if r.status_code == 200:
                payload = _read_json_payload(r)
                inner = payload.get("tweet") if isinstance(payload.get("tweet"), dict) else payload
                data = _parse_payload(inner)
                _cache_set(handle, status_id, data)
                return data
            elif r.status_code in (401, 403, 404):
                # Private / deleted / not found: no point retrying VX or FX aggressively
                raise CrownTALKError(
                    "Tweet is not accessible (deleted, private, or blocked)",
                    code="tweet_not_accessible",
                )
            elif r.status_code in (429, 500, 502, 503, 504):
                if r.status_code == 429 and attempt >= 2:
                    break
                _backoff_sleep(attempt)
                continue
            else:
                # unexpected status, fall through to FX
                break
        except requests.RequestException:
            logger.exception("VXTwitter request error")
            _backoff_sleep(attempt)

    fx_url = _FX_FMT.format(handle=handle, status_id=status_id)
    for attempt in range(1, 4):
        try:
            logger.info("Fetching FXTwitter data for %s -> %s", x_url, fx_url)
            r = _do_get_json(fx_url)
            last_status = r.status_code
            if r.status_code == 200:
                payload = _read_json_payload(r)
                inner = payload.get("tweet") if isinstance(payload.get("tweet"), dict) else payload
                data = _parse_payload(inner)
                _cache_set(handle, status_id, data)
                return data
            elif r.status_code in (401, 403, 404):
                raise CrownTALKError(
                    "Tweet is not accessible (deleted, private, or blocked)",
                    code="tweet_not_accessible",
                )
            elif r.status_code in (429, 500, 502, 503, 504):
                _backoff_sleep(attempt)
                continue
            else:
                break
        except requests.RequestException:
            logger.exception("FXTwitter request error")
            _backoff_sleep(attempt)

    # If we reach here, everything failed
    if last_status in (401, 403, 404):
        raise CrownTALKError(
            "Tweet is not accessible (deleted, private, or blocked)",
            code="tweet_not_accessible",
        )

    raise CrownTALKError(
        "Upstream is rate-limiting or unavailable; try fewer links or wait a minute",
        code="upstream_rate_limited",
    )
