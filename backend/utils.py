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

# -----------------------------
# Errors
# -----------------------------
class CrownTALKError(Exception):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code

# -----------------------------
# Lightweight global rate-limit
# (best effort across threads)
# -----------------------------
# Default: ~2 req/sec. Tune via env:
#   UPSTREAM_MIN_GAP_SECONDS=1.0  -> max ~1 req/sec
_MIN_GAP_SECONDS = float(os.environ.get("UPSTREAM_MIN_GAP_SECONDS", "0.5"))
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

# -----------------------------
# HTTP session
# -----------------------------
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "CrownTALK/1.0 (+https://crowndex.app)"
})
_DEFAULT_TIMEOUT = 12

# -----------------------------
# URL helpers
# -----------------------------
_X_DOMAINS = {"x.com", "twitter.com", "mobile.twitter.com", "m.twitter.com"}

def _extract_handle_and_id(url: str) -> Tuple[str, str]:
    """
    Accepts x.com/twitter.com URLs and returns (handle, status_id)
    """
    try:
        u = url.strip()
        if not u:
            raise ValueError("empty url")
        if not u.startswith("http"):
            u = "https://" + u
        p = urlparse(u)
        host = p.netloc.lower().split(":")[0]
        if host not in _X_DOMAINS:
            raise ValueError("not an x.com/twitter.com URL")

        parts = [seg for seg in p.path.split("/") if seg]
        # forms:
        # /<user>/status/<id>
        # /i/web/status/<id>  (rare)
        if len(parts) >= 3 and parts[1] == "status":
            handle = parts[0]
            status_id = parts[2]
        elif len(parts) >= 4 and parts[-2] == "status":
            handle = parts[-4]
            status_id = parts[-1]
        else:
            raise ValueError("couldn't parse status path")

        status_id = re.sub(r"[^\d]", "", status_id)
        if not handle or not status_id:
            raise ValueError("missing handle or id")

        return handle, status_id
    except Exception as e:
        raise CrownTALKError(f"Bad tweet URL: {url}", code="bad_tweet_url") from e

def _normalize_x_url(url: str) -> str:
    """Return canonical https://x.com/<user>/status/<id>"""
    handle, status_id = _extract_handle_and_id(url)
    return f"https://x.com/{handle}/status/{status_id}"

def clean_and_normalize_urls(urls: List[str]) -> List[str]:
    """
    - accepts an array (possibly multi-line strings)
    - extracts http(s) lines
    - keeps only x.com/twitter.com
    - canonicalizes and de-duplicates
    """
    out = []
    seen = set()
    for item in urls:
        if not item:
            continue
        # allow text blobs with newlines
        lines = str(item).splitlines()
        for raw in lines:
            raw = strip := raw.strip()
            raw = strip
            if not raw:
                continue
            if not raw.startswith("http"):
                # ignore non-links
                continue
            try:
                canon = _normalize_x_url(raw)
            except CrownTALKError:
                # ignore non-x links or malformed links
                continue
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    if not out:
        raise CrownTALKError("No valid X/Twitter links found", code="no_valid_urls")
    return out

# -----------------------------
# Upstream fetch (VX → FX)
# -----------------------------
@dataclass
class TweetData:
    text: str
    author_name: str | None
    lang: str | None

_VX_FMT = "https://api.vxtwitter.com/{handle}/status/{status_id}"
_FX_FMT = "https://api.fxtwitter.com/{handle}/status/{status_id}"

def _backoff_sleep(attempt: int, base: float = 0.35, cap: float = 6.0):
    sleep = min(cap, base * (2 ** (attempt - 1)))
    # tiny jitter
    sleep *= (0.9 + 0.2 * (attempt % 3))
    time.sleep(sleep)

def _do_get_json(url: str) -> requests.Response:
    _rate_limit_yield()
    return _SESSION.get(url, timeout=_DEFAULT_TIMEOUT)

def _read_json_payload(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        raise CrownTALKError("Bad JSON from upstream", code="upstream_bad_json")

def _parse_payload(payload: dict) -> TweetData:
    # VX/FX both expose a twitter-like object. Common fields:
    #  - text              : string
    #  - lang              : string (optional)
    #  - user_screen_name  : or payload["user"]["name"]
    #  - user_name         : human name (prefer)
    text = None
    lang = payload.get("lang") or payload.get("tweet", {}).get("lang")

    # Try common spots
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

    return TweetData(text=text, author_name=user_name, lang=lang)

def fetch_tweet_data(x_url: str) -> TweetData:
    """
    Fetch tweet contents from VXTwitter; on 429/>=500 or non-200, retry a few times
    and then try FXTwitter as a fallback.
    """
    handle, status_id = _extract_handle_and_id(x_url)

    # 1) Try VX
    vx_url = _VX_FMT.format(handle=handle, status_id=status_id)
    for attempt in range(1, 4):
        try:
            logger.info("Fetching VXTwitter data for %s -> %s", x_url, vx_url)
            r = _do_get_json(vx_url)
            if r.status_code == 200:
                payload = _read_json_payload(r)
                # VX sometimes wraps in {"tweet": {...}}
                inner = payload.get("tweet") if isinstance(payload.get("tweet"), dict) else payload
                return _parse_payload(inner)
            elif r.status_code in (429, 500, 502, 503, 504):
                logger.warning("VXTwitter non-200 status: %s", r.status_code)
                # If VX indicates hard rate-limit, break early to try FX after a small wait
                if r.status_code == 429 and attempt >= 2:
                    break
                _backoff_sleep(attempt)
                continue
            else:
                # 404, 410 etc. Try FX next.
                logger.warning("VXTwitter returned %s, will try FX fallback", r.status_code)
                break
        except requests.RequestException:
            logger.exception("VXTwitter request error")
            _backoff_sleep(attempt)

    # 2) Try FX fallback
    fx_url = _FX_FMT.format(handle=handle, status_id=status_id)
    for attempt in range(1, 4):
        try:
            logger.info("Fetching FXTwitter data for %s -> %s", x_url, fx_url)
            r = _do_get_json(fx_url)
            if r.status_code == 200:
                payload = _read_json_payload(r)
                inner = payload.get("tweet") if isinstance(payload.get("tweet"), dict) else payload
                return _parse_payload(inner)
            elif r.status_code in (429, 500, 502, 503, 504):
                logger.warning("FXTwitter non-200 status: %s", r.status_code)
                _backoff_sleep(attempt)
                continue
            else:
                break
        except requests.RequestException:
            logger.exception("FXTwitter request error")
            _backoff_sleep(attempt)

    # If we reach here, both failed
    raise CrownTALKError(
        "Upstream is rate-limiting or unavailable; try fewer links or wait a minute",
        code="upstream_rate_limited",
    )
