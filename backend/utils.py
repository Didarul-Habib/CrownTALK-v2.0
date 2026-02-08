from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger("utils")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class CrownTALKError(Exception):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------------------
# Global upstream pacing (VX/FX Twitter scrapers)
# ------------------------------------------------------------------------------

_MIN_GAP_SECONDS = float(os.environ.get("UPSTREAM_MIN_GAP_SECONDS", "0.5"))
_last_call_ts = 0.0
_rl_lock = threading.Lock()


def _rate_limit_yield() -> None:
    """Global process-level pacing for upstream calls."""
    global _last_call_ts
    with _rl_lock:
        now = time.time()
        delta = now - _last_call_ts
        if delta < _MIN_GAP_SECONDS:
            time.sleep(_MIN_GAP_SECONDS - delta)
        _last_call_ts = time.time()


# ------------------------------------------------------------------------------
# Tweet data model
# ------------------------------------------------------------------------------

@dataclass
class TweetData:
    text: str
    author_name: Optional[str]
    handle: Optional[str]
    tweet_id: Optional[str]
    lang: Optional[str]
    canonical_url: Optional[str] = None   # <— NEW: the real tweet URL from VX/FX


# VX/FX URL templates
_VX_FMT = "https://api.vxtwitter.com/{handle}/status/{status_id}"
_FX_FMT = "https://api.fxtwitter.com/{handle}/status/{status_id}"


def _extract_handle_and_id(url: str) -> Tuple[str, str]:
    """
    Extract handle and status_id from a Twitter/X URL.

    Supports:
      - https://x.com/handle/status/1234567890
      - https://twitter.com/handle/status/1234567890
      - https://mobile.twitter.com/handle/status/123...
      - https://x.com/i/status/1234567890
      - With or without query params (?s=20 etc.)
      - Bare "x.com/handle/status/..." without scheme (http/https auto-added)

    If the path is /i/status/123..., handle will be "i" and status_id is "123".
    """
    try:
        u = url.strip()
        if not u:
            raise ValueError("empty url")
        if not u.startswith(("http://", "https://")):
            u = "https://" + u

        parsed = urlparse(u)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]

        if host not in {
            "x.com",
            "twitter.com",
            "mobile.twitter.com",
            "m.twitter.com",
        }:
            raise ValueError(f"unsupported host {host}")

        parts = [p for p in parsed.path.split("/") if p]
        # expected patterns:
        #   /handle/status/id
        #   /i/status/id
        if len(parts) >= 3 and parts[-2] == "status":
            handle = parts[-3]
            status_id = parts[-1]
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


def _do_get_json(url: str) -> requests.Response:
    _rate_limit_yield()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        return requests.get(url, headers=headers, timeout=(5, 12))
    except Exception as e:
        raise CrownTALKError(f"Upstream error: {e}", code="upstream_error") from e


def _read_json_payload(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception as e:
        raise CrownTALKError(
            "Invalid JSON from upstream",
            code="upstream_invalid_json",
        ) from e


def _parse_payload(payload: dict) -> TweetData:
    """
    Extract tweet text, author name, handle, tweet_id, language and canonical URL
    from the VX/FX-style payload. We accept both top-level and nested 'tweet'
    structures.
    """
    # Some APIs wrap the tweet in payload["tweet"]
    base = payload.get("tweet") if isinstance(payload.get("tweet"), dict) else payload

    lang = base.get("lang") or payload.get("lang")

    text = (
        base.get("text")
        or base.get("full_text")
        or payload.get("text")
        or payload.get("full_text")
    )

    # X long-form / note tweets / articles (best-effort)
    note_obj = None
    for k in ("note_tweet", "noteTweet", "note_tweet_results", "noteTweetResults"):
        if isinstance(base.get(k), dict):
            note_obj = base.get(k)
            break
        if isinstance(payload.get(k), dict):
            note_obj = payload.get(k)
            break
    note_text = None
    if isinstance(note_obj, dict):
        note_text = (
            note_obj.get("text")
            or note_obj.get("full_text")
            or note_obj.get("fullText")
            or note_obj.get("note")
        )

    article_obj = base.get("article") if isinstance(base.get("article"), dict) else payload.get("article") if isinstance(payload.get("article"), dict) else None
    article_text = None
    if isinstance(article_obj, dict):
        article_text = article_obj.get("text") or article_obj.get("content") or article_obj.get("body")

    extra = None
    if note_text:
        extra = note_text
    elif article_text:
        extra = article_text

    if extra and isinstance(extra, str):
        extra = extra.strip()
        if extra and extra not in (text or ""):
            # If it's substantially longer, prefer it as the main text
            if len(extra) > len(text or "") + 40:
                text = extra
            else:
                text = (text or "") + "\n\n" + extra


    user_name = (
        base.get("user", {}).get("name")
        or base.get("user", {}).get("display_name")
        or base.get("user", {}).get("displayName")
        or base.get("user", {}).get("full_name")
        or base.get("user", {}).get("fullName")
        or payload.get("user", {}).get("name")
        or payload.get("user", {}).get("display_name")
        or payload.get("user", {}).get("displayName")
        or payload.get("user", {}).get("full_name")
        or payload.get("user", {}).get("fullName")
    )

    # FixTweet/FXTwitter style: author object (name + screen_name)
    author_obj = base.get("author") if isinstance(base.get("author"), dict) else None
    if not user_name and author_obj:
        user_name = author_obj.get("name") or author_obj.get("display_name") or author_obj.get("displayName") or user_name


    # Handle is very vendor-specific; try multiple common fields.
    handle = (
        base.get("user", {}).get("screen_name")
        or base.get("user", {}).get("username")
        or base.get("user_screen_name")
        or base.get("userScreenName")
        or payload.get("user", {}).get("screen_name")
        or payload.get("user", {}).get("username")
        or payload.get("user_screen_name")
        or payload.get("userScreenName")
    )

    if (not handle) and author_obj:
        handle = (
            author_obj.get("screen_name")
            or author_obj.get("username")
            or author_obj.get("screenName")
            or handle
        )


    tweet_id = (
        base.get("id_str")
        or base.get("id")
        or base.get("tweetID")
        or base.get("tweetId")
        or payload.get("id_str")
        or payload.get("id")
        or payload.get("tweetID")
        or payload.get("tweetId")
    )
    if tweet_id is not None:
        tweet_id = str(tweet_id)

    # Canonical URL from upstream (e.g., "tweetURL") – this is the clean form
    canonical_url = (
        base.get("tweetURL")
        or base.get("tweetUrl")
        or base.get("url")
        or payload.get("tweetURL")
        or payload.get("tweetUrl")
        or payload.get("url")
    )

    if not text:
        raise CrownTALKError(
            "Upstream payload missing text",
            code="upstream_missing_text",
        )

    return TweetData(
        text=text,
        author_name=user_name,
        handle=handle,
        tweet_id=tweet_id,
        lang=lang,
        canonical_url=canonical_url,
    )


# ------------------------------------------------------------------------------
# Small in-process cache
# ------------------------------------------------------------------------------

_TWEET_CACHE_TTL = float(os.environ.get("TWEET_CACHE_TTL_SECONDS", "120"))
_TWEET_CACHE: dict[tuple[str, str], tuple[float, TweetData]] = {}
_TWEET_CACHE_LOCK = threading.Lock()


def _cache_get(handle: str, status_id: str) -> Optional[TweetData]:
    key = (handle, status_id)
    now = time.time()
    with _TWEET_CACHE_LOCK:
        if key in _TWEET_CACHE:
            ts, val = _TWEET_CACHE[key]
            if now - ts <= _TWEET_CACHE_TTL:
                return val
            _TWEET_CACHE.pop(key, None)
    return None


def _cache_set(handle: str, status_id: str, data: TweetData) -> None:
    key = (handle, status_id)
    now = time.time()
    with _TWEET_CACHE_LOCK:
        _TWEET_CACHE[key] = (now, data)



# ------------------------------------------------------------------------------
# Fallback: Twitter syndication endpoint (often more reliable than VX/FX on hosts)
# ------------------------------------------------------------------------------

def _fetch_syndication_tweet(status_id: str) -> TweetData:
    """
    Fetch tweet info from Twitter syndication CDN endpoint.
    This often works for public tweets without auth.
    """
    url = f"https://cdn.syndication.twimg.com/tweet-result?id={status_id}&lang=en"
    r = _do_get_json(url)
    if r.status_code != 200:
        raise CrownTALKError(
            f"Syndication unexpected status {r.status_code}",
            code="upstream_error",
        )
    payload = _read_json_payload(r)

    text = (payload.get("text") or payload.get("full_text") or "").strip()
    if not text and isinstance(payload.get("quoted_tweet"), dict):
        text = (payload.get("quoted_tweet", {}).get("text") or "").strip()
    if not text:
        raise CrownTALKError("Upstream payload missing text", code="upstream_missing_text")

    user = payload.get("user") or {}
    author_name = (user.get("name") or "").strip() or None
    handle = (user.get("screen_name") or user.get("screenName") or "").strip() or None

    lang = payload.get("lang") or None
    canonical_url = f"https://x.com/{handle}/status/{status_id}" if handle else f"https://x.com/i/status/{status_id}"

    return TweetData(
        text=text,
        author_name=author_name,
        handle=handle,
        tweet_id=status_id,
        lang=lang,
        canonical_url=canonical_url,
    )
# ------------------------------------------------------------------------------
# Fetch tweet data via VX/FX
# ------------------------------------------------------------------------------

def fetch_tweet_data(x_url: str) -> TweetData:
    """
    Fetch TweetData for a given X/Twitter URL.

    - Normalizes / parses the URL to {handle, status_id}.
    - First tries VXTwitter, then FXTwitter as fallback.
    - Uses basic exponential backoff on transient errors (429/5xx).
    - Raises CrownTALKError with a structured .code for the caller.
    """
    handle, status_id = _extract_handle_and_id(x_url)

    cached = _cache_get(handle, status_id)
    if cached is not None:
        return cached

    last_status = None

    # Try VXTwitter
    vx_url = _VX_FMT.format(handle=handle, status_id=status_id)
    for attempt in range(1, 4):
        try:
            logger.info("Fetching VXTwitter data for %s -> %s", x_url, vx_url)
            r = _do_get_json(vx_url)
            last_status = r.status_code
            if r.status_code == 200:
                try:
                    payload = _read_json_payload(r)
                    data = _parse_payload(payload)
                except CrownTALKError as e:
                    # Some upstreams return HTML (200) behind bot protection. Treat invalid JSON as retryable/fallback.
                    if getattr(e, "code", None) == "upstream_invalid_json":
                        logger.warning("Invalid JSON from upstream (%s) for %s; will retry/fallback", r.url, x_url)
                        time.sleep(1 + attempt)
                        continue
                    raise
                _cache_set(handle, status_id, data)
                return data
            elif r.status_code in (401, 403, 404):
                raise CrownTALKError(
                    "Tweet not accessible via VXTwitter",
                    code="tweet_not_accessible",
                )
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(1 + attempt)
                continue
            else:
                raise CrownTALKError(
                    f"VXTwitter unexpected status {r.status_code}",
                    code="upstream_error",
                )
        except CrownTALKError:
            # Controlled, no retry here
            raise
        except Exception as e:
            logger.warning(
                "VXTwitter error on attempt %s for %s: %s",
                attempt,
                x_url,
                e,
            )
            time.sleep(1 + attempt)

    # Fallback: FXTwitter
    fx_url = _FX_FMT.format(handle=handle, status_id=status_id)
    for attempt in range(1, 4):
        try:
            logger.info("Fetching FXTwitter data for %s -> %s", x_url, fx_url)
            r = _do_get_json(fx_url)
            last_status = r.status_code
            if r.status_code == 200:
                try:
                    payload = _read_json_payload(r)
                    data = _parse_payload(payload)
                except CrownTALKError as e:
                    # Some upstreams return HTML (200) behind bot protection. Treat invalid JSON as retryable/fallback.
                    if getattr(e, "code", None) == "upstream_invalid_json":
                        logger.warning("Invalid JSON from upstream (%s) for %s; will retry/fallback", r.url, x_url)
                        time.sleep(1 + attempt)
                        continue
                    raise
                _cache_set(handle, status_id, data)
                return data
            elif r.status_code in (401, 403, 404):
                raise CrownTALKError(
                    "Tweet not accessible via FXTwitter",
                    code="tweet_not_accessible",
                )
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(1 + attempt)
                continue
            else:
                raise CrownTALKError(
                    f"FXTwitter unexpected status {r.status_code}",
                    code="upstream_error",
                )
        except CrownTALKError:
            raise
        except Exception as e:
            logger.warning(
                "FXTwitter error on attempt %s for %s: %s",
                attempt,
                x_url,
                e,
            )
            time.sleep(1 + attempt)
    # Final fallback: Twitter syndication CDN
    try:
        data = _fetch_syndication_tweet(status_id)
        _cache_set(handle, status_id, data)
        return data
    except CrownTALKError:
        pass
    except Exception as e:
        logger.warning("Syndication fallback error for %s: %s", x_url, e)

    raise CrownTALKError(
        f"Tweet could not be fetched (last status={last_status})",
        code="tweet_fetch_failed",
    )


# ------------------------------------------------------------------------------
# URL cleaning / normalization (messy input safe)
# ------------------------------------------------------------------------------

def clean_and_normalize_urls(urls: List[str]) -> List[str]:
    """
    Take a mixed list of strings (possibly multi-line, messy, without scheme),
    extract valid X/Twitter status URLs, normalize them, and dedupe.

    This helper is intentionally tolerant:

    - Accepts bare domains like "x.com/handle/status/123..."
    - Ignores any extra commentary text before/after the URL
      (e.g., "x.com/... *pls follow*").
    - Supports both classic "/{handle}/status/{id}" and "i/status/{id}" forms.
    - Always normalizes to https://x.com/… urls.
    """
    out: List[str] = []
    seen: set[str] = set()

    if not isinstance(urls, list):
        return out

    pattern = re.compile(
        r"(?P<scheme>https?://)?"
        r"(?P<domain>(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com))"
        r"/"
        r"(?:(?P<handle>[A-Za-z0-9_]{1,15})|i)"
        r"/status/"
        r"(?P<id>\d+)",
        flags=re.IGNORECASE,
    )

    for item in urls:
        if not item:
            continue
        text = str(item)

        # Scan the whole text for embedded status URLs
        for m in pattern.finditer(text):
            tweet_id = m.group("id")
            handle = m.group("handle")
            if handle:
                norm = f"https://x.com/{handle}/status/{tweet_id}"
            else:
                # /i/status/{id} – we keep 'i' here; later, once we fetch the tweet,
                # we will rewrite to https://x.com/{real_handle}/status/{id}
                norm = f"https://x.com/i/status/{tweet_id}"
            if norm not in seen:
                seen.add(norm)
                out.append(norm)

    return out


# ------------------------------------------------------------------------------
# Style fingerprint (for template memory)
# ------------------------------------------------------------------------------

def style_fingerprint(text: str) -> str:
    """
    Reduce a comment to a coarse 'style fingerprint' so we can remember
    templates without storing exact content.

    Example:
        "Love seeing real builders ship quietly, this is how it should be"
        -> "w w w w w w w w w"
    """
    t = (text or "").lower()
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    tokens = re.findall(r"\w+", t)
    tokens = tokens[:16]
    return " ".join("w" for _ in tokens)