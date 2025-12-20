import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("crowntalk")


class CrownTALKError(Exception):
    pass


VX_TOKEN = os.environ.get("VX_TOKEN", "")
FX_TOKEN = os.environ.get("FX_TOKEN", "")


@dataclass
class TweetData:
    url: str
    canonical_url: Optional[str]
    text: str
    author_name: Optional[str]
    author_handle: Optional[str]
    lang: Optional[str]


# --------------------------------------------------------------------------
# URL helpers
# --------------------------------------------------------------------------

def _normalize_single_url(raw: str) -> Optional[str]:
    """
    Normalize common X/Twitter URL formats into a canonical https://x.com/.../status/ID URL.

    - Accept x.com and twitter.com.
    - Accept /i/status and rewrite to /status.
    - Strip query params and fragments.
    """
    if not raw:
        return None

    s = raw.strip()

    # If user pasted something like `<https://x.com/...>` strip wrappers
    s = s.strip("<>")

    # Extract the first URL-ish substring
    m = re.search(r"https?://[^\s]+", s)
    if not m:
        return None
    s = m.group(0)

    # Normalize domains and /i/status
    s = s.replace("https://twitter.com", "https://x.com")
    s = s.replace("http://twitter.com", "https://x.com")
    s = s.replace("http://x.com", "https://x.com")

    # Rewrite /i/status/... -> /status/...
    s = s.replace("/i/status/", "/status/")

    # Strip query and fragment
    s = re.sub(r"[?#].*$", "", s)

    # Basic tweet URL pattern check
    if not re.search(r"https://x\.com/[^/]+/status/\d+", s):
        return None

    return s


def clean_and_normalize_urls(raw_text: str) -> List[str]:
    """
    Take arbitrary pasted text and return a list of clean, unique tweet URLs.

    - Handles multiple lines / spaces
    - Ignores non-URL junk
    - Normalizes twitter.com -> x.com and /i/status -> /status
    """
    if not raw_text:
        return []

    urls: List[str] = []

    for line in str(raw_text).splitlines():
        line = line.strip()
        if not line:
            continue

        # Some people paste numbered lists "1. https://..."
        line = re.sub(r"^\s*\d+[\).\-\:]\s*", "", line)

        maybe = _normalize_single_url(line)
        if maybe:
            urls.append(maybe)

    # Dedupe while preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------
# Tweet fetching via VX / FX
# --------------------------------------------------------------------------

def _vx_headers() -> Dict[str, str]:
    if not VX_TOKEN:
        raise CrownTALKError("VX_TOKEN not configured")
    return {"Authorization": f"Bearer {VX_TOKEN}"}


def _fx_headers() -> Dict[str, str]:
    if not FX_TOKEN:
        raise CrownTALKError("FX_TOKEN not configured")
    return {"Authorization": f"Bearer {FX_TOKEN}"}


def _vx_fetch(url: str) -> Dict[str, Any]:
    """
    Hit vx endpoint for tweet data.
    """
    api_url = "https://api.vx.dev/v1/tweet"
    try:
        resp = requests.get(api_url, headers=_vx_headers(), params={"url": url}, timeout=15)
    except Exception as exc:
        logger.warning("VX request error: %s", exc)
        raise CrownTALKError("vx_fetch_failed")

    if resp.status_code != 200:
        logger.warning("VX non-200: %s %s", resp.status_code, resp.text[:200])
        raise CrownTALKError("vx_fetch_non_200")

    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("VX invalid json: %s", exc)
        raise CrownTALKError("vx_invalid_json")

    return data


def _fx_fetch(url: str) -> Dict[str, Any]:
    """
    Hit fx endpoint for tweet data, with similar response.
    """
    api_url = "https://api.fx.dev/v1/tweet"
    try:
        resp = requests.get(api_url, headers=_fx_headers(), params={"url": url}, timeout=15)
    except Exception as exc:
        logger.warning("FX request error: %s", exc)
        raise CrownTALKError("fx_fetch_failed")

    if resp.status_code != 200:
        logger.warning("FX non-200: %s %s", resp.status_code, resp.text[:200])
        raise CrownTALKError("fx_fetch_non_200")

    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("FX invalid json: %s", exc)
        raise CrownTALKError("fx_invalid_json")

    return data


def fetch_tweet_data(url: str) -> TweetData:
    """
    Fetch tweet info using VX first, then FX fallback.

    We also try to read a canonical URL from the response so the
    frontend shows the “real” URL instead of /i/status, etc.
    """
    normalized = _normalize_single_url(url)
    if not normalized:
        raise CrownTALKError("invalid_tweet_url")

    errors: List[str] = []
    last_data: Optional[Dict[str, Any]] = None

    for source in ("vx", "fx"):
        try:
            if source == "vx":
                data = _vx_fetch(normalized)
            else:
                data = _fx_fetch(normalized)
            last_data = data
            break
        except CrownTALKError as exc:
            errors.append(f"{source}:{exc}")
        except Exception as exc:  # pragma: no cover
            errors.append(f"{source}:unknown:{exc}")

    if last_data is None:
        logger.warning("Both VX and FX failed: %s", errors)
        raise CrownTALKError("tweet_fetch_failed")

    tweet = last_data.get("tweet") or last_data

    text = tweet.get("text") or ""
    author_name = tweet.get("author_name") or tweet.get("user_name")
    author_handle = tweet.get("author_handle") or tweet.get("user_screen_name")
    lang = tweet.get("lang")

    canonical_url = tweet.get("url") or tweet.get("canonical_url") or normalized

    return TweetData(
        url=normalized,
        canonical_url=canonical_url,
        text=text,
        author_name=author_name,
        author_handle=author_handle,
        lang=lang,
    )


# --------------------------------------------------------------------------
# Style fingerprint (for offline generator)
# --------------------------------------------------------------------------

STYLE_PATTERNS = {
    "markets": re.compile(
        r"\b(apy|apr|yield|volume|market|cap|liquidity|supply|fdv|mc)\b",
        re.IGNORECASE,
    ),
    "nft": re.compile(
        r"\b(nft|mint|collection|pfp|floor|opensea|blur)\b",
        re.IGNORECASE,
    ),
    "airdrop": re.compile(
        r"\b(airdrop|claim|snapshot|retrodrop|farm|farming)\b",
        re.IGNORECASE,
    ),
    "infra": re.compile(
        r"\b(l2|rollup|zk|evm|bridge|rpc|indexer|sequencer)\b",
        re.IGNORECASE,
    ),
}


def style_fingerprint(tweet_text: str) -> str:
    """
    Very lightweight "intent" fingerprint for the tweet.

    We don't try to be clever here – just detect broad buckets for
    offline comment flavor (markets/NFT/airdrop/infra/etc).
    """
    if not tweet_text:
        return "unknown"

    txt = tweet_text.lower()

    hits: List[str] = []
    for key, pat in STYLE_PATTERNS.items():
        if pat.search(txt):
            hits.append(key)

    if not hits:
        if len(txt) < 40:
            return "short"
        if len(txt) > 240:
            return "long"
        return "generic"

    return "+".join(sorted(set(hits)))


# --------------------------------------------------------------------------
# Extra helpers used by older code paths / compatibility
# --------------------------------------------------------------------------

def maybe_sleep_backoff(attempt: int) -> None:
    """Tiny helper if you ever reintroduce retry loops."""
    delay = min(5.0, 0.2 * (2 ** attempt))
    time.sleep(delay)


# --------------------------------------------------------------------------
# Token / percent helpers for newer main versions (safe to keep even if unused)
# --------------------------------------------------------------------------

def extract_ticker_and_percent_summary(text: str) -> Dict[str, List[str]]:
    """Extract $tickers and percentage expressions from the original tweet text.

    This is used so later steps can preserve things like:
    - $HLS instead of HLS
    - 20% instead of 20
    - 3.4% instead of 3 4 or 3%

    Returns:
        { "tickers": [...], "percents": [...] }
    """
    if not text:
        return {"tickers": [], "percents": []}

    cleaned = " ".join(str(text).split())

    ticker_pattern = re.compile(r"\$[A-Za-z][A-Za-z0-9]{1,9}")
    tickers = list(dict.fromkeys(ticker_pattern.findall(cleaned)))

    percent_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*%")
    percents = list(dict.fromkeys(percent_pattern.findall(cleaned)))

    return {"tickers": tickers, "percents": percents}


def apply_token_style_fixes(comment: str, summary: Dict[str, List[str]]) -> str:
    """Post-process a generated comment.

    - Ensure tickers keep a single leading `$` if they appeared that way in the tweet.
    - Restore missing `%` in obvious APY/APR/yield contexts.
    - Strip a trailing period / exclamation point; keep `?` if present.
    """
    if not comment:
        return comment

    text = comment

    # --- Fix ticker style ($HLS vs HLS, and collapse $$HLS) ---
    tickers = (summary or {}).get("tickers") or []
    for ticker in tickers:
        core = ticker.lstrip("$")
        canonical = f"${core}"

        # Collapse '$$HLS' / '$$$HLS' -> '$HLS'
        text = re.sub(rf"\$+{re.escape(core)}\b", canonical, text)

        # Bare 'HLS' but not when already '$HLS'
        bare_pattern = re.compile(rf"(?<!\$)\b{re.escape(core)}\b")
        text = bare_pattern.sub(canonical, text)

    # --- Fix missing '%' where tweet had an exact percent ---
    percents = (summary or {}).get("percents") or []
    for pct in percents:
        number = pct.replace("%", "").strip()
        if not number:
            continue

        # If comment wrote '20 HLS APY' but tweet had '20% ... APY',
        # upgrade '20' to '20%' when it appears within a couple tokens of APY/APR/yield.
        pattern = re.compile(
            rf"\b{re.escape(number)}\b(?=(?:\s+\S+){{0,2}}\s+(apy|apr|yield)\b)",
            flags=re.IGNORECASE,
        )
        text = pattern.sub(pct, text)

    # --- Trim trailing punctuation (keep '?') ---
    text = text.rstrip()
    if text.endswith("?"):
        return text

    text = re.sub(r"[.!…]+$", "", text).rstrip()
    return text