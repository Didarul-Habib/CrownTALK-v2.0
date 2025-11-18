import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

VX_BASE = "https://api.vxtwitter.com"
DEFAULT_TIMEOUT = 8.0


class CrownTALKError(Exception):
    """Custom error to bubble up controlled failures."""
    def __init__(self, message: str, code: str = "crawling_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class TweetData:
    url: str
    text: str
    author_name: str
    lang: str
    raw: Dict[str, Any]


def _ensure_scheme(url: str) -> str:
    url = url.strip()
    if not url:
        raise CrownTALKError("Empty URL.", code="empty_url")
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url
    return url


def _normalize_domain(netloc: str) -> str:
    netloc = netloc.lower()
    replacements = {
        "www.twitter.com": "twitter.com",
        "www.x.com": "x.com",
    }
    return replacements.get(netloc, netloc)


def normalize_tweet_url(url: str) -> str:
    """
    Normalize a Twitter/X/Fx/FixVX URL and map to VX API.
    """
    url = _ensure_scheme(url)
    parsed = urlparse(url)
    netloc = _normalize_domain(parsed.netloc)

    if not parsed.path or parsed.path == "/":
        raise CrownTALKError("URL does not look like a tweet.", code="not_tweet")

    if netloc not in {"twitter.com", "x.com", "vxtwitter.com", "fixvx.com", "fxtwitter.com"}:
        raise CrownTALKError("Unsupported domain for tweet extraction.", code="unsupported_domain")

    api_url = VX_BASE + parsed.path
    if parsed.query:
        api_url += "?" + parsed.query
    return api_url


def fetch_tweet_data(url: str, timeout: float = DEFAULT_TIMEOUT) -> TweetData:
    """
    Fetch tweet data from VXTwitter.
    """
    api_url = normalize_tweet_url(url)
    logger.info("Fetching VXTwitter data for %s -> %s", url, api_url)

    try:
        resp = requests.get(
            api_url,
            timeout=timeout,
            headers={"User-Agent": "CrownTALK/EXTREME-v3"},
        )
    except Exception as e:
        logger.exception("Network error while contacting VXTwitter")
        raise CrownTALKError("Failed to contact VXTwitter API.", code="network_error") from e

    if resp.status_code != 200:
        logger.warning("VXTwitter non-200 status: %s", resp.status_code)
        raise CrownTALKError(f"VXTwitter returned status {resp.status_code}.", code="vx_http_error")

    try:
        data: Dict[str, Any] = resp.json()
    except Exception as e:
        logger.exception("Failed to parse VXTwitter JSON")
        raise CrownTALKError("Invalid response from VXTwitter.", code="vx_invalid_json") from e

    text = (
        data.get("tweet", {}).get("text")
        or data.get("full_text")
        or data.get("text")
        or ""
    ).strip()
    if not text:
        raise CrownTALKError("Could not extract tweet text.", code="no_text")

    author_name = (
        data.get("tweet", {}).get("user", {}).get("name")
        or data.get("user", {}).get("name")
        or ""
    ).strip()

    lang = (data.get("tweet", {}).get("lang") or data.get("lang") or "und")

    return TweetData(
        url=url,
        text=text,
        author_name=author_name,
        lang=lang,
        raw=data,
    )


def naive_lang_detect(text: str) -> str:
    """
    Tiny offline heuristic language detection.
    Returns: "en", "bn", "hi", "zh", or "other".
    """
    s = text.strip()
    if not s:
        return "other"

    bengali_chars = re.findall(r"[\u0980-\u09FF]", s)
    devanagari_chars = re.findall(r"[\u0900-\u097F]", s)
    cjk_chars = re.findall(r"[\u3040-\u30FF\u4E00-\u9FFF]", s)
    latin_letters = re.findall(r"[A-Za-z]", s)

    bn = len(bengali_chars)
    hi = len(devanagari_chars)
    zh = len(cjk_chars)
    en = len(latin_letters)

    if bn >= 4 and bn > hi and bn > zh:
        return "bn"
    if hi >= 4 and hi > bn and hi > zh:
        return "hi"
    if zh >= 4 and zh > bn and zh > hi:
        return "zh"
    if en >= 5:
        return "en"
    return "other"


def safe_excerpt(text: str, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def clean_and_normalize_urls(urls: Sequence[Any]) -> List[str]:
    """
    Clean user-provided URLs: trim, dedupe, and basic validation.
    """
    seen = set()
    cleaned: List[str] = []
    for raw in urls:
        if not isinstance(raw, str):
            raise CrownTALKError("All URLs must be strings.", code="invalid_url_type")

        candidate = raw.strip()
        if not candidate:
            continue

        candidate = _ensure_scheme(candidate)
        parsed = urlparse(candidate)
        normalized = urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                "",
                parsed.query,
                "",
            )
        )

        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    if not cleaned:
        raise CrownTALKError("No valid URLs after cleaning.", code="no_valid_urls")

    return cleaned


def chunk_list(seq: Iterable[Any], size: int) -> List[List[Any]]:
    """
    Yield chunks of the given size as a concrete list of lists.
    """
    bucket: List[Any] = []
    chunks: List[List[Any]] = []
    for item in seq:
        bucket.append(item)
        if len(bucket) >= size:
            chunks.append(bucket)
            bucket = []
    if bucket:
        chunks.append(bucket)
    return chunks
