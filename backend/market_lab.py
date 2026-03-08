"""
Market Post Lab module.

This module handles:
- Fetching lightweight market snapshots from CoinGecko.
- Building grounded prompts for asset-level market posts.
- Post-processing model output into single tweets or threads.

It intentionally mirrors the style of project_lab.py but stays simpler.
"""

from __future__ import annotations

import os
import random
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

from schemas import MarketPostRequest, MarketPostMode
from project_lab import normalize_project_post_text
from utils import CrownTALKError


COINGECKO_BASE = os.getenv("COINGECKO_API_BASE", "https://api.coingecko.com/api/v3")

# Small curated map of liquid large-cap assets we support for v1.
# Keep symbols in sync with the frontend MarketPostMode asset_id options.
ASSET_CATALOG: Dict[str, Dict[str, str]] = {
    "BTC": {"coingecko_id": "bitcoin", "name": "Bitcoin"},
    "ETH": {"coingecko_id": "ethereum", "name": "Ethereum"},
    "SOL": {"coingecko_id": "solana", "name": "Solana"},
    "BNB": {"coingecko_id": "binancecoin", "name": "BNB"},
}

SUPPORTED_ASSETS: List[str] = sorted(ASSET_CATALOG.keys())

# CoinGecko snapshot TTL: 5 minutes.  Avoids hammering free-tier rate limits.
_SNAPSHOT_TTL_SECONDS = int(os.getenv("COINGECKO_SNAPSHOT_TTL", "300"))
_snapshot_cache: Dict[str, tuple] = {}
_snapshot_lock = threading.Lock()


class MarketLabError(Exception):
    """Lightweight error type for Market Post Lab operations."""

    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class MarketSnapshot:
    symbol: str
    name: str
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    change_1h: Optional[float] = None
    change_24h: Optional[float] = None
    change_7d: Optional[float] = None


@dataclass
class MarketPostContext:
    """Structured context string passed to the user prompt builder."""

    context: str
    asset_code: str
    snapshot: Optional[MarketSnapshot] = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x:+.2f}%"
    except Exception:
        return "n/a"


def _fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        if x >= 1_000_000_000:
            return f"${x/1_000_000_000:.1f}B"
        if x >= 1_000_000:
            return f"${x/1_000_000:.1f}M"
        if x >= 1_000:
            return f"${x/1_000:.1f}K"
        return f"${x:,.2f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# CoinGecko fetch with TTL cache
# ---------------------------------------------------------------------------

def _fetch_coingecko_markets(symbols: List[str]) -> Dict[str, MarketSnapshot]:
    """
    Fetch market data for the given symbols from CoinGecko.

    Returns a dict mapping symbol (e.g. "BTC") to MarketSnapshot.
    Unknown symbols are silently omitted.  On rate-limit or network error,
    returns empty/partial snapshots so callers can still generate posts
    without live data (graceful degradation).
    """
    result: Dict[str, MarketSnapshot] = {}
    now = time.monotonic()

    # Split into cached vs. needs-fetch
    to_fetch: List[str] = []
    for sym in symbols:
        sym_upper = sym.upper()
        with _snapshot_lock:
            cached_entry = _snapshot_cache.get(sym_upper)
        if cached_entry and (now - cached_entry[0]) < _SNAPSHOT_TTL_SECONDS:
            result[sym_upper] = cached_entry[1]
        else:
            to_fetch.append(sym_upper)

    if not to_fetch:
        return result

    # Map symbols to CoinGecko IDs
    cg_ids: List[str] = []
    sym_by_id: Dict[str, str] = {}
    for sym in to_fetch:
        cat = ASSET_CATALOG.get(sym)
        if not cat:
            continue
        cg_id = cat["coingecko_id"]
        cg_ids.append(cg_id)
        sym_by_id[cg_id] = sym

    if not cg_ids:
        return result

    ids_param = ",".join(cg_ids)
    url = (
        f"{COINGECKO_BASE}/coins/markets"
        f"?vs_currency=usd"
        f"&ids={ids_param}"
        f"&order=market_cap_desc"
        f"&per_page={len(cg_ids)}"
        f"&page=1"
        f"&sparkline=false"
        f"&price_change_percentage=1h,24h,7d"
    )

    try:
        resp = requests.get(url, timeout=8, headers={"Accept": "application/json"})
        if resp.status_code == 429:
            # Rate limited; return stale cached data if available, else empty snapshots
            for sym in to_fetch:
                with _snapshot_lock:
                    stale = _snapshot_cache.get(sym)
                if stale:
                    result[sym] = stale[1]
                else:
                    cat = ASSET_CATALOG.get(sym, {})
                    result[sym] = MarketSnapshot(symbol=sym, name=cat.get("name", sym))
            return result
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        # On any fetch/parse error return empty snapshots for missing symbols
        # so callers can still generate a post with degraded context.
        for sym in to_fetch:
            if sym not in result:
                cat = ASSET_CATALOG.get(sym, {})
                result[sym] = MarketSnapshot(symbol=sym, name=cat.get("name", sym))
        return result

    fetch_time = time.monotonic()
    for item in data:
        cg_id = item.get("id", "")
        sym = sym_by_id.get(cg_id, "")
        if not sym:
            continue
        cat = ASSET_CATALOG.get(sym, {})
        snap = MarketSnapshot(
            symbol=sym,
            name=item.get("name") or cat.get("name", sym),
            price_usd=item.get("current_price"),
            market_cap=item.get("market_cap"),
            volume_24h=item.get("total_volume"),
            change_1h=item.get("price_change_percentage_1h_in_currency"),
            change_24h=item.get("price_change_percentage_24h"),
            change_7d=item.get("price_change_percentage_7d_in_currency"),
        )
        with _snapshot_lock:
            _snapshot_cache[sym] = (fetch_time, snap)
        result[sym] = snap

    # Fill any symbols that appeared in to_fetch but not in the CoinGecko response
    for sym in to_fetch:
        if sym not in result:
            cat = ASSET_CATALOG.get(sym, {})
            result[sym] = MarketSnapshot(symbol=sym, name=cat.get("name", sym))

    return result


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_market_context(snap: Optional[MarketSnapshot], asset_code: str) -> MarketPostContext:
    """
    Build a human-readable context string from a MarketSnapshot.

    If the snapshot has no live data (e.g. CoinGecko was unavailable) the
    context will still be usable but will indicate no live data is available.
    """
    name = (snap.name if snap else "") or asset_code
    lines: List[str] = []

    if snap and snap.price_usd is not None:
        lines.append(f"Asset: {name} ({asset_code})")
        lines.append(f"Price: {_fmt_usd(snap.price_usd)}")
        if snap.change_1h is not None:
            lines.append(f"1h change: {_fmt_pct(snap.change_1h)}")
        if snap.change_24h is not None:
            lines.append(f"24h change: {_fmt_pct(snap.change_24h)}")
        if snap.change_7d is not None:
            lines.append(f"7d change: {_fmt_pct(snap.change_7d)}")
        if snap.market_cap is not None:
            lines.append(f"Market cap: {_fmt_usd(snap.market_cap)}")
        if snap.volume_24h is not None:
            lines.append(f"24h volume: {_fmt_usd(snap.volume_24h)}")
    else:
        lines.append(f"Asset: {name} ({asset_code})")
        lines.append("Note: live price data unavailable; write from general market knowledge.")

    context_str = "\n".join(lines)
    return MarketPostContext(context=context_str, asset_code=asset_code, snapshot=snap)


# ---------------------------------------------------------------------------
# Thread splitter
# ---------------------------------------------------------------------------

def _split_thread(text: str) -> List[str]:
    """
    Split a thread-format LLM response into individual tweet strings.

    The model is instructed to separate tweets with blank lines.
    Returns a deduplicated list of non-empty tweet strings with leading
    numeric prefixes stripped.  Always returns at least one entry.
    """
    raw_parts = re.split(r"\n\s*\n", text.strip())

    tweets: List[str] = []
    seen: set = set()

    for part in raw_parts:
        cleaned = part.strip()
        # Strip leading numeric prefixes like "1.", "1)", "1:", "1 -"
        cleaned = re.sub(r"^\d+[.):\s\-]+", "", cleaned).strip()

        if not cleaned:
            continue

        norm = cleaned.lower()
        if norm in seen:
            continue
        seen.add(norm)

        tweets.append(cleaned)

    if not tweets:
        # Fallback: treat the whole text as a single tweet
        fallback = text.strip()
        if fallback:
            tweets.append(fallback)

    return tweets


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess_text(text: str) -> str:
    """Apply global post-processing rules to a single market post."""
    return normalize_project_post_text(text)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """System prompt for Market Post Lab.

    Persona: measured, risk-aware CT-native market commentator.
    """
    return (
        "You write short market commentary posts for crypto Twitter (CT).\n"
        "You sound like a thoughtful, sober trader / researcher, not a meme account.\n\n"
        "Core traits:\n"
        "- You are specific and anchored in the provided context.\n"
        "- You are risk-aware and avoid hypey language like 'moon', 'send it', 'ape', 'cheap'.\n"
        "- You NEVER add hashtags or cashtags, and you NEVER add emojis.\n"
        "- You do not add disclaimers like 'NFA' or 'DYOR'.\n"
        "- You do not speak about being an AI model.\n\n"
        "Output contract:\n"
        "- You always output either ONE X post or a short thread, depending on the mode.\n"
        "- For standard modes, output exactly one tweet worth of text, no bullets, no numbering.\n"
        "- For thread_4_6 mode, output 4-6 tweets separated by blank lines, "
        "with no numeric prefixes.\n"
    )


def _build_user_prompt(ctx: MarketPostContext, req: MarketPostRequest) -> str:
    """Build user prompt for market post generation."""
    asset_id = ctx.asset_code or (req.asset_id or "").strip().upper()
    # Resolve mode value safely regardless of whether it arrives as enum or bare string
    mode = req.post_mode.value if isinstance(req.post_mode, MarketPostMode) else str(req.post_mode)
    tone_hint = (req.tone or "").strip().lower()

    lines: list = []

    if asset_id:
        lines.append(f"You are writing a market post about: {asset_id}.")
    else:
        lines.append("You are writing a market post about the described market context.")

    lines.append("")
    lines.append("Raw market context (from upstream data or manual notes):")
    lines.append(ctx.context.strip())

    lines.append("")
    lines.append("High-level goals:")
    lines.append("- Summarise what matters right now in a way CT cares about.")
    lines.append("- Make it feel like a real, informed human wrote it.")
    lines.append("- Emphasise flows, positioning, and narrative over price prediction.")

    # Tone
    lines.append("")
    if tone_hint == "professional":
        lines.append(
            "Tone: more professional and measured – like a research desk comment or "
            "weekly note written for CT."
        )
    elif tone_hint == "casual":
        lines.append(
            "Tone: slightly more conversational, but still sober and non-cringe. "
            "Avoid meme slang and forced hype."
        )
    else:
        lines.append("Tone: credible CT-native voice. Calm, clear, and grounded.")

    # Mode-specific guidance — enum values: short_casual, medium_analysis, thread_4_6
    lines.append("")
    if mode == "short_casual":
        lines.append(
            "Post mode: short_casual. Write ONE compact X post (around 25-50 words) "
            "that captures the key idea or shift in a casual but informed voice."
        )
    elif mode == "medium_analysis":
        lines.append(
            "Post mode: medium_analysis. Write ONE X post (around 60-100 words) "
            "that briefly walks through what changed and why it matters. "
            "Stay analytical, not hype-driven."
        )
    elif mode == "thread_4_6":
        lines.append("Post mode: thread_4_6. Write a short thread of 4-6 tweets.")
        lines.append(
            "Each tweet should be on its own line, separated by a blank line. "
            "Do NOT number them. No 'Thread:' label."
        )
        lines.append(
            "The first tweet should be a hook. The rest unpack positioning, flows, "
            "and narrative in concrete terms."
        )
    else:
        lines.append("Post mode: default. Write ONE concise, self-contained X post.")

    lines.append("")
    lines.append("Global constraints:")
    lines.append("- Do not mention price targets or guarantees.")
    lines.append("- Do not explicitly tell people to buy or sell.")
    lines.append("- Do not use hashtags, cashtags, emojis, or bullet lists.")
    lines.append("- Output only the tweet text (or thread tweets), nothing else.")

    # Collapse consecutive blank lines
    result: list = []
    prev_empty = False
    for line in lines:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_market_post(
    req: MarketPostRequest,
    lang: str,
    qmode: str,
    chat_fn: Callable[..., Any],
) -> Dict[str, Any]:
    """Generate a market-level post or thread using the provided chat function."""
    # Asset selection
    asset_code = (req.asset_id or "").upper() if req.asset_id else None
    if asset_code and asset_code not in ASSET_CATALOG:
        raise MarketLabError(
            "unknown_asset",
            f"Unknown asset_id '{asset_code}'. Supported: {', '.join(SUPPORTED_ASSETS)}.",
            http_status=400,
        )

    if not asset_code:
        # Randomly pick one asset for "random market take" mode
        asset_code = random.choice(SUPPORTED_ASSETS)

    # Fetch market snapshot (TTL-cached; gracefully degrades on CoinGecko errors)
    snapshots = _fetch_coingecko_markets([asset_code])
    snap = snapshots.get(asset_code)

    # Build structured context, then prompts
    ctx = _build_market_context(snap, asset_code)
    system = _build_system_prompt()
    user_prompt = _build_user_prompt(ctx, req)

    # Prepend language directive
    target_lang = (lang or "en").strip() or "en"
    if target_lang != "en":
        lang_line = (
            f"Target output language: '{target_lang}'. "
            f"Write the final post entirely in this language."
        )
    else:
        lang_line = "Target output language: 'en'. Write the final post in clear, natural English."

    user_prompt = f"{lang_line}\n\n{user_prompt}"

    # Quality / token / temperature presets
    q = (qmode or "balanced").strip().lower() or "balanced"
    if q not in {"fast", "balanced", "pro"}:
        q = "balanced"

    if req.post_mode == MarketPostMode.SHORT_CASUAL:
        if q == "fast":
            max_tokens, temperature = 180, 0.70
        elif q == "pro":
            max_tokens, temperature = 260, 0.60
        else:
            max_tokens, temperature = 220, 0.65
    elif req.post_mode == MarketPostMode.MEDIUM_ANALYSIS:
        if q == "fast":
            max_tokens, temperature = 260, 0.72
        elif q == "pro":
            max_tokens, temperature = 380, 0.62
        else:
            max_tokens, temperature = 320, 0.68
    else:  # THREAD_4_6
        if q == "fast":
            max_tokens, temperature = 420, 0.70
        elif q == "pro":
            max_tokens, temperature = 720, 0.65
        else:
            max_tokens, temperature = 540, 0.68

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = chat_fn(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            model=None,
        )
    except CrownTALKError:
        raise
    except Exception as exc:
        raise MarketLabError(
            "llm_error", f"Upstream model error: {exc}", http_status=502
        ) from exc

    # Extract text from Groq / OpenAI-style response objects
    text: str = ""
    try:
        if hasattr(resp, "choices") and resp.choices:
            choice = resp.choices[0]
            msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
            if msg is not None:
                text = (getattr(msg, "content", None) or "").strip()
        elif isinstance(resp, dict):
            text = (resp.get("content") or resp.get("text") or "").strip()
    except Exception:
        text = ""

    if not text:
        raise MarketLabError("llm_empty", "Model returned empty content.", http_status=502)

    if req.post_mode == MarketPostMode.THREAD_4_6:
        tweets = _split_thread(text)
        if not tweets:
            raise MarketLabError(
                "llm_empty", "Model returned empty thread content.", http_status=502
            )
        return {
            "asset_id": asset_code,
            "post_mode": req.post_mode.value,
            "language": lang,
            "tweets": tweets,
            "meta": {"quality_mode": q},
        }

    processed = _postprocess_text(text)
    if not processed:
        raise MarketLabError(
            "llm_empty",
            "Model returned empty content after post-processing.",
            http_status=502,
        )

    return {
        "asset_id": asset_code,
        "post_mode": req.post_mode.value,
        "language": lang,
        "text": processed,
        "meta": {"quality_mode": q},
    }
