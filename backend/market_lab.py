"""
Market Post Lab module.

This module handles:
- Fetching lightweight market snapshots from CoinGecko.
- Building grounded prompts for asset-level market posts.
- Post-processing model output into single tweets or threads.

It intentionally mirrors the style of project_lab.py but stays simpler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import os
import random
import re
import textwrap
import time

import requests

from schemas import MarketPostRequest, MarketPostMode
from utils import CrownTALKError


COINGECKO_BASE = os.getenv("COINGECKO_API_BASE", "https://api.coingecko.com/api/v3")

# Small curated map of liquid large-cap assets we support for v1.
# You can extend this list over time – just keep symbols in sync with the frontend.
ASSET_CATALOG: Dict[str, Dict[str, str]] = {
    "BTC": {"coingecko_id": "bitcoin", "name": "Bitcoin"},
    "ETH": {"coingecko_id": "ethereum", "name": "Ethereum"},
    "SOL": {"coingecko_id": "solana", "name": "Solana"},
    "BNB": {"coingecko_id": "binancecoin", "name": "BNB"},
}

SUPPORTED_ASSETS: List[str] = sorted(ASSET_CATALOG.keys())


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


def _postprocess_text(text: str) -> str:
    """Apply global post-processing rules to a single post.

    Keep behaviour aligned with project_lab._postprocess_text.
    """
    if not isinstance(text, str):
        text = str(text or "")
    out = text.strip()

    # Normalize whitespace
    out = re.sub(r"\s+", " ", out)

    # Remove emojis (roughly: strip most characters outside basic planes).
    out = "".join(ch for ch in out if ord(ch) <= 0xFFFF)

    # Drop hashtags entirely.
    out = re.sub(r"\s*#\w+", "", out)

    # Ensure final punctuation.
    if out and out[-1] not in ".!?":
        out = out + "."

    # Soft ban on a few hype-y phrases.
    banned = ["WAGMI", "wen moon", "ape in"]
    for phrase in banned:
        out = re.sub(phrase, "", out, flags=re.IGNORECASE)

    return out.strip()


def _split_thread(raw: str) -> List[str]:
    """Split a model output into individual tweets for thread mode."""
    if not isinstance(raw, str):
        raw = str(raw or "")

    text = raw.strip()
    if not text:
        return []

    numbered_lines: List[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r"^\d+\)", ln):
            numbered_lines.append(ln)

    if numbered_lines:
        tweets = numbered_lines
    else:
        tweets = [ln.strip() for ln in text.splitlines() if ln.strip()]

    cleaned = [_postprocess_text(t) for t in tweets if t.strip()]
    if len(cleaned) > 6:
        cleaned = cleaned[:6]
    return cleaned


def _fetch_coingecko_markets(symbols: List[str]) -> Dict[str, MarketSnapshot]:
    """Fetch a minimal market snapshot for the given symbols.

    Returns a mapping from symbol -> MarketSnapshot. Missing symbols are omitted.
    """
    symbols = [s.upper() for s in symbols if isinstance(s, str) and s.strip()]
    wanted = [s for s in symbols if s in ASSET_CATALOG]
    if not wanted:
        return {}

    ids = [ASSET_CATALOG[s]["coingecko_id"] for s in wanted]
    params = {
        "vs_currency": "usd",
        "ids": ",".join(ids),
        "price_change_percentage": "1h,24h,7d",
        "per_page": len(ids) or 1,
        "page": 1,
        "sparkline": "false",
    }

    try:
        resp = requests.get(f"{COINGECKO_BASE}/coins/markets", params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json() or []
    except Exception as exc:
        # We deliberately fail soft: downstream will still generate generic posts.
        data = []
        # Avoid spamming logs from transient upstream issues.
        # Caller may log at a higher level if needed.

    out: Dict[str, MarketSnapshot] = {}
    by_id = {row.get("id"): row for row in data if isinstance(row, dict)}
    for sym in wanted:
        cfg = ASSET_CATALOG[sym]
        row = by_id.get(cfg["coingecko_id"], {})
        snap = MarketSnapshot(
            symbol=sym,
            name=cfg.get("name", sym),
            price_usd=row.get("current_price"),
            market_cap=row.get("market_cap"),
            volume_24h=row.get("total_volume"),
            change_1h=row.get("price_change_percentage_1h_in_currency"),
            change_24h=row.get("price_change_percentage_24h_in_currency"),
            change_7d=row.get("price_change_percentage_7d_in_currency"),
        )
        out[sym] = snap
    return out


def _build_system_prompt() -> str:
    return textwrap.dedent(
        """        You write short, human, professional posts for X (Twitter) about crypto markets.

        Rules:
        - No emojis.
        - No hashtags.
        - No hype slang (no "wen moon", "WAGMI", "ape in", etc.).
        - Short, clear sentences.
        - Never promise or imply guaranteed returns.
        - Keep everything grounded and realistic.
        """
    ).strip()


def _build_user_prompt(req: MarketPostRequest, snap: Optional[MarketSnapshot]) -> str:
    asset_code = (req.asset_id or "").upper() if req.asset_id else None
    mode = req.post_mode
    tone = (req.tone or "").strip().lower() or "casual"

    if asset_code and snap is not None:
        header = f"Write a market post about {snap.name} ({snap.symbol}) based on the latest data."
        meta_lines = [
            f"Price (USD): {_fmt_usd(snap.price_usd)}",
            f"Market cap: {_fmt_usd(snap.market_cap)}",
            f"24h volume: {_fmt_usd(snap.volume_24h)}",
            f"Change 1h: {_fmt_pct(snap.change_1h)}",
            f"Change 24h: {_fmt_pct(snap.change_24h)}",
            f"Change 7d: {_fmt_pct(snap.change_7d)}",
        ]
    elif asset_code:
        header = f"Write a market post about {asset_code} using realistic, generic context."
        meta_lines = [
            "Upstream market data was temporarily unavailable.",
            "Use general knowledge about liquid large-cap crypto assets.",
        ]
    else:
        header = "Write a market post about the overall crypto market."
        meta_lines = [
            "You do not know exact prices right now.",
            "Focus on structure of the market, flows, and behaviour of traders.",
        ]

    body_lines: List[str] = []
    body_lines.append(header)
    body_lines.append("")
    body_lines.append("Tone: casual but professional, realistic, no hype.")
    body_lines.append(f"Requested tone: {tone}.")
    body_lines.append("")

    if mode == MarketPostMode.SHORT_CASUAL:
        body_lines.append("Mode: SHORT_CASUAL (one tweet, ~20–35 words).")
        body_lines.append("Focus on one concrete observation or angle.")
    elif mode == MarketPostMode.MEDIUM_ANALYSIS:
        body_lines.append("Mode: MEDIUM_ANALYSIS (one tweet, ~40–80 words).")
        body_lines.append("Include: brief context, what is happening, and what matters for serious participants.")
    elif mode == MarketPostMode.THREAD_4_6:
        body_lines.append("Mode: THREAD_4_6 (4–6 tweets).")
        body_lines.append("Use a numbered thread: 1), 2), 3)... Each tweet 25–60 words.")
        body_lines.append("Cover: context, current situation, drivers, who is affected, realistic risks/caveats, closing thought.")

    body_lines.append("")
    body_lines.append("Market snapshot (if available):")
    for ln in meta_lines:
        body_lines.append(f"- {ln}")

    body_lines.append("")
    body_lines.append("Remember: no price predictions, no calls to buy/sell, no guarantees of future performance.")

    return "\n".join(body_lines).strip()


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
        raise MarketLabError("unknown_asset", f"Unknown asset_id '{asset_code}'.", http_status=400)

    if not asset_code:
        # Randomly pick one asset for "random market take" mode
        asset_code = random.choice(SUPPORTED_ASSETS)

    snapshots = _fetch_coingecko_markets([asset_code])
    snap = snapshots.get(asset_code)

    system = _build_system_prompt()
    user_prompt = _build_user_prompt(req, snap)

    max_tokens = 320
    temperature = 0.75
    if req.post_mode == MarketPostMode.THREAD_4_6:
        max_tokens = 640

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = chat_fn(messages=messages, max_tokens=max_tokens, temperature=temperature, n=1, model=None)
    except CrownTALKError:
        raise
    except Exception as exc:
        raise MarketLabError("llm_error", f"Upstream model error: {exc}", http_status=502) from exc

    # Extract text from Groq / OpenAI-style responses.
    text: str = ""
    try:
        # Groq client style
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
            raise MarketLabError("llm_empty", "Model returned empty content.", http_status=502)
        return {
            "asset_id": asset_code,
            "post_mode": req.post_mode.value,
            "language": lang,
            "tweets": tweets,
            "meta": {
                "quality_mode": qmode,
            },
        }

    processed = _postprocess_text(text)
    if not processed:
        raise MarketLabError("llm_empty", "Model returned empty content.", http_status=502)

    return {
        "asset_id": asset_code,
        "post_mode": req.post_mode.value,
        "language": lang,
        "text": processed,
        "meta": {
            "quality_mode": qmode,
        },
    }

