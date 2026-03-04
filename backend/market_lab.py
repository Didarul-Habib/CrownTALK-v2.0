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
from project_lab import normalize_project_post_text
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
    """Apply global post-processing rules to a single market post."""
    return normalize_project_post_text(text)




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
        "- For thread_4_6 mode, output 4-6 tweets separated by blank lines, with no numeric prefixes.\n"
    )


def _build_user_prompt(ctx: MarketPostContext, req: MarketPostRequest) -> str:
    """Build user prompt for market post generation."""

    asset_id = (req.asset_id or "").strip()
    mode = req.post_mode.value if isinstance(req.post_mode, MarketPostMode) else str(req.post_mode)
    tone_hint = (req.tone or "").strip().lower()

    lines: list[str] = []

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
        lines.append(
            "Tone: credible CT-native voice. Calm, clear, and grounded."
        )

    # Mode-specific guidance
    lines.append("")
    if mode == "short_comment":
        lines.append(
            "Post mode: short_comment. Write ONE compact X post (around 25-40 words) "
            "that captures the key idea or shift."
        )
    elif mode == "mini_blurb":
        lines.append(
            "Post mode: mini_blurb. Write ONE slightly longer X post (around 50-80 words) "
            "that briefly walks through what changed and why it matters."
        )
    elif mode == "thread_4_6":
        lines.append(
            "Post mode: thread_4_6. Write a short thread of 4-6 tweets."
        )
        lines.append(
            "Each tweet should be on its own line, separated by a blank line. Do NOT "
            "number them. No 'Thread:' label."
        )
        lines.append(
            "The first tweet should be a hook. The rest unpack positioning, flows, and "
            "narrative in concrete terms."
        )
    else:
        lines.append(
            "Post mode: default. Write ONE concise, self-contained X post."
        )

    lines.append("")
    lines.append("Global constraints:")
    lines.append("- Do not mention price targets or guarantees.")
    lines.append("- Do not explicitly tell people to buy or sell.")
    lines.append("- Do not use hashtags, cashtags, emojis, or bullet lists.")
    lines.append("- Output only the tweet text (or thread tweets), nothing else.")

    return "\n".join(l for l in lines if l.strip())


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

    # Language hint: default to English, allow explicit language codes.
    target_lang = (lang or "en").strip() or "en"
    lang_line = ""
    if target_lang and target_lang != "en":
        lang_line = f"Target output language: '{target_lang}'. Write the final post entirely in this language."
    elif target_lang:
        lang_line = "Target output language: 'en'. Write the final post in clear, natural English."

    if lang_line:
        user_prompt = f"{lang_line}\n\n{user_prompt}"

    # Quality / length presets.
    q = (qmode or "balanced").strip().lower() or "balanced"
    if q not in {"fast", "balanced", "pro"}:
        q = "balanced"

    if req.post_mode == MarketPostMode.SHORT_CASUAL:
        if q == "fast":
            max_tokens = 180
            temperature = 0.7
        elif q == "pro":
            max_tokens = 260
            temperature = 0.6
        else:
            max_tokens = 220
            temperature = 0.65
    elif req.post_mode == MarketPostMode.MEDIUM_ANALYSIS:
        if q == "fast":
            max_tokens = 260
            temperature = 0.72
        elif q == "pro":
            max_tokens = 380
            temperature = 0.62
        else:
            max_tokens = 320
            temperature = 0.68
    else:  # THREAD_4_6
        if q == "fast":
            max_tokens = 420
            temperature = 0.7
        elif q == "pro":
            max_tokens = 720
            temperature = 0.65
        else:
            max_tokens = 540
            temperature = 0.68

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
                "quality_mode": q,
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
            "quality_mode": q,
        },
    }

