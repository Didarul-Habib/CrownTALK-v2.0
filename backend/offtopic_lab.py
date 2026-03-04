"""
Off-topic / general CT post generator.

Used for:
- GM / GN / time-of-day greetings with light CT context.
- Random crypto-adjacent thoughts that are not tied to a specific project or asset.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import random
import re
import textwrap
import time

from schemas import OfftopicPostRequest, OfftopicKind
from project_lab import normalize_project_post_text
from utils import CrownTALKError


class OfftopicLabError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status




def _postprocess_text(text: str) -> str:
    """Post-process off-topic posts.

    We keep line breaks but normalise each non-empty line using the shared project
    normaliser so CT style is consistent.
    """
    if not text:
        return ""

    lines: list[str] = []
    for raw_line in str(text).splitlines():
        if not raw_line.strip():
            continue
        norm = normalize_project_post_text(raw_line)
        if norm:
            lines.append(norm)
    return "\n".join(lines)


def _build_system_prompt() -> str:
    """System prompt for Off-topic Post Lab.

    Persona: CT-native GM/GN + mindset poster, but sober and non-cringe.
    """

    return (
        "You write off-topic posts (GM, GN, mindset, general life/builder notes) for crypto Twitter (CT).\n"
        "You sound like a real person who trades or builds in the space, not a guru or bot.\n\n"
        "Core traits:\n"
        "- You are grounded, specific, and not overly motivational.\n"
        "- You avoid cringe hustle memes, 'grindset', and fake positivity.\n"
        "- You NEVER add hashtags or cashtags, and you NEVER add emojis.\n"
        "- You do not add disclaimers like 'NFA' or 'DYOR'.\n"
        "- You do not speak about being an AI model.\n\n"
        "Output contract:\n"
        "- For all kinds, you output a small set of lines that together read as one post.\n"
        "- Lines are separated by newlines, not bullets and not numbering.\n"
        "- For GM/GN, default to 2-4 short lines.\n"
        "- Output only the post text, nothing else.\n"
    )


def _build_user_prompt(req: OfftopicPostRequest) -> str:
    """Build user prompt for off-topic GM/GN and mindset posts."""

    kind = req.kind.value if isinstance(req.kind, OfftopicKind) else str(req.kind)
    tone_hint = (req.tone or "").strip().lower()
    topic = (req.topic or "").strip()
    language = (req.language or "").strip() or "en"

    lines: list[str] = []

    if kind in {"gm", "gn"}:
        when = "morning" if kind == "gm" else "night"
        lines.append(f"You are writing a {kind.upper()} post for crypto Twitter ({when}).")
    else:
        lines.append("You are writing an off-topic / mindset style post for CT.")

    if topic:
        lines.append("")
        lines.append("Context / topic for this post:")
        lines.append(topic)

    lines.append("")
    lines.append("High-level goals:")
    if kind == "gm":
        lines.append("- Help CT start the day with a clear, grounded frame.")
        lines.append("- Acknowledge the grind of trading/building without glorifying burnout.")
    elif kind == "gn":
        lines.append("- Help CT close the day with reflection and decompression.")
        lines.append("- Acknowledge both wins and losses without drama.")
    else:
        lines.append("- Share a short reflection that would feel natural on a CT timeline.")
        lines.append("- Make it feel honest and specific, not generic self-help.")

    # Tone guidance
    lines.append("")
    if tone_hint == "professional":
        lines.append(
            "Tone: more measured and low-key. Think calm, grounded builder or fund "
            "voice talking to peers."
        )
    elif tone_hint == "casual":
        lines.append(
            "Tone: conversational and human, but still non-cringe and free of forced memes."
        )
    else:
        lines.append(
            "Tone: credible CT-native voice. Calm, clear, and grounded."
        )

    # Kind-specific structure
    lines.append("")
    if kind == "gm":
        lines.append(
            "Structure: write 2-4 short lines. Line 1 sets the GM and frame. "
            "Later lines add one or two concrete thoughts about how to approach the day."
        )
    elif kind == "gn":
        lines.append(
            "Structure: write 2-4 short lines. Line 1 sets the GN and mood. "
            "Later lines reflect on what matters over a longer arc than today's PnL."
        )
    else:
        lines.append(
            "Structure: write 2-4 short lines that together read as a single post. "
            "Keep each line tight – no long paragraphs."
        )

    lines.append("")
    lines.append("Global constraints:")
    lines.append("- Do not use hashtags or cashtags.")
    lines.append("- Do not use emojis.")
    lines.append("- Do not use bullet lists or numbered lists.")
    lines.append("- Do not add disclaimers (NFA, DYOR, etc.).")
    lines.append("- Output only the final multi-line post text, nothing else.")

    lines.append("")
    if language == "en":
        lines.append("Write the post in English.")
    else:
        lines.append(f"Write the post in {language}.")

    return "\n".join(l for l in lines if l.strip())


def generate_offtopic_post(
    req: OfftopicPostRequest,
    lang: str,
    qmode: str,
    chat_fn: Callable[..., Any],
) -> Dict[str, Any]:
    """Generate an off-topic / random CT-style post."""
    system = _build_system_prompt()
    user_prompt = _build_user_prompt(req)

    # Language hint: default to clean English, allow explicit language codes.
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

    if req.post_mode == "short":
        if q == "fast":
            max_tokens = 160
            temperature = 0.78
        elif q == "pro":
            max_tokens = 260
            temperature = 0.7
        else:
            max_tokens = 200
            temperature = 0.75
    else:  # semi_mid
        if q == "fast":
            max_tokens = 220
            temperature = 0.78
        elif q == "pro":
            max_tokens = 380
            temperature = 0.7
        else:
            max_tokens = 320
            temperature = 0.75

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = chat_fn(messages=messages, max_tokens=max_tokens, temperature=temperature, n=1, model=None)
    except CrownTALKError:
        raise
    except Exception as exc:
        raise OfftopicLabError("llm_error", f"Upstream model error: {exc}", http_status=502) from exc

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
        raise OfftopicLabError("llm_empty", "Model returned empty content.", http_status=502)

    processed = _postprocess_text(text)
    if not processed:
        raise OfftopicLabError("llm_empty", "Model returned empty content.", http_status=502)

    return {
        "kind": req.kind.value,
        "post_mode": req.post_mode,
        "language": lang,
        "text": processed,
        "meta": {
            "quality_mode": q,
        },
    }

