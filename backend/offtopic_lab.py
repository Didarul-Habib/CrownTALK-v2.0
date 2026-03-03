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
from utils import CrownTALKError


class OfftopicLabError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def _postprocess_text(text: str) -> str:
    """Shared post-processing with market/project posts."""
    if not isinstance(text, str):
        text = str(text or "")
    out = text.strip()

    out = re.sub(r"\s+", " ", out)
    out = "".join(ch for ch in out if ord(ch) <= 0xFFFF)
    out = re.sub(r"\s*#\w+", "", out)
    if out and out[-1] not in ".!?":
        out = out + "."
    banned = ["WAGMI", "wen moon", "ape in"]
    for phrase in banned:
        out = re.sub(phrase, "", out, flags=re.IGNORECASE)
    return out.strip()


def _build_system_prompt() -> str:
    return textwrap.dedent(
        """        You write short, human, professional posts for X (Twitter) that are *not*
        tied to a single project or token.

        These are time-of-day greetings and random CT-adjacent thoughts.

        Rules:
        - No emojis.
        - No hashtags.
        - No hype slang (no "wen moon", "WAGMI", "ape in", etc.).
        - Short, clear sentences.
        - Never promise or imply guaranteed returns.
        - Keep everything grounded and realistic.
        """
    ).strip()


def _kind_label(kind: OfftopicKind) -> str:
    if kind == OfftopicKind.GM_MORNING:
        return "good morning"
    if kind == OfftopicKind.NOON:
        return "midday"
    if kind == OfftopicKind.AFTERNOON:
        return "afternoon"
    if kind == OfftopicKind.EVENING:
        return "evening"
    if kind == OfftopicKind.GN_NIGHT:
        return "good night"
    return "random"


def _build_user_prompt(req: OfftopicPostRequest) -> str:
    kind_label = _kind_label(req.kind)
    tone = (req.tone or "").strip().lower() or "casual"

    base_lines: List[str] = []
    base_lines.append(f"Write an off-topic CT post for {kind_label}.")
    base_lines.append("It should feel like a real X user posting, not a brand account.")
    base_lines.append("Keep it grounded in builder / trader reality: focus, execution, risk management, learning.")
    base_lines.append("")
    base_lines.append(f"Requested tone: {tone}.")
    base_lines.append("No emojis, no hashtags, no degen slang.")
    base_lines.append("")

    if req.kind == OfftopicKind.GM_MORNING:
        base_lines.append("Angle suggestions: starting the day, showing up, building consistently.")
    elif req.kind == OfftopicKind.NOON:
        base_lines.append("Angle suggestions: mid-day check-in, progress vs distraction, staying focused.")
    elif req.kind == OfftopicKind.AFTERNOON:
        base_lines.append("Angle suggestions: fatigue vs discipline, finishing the day properly.")
    elif req.kind == OfftopicKind.EVENING:
        base_lines.append("Angle suggestions: reflection on the day, lessons learned, planning tomorrow.")
    elif req.kind == OfftopicKind.GN_NIGHT:
        base_lines.append("Angle suggestions: winding down, logging off, and quietly setting up tomorrow.")
    else:
        base_lines.append("Angle suggestions: thoughtful, slightly contrarian, but not edgy or toxic.")

    if req.post_mode == "short":
        base_lines.append("Mode: SHORT (one tweet, ~15–30 words).")
        base_lines.append("Keep it punchy, one core idea.")
    else:
        base_lines.append("Mode: SEMI_MID (one tweet, ~35–70 words).")
        base_lines.append("Allow a bit more context and nuance, but keep it readable.")

    base_lines.append("Avoid giving explicit trading calls. No 'buy' / 'sell' / targets.")
    return "\n".join(base_lines).strip()


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

