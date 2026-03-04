"""
Project Post Lab support module.

This module is intentionally separate from main.py to keep the main file smaller.
It is responsible for:
- Loading PROJECT_POST_CARD_V1 text files from a directory.
- Exposing an in-memory catalog of projects.
- Providing a helper to generate project-level posts/threads using an injected chat function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict

import re
import textwrap


from schemas import ProjectPostRequest, ProjectPostMode
from utils import CrownTALKError


class ProjectLabError(Exception):
    """Lightweight error type for Project Post Lab operations."""

    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class ProjectPostSections(TypedDict, total=False):
    core_narrative: str
    product_details: str
    key_metrics: str
    risks_realism: str
    ct_angles: str
    hooks_short: List[str]
    hooks_thread: List[str]
    thread_skeleton: List[str]
    dos_donts: str


class ProjectPostCard(TypedDict):
    id: str
    slug: str
    name: str
    ticker: str
    primary_chain: str
    category: str
    stage: str
    one_line_pitch: str
    summary: str
    raw_text: str
    sections: ProjectPostSections


def _split_sections(raw: str) -> Tuple[Dict[str, str], str]:
    """
    Split a PROJECT_POST_CARD_V1 file into header key/values and section blocks.

    The expected format is:

        PROJECT_POST_CARD_V1
        ====================

        KEY: VALUE
        KEY: VALUE

        SECTION_NAME:
        content...

    We treat everything after the header key/value block as free-form sections.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    header: Dict[str, str] = {}
    body_lines: List[str] = []

    # Skip the first line + underline if present
    i = 0
    if i < len(lines) and lines[i].strip().upper().startswith("PROJECT_POST_CARD_V1"):
        i += 1
        if i < len(lines) and set(lines[i].strip()) == {"="}:
            i += 1

    # Consume blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    header_pattern = re.compile(r"^(?P<key>[A-Z0-9_]+):\s*(?P<value>.*)$")

    # Parse header key/value lines until we hit a blank line or non-matching line
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            break
        m = header_pattern.match(line.strip())
        if not m:
            # Not a header line; treat as body from here
            break
        key = m.group("key").strip().upper()
        value = m.group("value").strip()
        header[key] = value
        i += 1

    # Remaining lines are treated as free-form section text
    body_lines = lines[i:]
    body = "\n".join(body_lines).strip("\n")

    return header, body


_SECTION_KEYS = {
    "ONE_LINE_PITCH": "one_line_pitch",
    "CORE_NARRATIVE": "core_narrative",
    "PRODUCT_DETAILS": "product_details",
    "KEY_METRICS": "key_metrics",
    "RISKS_AND_REALISM": "risks_realism",
    "RISKS_AND_REALISM:": "risks_realism",  # tolerate accidental colon in header
    "CT_ANGLES_HOW_TO_TALK": "ct_angles",
    "HOOK_IDEAS_SHORT": "hooks_short",
    "HOOK_IDEAS_THREAD": "hooks_thread",
    "THREAD_SKELETON_4_6_TWEETS": "thread_skeleton",
    "DOS_AND_DONTS": "dos_donts",
}

SCORE_UPDATE_SUPPORTED_PROJECT_IDS = {
    "WALLCHAIN",
}



def _parse_body_sections(body: str) -> ProjectPostSections:
    """
    Parse the free-form body into named sections.

    Sections are delimited by lines like:

        SECTION_NAME:

    Everything until the next SECTION_NAME: belongs to that section.
    """
    sections: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    current_lines: List[str] = []

    header_re = re.compile(r"^([A-Z0-9_]+):\s*$")

    for raw_line in body.split("\n"):
        line = raw_line.rstrip("\n")
        m = header_re.match(line.strip())
        if m:
            # Flush previous section
            if current_key is not None:
                sections[current_key] = current_lines
            current_key = m.group(1).upper()
            current_lines = []
        else:
            if current_key is None:
                # Ignore leading text before first section header
                continue
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = current_lines

    parsed: ProjectPostSections = {}

    def _normalize_block(lines: List[str]) -> str:
        block = "\n".join(lines).strip("\n")
        # Dedent to avoid over-indentation in prompts
        return textwrap.dedent(block).strip()

    for raw_key, lines in sections.items():
        key = raw_key.upper()
        mapped = _SECTION_KEYS.get(key)
        if not mapped:
            continue
        if mapped in ("hooks_short", "hooks_thread", "thread_skeleton"):
            bullets: List[str] = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                if ln.startswith(("-", "*")):
                    ln = ln.lstrip("-*").strip()
                bullets.append(ln)
            parsed[mapped] = bullets  # type: ignore[assignment]
        else:
            parsed[mapped] = _normalize_block(lines)  # type: ignore[assignment]

    return parsed


def load_project_posts(directory: str) -> Dict[str, ProjectPostCard]:
    """
    Load all PROJECT_POST_CARD_V1 files from a directory.

    Returns a mapping from PROJECT_ID (uppercased) to ProjectPostCard.
    """
    import os

    cards: Dict[str, ProjectPostCard] = {}

    if not os.path.isdir(directory):
        return cards

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".txt"):
            continue
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            continue

        header, body = _split_sections(raw)
        sections = _parse_body_sections(body)

        project_id = (header.get("PROJECT_ID") or "").strip() or filename.rsplit(".", 1)[0]
        slug = (header.get("SLUG") or project_id).strip()
        name = (header.get("NAME") or slug).strip()
        ticker = (header.get("TICKER") or "").strip()
        primary_chain = (header.get("PRIMARY_CHAIN") or "").strip()
        category = (header.get("CATEGORY") or "").strip()
        stage = (header.get("STAGE") or "").strip()

        one_line_pitch = sections.get("one_line_pitch") or header.get("ONE_LINE_PITCH") or ""
        one_line_pitch = (one_line_pitch or "").strip()

        # Build a short summary for catalog display
        ct_angles = sections.get("ct_angles") or ""
        summary_parts: List[str] = []
        if one_line_pitch:
            summary_parts.append(one_line_pitch)
        if ct_angles:
            first_line = str(ct_angles).splitlines()[0].strip()
            if first_line:
                summary_parts.append(first_line)
        summary = " ".join(summary_parts).strip()

        card: ProjectPostCard = {
            "id": project_id.upper(),
            "slug": slug,
            "name": name,
            "ticker": ticker,
            "primary_chain": primary_chain,
            "category": category,
            "stage": stage,
            "one_line_pitch": one_line_pitch,
            "summary": summary or one_line_pitch,
            "raw_text": raw,
            "sections": sections,
        }

        cards[card["id"]] = card

    return cards



def _build_system_prompt() -> str:
    """System prompt for Project Post Lab.

    This persona writes sober, CT-native project posts grounded in the research card.
    """

    return (
        "You are the writer behind a crypto-native account that posts thoughtful,\n"
        "sober commentary about projects. You write like a human, not a bot.\n\n"
        "Core traits:\n"
        "- You are specific, grounded in the provided project card and sections.\n"
        "- You avoid hype, memes, and cringe 'shill' language.\n"
        "- You NEVER add hashtags or cashtags, and you NEVER add emojis.\n"
        "- You do not add disclaimers like 'NFA', 'DYOR', or 'not financial advice'.\n"
        "- You do not add labels such as 'Thread:', 'Summary:', or 'Takeaways:'.\n"
        "- You do not speak about the author being an AI model.\n\n"
        "Output contract:\n"
        "- For standard modes you output exactly ONE X post string. No bullet list,\n"
        "  no numbering, no extra commentary.\n"
        "- For thread_4_6 mode you output 4-6 separate tweets separated by blank lines.\n"
        "  Each tweet is plain text with no numeric prefixes (no '1)', '2)', '3)').\n"
        "- For score_update mode you output a short multi-line update (3-4 lines);\n"
        "  each line is a complete sentence or clause.\n"
        "- All outputs must be written in the requested target language.\n"
    )



def _build_user_prompt(card: ProjectPostCard, req: ProjectPostRequest) -> str:
    """Build user prompt for project post generation."""

    sections = card.get("sections", {}) or {}

    def get_section(key: str) -> str:
        value = sections.get(key)
        if not value:
            return ""
        if isinstance(value, list):
            return " ".join(str(v).strip() for v in value if str(v).strip())
        return str(value).strip()

    name = card.get("name") or card.get("id") or ""
    ticker = card.get("ticker") or ""
    primary_chain = card.get("primary_chain") or ""
    category = card.get("category") or ""
    stage = card.get("stage") or ""

    one_liner = get_section("one_line_pitch")
    core_narrative = get_section("core_narrative")
    product_details = get_section("product_details")
    ct_angles = get_section("ct_angles")
    key_metrics = get_section("key_metrics")
    risks_realism = get_section("risks_realism")

    # Angle mapping
    angle = (req.angle or "").strip().lower() or "balanced"
    angle_hints = {
        "balanced": (
            "Balanced overview: connect narrative, product, and real metrics into "
            "one coherent CT-ready post. Make it feel grounded but still optimistic."
        ),
        "how_to_use": (
            "How-to angle: show how a real user or team would actually use this "
            "product today. Emphasise concrete actions and UX, not just vision."
        ),
        "narrative": (
            "Narrative angle: explain where this project sits inside the current "
            "crypto meta and why it matters for the broader story."
        ),
        "risk": (
            "Risk-first angle: highlight execution, market, and structural risks "
            "while still acknowledging why people care about the project."
        ),
        "builder": (
            "Builder angle: focus on what is being shipped, what is live, how the "
            "product and team are iterating, and why that matters."
        ),
    }
    angle_explainer = angle_hints.get(angle, angle_hints["balanced"])

    mode = req.post_mode.value if isinstance(req.post_mode, ProjectPostMode) else str(req.post_mode)
    tone_hint = (req.tone or "").strip().lower()

    # Meta + card context
    lines: list[str] = []

    lines.append("You are generating an X post about the following project.")
    lines.append("")
    lines.append("Project meta:")
    if name:
        lines.append(f"- Name: {name}")
    if ticker:
        lines.append(f"- Ticker: {ticker}")
    if primary_chain:
        lines.append(f"- Primary chain: {primary_chain}")
    if category:
        lines.append(f"- Category: {category}")
    if stage:
        lines.append(f"- Stage: {stage}")

    if one_liner:
        lines.append("")
        lines.append("One line pitch:")
        lines.append(one_liner)

    if core_narrative:
        lines.append("")
        lines.append("Core narrative:")
        lines.append(core_narrative)

    if product_details:
        lines.append("")
        lines.append("Product / UX details:")
        lines.append(product_details)

    if key_metrics:
        lines.append("")
        lines.append("Key metrics and traction signals:")
        lines.append(key_metrics)

    if ct_angles:
        lines.append("")
        lines.append("How CT should talk about it (angles):")
        lines.append(ct_angles)

    if risks_realism:
        lines.append("")
        lines.append("Risks and realism:")
        lines.append(risks_realism)

    lines.append("")
    lines.append("Angle focus:")
    lines.append(angle_explainer)

    # Tone guidance
    tone_lines: list[str] = []
    if tone_hint == "professional":
        tone_lines.append(
            "Tone: more professional and measured, like a thoughtful fund / "
            "research account posting for CT."
        )
    elif tone_hint == "casual":
        tone_lines.append(
            "Tone: slightly more casual and conversational, but still sober and "
            "non-cringe. Avoid meme slang and over-the-top hype."
        )
    else:
        tone_lines.append(
            "Tone: credible CT-native voice. Calm, clear, and concrete. No memes, "
            "no cringe, no fake hype."
        )

    # Mode-specific instructions
    mode_lines: list[str] = []
    if mode == "short_casual":
        mode_lines.append(
            "Post mode: short_casual. Write ONE short X post (around 20-35 words)."
        )
        mode_lines.append(
            "Focus on a single sharp idea or hook rather than covering everything."
        )
    elif mode in {"medium_casual", "medium_professional"}:
        mode_lines.append(
            "Post mode: medium. Write ONE medium-length X post (roughly 45-80 words)."
        )
        mode_lines.append(
            "You may connect 2-3 key ideas (narrative + product + metric), but keep "
            "it tight enough to feel like a single tweet."
        )
    elif mode == "long_detailed":
        mode_lines.append(
            "Post mode: long_detailed. Write ONE longer X post (roughly 90-150 words)."
        )
        mode_lines.append(
            "You can unpack more nuance, but still keep it readable in a single tweet."
        )
    elif mode == "thread_4_6":
        mode_lines.append(
            "Post mode: thread_4_6. Write a short thread of 4-6 tweets."
        )
        mode_lines.append(
            "Each tweet should be on its own line, and tweets must be separated by a "
            "blank line. Do NOT number them, and do NOT include 'Thread:' or similar."
        )
        mode_lines.append(
            "The first tweet should work as a strong hook. Subsequent tweets deepen "
            "the narrative with concrete details and metrics."
        )
    elif mode == "score_update":
        mode_lines.append(
            "Post mode: score_update. Write a GM-style multi-line update (3-4 lines) "
            "about an X Score move, targeting serious CT readers."
        )
        mode_lines.append(
            "Each line should be a complete, compact sentence or clause. Keep it calm "
            "and credible, not hypey."
        )

    # Optional score update payload details
    if mode == "score_update" and req.score_update:
        su = req.score_update
        lines.append("")
        lines.append("Score update context:")
        lines.append("- Metric: X Score (0-100 scale).")
        lines.append(f"- Previous value: {su.from_value}.")
        lines.append(f"- New value: {su.to_value}.")
        if su.period_label:
            lines.append(f"- Period: {su.period_label}.")
        lines.append(
            "- Treat this as a scoreboard for traction and execution, not a price signal."
        )
        lines.append(
            "- Do NOT mention price, pumps, or 'moon'. Do NOT tell people to buy or sell."
        )

    lines.append("")
    lines.append("Global style rules:")
    lines.append("- Do not use hashtags or cashtags.")
    lines.append("- Do not use emojis.")
    lines.append("- Do not use bullet lists or numbered lists in the final output.")
    lines.append("- Do not add disclaimers (NFA, DYOR, etc.).")
    lines.append(
        "- Output only the final X post text (or thread tweets), nothing else."
    )

    lines.append("")
    lines.extend(tone_lines)
    lines.extend(mode_lines)

    return "\n".join(l for l in lines if l.strip())



def normalize_project_post_text(text: str) -> str:
    """Normalize a single project post or tweet into a clean, CT-ready line."""
    if text is None:
        return ""

    t = str(text).strip()
    if not t:
        return ""

    # Strip wrapping quotes if the whole string is quoted
    if (
        (t.startswith('"') and t.endswith('"'))
        or (t.startswith("“") and t.endswith("”"))
        or (t.startswith("'") and t.endswith("'"))
    ) and len(t) >= 2:
        t = t[1:-1].strip()

    # Trim stray leading/trailing quote characters
    while t and t[0] in {'"', "“", "’", "'"}:
        t = t[1:].lstrip()
    while t and t[-1] in {'"', "”", "’", "'"}:
        t = t[:-1].rstrip()

    # Remove standalone hashtags but keep the words
    words = t.split()
    cleaned_words: list[str] = []
    for w in words:
        if w.startswith("#"):
            # drop the hash but keep the token content if any
            if len(w) > 1:
                cleaned_words.append(w[1:])
            continue
        cleaned_words.append(w)
    t = " ".join(cleaned_words)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Ensure a terminal punctuation mark for single-line posts
    if t and t[-1] not in ".!?":
        t += "."

    return t


def _postprocess_score_update(text: str) -> str:
    """Post-process a multi-line score_update message.

    We keep line breaks but normalise each non-empty line.
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


def _postprocess_text(text: str) -> str:
    """Apply global post-processing rules to a single project post."""
    return normalize_project_post_text(text)


def _split_thread(raw: str) -> List[str]:
    """
    Split a model output into individual tweets for thread mode.

    We prefer lines that start with '1)', '2)', etc., but fall back to
    splitting on newlines.
    """
    if not isinstance(raw, str):
        raw = str(raw or "")

    text = raw.strip()
    if not text:
        return []

    # Prefer numbered lines
    numbered_lines = []
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

    # Post-process each tweet
    cleaned = [_postprocess_text(t) for t in tweets if t.strip()]
    # Enforce 4–6 tweets if possible
    if len(cleaned) > 6:
        cleaned = cleaned[:6]
    return cleaned



def generate_project_post(
    card: ProjectPostCard,
    req: ProjectPostRequest,
    lang: str,
    qmode: str,
    chat_fn,
) -> dict:
    """Generate a project post or thread based on a project card and request."""

    mode = req.post_mode.value if isinstance(req.post_mode, ProjectPostMode) else str(req.post_mode)
    mode = mode or "medium_casual"

    # Score update is only allowed for specific projects.
    if mode == "score_update":
        project_label = (card.get("id") or card.get("slug") or card.get("name") or "").upper()
        if not any(key in project_label for key in SCORE_UPDATE_SUPPORTED_PROJECT_IDS):
            raise ProjectLabError(
                "score_update_unsupported_project",
                "Score update posts are only supported for specific projects (currently WALLCHAIN).",
            )
        if not req.score_update:
            raise ProjectLabError(
                "score_update_missing_payload",
                "score_update payload is required when post_mode == 'score_update'.",
            )

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(card, req)

    target_lang = (lang or "en").strip() or "en"
    if target_lang == "en":
        lang_line = "Write the output in English."
    else:
        lang_line = f"Write the output in {target_lang}."
    user_prompt = f"{lang_line}\n\n{user_prompt}"

    quality = (qmode or "balanced").strip().lower()
    max_tokens: int
    temperature: float

    if mode == "short_casual":
        max_tokens = 96 if quality == "fast" else 128
        temperature = 0.7 if quality == "fast" else 0.75
    elif mode in {"medium_casual", "medium_professional"}:
        max_tokens = 160 if quality == "fast" else 220
        temperature = 0.7 if quality == "fast" else 0.72
    elif mode == "long_detailed":
        max_tokens = 260 if quality == "fast" else 320
        temperature = 0.68 if quality == "fast" else 0.7
    elif mode == "thread_4_6":
        max_tokens = 360 if quality == "fast" else 420
        temperature = 0.72 if quality == "fast" else 0.74
    elif mode == "score_update":
        # Short multi-line GM-style update
        max_tokens = 140 if quality == "fast" else 180
        temperature = 0.65 if quality == "fast" else 0.68
    else:
        max_tokens = 200
        temperature = 0.7

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = chat_fn(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except CrownTALKError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ProjectLabError("llm_error", f"Upstream LLM error: {exc!r}") from exc

    text: str | None = None
    # Groq/OpenAI-compatible response shape
    if hasattr(resp, "choices"):
        choice = resp.choices[0]
        if hasattr(choice, "message"):
            text = choice.message.content or ""
    elif isinstance(resp, dict):
        try:
            text = resp["choices"][0]["message"]["content"]
        except Exception:
            text = None

    if not text or not str(text).strip():
        raise ProjectLabError("empty_response", "Model returned empty text for project post.")

    # Mode-specific post-processing
    if mode == "thread_4_6":
        tweets = _split_thread(text)
        if not tweets:
            raise ProjectLabError("thread_parse_error", "Could not parse thread into tweets.")
        return {
            "project_id": card.get("id"),
            "post_mode": mode,
            "language": target_lang,
            "kind": "thread_4_6",
            "tweets": tweets,
            "meta": {
                "quality_mode": quality,
            },
        }

    if mode == "score_update":
        processed = _postprocess_score_update(text)
        if not processed:
            raise ProjectLabError(
                "postprocess_error",
                "Post-processing produced empty score_update text.",
            )
        return {
            "project_id": card.get("id"),
            "post_mode": mode,
            "language": target_lang,
            "kind": "single",
            "text": processed,
            "meta": {
                "quality_mode": quality,
                "mode": mode,
            },
        }

    processed = _postprocess_text(text)
    if not processed:
        raise ProjectLabError("postprocess_error", "Post-processing produced empty text.")

    return {
        "project_id": card.get("id"),
        "post_mode": mode,
        "language": target_lang,
        "kind": "single",
        "text": processed,
        "meta": {
            "quality_mode": quality,
        },
    }
