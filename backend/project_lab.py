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
    return (
        "You write content for X (Twitter) about crypto and Web3 projects.\n"
        "Your goal is to write short, human, professional posts that a CT user can copy and lightly edit.\n"
        "No emojis. No hashtags.\n"
        "Avoid hype slang and overpromising.\n"
        "Use short, clear sentences and a realistic, grounded tone.\n"
    )


def _build_user_prompt(card: ProjectPostCard, req: ProjectPostRequest) -> str:
    meta_lines = [
        f"Project: {card['name']} ({card['ticker']})",
        f"Chain: {card['primary_chain'] or 'n/a'}",
        f"Category: {card['category'] or 'n/a'}",
        f"Stage: {card['stage'] or 'n/a'}",
    ]
    if card["one_line_pitch"]:
        meta_lines.append(f"One-line pitch: {card['one_line_pitch']}")
    meta_lines.append("")

    s = card["sections"]
    sections_lines: List[str] = []

    def add_block(label: str, key: str) -> None:
        val = s.get(key)
        if not val:
            return
        sections_lines.append(f"{label}:")
        sections_lines.append(str(val).strip())
        sections_lines.append("")

    add_block("Core narrative", "core_narrative")
    add_block("Product details", "product_details")
    add_block("Key metrics", "key_metrics")
    add_block("Risks and realism", "risks_realism")
    add_block("How to talk about it on CT", "ct_angles")

    hooks_short = s.get("hooks_short") or []
    hooks_thread = s.get("hooks_thread") or []
    thread_skeleton = s.get("thread_skeleton") or []

    if hooks_short:
        sections_lines.append("Short hook ideas (for single-tweet posts):")
        for h in hooks_short:
            sections_lines.append(f"- {h}")
        sections_lines.append("")

    if hooks_thread:
        sections_lines.append("Thread hook ideas (for 4–6 tweet threads):")
        for h in hooks_thread:
            sections_lines.append(f"- {h}")
        sections_lines.append("")

    if thread_skeleton:
        sections_lines.append("Recommended thread structure (4–6 tweets):")
        for step in thread_skeleton:
            sections_lines.append(f"- {step}")
        sections_lines.append("")

    dos_donts = s.get("dos_donts")
    if dos_donts:
        sections_lines.append("Do and don't guidelines:")
        sections_lines.append(str(dos_donts).strip())
        sections_lines.append("")

    sections_lines.append(
        "Use the ideas above as context and inspiration. Do NOT copy the bullet points verbatim."
    )

    prompt_lines: List[str] = []
    prompt_lines.append("\n".join(meta_lines).strip())
    prompt_lines.append("")
    prompt_lines.extend(sections_lines)

    # Mode-specific instructions
    prompt_lines.append("")
    prompt_lines.append(f"Requested post mode: {req.post_mode}")
    tone = req.tone or "auto"
    prompt_lines.append(f"Requested tone: {tone}")
    prompt_lines.append("")

    mode = req.post_mode

    if mode == "short_casual":
        prompt_lines.append(
            "Write exactly ONE tweet. Length: ~20–35 words. Tone: casual but still professional."
        )
        prompt_lines.append(
            "Use one short hook idea if helpful. Mention ONE key angle or benefit. No emojis, no hashtags."
        )
    elif mode == "medium_casual":
        prompt_lines.append(
            "Write exactly ONE tweet. Length: ~40–80 words. Tone: slightly relaxed, conversational."
        )
        prompt_lines.append(
            "Include: brief problem context, what the project does, and one concrete benefit or use case."
        )
    elif mode == "medium_professional":
        prompt_lines.append(
            "Write exactly ONE tweet. Length: ~40–80 words. Tone: more serious, professional CT tone."
        )
        prompt_lines.append(
            "Include: problem context, what the project does, 1–2 concrete features or benefits, and optionally a realistic mention of risks or tradeoffs."
        )
    elif mode == "long_detailed":
        prompt_lines.append(
            "Write exactly ONE tweet. Length: ~120–200 words. It should feel like a mini-explainer compressed into a single post."
        )
        prompt_lines.append(
            "Cover: problem, solution (the project), key features/mechanics, who it’s for, and 1–2 realistic risks or caveats."
        )
    elif mode == "thread_4_6":
        prompt_lines.append(
            "Write a numbered thread of 4–6 tweets. Use the recommended thread structure as the backbone."
        )
        prompt_lines.append(
            "Format each tweet as a numbered line like '1) ...', '2) ...'. Each tweet should be ~25–60 words."
        )
        prompt_lines.append(
            "Cover: problem/context, what the project does, how it works, who it’s for / how to use it, realistic risks/tradeoffs, and an optional closing or CTA."
        )
    else:
        # This should be guarded before calling, but keep a fallback.
        raise ProjectLabError("project_post_invalid_mode", f"Unsupported post_mode: {mode}")

    prompt_lines.append("")
    prompt_lines.append("General rules:")
    prompt_lines.append("- No emojis.")
    prompt_lines.append("- No hashtags.")
    prompt_lines.append('- No hype memes such as "wen moon", "WAGMI", or "ape in".')
    prompt_lines.append("- Keep claims realistic. Do not promise returns or guaranteed upside.")
    prompt_lines.append("- Prefer short, clean sentences. Avoid unnecessary jargon.")

    return "\n".join(prompt_lines).strip()


def _postprocess_text(text: str) -> str:
    """Apply global post-processing rules to a single post."""
    if not isinstance(text, str):
        text = str(text or "")
    out = text.strip()

    # Normalize whitespace
    out = re.sub(r"\s+", " ", out)

    # Remove emojis (very rough pass: strip most characters outside basic planes).
    out = "".join(ch for ch in out if ord(ch) <= 0xFFFF)

    # Drop hashtags entirely.
    out = re.sub(r"\s*#\w+", "", out)

    # Ensure final punctuation.
    if out and out[-1] not in ".!?":
        out = out + "."
    return out.strip()


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
    chat_fn: Callable[[List[Dict[str, str]]], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a project-level post or thread using the provided chat function.

    chat_fn is expected to take a list of messages and return an object with at
    least a 'content' field holding the LLM text.
    """
    mode = req.post_mode
    system = _build_system_prompt()
    user_prompt = _build_user_prompt(card, req)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = chat_fn(messages)
    except CrownTALKError:
        # Bubble up known errors unchanged
        raise
    except Exception as exc:
        raise ProjectLabError("llm_error", f"LLM call failed: {exc}", http_status=502)

    text = getattr(resp, "content", None)
    if text is None and isinstance(resp, dict):
        text = resp.get("content")
    if text is None:
        raise ProjectLabError("llm_empty", "Model returned empty content.", http_status=502)

    if mode == "thread_4_6":
        tweets = _split_thread(text)
        if not tweets:
            raise ProjectLabError("llm_empty", "Model returned empty thread.", http_status=502)
        return {
            "project_id": card["id"],
            "post_mode": mode,
            "language": lang,
            "tweets": tweets,
            "meta": {
                "quality_mode": qmode,
            },
        }

    processed = _postprocess_text(text)
    if not processed:
        raise ProjectLabError("llm_empty", "Model returned empty content.", http_status=502)

    return {
        "project_id": card["id"],
        "post_mode": mode,
        "language": lang,
        "text": processed,
        "meta": {
            "quality_mode": qmode,
        },
    }
