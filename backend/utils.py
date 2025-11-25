# utils.py
import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("crowntalk.utils")
logger.setLevel(logging.INFO)

# -------------------------------
# General helpers
# -------------------------------
_URL_RE = re.compile(r"^https?://", re.I)

def is_valid_url(url: str) -> bool:
    return bool(url and _URL_RE.match(url))

def safe_json(body: bytes) -> dict:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return {}

def log_exc(prefix: str, err: Exception):
    logger.exception("%s: %s", prefix, err)

# -------------------------------
# Provider availability / order
# -------------------------------
def provider_available(name: str) -> bool:
    """A provider is considered 'available' if its minimal API key is present."""
    name = (name or "").lower()
    if name == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if name in ("openai", "oai"):
        return bool(os.getenv("OPENAI_API_KEY"))
    if name in ("gemini", "google", "googleai"):
        return bool(os.getenv("GEMINI_API_KEY"))
    return False

def resolve_provider_order() -> List[str]:
    """
    Get desired order from PROVIDER_ORDER, but only keep available ones.
    Example: PROVIDER_ORDER="groq,openai,gemini"
    Fallback default order tries fast+cheap first.
    """
    env_order = os.getenv("PROVIDER_ORDER", "")
    if env_order.strip():
        candidates = [p.strip().lower() for p in env_order.split(",") if p.strip()]
    else:
        candidates = ["groq", "openai", "gemini"]

    # Keep only available
    ordered = [p for p in candidates if provider_available(p)]

    # If none configured, return empty -> caller will use offline fallback
    return ordered

# -------------------------------
# Provider clients (lazy init)
# -------------------------------
def _get_groq_client():
    """Return GROQ client or None if not configured."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception as e:
        log_exc("GROQ client init failed", e)
        return None

def _get_openai_client():
    """Return OpenAI client or None if not configured."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception as e:
        log_exc("OpenAI client init failed", e)
        return None

def _get_gemini_model():
    """Return Gemini GenerativeModel or None if not configured."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        # fast, inexpensive default
        return genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    except Exception as e:
        log_exc("Gemini model init failed", e)
        return None

# -------------------------------
# Prompt builder
# -------------------------------
def build_system_prompt() -> str:
    return (
        "You are CrownTALK, a social reply generator. "
        "Write short, humanlike, friendly comment variations for an X (Twitter) post URL. "
        "No hashtags, no emojis unless clearly appropriate. Avoid sounding generic. "
        "Return concise 1–2 sentence replies."
    )

def build_user_prompt(url: str, lang_hint: Optional[str] = None) -> str:
    base = f"Generate 2–3 different comment options that would fit under this X post:\nURL: {url}"
    if lang_hint and lang_hint.lower() != "en":
        base += f"\nPrimary language: {lang_hint}"
    base += "\nReturn only the raw comments, one per line."
    return base

# -------------------------------
# Single-call wrappers to each provider
# -------------------------------
def _call_groq(url: str, lang_hint: Optional[str]) -> List[Dict]:
    client = _get_groq_client()
    if not client:
        raise RuntimeError("GROQ not configured")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    system = build_system_prompt()
    user = build_user_prompt(url, lang_hint)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=200,
        n=1,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _split_lines_to_comments(text, lang_hint)

def _call_openai(url: str, lang_hint: Optional[str]) -> List[Dict]:
    client = _get_openai_client()
    if not client:
        raise RuntimeError("OpenAI not configured")
    # sensible light default; allow override via env
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system = build_system_prompt()
    user = build_user_prompt(url, lang_hint)

    # Using Chat Completions for broad SDK compatibility
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=220,
        n=1,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _split_lines_to_comments(text, lang_hint)

def _call_gemini(url: str, lang_hint: Optional[str]) -> List[Dict]:
    model = _get_gemini_model()
    if not model:
        raise RuntimeError("Gemini not configured")
    system = build_system_prompt()
    user = build_user_prompt(url, lang_hint)
    # Gemini has no native 'system' field — prepend system to content
    prompt = system + "\n\n" + user

    resp = model.generate_content(prompt)
    # SDK returns either .text or candidates; stick to .text for simplicity
    text = (getattr(resp, "text", None) or "").strip()
    if not text and getattr(resp, "candidates", None):
        text = (resp.candidates[0].content.parts[0].text or "").strip()
    return _split_lines_to_comments(text, lang_hint)

# -------------------------------
# Parse LLM output → comments[]
# -------------------------------
def _split_lines_to_comments(text: str, lang_hint: Optional[str]) -> List[Dict]:
    """
    Convert model's multi-line output into: [{text, lang}]
    Lines like "1) ..." or "- ..." are normalized.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines()]
    out: List[Dict] = []
    for ln in lines:
        # strip common list markers
        ln = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln)
        if ln:
            out.append({"text": ln, "lang": (lang_hint or "en")})
    return out[:6]  # cap to a reasonable number

# -------------------------------
# Offline fallback
# -------------------------------
def offline_fallback_comments(url: str, lang_hint: Optional[str]) -> List[Dict]:
    """
    Very small deterministic generator used when all providers fail.
    Keeps tone friendly & short to mimic the live model.
    """
    base = re.sub(r"https?://", "", url).split("/")[0] if url else "this"
    samples = [
        f"Love the take — curious to see where this goes 👀",
        f"Well said. {base} never disappoints.",
        "This is such a clean breakdown — thanks for sharing!",
        "Solid points here. Bookmarked to revisit.",
        "Yep, this hits the nail on the head.",
    ]
    lang = (lang_hint or "en")
    return [{"text": s, "lang": lang} for s in samples[:3]]

# -------------------------------
# Orchestrator: try providers in order
# -------------------------------
def generate_for_url(url: str, lang_hint: Optional[str] = None) -> Tuple[List[Dict], str]:
    """
    Try providers in resolved order; return (comments, provider_used).
    If none succeed, return offline fallback with provider_used="offline".
    """
    if not is_valid_url(url):
        raise ValueError("Invalid URL")

    order = resolve_provider_order()
    errors: List[str] = []

    for name in order:
        try:
            if name == "groq":
                comments = _call_groq(url, lang_hint)
            elif name in ("openai", "oai"):
                comments = _call_openai(url, lang_hint)
            elif name in ("gemini", "google", "googleai"):
                comments = _call_gemini(url, lang_hint)
            else:
                continue

            if comments:
                return comments, name
        except Exception as e:
            errors.append(f"{name}: {e}")
            # brief backoff before trying next provider
            time.sleep(0.15)

    # If we reach here, all providers failed → offline fallback
    logger.warning("All providers failed for %s; errors=%s", url, "; ".join(errors))
    return offline_fallback_comments(url, lang_hint), "offline"

# -------------------------------
# Health / warm helpers
# -------------------------------
def health_payload() -> dict:
    order = resolve_provider_order()
    return {
        "ok": True,
        "providers_configured": order,
        "env": {
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
        },
    }
