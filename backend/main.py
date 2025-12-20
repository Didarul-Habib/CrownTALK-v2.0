import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from utils import (
    CrownTALKError,
    fetch_tweet_data,
    clean_and_normalize_urls,
    style_fingerprint,
)

# -----------------------------------------------------------------------------
# Logging / Config
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

ENABLE_OPENAI = bool(OPENAI_API_KEY)
ENABLE_GROQ = bool(GROQ_API_KEY)
ENABLE_GEMINI = bool(GEMINI_API_KEY)
ENABLE_DEEPSEEK = bool(DEEPSEEK_API_KEY)
ENABLE_FIREWORKS = bool(FIREWORKS_API_KEY)
ENABLE_TOGETHER = bool(TOGETHER_API_KEY)
ENABLE_XAI = bool(XAI_API_KEY)

# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)


# -----------------------------------------------------------------------------
# Greeting Heuristic
# -----------------------------------------------------------------------------

class GreetingHeuristic:
    """Detect GM / GA / GE / GN tweets and return a normalized tag."""

    GREETING_PATTERNS = {
        "gm": re.compile(r"\b(gm|good\s+morning|morning)\b", re.IGNORECASE),
        "ga": re.compile(r"\b(ga|good\s+afternoon)\b", re.IGNORECASE),
        "ge": re.compile(r"\b(ge|good\s+evening)\b", re.IGNORECASE),
        "gn": re.compile(r"\b(gn|good\s+night|good\s+nite)\b", re.IGNORECASE),
    }

    @classmethod
    def detect(cls, text: str) -> Optional[str]:
        if not text:
            return None
        for tag, pat in cls.GREETING_PATTERNS.items():
            if pat.search(text):
                return tag
        return None


def _extract_first_name_from_display(author: Optional[str]) -> Optional[str]:
    """
    Try to get a human-looking first name from display name like:
    - "John.base.eth" -> "John"
    - "King Satoshi"  -> "King"
    """
    if not author:
        return None

    s = str(author).strip()
    # Remove emojis & weird unicode punctuation roughly
    s = re.sub(r"[^\w\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

    # If it looks like "john.base.eth" -> split on '.' and keep first segment
    if "." in s and " " not in s:
        s = s.split(".")[0]

    # Final: keep the first token
    parts = s.split()
    if not parts:
        return None

    first = parts[0].strip()
    if not first:
        return None

    # Basic sanity: avoid returning '0x...' as a "name"
    if first.lower().startswith("0x"):
        return None

    return first


def _greeting_prefix_from_tag(tag: Optional[str], author: Optional[str]) -> Optional[str]:
    """Convert greeting tag + author into a small prefix line."""
    if not tag:
        return None

    # Map tag -> phrase
    base = {
        "gm": "GM",
        "ga": "Good afternoon",
        "ge": "Good evening",
        "gn": "Good night",
    }.get(tag, "GM")

    name = _extract_first_name_from_display(author)
    if name:
        return f"{base} {name}"
    return base


def _apply_greeting_to_first_comment(
    comments: List[str],
    tweet_text: str,
    author: Optional[str],
) -> List[str]:
    """
    If tweet is a greeting (GM/GA/GE/GN),
    force the first comment to greet the author in a natural KOL way.
    """
    if not comments:
        return comments

    tag = GreetingHeuristic.detect(tweet_text)
    if not tag:
        return comments

    prefix = _greeting_prefix_from_tag(tag, author)
    if not prefix:
        return comments

    new_comments: List[str] = []
    for i, c in enumerate(comments):
        c = (c or "").strip()
        if not c:
            new_comments.append(c)
            continue
        if i == 0:
            # Don't double-greet if it's already starting with a greeting
            lowered = c.lower()
            if not lowered.startswith(("gm", "good morning", "good afternoon", "good evening", "good night", "gn", "ga", "ge")):
                c = f"{prefix} — {c}"
        new_comments.append(c)
    return new_comments


# -----------------------------------------------------------------------------
# Basic helpers for text post-processing
# -----------------------------------------------------------------------------

def _ensure_no_trailing_dot(text: str) -> str:
    """
    Drop a trailing '.' or '!' etc; keep '?' because user wants questions
    to keep their '?' when needed.
    """
    if not text:
        return text
    t = text.rstrip()
    if t.endswith("?"):
        return t
    t = re.sub(r"[.!…]+$", "", t).rstrip()
    return t


def _tweet_cashtags(tweet_text: str) -> List[str]:
    """
    Extract $TICKER tokens from the original tweet text so we can
    normalize style in generated comments.
    """
    if not tweet_text:
        return []
    return list(
        dict.fromkeys(re.findall(r"\$[A-Za-z][A-Za-z0-9]{1,9}", tweet_text))
    )


def _apply_cashtag_fix(
    comments: List[str],
    tweet_text: str,
) -> List[str]:
    """
    Make sure comments respect '$' prefix for tickers that appeared
    in the original tweet.

    - If tweet had '$HLS', comment shouldn't say just 'HLS'.
    - If comment has '$$HLS', collapse to '$HLS'.
    """
    cashtags = _tweet_cashtags(tweet_text)
    if not cashtags:
        return comments

    fixed: List[str] = []
    for c in comments:
        t = c or ""
        for tag in cashtags:
            core = tag.lstrip("$")
            canonical = f"${core}"

            # collapse '$$HLS'/'$$$HLS' -> '$HLS'
            t = re.sub(rf"\$+{re.escape(core)}\b", canonical, t)

            # bare 'HLS' (but not when already '$HLS')
            t = re.sub(
                rf"(?<!\$)\b{re.escape(core)}\b",
                canonical,
                t,
            )
        fixed.append(t)
    return fixed


def _extract_percent_values(tweet_text: str) -> List[str]:
    """
    Return list of percent numbers (without %) that appear in tweet.
    Example: "20% APY, 3.4% APR" -> ['20', '3.4']
    """
    if not tweet_text:
        return []
    percents = re.findall(r"\b(\d+(?:\.\d+)?)\s*%", tweet_text)
    return list(dict.fromkeys(percents))


def _apply_percent_fix(
    comments: List[str],
    tweet_text: str,
) -> List[str]:
    """
    If tweet has '20%' but comment wrote '20 APY', fix it to '20% APY'.

    We only adjust in obvious APY/APR/yield contexts and only for
    numbers that appeared with '%' in the tweet.
    """
    percents = _extract_percent_values(tweet_text)
    if not percents:
        return comments

    fixed: List[str] = []
    for c in comments:
        t = c or ""
        for value in percents:
            # '20 HLS APY' -> '20% HLS APY'
            pattern = re.compile(
                rf"\b{re.escape(value)}\b(?=(?:\s+\S+){{0,2}}\s+(apy|apr|yield)\b)",
                flags=re.IGNORECASE,
            )
            t = pattern.sub(f"{value}%", t)
        fixed.append(t)
    return fixed


# -----------------------------------------------------------------------------
# Offline comment generator
# -----------------------------------------------------------------------------

@dataclass
class OfflineContext:
    tweet_text: str
    author: Optional[str]
    style: str


class OfflineCommentGenerator:
    """
    Lightweight offline comment generator as a fallback and for diversity.
    It tries to read the tweet intent (markets, NFT, etc) and
    respond in KOL / degen-ish style without sounding like a template.
    """

    def __init__(self) -> None:
        self.rng = random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_two(
        self,
        tweet_text: str,
        author: Optional[str],
        handle: Optional[str],
        lang: Optional[str],
        url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ctx = OfflineContext(
            tweet_text=tweet_text,
            author=author,
            style=style_fingerprint(tweet_text),
        )
        lines: List[str] = []

        # Some randomness in how we react
        for _ in range(4):
            variant = self._one_comment(ctx)
            if variant:
                lines.append(variant)

        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for t in lines:
            t = (t or "").strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(_ensure_no_trailing_dot(t))

        out: List[Dict[str, Any]] = []
        for t in uniq[:2]:
            out.append({"lang": lang or "en", "text": t})
        return out

    # ------------------------------------------------------------------
    # Comment synthesis
    # ------------------------------------------------------------------

    def _one_comment(self, ctx: OfflineContext) -> str:
        text = ctx.tweet_text.lower()

        # micro-heuristics
        is_nft = " nft" in text or "collection" in text or "mint" in text
        is_giveaway = "giveaway" in text or "wl" in text or "whitelist" in text
        is_market = any(k in text for k in ["apy", "%", "yield", "volume", "market"])

        buckets: List[str] = []

        focus = self._extract_focus(ctx.tweet_text)

        if is_market:
            buckets.extend(self._topic_buckets_markets())
        if is_nft:
            buckets.extend(self._topic_buckets_nft())
        if is_giveaway:
            buckets.extend(self._topic_buckets_giveaway())

        buckets.extend(self._topic_buckets_generic())

        if not buckets:
            return ""

        template = self.rng.choice(buckets)
        out = template.format(focus=focus).strip()
        return out

    # ------------------------------------------------------------------
    # Focus extraction
    # ------------------------------------------------------------------

    def _extract_focus(self, text: str) -> str:
        """
        Try to pull a key noun/phrase to reference in the comment.
        """
        if not text:
            return "this"

        # prefer cashtags/project names
        m = re.search(r"\$[A-Za-z][A-Za-z0-9]{1,9}", text)
        if m:
            return m.group(0)

        # simple heuristic: last capitalized word
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        candidate = ""
        for tok in tokens:
            if tok[0].isupper():
                candidate = tok
        if candidate:
            return candidate

        return "this"

    # ------------------------------------------------------------------
    # Buckets for different tweet types
    # ------------------------------------------------------------------

    def _topic_buckets_markets(self) -> List[str]:
        return [
            "Risk/reward on {focus} feels way better than most are pricing in ngl",
            "If {focus} holds this zone, the whole setup can flip fast",
            "Everyone stares at price but very few watch how {focus} actually moves liquidity",
            "The structure around {focus} tells you more than any single green candle",
            "Narratives rotate quickly once {focus} starts leading flows instead of lagging them",
            "Quietly watching how {focus} trades into real volume before the crowd catches on",
            "Once {focus} gets proper liquidity, the real game starts for early degens",
        ]

    def _topic_buckets_nft(self) -> List[str]:
        return [
            "Art aside, the {focus} meta they’re building around this collection is actually interesting",
            "If the {focus} angle sticks, this ages way better than most quick-flip mints",
            "The way you frame {focus} here feels like someone who actually lives NFT markets",
            "Cool to see someone talk {focus} instead of pure floor-price hopium",
            "If {focus} stays aligned with real collectors, this set could surprise a lot of people",
        ]

    def _topic_buckets_giveaway(self) -> List[str]:
        return [
            "People always fade the value of {focus} in these drops until it’s too late",
            "If {focus} is legit, this WL ends up meaning more than just free entries",
            "{focus} tied back to real community skin in the game is how you build stickiness",
            "The {focus} angle makes this feel less like a random promo and more like a filter for real ones",
        ]

    def _topic_buckets_generic(self) -> List[str]:
        return [
            "You frame {focus} in a way most people scrolling past will totally miss",
            "If they execute on {focus}, the rest starts to look like a side quest",
            "Long term, {focus} is what decides who actually survives this cycle",
            "Most timelines ignore the {focus} part, but that’s where the real edge probably sits",
            "The conviction around {focus} here hits harder than people think at first glance",
            "Builders who understand {focus} tend to outlast the noise every time",
            "This take on {focus} feels like it’s coming from someone actually in the trenches",
            "You’re basically front-running the next narrative around {focus} with this thread",
        ]


offline_generator = OfflineCommentGenerator()

# -----------------------------------------------------------------------------
# Provider clients
# -----------------------------------------------------------------------------


def _call_openai_comment_model(tweet_text: str, author: Optional[str]) -> List[str]:
    if not ENABLE_OPENAI:
        return []

    try:
        import openai
    except Exception as exc:  # pragma: no cover
        logger.warning("OpenAI import failed: %s", exc)
        return []

    openai.api_key = OPENAI_API_KEY

    system_prompt = (
        "You are CrownTALK, a degen-savvy KOL commenting engine.\n"
        "Task: write 2 short, natural English comments reacting to the tweet.\n"
        "- Sound like an experienced Web3/KOL user, not a bot\n"
        "- Use modern crypto slang sparingly (ngl, anon, degen, etc.)\n"
        "- No emojis, no hashtags, no numbered lists\n"
        "- No quote of the tweet, just your reaction\n"
        "- 1 sentence per comment, no trailing period if you can avoid it\n"
        "- End with '?' only when you’re genuinely asking something\n"
    )

    user_prompt = f"Tweet by {author or 'anon'}:\n\"{tweet_text}\"\n\nReturn 2 separate comments."

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=160,
        )
    except Exception as exc:
        logger.warning("OpenAI request failed: %s", exc)
        return []

    text = resp["choices"][0]["message"]["content"]
    lines = _split_raw_llm_comments(text)
    return lines[:4]


def _call_groq_comment_model(tweet_text: str, author: Optional[str]) -> List[str]:
    if not ENABLE_GROQ:
        return []

    import groq  # type: ignore

    client = groq.Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "You are CrownTALK, a sharp Web3 degen commentator.\n"
        "Write 2 tight, human-sounding replies to the tweet.\n"
        "- No bullet points, no numbering\n"
        "- No emojis or hashtags\n"
        "- Sound like an engaged KOL who actually read the tweet\n"
        "- Vary your tone: sometimes curious, sometimes confident\n"
        "- Avoid repeating the same phrases across comments\n"
    )

    user_prompt = f"Tweet by {author or 'anon'}:\n\"{tweet_text}\"\n\nReturn 2 comments."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.95,
            max_tokens=200,
        )
    except Exception as exc:
        logger.warning("Groq request failed: %s", exc)
        return []

    text = chat_completion.choices[0].message.content or ""
    lines = _split_raw_llm_comments(text)
    return lines[:4]


def _call_gemini_comment_model(tweet_text: str, author: Optional[str]) -> List[str]:
    if not ENABLE_GEMINI:
        return []

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:
        logger.warning("Gemini import failed: %s", exc)
        return []

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = (
        "You are CrownTALK, a Web3-native commentator.\n"
        "Write 2 short comments reacting to the tweet below.\n"
        "Guidelines:\n"
        "- Write in natural, human English with light degen slang\n"
        "- No emojis, no hashtags, no bullet points\n"
        "- 1 sentence per comment\n"
        "- Avoid generic phrases like 'great project' or 'love this'\n"
        "- Keep it punchy, like you're replying directly on X\n\n"
        f"Tweet by {author or 'anon'}:\n\"{tweet_text}\"\n\nReturn 2 comments."
    )

    try:
        result = model.generate_content(prompt)
        text = result.text or ""
    except Exception as exc:
        logger.warning("Gemini request failed: %s", exc)
        return []

    lines = _split_raw_llm_comments(text)
    return lines[:4]


def _call_deepseek_comment_model(tweet_text: str, author: Optional[str]) -> List[str]:
    if not ENABLE_DEEPSEEK:
        return []

    try:
        import openai
    except Exception as exc:
        logger.warning("DeepSeek(openai-compatible) import failed: %s", exc)
        return []

    client = openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    system_prompt = (
        "You are CrownTALK, a degen crypto commentator.\n"
        "Write 2 different, short comments for the tweet below.\n"
        "- No emojis, no hashtags\n"
        "- 1 sentence each, KOL tone\n"
        "- Avoid sounding like a template\n"
    )

    user_prompt = f"Tweet by {author or 'anon'}:\n\"{tweet_text}\"\n\nReturn 2 comments."

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.95,
            max_tokens=200,
        )
    except Exception as exc:
        logger.warning("DeepSeek request failed: %s", exc)
        return []

    text = resp.choices[0].message.content or ""
    lines = _split_raw_llm_comments(text)
    return lines[:4]


def _call_fireworks_comment_model(tweet_text: str, author: Optional[str]) -> List[str]:
    if not ENABLE_FIREWORKS:
        return []

    try:
        import openai
    except Exception as exc:
        logger.warning("Fireworks(openai-compatible) import failed: %s", exc)
        return []

    client = openai.OpenAI(
        api_key=FIREWORKS_API_KEY,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    system_prompt = (
        "You are CrownTALK, a sharp Web3 KOL.\n"
        "Write 2 succinct comments reacting to this tweet.\n"
        "- Use crypto-Twitter tone\n"
        "- No emojis, hashtags or numbering\n"
        "- 1 sentence each\n"
    )

    user_prompt = f"Tweet by {author or 'anon'}:\n\"{tweet_text}\"\n\nReturn 2 comments."

    try:
        resp = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=220,
        )
    except Exception as exc:
        logger.warning("Fireworks request failed: %s", exc)
        return []

    text = resp.choices[0].message.content or ""
    lines = _split_raw_llm_comments(text)
    return lines[:4]


def _call_together_comment_model(tweet_text: str, author: Optional[str]) -> List[str]:
    if not ENABLE_TOGETHER:
        return []

    try:
        import openai
    except Exception as exc:
        logger.warning("Together(openai-compatible) import failed: %s", exc)
        return []

    client = openai.OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )

    system_prompt = (
        "You are CrownTALK, a Web3 commentator.\n"
        "Write 2 natural comments reacting to the tweet.\n"
        "- Short, punchy, no emojis\n"
        "- Crypto Twitter tone\n"
        "- 1 sentence each\n"
    )

    user_prompt = f"Tweet by {author or 'anon'}:\n\"{tweet_text}\"\n\nReturn 2 comments."

    try:
        resp = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.95,
            max_tokens=220,
        )
    except Exception as exc:
        logger.warning("Together request failed: %s", exc)
        return []

    text = resp.choices[0].message.content or ""
    lines = _split_raw_llm_comments(text)
    return lines[:4]


def _call_xai_comment_model(tweet_text: str, author: Optional[str]) -> List[str]:
    if not ENABLE_XAI:
        return []

    try:
        import openai
    except Exception as exc:
        logger.warning("xAI(openai-compatible) import failed: %s", exc)
        return []

    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    system_prompt = (
        "You are CrownTALK, a Web3 degen commentator.\n"
        "Write 2 natural comments reacting to the tweet.\n"
        "- Short, one sentence each\n"
        "- Crypto Twitter tone\n"
        "- No emojis, no hashtags, no lists\n"
    )

    user_prompt = f"Tweet by {author or 'anon'}:\n\"{tweet_text}\"\n\nReturn 2 comments."

    try:
        resp = client.chat.completions.create(
            model="grok-2-1212",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=220,
        )
    except Exception as exc:
        logger.warning("xAI request failed: %s", exc)
        return []

    text = resp.choices[0].message.content or ""
    lines = _split_raw_llm_comments(text)
    return lines[:4]


def _split_raw_llm_comments(text: str) -> List[str]:
    """
    LLMs might return:
    - bullet list
    - numbered list
    - plain text with newlines

    We normalize into a list of candidate comment strings.
    """
    if not text:
        return []

    raw = text.strip()

    # Split on newlines, strip bullets/numbers
    lines: List[str] = []
    for line in raw.splitlines():
        ln = line.strip()
        if not ln:
            continue
        ln = re.sub(r"^[\-\*\d\.\)]+\s*", "", ln)
        if ln:
            lines.append(ln)

    # If it's single long line, try splitting on '1.' '2.' patterns
    if len(lines) == 1:
        tmp = re.split(r"\s*\d+\.\s*", raw)
        tmp = [t.strip() for t in tmp if t.strip()]
        if len(tmp) > 1:
            lines = tmp

    # Final cleanup
    cleaned: List[str] = []
    for ln in lines:
        ln = ln.strip()
        ln = re.sub(r"\s+", " ", ln)
        if ln:
            cleaned.append(_ensure_no_trailing_dot(ln))

    return cleaned


# -----------------------------------------------------------------------------
# Provider orchestration
# -----------------------------------------------------------------------------


def enforce_unique(candidates: List[str], tweet_text: str) -> List[str]:
    """Lower-case dedupe + avoid exact tweet repetition."""
    seen = set()
    out: List[str] = []
    normalized_tweet = (tweet_text or "").strip().lower()

    for c in candidates:
        c = (c or "").strip()
        if not c:
            continue
        key = c.lower()
        if key == normalized_tweet:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(_ensure_no_trailing_dot(c))
    return out


def _available_providers() -> List[Tuple[str, Any]]:
    providers: List[Tuple[str, Any]] = []

    if ENABLE_OPENAI:
        providers.append(("openai", _call_openai_comment_model))
    if ENABLE_GROQ:
        providers.append(("groq", _call_groq_comment_model))
    if ENABLE_GEMINI:
        providers.append(("gemini", _call_gemini_comment_model))
    if ENABLE_DEEPSEEK:
        providers.append(("deepseek", _call_deepseek_comment_model))
    if ENABLE_FIREWORKS:
        providers.append(("fireworks", _call_fireworks_comment_model))
    if ENABLE_TOGETHER:
        providers.append(("together", _call_together_comment_model))
    if ENABLE_XAI:
        providers.append(("xai", _call_xai_comment_model))

    return providers


def generate_two_comments_with_providers(
    tweet_text: str,
    author: Optional[str],
    handle: Optional[str],
    lang: Optional[str],
    url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid provider strategy with randomness:

    - Randomize provider order per request.
    - Try them in that random order, collecting candidates.
    - Stop once we have 2 solid comments.
    - If still < 2, fall back to offline generator.
    """
    candidates: List[str] = []

    providers = _available_providers()
    random.shuffle(providers)

    # 1) Try online providers
    for name, fn in providers:
        if len(candidates) >= 2:
            break
        try:
            got = fn(tweet_text, author)
            candidates = enforce_unique(candidates + got, tweet_text=tweet_text)
        except Exception as exc:
            logger.warning("%s provider failed: %s", name, exc)

    # 2) If providers didn't give enough, extend with offline
    if len(candidates) < 2:
        try:
            offline_items = offline_generator.generate_two(
                tweet_text,
                author or None,
                handle,
                lang,
                url=url,
            )
            offline_lines = [item.get("text") or "" for item in offline_items]
            candidates = enforce_unique(candidates + offline_lines, tweet_text=tweet_text)
        except Exception as exc:
            logger.warning("offline generator failed: %s", exc)

    # 3) Final guardrail: still nothing? simple rescue
    if not candidates:
        candidates = _rescue_two(tweet_text)
        candidates = enforce_unique(candidates, tweet_text=tweet_text) or candidates

    # 4) Hard cap to exactly 2 strings
    candidates = [c for c in candidates if c][:2]

    # 5) Apply GM/GA/GE/GN greeting logic to first comment if applicable
    if candidates:
        candidates = _apply_greeting_to_first_comment(
            candidates,
            tweet_text,
            author,
        )

    # 6) Apply cashtag / percent normalization
    candidates = _apply_cashtag_fix(candidates, tweet_text)
    candidates = _apply_percent_fix(candidates, tweet_text)

    # 7) Wrap with language info
    out: List[Dict[str, Any]] = []
    for c in candidates:
        out.append({"lang": lang or "en", "text": c})

    return out[:2]


def _rescue_two(tweet_text: str) -> List[str]:
    """
    Very simple rescue comments when everything else fails.
    """
    tt = (tweet_text or "").strip()
    if not tt:
        return [
            "Curious to see where this goes anon",
            "Let’s see how this plays out on-chain",
        ]

    short = tt[:60]
    return [
        f"Interesting angle on this ngl — {short}",
        "Curious how this actually plays out once real volume shows up",
    ]


# -----------------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/comment", methods=["POST"])
def comment() -> Any:
    """
    Main entry: expects JSON with { urls: [..] } or plain text body of URLs.

    1. Clean & normalize URLs (strip extra text, dedupe, fix /i/status).
    2. Fetch tweet metadata for each URL.
    3. Generate 2 comments via provider cascade.
    4. Return comments + normalized URL.
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}

    raw_text = ""
    urls: List[str] = []

    if isinstance(payload, dict):
        if "urls" in payload and isinstance(payload["urls"], list):
            urls = [str(u) for u in payload["urls"]]
        elif "text" in payload:
            raw_text = str(payload["text"] or "")
        else:
            raw_text = ""
    elif isinstance(payload, list):
        urls = [str(u) for u in payload]
    else:
        raw_text = ""

    # if user pasted raw text with URLs, extract/normalize here
    if raw_text and not urls:
        urls = clean_and_normalize_urls(raw_text)

    urls = clean_and_normalize_urls("\n".join(urls))

    if not urls:
        raise CrownTALKError("No valid tweet URLs found in request")

    results: List[Dict[str, Any]] = []

    for url in urls:
        try:
            td = fetch_tweet_data(url)
        except CrownTALKError as exc:
            logger.warning("tweet fetch failed for %s: %s", url, exc)
            continue

        two = generate_two_comments_with_providers(
            tweet_text=td.text,
            author=td.author_name,
            handle=td.author_handle,
            lang=td.lang,
            url=td.canonical_url or td.url,
        )

        results.append(
            {
                "url": td.canonical_url or td.url,
                "author": td.author_name,
                "handle": td.author_handle,
                "lang": td.lang,
                "text": td.text,
                "comments": two,
            }
        )

    return jsonify({"results": results})


@app.errorhandler(CrownTALKError)
def handle_crowntalk_error(err: CrownTALKError) -> Any:
    logger.warning("CrownTALKError: %s", err)
    return jsonify({"error": str(err)}), 400


@app.errorhandler(Exception)
def handle_general_error(err: Exception) -> Any:
    logger.exception("Unhandled error: %s", err)
    return jsonify({"error": "internal_error"}), 500


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)