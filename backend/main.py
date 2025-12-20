from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from collections import Counter
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify

from utils import (
    clean_and_normalize_urls,
    fetch_tweet_data,
    TweetData,
    extract_ticker_and_percent_summary,
    random_user_agent,
)

# -------------------------
# Basic Flask app + logging
# -------------------------

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

# -------------------------
# SQLite for simple memory
# -------------------------

DB_PATH = os.environ.get("CROWNTALK_DB", "crowntalk.db")


def _ensure_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS comments_seen (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS comments_openers_seen (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                opener TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
    finally:
        conn.close()


_ensure_db()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def remember_comment(text: str) -> None:
    try:
        h = _hash_text(text)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO comments_seen(text_hash) VALUES (?)", (h,)
            )
            conn.commit()
    except Exception as e:
        logger.warning("remember_comment failed: %s", e)


def comment_seen(text: str) -> bool:
    try:
        h = _hash_text(text)
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT 1 FROM comments_seen WHERE text_hash = ? LIMIT 1", (h,)
            )
            return cur.fetchone() is not None
    except Exception as e:
        logger.warning("comment_seen failed: %s", e)
        return False


def remember_opener(opener: str) -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO comments_openers_seen(opener) VALUES (?)",
                (opener.lower().strip(),),
            )
            conn.commit()
    except Exception as e:
        logger.warning("remember_opener failed: %s", e)


def opener_seen(opener: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT 1 FROM comments_openers_seen WHERE opener = ? LIMIT 1",
                (opener.lower().strip(),),
            )
            return cur.fetchone() is not None
    except Exception as e:
        logger.warning("opener_seen failed: %s", e)
        return False


# -------------------------
# OpenAI / Groq / Gemini clients
# -------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()


def _has_openai() -> bool:
    return bool(OPENAI_API_KEY)


def _has_groq() -> bool:
    return bool(GROQ_API_KEY)


def _has_gemini() -> bool:
    return bool(GEMINI_API_KEY)


# --------------
# Text utilities
# --------------


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def strip_emoji(s: str) -> str:
    if not s:
        return s
    return re.sub(
        r"[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF]+", "", s, flags=re.UNICODE
    )


def strip_trailing_mentions(s: str) -> str:
    if not s:
        return s
    return re.sub(r"(?:\s*@\w+)+\s*$", "", s).strip()


def shorten_username(display_name: str) -> str:
    """
    Turn 'john.base.eth' -> 'john', 'Satoshi Nakamoto' -> 'Satoshi'
    """
    if not display_name:
        return ""
    name = display_name.strip()
    name = re.split(r"[|·•\-–—|]", name)[0].strip()
    if "." in name and " " not in name:
        name = name.split(".")[0]
    parts = name.split()
    if len(parts) > 0:
        return parts[0]
    return name


def is_gm_style_tweet(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    gm_words = [
        "gm",
        "good morning",
        "goodmorning",
        "good afternoon",
        "ga",
        "good evening",
        "ge",
        "good night",
        "gn",
    ]
    return any(w in t for w in gm_words)


def enforce_word_count_natural(
    line: str, min_words: int = 7, max_words: int = 26
) -> str:
    """
    Keep comments in a natural length range, avoid one-word or essay comments.
    Also:
    - strip leading/trailing quotes
    - remove emojis
    - allow '?' but avoid '.' at the very end
    """
    if not line:
        return ""

    line = strip_emoji(line)
    line = normalize_whitespace(line)

    # remove surrounding quotes
    line = line.strip(" \"'“”‘’")

    # kill trailing punctuation except '?'
    while len(line) > 0 and line[-1] in ".!,;:…":
        line = line[:-1].rstrip()

    # If line is just '?', it's useless
    if line == "?":
        return ""

    words = line.split()
    if len(words) < min_words:
        # allow shorter comments if they are still meaningful, but not 1-2 words
        if len(words) < 3:
            return ""
        # don't force-fill, just accept slightly short comments
        return line

    if len(words) > max_words:
        words = words[:max_words]
        line = " ".join(words)

    return line


GENERIC_PHRASES = [
    "this kind of breakdown is exactly what new folks need",
    "this kind of breakdown is what new folks need most",
    "this kind of breakdown is exactly what new folks need to see",
    "the nuance here hits harder than most people think",
    "the nuance here hits harder than people think",
    "the nuance here hits harder",
    "tells you everything you need to know",
    "says a lot without saying much",
    "worth saving this thread for later",
    "bookmarking this one fr",
    "bookmarking this one, fr",
    "bookmarking this one for later",
    "bookmark-worthy thread tbh",
    "bookmark-worthy thread ngl",
    "bookmark-worthy alpha ngl",
    "bookmark-worthy alpha tbh",
    "this is the kind of alpha you only get by actually building",
    "you only get takes like this by actually building",
    "you only get this perspective by actually building",
    "love how you frame this",
    "love how you broke this down",
    "love how you laid this out",
    "love how you're separating signal from noise here",
    "clean breakdown of the moving parts",
    "clean breakdown of how this all fits together",
    "clean breakdown of the different moving parts",
    "clean way to frame the moving pieces",
    "clean way to frame what’s happening",
    "clean way to frame the risk/reward",
    "clean way to frame the tradeoff",
    "clean way to frame the problem",
    "clean way to frame the opportunity",
    "clean way to frame what’s really happening",
    "this hits different for the people actually building",
    "this hits different for anyone actually building",
    "this hits different if you’ve been here a minute",
    "this hits different if you’ve been paying attention",
    "this hits different if you understand the game",
    "this hits different if you’ve watched a few cycles",
    "this hits different if you've watched a few cycles",
    "quietly one of the most important",
    "quietly one of the more important",
    "quietly one of the biggest",
    "quietly one of the cleanest",
    "quietly one of the most interesting",
    "quietly stacking",
    "everyone talks",
    "low-key bullish",
    "low key bullish",
    "lowkey bullish",
    "next real edge probably sits",
    "is exactly where the next real edge probably sits",
    "weekend, very few are actually shipping it",
    "meet like this is how you win long term",
    "is how you win long term",
    "this stack pans out in real environments",
]


def contains_generic_phrase(line: str) -> bool:
    low = line.lower()
    for phrase in GENERIC_PHRASES:
        if phrase in low:
            return True
    return False


BAD_OPENERS = [
    "low-key bullish",
    "low key bullish",
    "lowkey bullish",
    "everyone talks",
    "quietly stacking",
    "interesting take",
    "interesting how",
    "interesting seeing",
    "interesting to see",
    "interesting,",
    "interesting.",
    "big promises",
    "the angle on",
    "the nuance around",
    "still waiting to see how",
    "still waiting to see",
    "hard not to keep watching",
]


def _openers(text: str) -> str:
    low = text.strip().lower()
    for op in BAD_OPENERS:
        if low.startswith(op):
            # return the base opener phrase
            return op
    # also grab first 3 words as a soft opener
    words = low.split()
    if len(words) >= 3:
        return " ".join(words[:3])
    return ""


def tweet_keywords_for_scoring(tweet_text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9$#]+", tweet_text.lower())
    return set(toks)


def tweet_keywords_for_scoring(tweet_text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9$#]+", tweet_text.lower())
    return set(toks)


def _percent_patterns_from_tweet(tweet_text: str) -> List[str]:
    """
    Extract patterns like "20%" or "3.4%" from the tweet so we can enforce them in comments.
    """
    out: List[str] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", tweet_text):
        num = m.group(1)
        out.append(num)
    return out


def _apply_percent_fix(tweet_text: str, comment: str) -> str:
    if not tweet_text or not comment:
        return comment

    patterns = _percent_patterns_from_tweet(tweet_text)
    if not patterns:
        return comment

    fixed = comment

    for num in patterns:
        # 1) Fix simple "20" -> "20%"
        simple_pat = re.compile(
            rf"(?<![\d.]){re.escape(num)}(?![\d.%])", flags=re.IGNORECASE
        )
        fixed = simple_pat.sub(f"{num}%", fixed)

        # 2) Fix "3 4" -> "3.4%" when tweet had "3.4%"
        if "." in num:
            a, b = num.split(".", 1)
            split_pat = re.compile(
                rf"(?<!\d){re.escape(a)}\s+{re.escape(b)}(?!\d)", flags=re.IGNORECASE
            )
            fixed = split_pat.sub(f"{num}%", fixed)

    return fixed


CASHTAG_RE = re.compile(r"\$([A-Z][A-Z0-9]{1,15})\b")


def _cashtags_from_tweet(tweet_text: str) -> List[str]:
    return list({m.group(1) for m in CASHTAG_RE.finditer(tweet_text or "")})


def _apply_cashtag_fix(tweet_text: str, comment: str) -> str:
    """
    Ensure tickers in comments look like in tweet:
    - If tweet has $HLS and comment has bare "HLS", we upgrade to "$HLS"
    - If comment has "$$HLS", collapse to "$HLS"
    """
    if not tweet_text or not comment:
        return comment

    cashtags = _cashtags_from_tweet(tweet_text)
    fixed = comment

    # Collapse $$ABC -> $ABC
    fixed = re.sub(
        r"\${2,}([A-Z][A-Z0-9]{1,15})\b", lambda m: f"${m.group(1)}", fixed
    )

    for tag in cashtags:
        bare_pat = re.compile(rf"\b{re.escape(tag)}\b")
        if bare_pat.search(fixed) and f"${tag}" not in fixed:
            fixed = bare_pat.sub(f"${tag}", fixed)

    return fixed


def trigram_set(s: str) -> set[str]:
    s = re.sub(r"\s+", " ", s.strip().lower())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def trigram_overlap(a: str, b: str) -> float:
    A = trigram_set(a)
    B = trigram_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union


def trigram_overlap_bad(a: str, b: str, threshold: float = 0.85) -> bool:
    return trigram_overlap(a, b) >= threshold


def _random_delay_between(a: float, b: float) -> None:
    time.sleep(random.uniform(a, b))


# ----------------------
# Provider-style prompts
# ----------------------

SYSTEM_PROMPT = """You are CrownTALK, a Web3-native KOL and crypto degen.
You write SHORT, punchy, human style quotes as if replying to tweets.
You never explain like a chatbot. You write like a real X user who lives on-chain.

Rules:
- 1 or 2 short sentences max (ideally 8–22 words total)
- Crypto-native slang is fine (gm, ngl, anon, degen, L2, airdrop, CT, onchain, etc.)
- You react to the **core idea** of the tweet, not just restating it
- Sometimes you ask a sharp question, sometimes you make a bold statement
- Avoid generic filler like "great thread", "nice breakdown", "interesting", "thanks for sharing"
- Avoid emoji spam; 0–1 emoji max, only if it adds vibe
- Avoid hashtags
- Use correct cashtags with a leading "$" for tokens.
- Preserve key numbers (APY, %, caps, volumes) exactly as in the tweet when you mention them.
- No URLs in the reply.

Tone:
- Mix of curious builder, early degen, and thoughtful KOL
- Confident but not cringe
- You can be skeptical if something feels like pure hype with no execution
- You notice design, incentives, infra, and where the real edge is

Output:
Return exactly TWO separate reply lines, each on its own line.
No numbering, no bullets, no quotes, no extra commentary before or after.
"""


def _build_prompt(tweet_text: str, author: Optional[str]) -> str:
    base = f"Tweet from @{author or 'anon'}:\n{tweet_text.strip()}\n\nWrite 2 different human replies."
    return base


# -------------
# Model clients
# -------------


def _openai_two(tweet_text: str, author: Optional[str]) -> List[str]:
    if not _has_openai():
        return []

    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = _build_prompt(tweet_text, author)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=140,
        n=1,
    )
    text = resp.choices[0].message.content or ""
    lines = [normalize_whitespace(l) for l in text.split("\n") if l.strip()]
    return lines[:4]


def _groq_two(tweet_text: str, author: Optional[str]) -> List[str]:
    if not _has_groq():
        return []

    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    prompt = _build_prompt(tweet_text, author)

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.95,
        max_tokens=160,
        n=1,
    )
    text = resp.choices[0].message.content or ""
    lines = [normalize_whitespace(l) for l in text.split("\n") if l.strip()]
    return lines[:4]


def _gemini_two(tweet_text: str, author: Optional[str]) -> List[str]:
    if not _has_gemini():
        return []

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = SYSTEM_PROMPT + "\n\n" + _build_prompt(tweet_text, author)

    resp = model.generate_content(prompt)
    text = resp.text or ""
    lines = [normalize_whitespace(l) for l in text.split("\n") if l.strip()]
    return lines[:4]


# -----------------------
# Offline candidates
# -----------------------


class OfflineGenerator:
    """
    Simple template-based generator for when APIs fail or are disabled.
    Keeps some KOL-style voice but avoids repetition.
    """

    def __init__(self):
        self.rng = random.Random()

    #  --- topic buckets ---

    def _topic_buckets_markets(self) -> list[str]:
        return [
            "Risk/reward on {focus} is way better than people admit",
            "If {focus} holds, this whole setup flips fast",
            "Most are staring at price, few watch {focus}",
            "The {focus} structure tells you more than the chart ngl",
            "Narratives change quick once {focus} starts moving",
        ]

    def _topic_buckets_nft(self) -> list[str]:
        return [
            "Art aside, the {focus} meta is actually interesting here",
            "If {focus} sticks, this collection could age crazy well",
            "The way you talk about {focus} feels way more grown than usual CT",
            "Nice to see someone talk {focus} instead of pure floor price hopium",
        ]

    def _topic_buckets_giveaway(self) -> list[str]:
        return [
            "People sleep on the value of {focus} in these drops fr",
            "If {focus} is legit, this WL hits way different",
            "Tying {focus} back to actual community is how you keep it alive",
            "The {focus} angle makes this feel less like pure freebie farming",
        ]

    def _topic_buckets_ai(self) -> list[str]:
        return [
            "Real edge with AI is who controls {focus}, not just who ships a bot",
            "If {focus} compounds, a lot of current infra looks ancient fast",
            "{focus} is exactly where most teams underestimate the risk",
            "Most talk AI vibes, very few design around {focus} properly",
        ]

    def _topic_buckets_infra(self) -> list[str]:
        return [
            "{focus} infra is the boring part that quietly decides who wins",
            "If {focus} gets solved, a lot of current pain points disappear",
            "Builders watching {focus} closely are going to front-run the next cycle",
            "Most people ignore {focus} until it breaks in prod",
        ]

    def _topic_buckets_risk(self) -> list[str]:
        return [
            "Biggest risk here is still {focus}, even if nobody wants to say it",
            "If {focus} goes wrong, the whole story changes quick",
            "Everyone’s chasing upside, but {focus} is where survival actually sits",
            "{focus} risk is where grown-ups quietly do their homework",
        ]

    def _generic_bucket(self) -> list[str]:
        return [
            "The way you frame this is low-key useful for people actually building",
            "You’re connecting incentives and infra in a way most timelines miss",
            "You’re talking about the parts that decide who survives next cycle",
            "This is the type of convo CT needs more than another airdrop thread",
            "You’re clearly thinking past the usual farm and dump meta",
        ]

    def _topics_from_tweet(self, tweet_text: str) -> List[str]:
        low = tweet_text.lower()
        topics = []

        if any(w in low for w in ["nft", "collection", "mint", "pfp"]):
            topics.append("nft")
        if any(w in low for w in ["airdrop", "wl", "whitelist", "giveaway", "raffle"]):
            topics.append("giveaway")
        if any(w in low for w in ["ai", "agent", "llm", "model"]):
            topics.append("ai")
        if any(
            w in low
            for w in [
                "infra",
                "infrastructure",
                "rollup",
                "l2",
                "bridge",
                "module",
                "sdk",
                "builder",
            ]
        ):
            topics.append("infra")
        if any(
            w in low
            for w in ["risk", "volatility", "drawdown", "hedge", "depeg", "rug"]
        ):
            topics.append("risk")
        if any(
            w in low
            for w in [
                "yield",
                "apy",
                "apr",
                "farm",
                "liquidity",
                "volume",
                "market cap",
                "marketcap",
            ]
        ):
            topics.append("markets")

        if not topics:
            topics.append("generic")
        return topics

    def _bucket_for_topic(self, topic: str) -> list[str]:
        topic = topic.lower()
        if topic == "markets":
            return self._topic_buckets_markets()
        if topic == "nft":
            return self._topic_buckets_nft()
        if topic == "giveaway":
            return self._topic_buckets_giveaway()
        if topic == "ai":
            return self._topic_buckets_ai()
        if topic == "infra":
            return self._topic_buckets_infra()
        if topic == "risk":
            return self._topic_buckets_risk()
        return self._generic_bucket()

    def _sample_focus_from_tweet(self, tweet_text: str) -> str:
        """
        Try to pick a key phrase from the tweet to plug into templates: token, product,
        or concept.
        """
        low = tweet_text.lower()

        # Prioritize cashtags and tickers
        cashtags = re.findall(r"\$[a-z0-9_]+", low)
        if cashtags:
            return cashtags[0]

        # Look for obvious product / protocol names (capitalized words)
        words = re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", tweet_text)
        if words:
            return words[0]

        # Fallback: pick a mid-length noun-ish word
        tokens = re.findall(r"[a-z]{4,}", low)
        if tokens:
            return tokens[len(tokens) // 2]

        return "this"

    def _build_offline_comment(self, tweet_text: str) -> str:
        topics = self._topics_from_tweet(tweet_text)
        topic = self.rng.choice(topics)
        bucket = self._bucket_for_topic(topic)

        tmpl = self.rng.choice(bucket)
        focus = self._sample_focus_from_tweet(tweet_text)

        line = tmpl.format(focus=focus)
        line = enforce_word_count_natural(line)
        return line

    def generate_two(self, tweet_text: str, author: Optional[str]) -> List[Dict[str, str]]:
        """
        Generate two offline comments in the usual structure: {"lang": "en", "text": "..."}.
        """
        out: List[Dict[str, str]] = []
        seen: set[str] = set()

        for _ in range(8):
            line = self._build_offline_comment(tweet_text)
            if not line:
                continue
            low = line.lower()
            if low in seen:
                continue
            if contains_generic_phrase(low):
                continue
            seen.add(low)
            out.append({"lang": "en", "text": line})
            if len(out) >= 2:
                break

        if not out:
            # Super hard fallback, just echo something minimal
            base = normalize_whitespace(tweet_text)[:140]
            if base:
                out.append(
                    {
                        "lang": "en",
                        "text": enforce_word_count_natural(
                            f"Wild to see how fast this space moves around: {base}"
                        ),
                    }
                )

        return out[:2]


generator = OfflineGenerator()


def _available_providers():
    providers = []
    if _has_openai():
        providers.append(("openai", _openai_two))
    if _has_groq():
        providers.append(("groq", _groq_two))
    if _has_gemini():
        providers.append(("gemini", _gemini_two))
    return providers


def enforce_unique(candidates: List[str], tweet_text: str) -> List[str]:
    """
    Filter and rank comments for natural, non-botty feel.
    """
    tweet_keys = tweet_keywords_for_scoring(tweet_text)
    out: List[str] = []
    seen: set[str] = set()
    seen_openers_local: set[str] = set()

    for c in candidates:
        c = enforce_word_count_natural(c)
        if not c:
            continue

        low = c.lower().strip()

        # kill obvious generic phrases again (cheap check)
        if contains_generic_phrase(low):
            continue

        op = _openers(low)
        if op and op in seen_openers_local:
            continue

        if comment_seen(c):
            continue

        # Avoid heavy trigram overlap with already chosen ones
        if any(trigram_overlap_bad(c, existing) for existing in out):
            continue

        # scoring: prefer overlap with tweet keywords but not trivial echo
        toks = set(re.findall(r"[a-z0-9$#]+", low))
        overlap = len(toks & tweet_keys)

        # mild penalty for generic question-only comments
        is_question = c.endswith("?")
        score = overlap
        if is_question:
            score -= 0.2

        out.append((score, c, op))

    # sort: higher score first, then random jitter
    out.sort(key=lambda x: (x[0], random.random()), reverse=True)

    final: List[str] = []
    for score, text, op in out:
        line = text

        if contains_generic_phrase(line):
            continue

        final.append(line)
        if op:
            seen_openers_local.add(op)
            remember_opener(op)
        remember_comment(line)

        if len(final) >= 6:
            break

    return final


def _rescue_two(tweet_text: str) -> List[str]:
    base = normalize_whitespace(tweet_text)[:120]
    if not base:
        return [
            "Still trying to decide how I feel about this tbh",
            "Curious how this actually plays out once the hype fades",
        ]

    out = [
        enforce_word_count_natural(
            f"Hard not to keep watching this play out ngl: {base}"
        ),
        enforce_word_count_natural(
            f"Real question is how this behaves once the easy money leaves: {base}"
        ),
    ]
    return [x for x in out if x]


def _apply_greeting_to_first_comment(
    candidates: List[str], tweet_text: str, author_display: Optional[str]
) -> List[str]:
    """
    If tweet is a GM/GA/GE/GN style greeting, force first comment
    to greet the author by display name (shortened).
    """
    if not candidates:
        return candidates

    if not is_gm_style_tweet(tweet_text):
        return candidates

    name = shorten_username(author_display or "")
    if not name:
        name = "gm"

    first = candidates[0]
    base = first.lstrip()

    if base.lower().startswith(("gm ", "gn ", "ga ", "ge ", "good ")):
        return candidates

    prefix_opts = [
        f"gm {name}",
        f"gm {name},",
        f"gm {name} anon",
        f"gm {name} frens",
    ]
    prefix = random.choice(prefix_opts)
    rebuilt = f"{prefix} {base}"
    candidates[0] = enforce_word_count_natural(rebuilt) or rebuilt
    return candidates

def generate_two_comments_with_providers(
    tweet_text: str,
    author: Optional[str],
    handle: Optional[str],
    lang: Optional[str],
    url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid provider strategy with randomness:

    - For each request, randomize the order of enabled providers.
    - Try them in that random order, accumulating comments.
    - As soon as we have 2 solid comments, stop.
    - If all fail or give < 2, fall back to offline generator.
    """
    candidates: list[str] = []

    providers = _available_providers()
    random.shuffle(providers)

    for name, fn in providers:
        if len(candidates) >= 2:
            break
        try:
            got = fn(tweet_text, author)
            candidates = enforce_unique(candidates + got, tweet_text=tweet_text)
        except Exception as e:
            logger.warning("%s provider failed: %s", name, e)

    # If providers didn't give enough, extend with offline
    if len(candidates) < 2:
        try:
            offline = generator.generate_two(tweet_text, author)
            offline_texts = [item.get("text", "") for item in offline]
            candidates = enforce_unique(
                candidates + offline_texts, tweet_text=tweet_text
            )
        except Exception as e:
            logger.warning("offline generator failed: %s", e)

    # If still nothing, hard fallback to 2 simple offline lines
    if not candidates:
        raw = _rescue_two(tweet_text)
        candidates = enforce_unique(raw, tweet_text=tweet_text) or raw

    # apply numeric & ticker fixes
    fixed_candidates: List[str] = []
    for c in candidates:
        if not c:
            continue
        c2 = _apply_percent_fix(tweet_text, c)
        c2 = _apply_cashtag_fix(tweet_text, c2)
        fixed_candidates.append(c2)

    candidates = [c for c in fixed_candidates if c][:2]

    # Build normalized dicts
    out: List[Dict[str, Any]] = []
    for c in candidates:
        out.append({"lang": lang or "en", "text": c})

    # If tweet is a GM/GA/GE/GN style greeting,
    # force first comment to greet the author by display name.
    if out:
        texts_only = [item["text"] for item in out]
        texts_only = _apply_greeting_to_first_comment(
            texts_only, tweet_text, author
        )
        for i, t in enumerate(texts_only):
            out[i]["text"] = t

    # If somehow we still ended up with < 2 dicts, ask offline generator directly
    if len(out) < 2:
        try:
            extra_items = generator.generate_two(tweet_text, author or None)
            for item in extra_items:
                if len(out) >= 2:
                    break
                txt = (item.get("text") or "").strip()
                if txt:
                    txt = _apply_percent_fix(tweet_text, txt)
                    txt = _apply_cashtag_fix(tweet_text, txt)
                    out.append({"lang": item.get("lang") or lang or "en", "text": txt})
        except Exception as e:
            logger.exception("Total failure in provider cascade: %s", e)

    # Final hard cap: exactly 2
    return out[:2]


# ------------------------
# VXTwitter / Tweet helper
# ------------------------


def _canonical_x_url_from_tweet(original_url: str, t: TweetData) -> str:
    """
    Build a stable https://x.com/{handle}/status/{id} URL if we can,
    instead of /i/status or mobile variants.
    """
    if t.handle and t.tweet_id:
        return f"https://x.com/{t.handle}/status/{t.tweet_id}"
    return original_url


def _raw_x_url(original_url: str) -> str:
    """
    For display, we keep the original URL, but if it's vx or fxtwitter, show the x.com url.
    """
    try:
        parsed = urlparse(original_url)
    except Exception:
        return original_url

    host = parsed.netloc.lower()
    if any(
        s in host
        for s in ["vxtwitter.com", "fxtwitter.com", "twittpr.com", "fixupx.com"]
    ):
        return "https://x.com" + parsed.path
    if host.endswith("x.com") or host.endswith("twitter.com"):
        return "https://" + host + parsed.path
    return original_url


# ----------------------
# Comment normalization
# ----------------------


def _normalize_comment_items(
    items: List[Any],
    tweet_text: str,
    author: Optional[str],
    lang: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Take a list of raw comment items (strings or dicts) and
    return a clean [ {lang, text}, ... ] list of length <= 2.
    """
    normalized: List[Dict[str, Any]] = []

    if not items:
        try:
            offline_items = generator.generate_two(tweet_text, author or None)
            items = offline_items
        except Exception as e:
            logger.warning("offline in _normalize_comment_items failed: %s", e)
            items = []

    for it in items or []:
        if isinstance(it, dict):
            raw = str(it.get("text") or "").strip()
        else:
            raw = str(it or "").strip()
        if not raw:
            continue

        txt = enforce_word_count_natural(raw)
        if not txt:
            continue
        if contains_generic_phrase(txt):
            continue

        txt = _apply_percent_fix(tweet_text, txt)
        txt = _apply_cashtag_fix(tweet_text, txt)

        normalized.append(
            {
                "lang": (it.get("lang") if isinstance(it, dict) else lang) or "en",
                "text": txt,
            }
        )

    texts = [n["text"] for n in normalized]
    texts = enforce_unique(texts, tweet_text=tweet_text)
    texts = texts[:2]

    final: List[Dict[str, Any]] = []
    for t in texts:
        final.append({"lang": lang or "en", "text": t})

    if len(final) < 2:
        rescue = _rescue_two(tweet_text)
        for r in rescue:
            if len(final) >= 2:
                break
            txt = enforce_word_count_natural(r)
            if txt:
                final.append({"lang": lang or "en", "text": txt})

    if final:
        texts_only = [item["text"] for item in final]
        texts_only = _apply_greeting_to_first_comment(texts_only, tweet_text, author)
        for i, t in enumerate(texts_only):
            final[i]["text"] = t

    return final[:2]


# -------------
# Flask routes
# -------------


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/comment", methods=["POST", "OPTIONS"])
def comment_endpoint():
    if request.method == "OPTIONS":
        return add_cors_headers(jsonify({"ok": True}))

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        payload = {}

    raw_urls = payload.get("urls") or payload.get("url") or ""
    lang = payload.get("lang") or "en"

    urls = clean_and_normalize_urls(raw_urls)
    if not urls:
        return add_cors_headers(
            jsonify({"results": [], "failed": [], "error": "No valid URLs provided"})
        )

    results: List[Dict[str, Any]] = []
    failed: List[str] = []

    for url in urls:
        try:
            tweet = fetch_tweet_data(url)
            if not tweet or not tweet.text:
                failed.append(url)
                continue

            tweet_text = tweet.text
            author_display = tweet.author or ""
            handle = tweet.handle or ""

            display_url = _canonical_x_url_from_tweet(url, tweet)
            display_url = _raw_x_url(display_url)

            two = generate_two_comments_with_providers(
                tweet_text,
                author_display,
                handle,
                lang,
                url=display_url,
            )

            if not two:
                failed.append(url)
                continue

            results.append(
                {
                    "url": display_url,
                    "author": author_display,
                    "handle": handle,
                    "tweet_text": tweet_text,
                    "comments": two,
                }
            )

            _random_delay_between(0.6, 1.0)
        except Exception as e:
            logger.exception("Error generating comments for %s: %s", url, e)
            failed.append(url)

    return add_cors_headers(jsonify({"results": results, "failed": failed}))


@app.route("/reroll", methods=["POST", "OPTIONS"])
def reroll_endpoint():
    if request.method == "OPTIONS":
        return add_cors_headers(jsonify({"ok": True}))

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        payload = {}

    url = payload.get("url") or ""
    lang = payload.get("lang") or "en"

    if not url:
        return add_cors_headers(
            jsonify({"error": "Missing url", "comments": []}), 400
        )

    try:
        tweet = fetch_tweet_data(url)
        if not tweet or not tweet.text:
            return add_cors_headers(
                jsonify({"error": "Unable to fetch tweet", "comments": []}), 400
            )

        tweet_text = tweet.text
        author_display = tweet.author or ""
        handle = tweet.handle or ""

        display_url = _canonical_x_url_from_tweet(url, tweet)
        display_url = _raw_x_url(display_url)

        two = generate_two_comments_with_providers(
            tweet_text,
            author_display,
            handle,
            lang,
            url=display_url,
        )

        if not two:
            two = generator.generate_two(tweet_text, author_display)

        return add_cors_headers(
            jsonify(
                {
                    "url": display_url,
                    "author": author_display,
                    "handle": handle,
                    "tweet_text": tweet_text,
                    "comments": two,
                }
            )
        )
    except Exception as e:
        logger.exception("Error in reroll for %s: %s", url, e)
        return add_cors_headers(
            jsonify({"error": "Internal error", "comments": []}), 500
        )

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "CrownTALK backend"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)