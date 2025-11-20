# main.py
from __future__ import annotations

from flask import Flask, request, jsonify
import threading
import time
import os
import re
import random
import sqlite3
import hashlib
import logging
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

# Use your existing utils (unchanged)
from utils import CrownTALKError, fetch_tweet_data, clean_and_normalize_urls

# ------------------------------------------------------------------------------
# APP / CONFIG
# ------------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

PORT = int(os.environ.get("PORT", "10000"))
DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "")
BATCH_SIZE = 2
KEEP_ALIVE_INTERVAL = 600
MAX_URLS_PER_REQUEST = 8

# ------------------------------------------------------------------------------
# KEEPALIVE (Render free)
# ------------------------------------------------------------------------------
def keep_alive() -> None:
    if not BACKEND_PUBLIC_URL:
        return
    while True:
        try:
            import requests
            requests.get(f"{BACKEND_PUBLIC_URL}/", timeout=5)
        except Exception:
            pass
        time.sleep(KEEP_ALIVE_INTERVAL)

# ------------------------------------------------------------------------------
# DB INIT (safe across workers)
# ------------------------------------------------------------------------------
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)


def _locked_init(fn):
    if not _HAS_FCNTL:
        return fn()
    lock_path = "/tmp/crowntalk.db.lock"
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _do_init() -> None:
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                lang TEXT,
                text TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_comments_url ON comments(url);

            CREATE TABLE IF NOT EXISTS comments_seen(
                hash TEXT PRIMARY KEY,
                created_at INTEGER
            );

            -- OTP pattern guards
            CREATE TABLE IF NOT EXISTS comments_openers_seen(
                opener TEXT PRIMARY KEY,
                created_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS comments_ngrams_seen(
                ngram TEXT PRIMARY KEY,
                created_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS comments_templates_seen(
                thash TEXT PRIMARY KEY,
                created_at INTEGER
            );
            """
        )


def init_db() -> None:
    def _safe():
        try:
            _do_init()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(1.0)
                _do_init()
            else:
                raise

    _locked_init(_safe) if _HAS_FCNTL else _safe()

# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------
def now_ts() -> int:
    return int(time.time())


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        m = re.search(
            r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/([^/]+)/status/",
            url,
            re.I,
        )
        return m.group(1) if m else None
    except Exception:
        return None

# ------------------------------------------------------------------------------
# LIGHT MEMORY / OTP GUARDS
# ------------------------------------------------------------------------------
def _normalize_for_memory(text: str) -> str:
    t = normalize_ws(text).lower()
    t = re.sub(r"[^\w\s']+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def comment_seen(text: str) -> bool:
    norm = _normalize_for_memory(text)
    if not norm:
        return False
    h = sha256(norm)
    try:
        with get_conn() as c:
            return (
                c.execute("SELECT 1 FROM comments_seen WHERE hash=? LIMIT 1", (h,)).fetchone()
                is not None
            )
    except Exception:
        return False


def remember_comment(text: str, url: str = "", lang: Optional[str] = None) -> None:
    try:
        norm = _normalize_for_memory(text)
        if not norm:
            return
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_seen(hash, created_at) VALUES(?,?)",
                (sha256(norm), now_ts()),
            )
            c.execute("INSERT INTO comments(url, lang, text) VALUES (?,?,?)", (url, lang, text))
    except Exception:
        pass
    try:
        remember_ngrams(text)
        remember_opener(_openers(text))
    except Exception:
        pass


def _openers(text: str) -> str:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return " ".join(w[:3])


def _trigrams(text: str) -> List[str]:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [" ".join(w[i : i + 3]) for i in range(len(w) - 2)]


def opener_seen(opener: str) -> bool:
    try:
        with get_conn() as c:
            return (
                c.execute(
                    "SELECT 1 FROM comments_openers_seen WHERE opener=? LIMIT 1", (opener,)
                ).fetchone()
                is not None
            )
    except Exception:
        return False


def remember_opener(opener: str) -> None:
    try:
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_openers_seen(opener, created_at) VALUES (?,?)",
                (opener, now_ts()),
            )
    except Exception:
        pass


def trigram_overlap_bad(text: str, threshold: int = 2) -> bool:
    grams = _trigrams(text)
    if not grams:
        return False
    hits = 0
    try:
        with get_conn() as c:
            for g in grams:
                if c.execute(
                    "SELECT 1 FROM comments_ngrams_seen WHERE ngram=? LIMIT 1", (g,)
                ).fetchone():
                    hits += 1
                    if hits >= threshold:
                        return True
    except Exception:
        return False
    return False


def remember_ngrams(text: str) -> None:
    grams = _trigrams(text)
    if not grams:
        return
    try:
        with get_conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO comments_ngrams_seen(ngram, created_at) VALUES (?,?)",
                [(g, now_ts()) for g in grams],
            )
    except Exception:
        pass


def template_burned(tmpl: str) -> bool:
    thash = sha256(tmpl)
    try:
        with get_conn() as c:
            return (
                c.execute(
                    "SELECT 1 FROM comments_templates_seen WHERE thash=? LIMIT 1", (thash,)
                ).fetchone()
                is not None
            )
    except Exception:
        return False


def remember_template(tmpl: str) -> None:
    try:
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_templates_seen(thash, created_at) VALUES (?,?)",
                (sha256(tmpl), now_ts()),
            )
    except Exception:
        pass


def _word_trigrams(s: str) -> set:
    w = re.findall(r"[A-Za-z0-9']+", s.lower())
    return set(" ".join(w[i : i + 3]) for i in range(max(0, len(w) - 2)))


def too_similar_to_recent(text: str, threshold: float = 0.62, sample: int = 300) -> bool:
    """Jaccard(word-3grams) vs last N comments."""
    try:
        with get_conn() as c:
            rows = c.execute(
                "SELECT text FROM comments ORDER BY id DESC LIMIT ?", (sample,)
            ).fetchall()
    except Exception:
        return False
    here = _word_trigrams(text)
    if not here:
        return False
    for (t,) in rows:
        there = _word_trigrams(t)
        if not there:
            continue
        inter = len(here & there)
        uni = len(here | there)
        if uni and (inter / uni) >= threshold:
            return True
    return False

# ------------------------------------------------------------------------------
# CORS + HEALTH
# ------------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ------------------------------------------------------------------------------
# TOPIC / KEYWORDS
# ------------------------------------------------------------------------------
EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "to",
    "in",
    "on",
    "of",
    "for",
    "with",
    "at",
    "from",
    "by",
    "about",
    "as",
    "into",
    "like",
    "through",
    "after",
    "over",
    "between",
    "out",
    "against",
    "during",
    "without",
    "before",
    "under",
    "around",
    "among",
    "is",
    "are",
    "be",
    "am",
    "was",
    "were",
    "it",
    "its",
    "that",
    "this",
    "so",
    "very",
    "really",
}

AI_BLOCKLIST = {
    "amazing",
    "awesome",
    "incredible",
    "empowering",
    "game changer",
    "game-changing",
    "transformative",
    "paradigm shift",
    "as an ai",
    "as a language model",
    "in conclusion",
    "in summary",
    "furthermore",
    "moreover",
    "navigate this landscape",
    "ever-evolving landscape",
    "leverage this insight",
    "cutting edge",
    "state of the art",
    "unprecedented",
    "unleash",
    "harness the power",
    "embark on this journey",
    "revolutionize",
    "disruptive",
    "bestie",
    "like and retweet",
    "thoughts?",
    "agree?",
    "who's with me",
    "drop your thoughts",
    "smash that like button",
    "link in bio",
    "in case you missed it",
    "i think",
    "i believe",
    "great point",
    "just saying",
    "according to",
    "to be honest",
    "actually",
    "literally",
    "personally i think",
    "my take",
    "as someone who",
    "at the end of the day",
    "moving forward",
    "synergy",
    "circle back",
    "bandwidth",
    "double down",
    "let that sink in",
    "on so many levels",
    "tbh",
    "this resonates",
    "food for thought",
    "hit different",
}

STARTER_BLOCKLIST = {
    "yeah this",
    "honestly this",
    "kind of",
    "nice to",
    "hard to",
    "feels like",
    "this is",
    "short line",
    "funny how",
    "appreciate that",
    "interested to",
    "curious where",
    "nice to see",
    "chill sober",
    "good reminder",
    "yeah that",
    "good to see the boring",
}

EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", re.UNICODE
)


def detect_topic(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("gm ", "gn ", "good morning", "good night")):
        return "greeting"
    if any(k in t for k in ("airdrop", "whitelist", "wl spot", "mint is live")):
        return "giveaway"
    if any(
        k in t
        for k in (
            "chart",
            "support",
            "resistance",
            "ath",
            "price target",
            "%",
            "market cap",
            "mc",
        )
    ):
        return "chart"
    if any(k in t for k in ("bug", "issue", "broken", "down again", "wtf", "why is", "tired of")):
        return "complaint"
    if any(
        k in t
        for k in ("announcing", "announcement", "we're live", "we are live", "launching", "we shipped")
    ):
        return "announcement"
    if any(k in t for k in ("meme", "shitpost", "ratioed", "memeing")) or "lol" in t:
        return "meme"
    if "🧵" in text or len(text) > 220:
        return "thread"
    if len(text) < 80:
        return "one_liner"
    return "generic"


def is_crypto_tweet(text: str) -> bool:
    t = text.lower()
    crypto_keywords = [
        "crypto",
        "defi",
        "nft",
        "airdrop",
        "token",
        "coin",
        "chain",
        "l1",
        "l2",
        "staking",
        "yield",
        "dex",
        "cex",
        "onchain",
        "on-chain",
        "gas fees",
        "btc",
        "eth",
        "sol",
        "arb",
        "layer two",
        "mainnet",
    ]
    return any(k in t for k in crypto_keywords) or bool(re.search(r"\$\w{2,8}", text))


def extract_keywords(text: str) -> list[str]:
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"[@#]\S+", "", cleaned)
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", cleaned)
    if not tokens:
        return []
    filtered = [t for t in tokens if t.lower() not in EN_STOPWORDS and len(t) > 2]
    if not filtered:
        filtered = tokens
    counts = Counter([t.lower() for t in filtered])
    sorted_tokens = sorted(filtered, key=lambda w: (-counts[w.lower()], -len(w)))
    seen, out = set(), []
    for w in sorted_tokens:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(w)
    return out[:10]


def pick_focus_token(tokens: List[str]) -> Optional[str]:
    if not tokens:
        return None
    upperish = [t for t in tokens if t.isupper() or t[0].isupper()]
    return random.choice(upperish) if upperish else random.choice(tokens)

# ------------------------------------------------------------------------------
# CONTEXT PROFILE
# ------------------------------------------------------------------------------
def build_context_profile(
    raw_text: str, url: Optional[str] = None, tweet_author: Optional[str] = None, handle: Optional[str] = None
) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    projects, keywords, numbers = set(), set(), set()
    is_question = "?" in text

    projects |= set(re.findall(r"#([A-Za-z0-9_]+)", text))
    projects |= set(re.findall(r"\$[A-Za-z]{2,6}", text))
    projects |= set(re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b", text))
    projects |= set(re.findall(r"\b[A-Z]{3,}\b", text))
    for m in re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", text):
        numbers.add(m)
    for m in re.findall(r"(?<!\.)\s([A-Z][a-z]{2,})", text):
        keywords.add(m)

    if url and not handle:
        try:
            p = urlparse(url)
            segs = [s for s in p.path.split("/") if s]
            if segs:
                handle = segs[0]
        except Exception:
            pass

    script = "latn"
    blocks = [
        ("bn", r"[\u0980-\u09FF]"),
        ("hi", r"[\u0900-\u097F]"),
        ("ar", r"[\u0600-\u06FF]"),
        ("ta", r"[\u0B80-\u0BFF]"),
        ("te", r"[\u0C00-\u0C7F]"),
        ("ur", r"[\u0600-\u06FF]"),
    ]
    text_no_urls = re.sub(r"https?://\S+", "", text)
    total_letters = len(re.findall(r"[^\W\d_]", text_no_urls, flags=re.UNICODE))
    for code, pat in blocks:
        cnt = len(re.findall(pat, text_no_urls))
        if total_letters and cnt / max(1, total_letters) >= 0.25:
            script = code
            break

    return {
        "author_name": (tweet_author or "").strip() or None,
        "handle": (handle or "").strip() or None,
        "projects": list(projects)[:6],
        "keywords": list(keywords)[:6],
        "numbers": list(numbers)[:6],
        "is_question": is_question,
        "script": script,
    }

# ------------------------------------------------------------------------------
# VARIETY: fixed buckets + combinator
# ------------------------------------------------------------------------------
LEADINS = [
    "short answer:",
    "zooming out,",
    "if you're weighing",
    "plainly,",
    "real talk:",
    "on the math,",
    "from experience,",
    "quick take:",
    "low key,",
    "no fluff:",
    "in practice,",
    "gut check:",
    "signal over noise:",
    "nuts and bolts:",
    "from the builder side,",
    "first principles:",
]
CLAIMS = [
    "{focus} is doing more work than the headline",
    "{focus} is where the thesis tightens",
    "{focus} is the part that moves things",
    "{focus} is the practical hinge",
    "{focus} is the constraint to solve",
    "{focus} tells you the next step",
    "it lives or dies on {focus}",
    "risk mostly hides in {focus}",
    "execution shows up as {focus}",
    "watch how {focus} trends, not the hype",
    "{focus} is the boring piece that decides outcomes",
    "{focus} sets the real ceiling",
    "{focus} is the bit with actual leverage",
    "most errors start before {focus} is clear",
]
NUANCE = [
    "separate it from optics",
    "strip the hype and check it",
    "ignore the noise and test it",
    "details beat slogans here",
    "context > theatrics",
    "measure it in weeks, not likes",
    "model it once and the picture clears",
    "ship first, argue later",
    "constraints explain the behavior",
    "once {focus} holds, the plan is simple",
    "touch grass and look at {focus}",
]
CLOSERS = [
    "then the plan makes sense",
    "and the whole picture clicks",
    "and entries/exits get cleaner",
    "and you avoid dumb errors",
    "and the convo gets useful",
    "and incentives line up",
    "and the path forward writes itself",
    "and the take stops being vibes-only",
]


def _combinator(ctx: Dict[str, Any], key_tokens: List[str]) -> str:
    focus = pick_focus_token(key_tokens) or "this"
    handle = ctx.get("handle")
    author = ctx.get("author_name")
    prefix = ""
    r = random.random()
    if handle and r < 0.25:
        prefix = f"@{handle} "
    elif author and r < 0.40:
        prefix = f"{author.split()[0]}, "

    mode = random.choice(["lead+claim", "claim+nuance", "claim+closer", "two"])
    if mode == "lead+claim":
        s = f"{random.choice(LEADINS)} {random.choice(CLAIMS).format(focus=focus)}"
    elif mode == "claim+nuance":
        s = f"{random.choice(CLAIMS).format(focus=focus)} — {random.choice(NUANCE).replace('{focus}', focus)}"
    elif mode == "claim+closer":
        s = f"{random.choice(CLAIMS).format(focus=focus)}, {random.choice(CLOSERS)}"
    else:
        a = random.choice(CLAIMS).format(focus=focus)
        b = random.choice(NUANCE + CLOSERS)
        join = " — " if random.random() < 0.5 else ", "
        s = a + join + b.replace("{focus}", focus)

    out = normalize_ws(prefix + s)
    out = re.sub(r"\s([,.;:?!])", r"\1", out)
    out = re.sub(r"[.!?;:…]+$", "", out)
    words = out.split()
    if len(words) < 8:
        out += " — keep it tight"
    elif len(words) > 22:
        out = " ".join(words[:22])
    return out

# ------------------------------------------------------------------------------
# GENERATOR
# ------------------------------------------------------------------------------
class OfflineCommentGenerator:
    def __init__(self) -> None:
        self.random = random.Random()

    def _violates_ai_blocklist(self, text: str) -> bool:
        low = text.lower()
        if any(p in low 
