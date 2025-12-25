from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from collections import Counter
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from contextvars import ContextVar

import requests
from flask import Flask, request, jsonify

# Helpers from utils.py (already deployed)
from utils import CrownTALKError, fetch_tweet_data, clean_and_normalize_urls

# ------------------------------------------------------------------------------
# App / Logging / Config
# ------------------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------------------
# CORS (needed for Netlify -> Render calls)
# ------------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

PORT = int(os.environ.get("PORT", "10000"))
DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "https://crowntalk.onrender.com")

# Batch & pacing (env-tunable)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))                 # ← process N at a time
PER_URL_SLEEP = float(os.environ.get("PER_URL_SLEEP_SECONDS", "0.1"))  # ← sleep after every URL
MAX_URLS_PER_REQUEST = int(os.environ.get("MAX_URLS_PER_REQUEST", "25"))  # ← hard cap per request

KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))


# ------------------------------------------------------------------------------
# Upgrade add-ons (added without removing existing logic)
# Disabled by default so your current behavior stays the same until you enable.
# ------------------------------------------------------------------------------
ENABLE_VOICE_ROULETTE = os.getenv("ENABLE_VOICE_ROULETTE", "0").strip() == "1"
ENABLE_THREAD_CONTEXT = os.getenv("ENABLE_THREAD_CONTEXT", "0").strip() == "1"
ENABLE_RESEARCH = os.getenv("ENABLE_RESEARCH", "0").strip() == "1"
PRO_KOL_MODE = os.getenv("PRO_KOL_MODE", "0").strip() == "1"
ANTI_HALLUCINATION = os.getenv("ANTI_HALLUCINATION", "0").strip() == "1"

# Optional (rate-limited). Keep OFF unless you really want it.
ENABLE_COINGECKO = os.getenv("ENABLE_COINGECKO", "0").strip() == "1"
COINGECKO_DEMO_KEY = os.getenv("COINGECKO_DEMO_KEY", "").strip()

# Research caching
RESEARCH_CACHE_TTL_SEC = int(os.getenv("RESEARCH_CACHE_TTL_SEC", "900"))      # 15 minutes
LLAMA_PROTOCOLS_CACHE_TTL_SEC = int(os.getenv("LLAMA_PROTOCOLS_CACHE_TTL_SEC", "21600"))  # 6 hours
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "2.2"))

# Request-scoped context (set once per tweet request)
REQUEST_THREAD_CTX: ContextVar[dict] = ContextVar("REQUEST_THREAD_CTX", default={})
REQUEST_RESEARCH_CTX: ContextVar[dict] = ContextVar("REQUEST_RESEARCH_CTX", default={})
REQUEST_VOICE: ContextVar[str] = ContextVar("REQUEST_VOICE", default="")


# ------------------------------------------------------------------------------
# Optional Groq (free-tier). If not set, we run fully offline.
# ------------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)
if USE_GROQ:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        _groq_client = None
        USE_GROQ = False

# ------------------------------------------------------------------------------
# Optional OpenAI (if you have key). If not set, we run offline / Groq / Gemini.
# ------------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)
_openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _openai_client = None
        USE_OPENAI = False
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------------------------------------------------------------------
# Optional Gemini
# ------------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USE_GEMINI = bool(GEMINI_API_KEY)
_gemini_model = None
if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        USE_GEMINI = False
        _gemini_model = None


# ------------------------------------------------------------------------------
# DB helpers (SQLite) for dedupe / history
# ------------------------------------------------------------------------------
_db_lock = threading.Lock()

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def now_ts() -> int:
    return int(time.time())

def sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def with_db_lock(fn):
    with _db_lock:
        return fn()

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

    return with_db_lock(_safe)

def _do_init() -> None:
    with get_conn() as conn:
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            pass
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

            -- Upgrade add-on: entity memory ($TICKER/name -> DefiLlama slug)
            CREATE TABLE IF NOT EXISTS entity_map(
                kind TEXT NOT NULL,
                k TEXT NOT NULL,
                slug TEXT NOT NULL,
                updated_at INTEGER,
                PRIMARY KEY(kind, k)
            );
            """
        )

init_db()


def seen_hash(h: str) -> bool:
    try:
        with get_conn() as c:
            return c.execute("SELECT 1 FROM comments_seen WHERE hash=? LIMIT 1", (h,)).fetchone() is not None
    except Exception:
        return False

def remember_hash(h: str) -> None:
    try:
        with get_conn() as c:
            c.execute("INSERT OR REPLACE INTO comments_seen(hash, created_at) VALUES(?,?)", (h, now_ts()))
    except Exception:
        pass

def _openers(text: str) -> str:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return " ".join(w[:3])

def _trigrams(text: str) -> List[str]:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [" ".join(w[i:i+3]) for i in range(len(w) - 2)]

def opener_seen(opener: str) -> bool:
    try:
        with get_conn() as c:
            return c.execute("SELECT 1 FROM comments_openers_seen WHERE opener=? LIMIT 1", (opener,)).fetchone() is not None
    except Exception:
        return False

def remember_opener(opener: str) -> None:
    try:
        with get_conn() as c:
            c.execute("INSERT OR REPLACE INTO comments_openers_seen(opener, created_at) VALUES(?,?)", (opener, now_ts()))
    except Exception:
        pass

def trigram_seen(tri: str) -> bool:
    try:
        with get_conn() as c:
            return c.execute("SELECT 1 FROM comments_ngrams_seen WHERE ngram=? LIMIT 1", (tri,)).fetchone() is not None
    except Exception:
        return False

def remember_ngrams(text: str) -> None:
    try:
        tris = _trigrams(text)
        with get_conn() as c:
            for tri in tris:
                c.execute("INSERT OR REPLACE INTO comments_ngrams_seen(ngram, created_at) VALUES(?,?)", (tri, now_ts()))
    except Exception:
        pass


# ------------------------------------------------------------------------------
# comment filtering utilities
# ------------------------------------------------------------------------------
AI_BLOCKLIST = [
    "as an ai", "as a language model", "i can't browse", "i cannot browse",
    "i don't have access", "my training data", "i am an ai", "i'm an ai",
    "this is a complex topic", "it depends", "in conclusion", "overall",
    "let me know if", "feel free to ask", "i hope this helps",
]

GENERIC_PHRASES = [
    "thanks for sharing", "love this", "so true", "great post", "nice thread",
    "this is huge", "big if true", "wild", "insane", "amazing",
]

def sanitize_comment(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[@#]\S+", "", t)
    t = t.strip(" -–—|•")
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def violates_ai_blocklist(text: str) -> bool:
    t = (text or "").lower()
    return any(b in t for b in AI_BLOCKLIST)

def is_generic(text: str) -> bool:
    t = (text or "").lower()
    if len(t) < 3:
        return True
    if any(p in t for p in GENERIC_PHRASES):
        return True
    return False

def too_similar_to_recent(text: str, window: int = 60) -> bool:
    # naive similarity: if any trigram already seen, treat as too similar
    tris = _trigrams(text)
    hits = 0
    for tri in tris:
        if trigram_seen(tri):
            hits += 1
    return hits >= 2

def trigram_overlap_bad(text: str, threshold: int = 2) -> bool:
    tris = _trigrams(text)
    if not tris:
        return False
    overlap = sum(1 for tri in tris if trigram_seen(tri))
    return overlap >= threshold


# ------------------------------------------------------------------------------
# Comment format guards / parsing
# ------------------------------------------------------------------------------
def parse_two_comments_flex(raw: Any) -> List[str]:
    """
    Accept either:
      - JSON array: ["a", "b"]
      - Two lines: "a\nb"
      - Dict with items: {"items":[{"text":"a"}, {"text":"b"}]}
      - List of dicts: [{"text":"a"}, {"text":"b"}]
      - Single string that contains '1)' / '2)' or similar markers
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("comment") or item.get("value")
                if isinstance(t, str):
                    out.append(t)
        return out[:2]

    if isinstance(raw, dict):
        items = raw.get("items") or raw.get("data") or raw.get("results")
        if isinstance(items, list):
            return parse_two_comments_flex(items)
        # sometimes dict is itself a single item
        if "text" in raw and isinstance(raw.get("text"), str):
            return [raw["text"]]

    if isinstance(raw, str):
        s = raw.strip()
        # try JSON array
        try:
            j = json.loads(s)
            return parse_two_comments_flex(j)
        except Exception:
            pass

        # try split lines
        lines = [l.strip() for l in s.splitlines() if l.strip()]
        if len(lines) >= 2:
            return lines[:2]

        # try 1) 2) patterns
        m = re.split(r"(?:\n|^)\s*(?:1[\)\.\-]|•)\s*", s)
        if len(m) >= 2:
            rest = m[1]
            m2 = re.split(r"(?:\n|^)\s*(?:2[\)\.\-]|•)\s*", rest)
            if len(m2) >= 2:
                a = m2[0].strip()
                b = m2[1].strip()
                if a and b:
                    return [a, b]

        # last resort: sentence split
        parts = re.split(r"[.!?]\s+", s)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return [parts[0], parts[1]]

        return [s] if s else []

    return []


WORD_RE = re.compile(r"(?:\$\w{2,15}|\d+(?:\.\d+)?|[A-Za-z0-9’']+(?:-[A-Za-z0-9’']+)*)")

def words(t: str) -> list[str]:
    return WORD_RE.findall(t or "")


def enforce_word_count_natural(text: str, min_words: int = 6, max_words: int = 13) -> str:
    t = sanitize_comment(text)
    w = words(t)
    if not w:
        return ""

    # keep tokens like $SOL and decimals intact by trimming by token count
    if len(w) > max_words:
        w = w[:max_words]
        t = " ".join(w)

    if len(w) < min_words:
        # expand slightly without adding fluff
        # (do not add numbers or cashtags)
        if not t.endswith("?") and not t.endswith("."):
            t = t + "."
    return t.strip()


def enforce_unique(candidates: List[str], tweet_text: str = "") -> List[str]:
    out: List[str] = []
    for c in candidates:
        c = sanitize_comment(c)
        if not c:
            continue
        if violates_ai_blocklist(c):
            continue
        if is_generic(c):
            continue

        h = sha256(c.lower())
        if seen_hash(h):
            continue

        # structural repetition guards
        if opener_seen(_openers(c)) or trigram_overlap_bad(c, threshold=2) or too_similar_to_recent(c):
            continue

        # Upgrade add-on: block repeated sentence skeletons
        if template_burned(c):
            continue

        # Upgrade add-on: anti-hallucination guards (opt-in)
        if ANTI_HALLUCINATION and tweet_text:
            source_blob = (tweet_text or "")
            try:
                tc = REQUEST_THREAD_CTX.get() or {}
                source_blob = " ".join([
                    tweet_text or "",
                    tc.get("quoted_text",""),
                    tc.get("parent_text",""),
                ]).strip()
            except Exception:
                pass
            if violates_hallucination_guards(c, source_blob):
                continue

        # remember patterns
        remember_hash(h)
        remember_opener(_openers(c))
        remember_ngrams(c)
        remember_template(c)
        out.append(c)

        if len(out) >= 2:
            break

    # if less than 2, try mild variations from remaining
    if len(out) < 2:
        for c in candidates:
            if len(out) >= 2:
                break
            alt = sanitize_comment(c)
            if not alt or alt in out:
                continue
            if violates_ai_blocklist(alt) or is_generic(alt):
                continue

            h = sha256(alt.lower())
            if seen_hash(h):
                continue

            if opener_seen(_openers(alt)) or trigram_overlap_bad(alt, threshold=2) or too_similar_to_recent(alt):
                continue
            if template_burned(alt):
                continue

            remember_hash(h)
            remember_opener(_openers(alt))
            remember_ngrams(alt)
            remember_template(alt)
            out.append(alt)

    return out


# ------------------------------------------------------------------------------
# Topic detection + prompt building (offline generator)
# ------------------------------------------------------------------------------
def detect_topic(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("chart", "support", "resistance", "liquidity", "funding", "oi", "entry", "tp", "sl")):
        return "chart"
    if any(k in t for k in ("nft", "mint", "floor", "collection", "art", "jpeg")):
        return "nft"
    if any(k in t for k in ("defi", "yield", "apy", "tvl", "protocol", "lending", "perps", "dex")):
        return "defi"
    if any(k in t for k in ("airdrop", "claim", "whitelist", "allowlist")):
        return "airdrop"
    if any(k in t for k in ("meme", "degenerate", "ape", "gm", "ngmi", "wagmi")):
        return "meme"
    if any(k in t for k in ("giveaway", "prize", "winner", "raffle")):
        return "giveaway"
    return "general"


# ------------------------------------------------------------------------------
# Upgrade add-on: template fingerprinting (prevents “same skeleton, new topic”)
# ------------------------------------------------------------------------------
_STYLE_STOPWORDS = {
    "i","you","we","they","it","this","that","these","those",
    "is","are","was","were","be","been","being",
    "a","an","the","and","or","but","so","because",
    "to","of","in","on","for","with","at","from","by","as",
    "if","then","once","until","unless","when","while",
    "not","no","too","very","more","most","less",
    "what","why","how","where","who",
}
_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Local tokenizer so this works even before WORD_RE is defined below
_STYLE_TOKEN_RE = re.compile(r"(?:\$\w{2,15}|\d+(?:\.\d+)?|[A-Za-z0-9’']+(?:-[A-Za-z0-9’']+)*)")

def style_fingerprint(text: str) -> str:
    """Return a short structure signature for duplicate-template blocking."""
    t = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not t:
        return ""
    toks = _STYLE_TOKEN_RE.findall(t)
    if not toks:
        return ""

    mapped: list[str] = []
    for tok in toks:
        if tok in _STYLE_STOPWORDS:
            mapped.append(tok)
        elif tok.startswith("$"):
            mapped.append("$T")
        elif _NUM_RE.match(tok):
            mapped.append("N")
        else:
            mapped.append("W")

    # Compress consecutive W to reduce sensitivity to topic words
    out: list[str] = []
    for m in mapped:
        if out and m == "W" and out[-1] == "W":
            continue
        out.append(m)

    fp = " ".join(out).strip()
    return fp[:140]

def template_burned(tmpl: str) -> bool:
    try:
        fp = style_fingerprint(tmpl)
        if not fp:
            return False
        th = sha256(fp)
        with get_conn() as c:
            return c.execute("SELECT 1 FROM comments_templates_seen WHERE thash=? LIMIT 1", (th,)).fetchone() is not None
    except Exception:
        return False

def remember_template(tmpl: str) -> None:
    try:
        fp = style_fingerprint(tmpl)
        if not fp:
            return
        th = sha256(fp)
        with get_conn() as c:
            c.execute("INSERT OR REPLACE INTO comments_templates_seen(thash, created_at) VALUES(?,?)", (th, now_ts()))
    except Exception:
        pass


# ------------------------------------------------------------------------------
# Upgrade add-ons: voice roulette + thread context + lightweight research
# (All are opt-in via env flags; they do not remove your existing buckets/rules.)
# ------------------------------------------------------------------------------

VOICE_CARDS: Dict[str, str] = {
    "trader": "Voice: trader. Focus on flow, levels, risk, timeframe. Calm, not hype.",
    "builder": "Voice: builder. Practical mechanics, constraints, and execution details.",
    "researcher": "Voice: researcher. Precise, skeptical, one sharp question if unclear.",
    "skeptic": "Voice: skeptic. Incentives, what breaks, second-order risks. No hype.",
    "curious_friend": "Voice: curious friend. Casual, grounded, real human reaction.",
    "deadpan_meme": "Voice: deadpan meme. One witty line allowed if tweet is funny.",
}

def choose_voice(text: str) -> str:
    t = (text or "").lower()
    topic = detect_topic(text or "")

    if topic == "meme":
        return "deadpan_meme"

    weights: Dict[str, float] = {
        "trader": 1.0,
        "builder": 1.0,
        "researcher": 1.0,
        "skeptic": 1.0,
        "curious_friend": 1.0,
    }

    if topic in ("chart", "trading") or any(k in t for k in ("chart", "support", "resistance", "oi", "funding", "liq", "entry", "tp", "sl")):
        weights["trader"] += 2.0
    if any(k in t for k in ("protocol", "amm", "perps", "lending", "audit", "exploit", "bridge", "fees", "mainnet", "testnet")):
        weights["builder"] += 1.5
        weights["skeptic"] += 1.0
    if any(k in t for k in ("thread", "deep dive", "paper", "research", "mechanics", "tokenization")) or len(text or "") > 220:
        weights["researcher"] += 2.0
    if any(k in t for k in ("rug", "scam", "risk", "concern", "fragile", "attack", "hack")):
        weights["skeptic"] += 2.0

    names = list(weights.keys())
    probs = [max(0.1, weights[n]) for n in names]
    total = sum(probs)
    r = random.random() * total
    acc = 0.0
    for n, p in zip(names, probs):
        acc += p
        if r <= acc:
            return n
    return "curious_friend"


def extract_thread_context(tweet_obj: Any) -> Dict[str, str]:
    """Best-effort extraction of quote/parent tweet text from the TweetData object."""
    if not tweet_obj:
        return {}

    # Direct attributes (if utils provides them)
    out: Dict[str, str] = {}
    for attr, key in (
        ("quoted_text", "quoted_text"),
        ("quote_text", "quoted_text"),
        ("parent_text", "parent_text"),
        ("reply_to_text", "parent_text"),
        ("thread_text", "parent_text"),
    ):
        val = getattr(tweet_obj, attr, None)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip()

    # Raw dict payloads (different upstream shapes)
    raw = getattr(tweet_obj, "raw", None) or getattr(tweet_obj, "data", None) or getattr(tweet_obj, "payload", None)
    if isinstance(raw, dict):
        # Common shapes: {"quote": {...}}, {"quotedTweet": {...}}, {"retweetedTweet": {...}}, {"replyingTo": {...}}
        def _dig_text(d: Any) -> str:
            if isinstance(d, dict):
                for k in ("text", "full_text", "content"):
                    if isinstance(d.get(k), str) and d.get(k).strip():
                        return d.get(k).strip()
            return ""

        for k in ("quote", "quotedTweet", "quoted_tweet", "quoted", "retweetedTweet", "retweeted_tweet"):
            txt = _dig_text(raw.get(k))
            if txt and "quoted_text" not in out:
                out["quoted_text"] = txt

        for k in ("replyingTo", "in_reply_to", "parent", "conversation", "thread"):
            txt = _dig_text(raw.get(k))
            if txt and "parent_text" not in out:
                out["parent_text"] = txt

        # VXTwitter-like nesting
        # but sometimes it's raw["tweet"]["quotedTweet"]
        tweet_node = raw.get("tweet") if isinstance(raw.get("tweet"), dict) else None
        if tweet_node:
            qtxt = _dig_text(tweet_node.get("quotedTweet") or tweet_node.get("quote") or tweet_node.get("quoted_tweet"))
            if qtxt and "quoted_text" not in out:
                out["quoted_text"] = qtxt
            ptxt = _dig_text(tweet_node.get("replyingTo") or tweet_node.get("parent"))
            if ptxt and "parent_text" not in out:
                out["parent_text"] = ptxt

    # Keep short to avoid prompt bloat
    if out.get("quoted_text"):
        out["quoted_text"] = out["quoted_text"][:800]
    if out.get("parent_text"):
        out["parent_text"] = out["parent_text"][:800]
    return out


def entity_map_get(kind: str, k: str) -> Optional[str]:
    try:
        with get_conn() as c:
            row = c.execute("SELECT slug FROM entity_map WHERE kind=? AND k=? LIMIT 1", (kind, k)).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def entity_map_set(kind: str, k: str, slug: str) -> None:
    try:
        with get_conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO entity_map(kind, k, slug, updated_at) VALUES(?,?,?,?)",
                (kind, k, slug, now_ts()),
            )
    except Exception:
        pass


# In-memory HTTP/cache helpers (best-effort, short timeouts)
_mem_cache: Dict[str, Any] = {}
_mem_cache_exp: Dict[str, float] = {}
_mem_cache_lock = threading.Lock()

def _cache_get(key: str) -> Any:
    with _mem_cache_lock:
        exp = _mem_cache_exp.get(key)
        if exp and exp > time.time():
            return _mem_cache.get(key)
        return None

def _cache_set(key: str, val: Any, ttl: int) -> None:
    with _mem_cache_lock:
        _mem_cache[key] = val
        _mem_cache_exp[key] = time.time() + max(1, int(ttl))

def _http_get_json(url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def _get_defillama_protocols() -> List[dict]:
    cached = _cache_get("llama_protocols")
    if cached is not None:
        return cached
    data = _http_get_json("https://api.llama.fi/protocols")
    if isinstance(data, list):
        _cache_set("llama_protocols", data, LLAMA_PROTOCOLS_CACHE_TTL_SEC)
        return data
    return []


_CASHTAG_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9]{1,14}")
_NUMTOK_RE = re.compile(r"\d+(?:\.\d+)?")

def _extract_cashtags(text: str) -> set[str]:
    return set(m.group(0) for m in _CASHTAG_RE.finditer(text or ""))

def _extract_numbers(text: str) -> set[str]:
    return set(m.group(0) for m in _NUMTOK_RE.finditer(text or ""))

def violates_hallucination_guards(comment: str, source_text: str) -> bool:
    """Reject if comment invents new cashtags or big/decimal numbers not present in source."""
    if not comment or not source_text:
        return False
    src_tags = _extract_cashtags(source_text)
    c_tags = _extract_cashtags(comment)
    if any(t not in src_tags for t in c_tags):
        return True

    src_nums = _extract_numbers(source_text)
    for num in _extract_numbers(comment):
        # allow tiny integers 1-10 as conversational glue
        try:
            if "." not in num and int(num) <= 10:
                continue
        except Exception:
            pass
        if num not in src_nums:
            return True
    return False


def resolve_defillama_slug(text: str) -> Optional[str]:
    """Try to resolve a project mention to a DefiLlama slug (cached + entity memory)."""
    t = (text or "").strip()
    if not t:
        return None

    # Prefer $TICKER mapping if present
    tags = list(_extract_cashtags(t))
    for tag in tags:
        slug = entity_map_get("cashtag", tag.upper())
        if slug:
            return slug

    # If no mapping, try name matching against protocol list
    protocols = _get_defillama_protocols()
    low = t.lower()
    best = None
    best_score = 0

    for p in protocols:
        name = str(p.get("name") or "")
        slug = str(p.get("slug") or "")
        if not name or not slug:
            continue
        nlow = name.lower()

        score = 0
        if nlow == low:
            score = 100
        elif nlow in low or low in nlow:
            score = 60
        elif len(low) >= 4 and low.split()[0] in nlow:
            score = 30

        if score > best_score:
            best_score = score
            best = slug

    return best if best_score >= 60 else None


def fetch_research_context(tweet_text: str, thread_ctx: Dict[str, str]) -> Dict[str, Any]:
    """Best-effort project research (DefiLlama; optional CoinGecko). Cached."""
    # Cache by a stable key so repeated tweets don't refetch
    key = "research:" + sha256((tweet_text or "")[:800] + "|" + (thread_ctx.get("quoted_text","")[:300] if thread_ctx else ""))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # Try resolve a slug from tweet + thread context
    search_blob = " ".join([
        tweet_text or "",
        thread_ctx.get("quoted_text","") if thread_ctx else "",
        thread_ctx.get("parent_text","") if thread_ctx else "",
    ]).strip()

    slug = resolve_defillama_slug(search_blob)
    out: Dict[str, Any] = {"slug": None, "tvl": None, "category": None, "chain": None, "source": []}

    if slug:
        out["slug"] = slug
        try:
            detail = _http_get_json(f"https://api.llama.fi/protocol/{slug}")
            if isinstance(detail, dict):
                out["category"] = detail.get("category")
                out["chain"] = detail.get("chain")
                out["tvl"] = detail.get("tvl") or detail.get("currentChainTvls")
                out["source"].append("defillama:protocol")
        except Exception:
            pass

    # Optional: CoinGecko (keep OFF unless you enable)
    if ENABLE_COINGECKO and COINGECKO_DEMO_KEY:
        try:
            # Try cashtag first
            tags = list(_extract_cashtags(search_blob))
            q = tags[0].lstrip("$") if tags else (slug or "")
            if q:
                params = {"query": q}
                headers = {"x-cg-demo-api-key": COINGECKO_DEMO_KEY}
                cg = _http_get_json("https://api.coingecko.com/api/v3/search", headers=headers, params=params)
                if isinstance(cg, dict) and cg.get("coins"):
                    top = cg["coins"][0]
                    out["coingecko_id"] = top.get("id")
                    out["source"].append("coingecko:search")
        except Exception:
            pass

    _cache_set(key, out, RESEARCH_CACHE_TTL_SEC)
    return out


def build_context_blob(tweet_text: str, author: Optional[str], url: Optional[str]) -> str:
    """Build a short context JSON that all providers can use."""
    thread_ctx = REQUEST_THREAD_CTX.get() or {}
    research_ctx = REQUEST_RESEARCH_CTX.get() or {}
    voice = REQUEST_VOICE.get() or ""

    ctx = {
        "url": url,
        "author": author,
        "voice": voice,
        "voice_card": VOICE_CARDS.get(voice, "") if voice else "",
        "thread": {
            "quoted_text": thread_ctx.get("quoted_text", ""),
            "parent_text": thread_ctx.get("parent_text", ""),
        },
        "research": research_ctx,
        "hard_rules": [
            "Preserve decimals exactly (e.g., 17.99 must stay 17.99).",
            "Preserve cashtags exactly (e.g., $SOL must stay $SOL).",
            "No new cashtags or big numbers not present in the tweet/thread context.",
            "If research is empty/unclear, ask a question instead of making claims.",
        ],
    }

    return "CONTEXT_JSON:\n" + json.dumps(ctx, ensure_ascii=False) + "\n"

def llm_extra_system_rules() -> str:
    parts = []
    parts.append("- Preserve decimals and cashtags exactly as written in the post.")
    parts.append("- If CONTEXT_JSON research is empty/unclear, ask a question instead of asserting facts.")
    if PRO_KOL_MODE:
        parts.append("- Professional KOL tone: grounded, specific, no empty hype (avoid 'wow', 'huge', 'exciting').")
    return "\n".join(parts) + "\n"


class OfflineCommentGenerator:
    def __init__(self):
        self.rng = random.Random()

        # buckets / variants (kept as-is)
        self.buckets = {
            "general": [
                ["Relatable Friend Mode", "Human Expertise Mode"],
                ["First Person Human Perspective", "Deep Human Reasoning Mode"],
            ],
            "defi": [
                ["Human Expertise Mode", "Deep Human Reasoning Mode"],
                ["Relatable Friend Mode", "First Person Human Perspective"],
            ],
            "nft": [
                ["Relatable Friend Mode", "deadpan_meme"],
                ["First Person Human Perspective", "curious_friend"],
            ],
            "chart": [
                ["trader", "skeptic"],
                ["researcher", "curious_friend"],
            ],
            "airdrop": [
                ["skeptic", "researcher"],
                ["curious_friend", "builder"],
            ],
            "meme": [
                ["deadpan_meme", "curious_friend"],
                ["Relatable Friend Mode", "First Person Human Perspective"],
            ],
            "giveaway": [
                ["curious_friend", "skeptic"],
                ["Relatable Friend Mode", "researcher"],
            ],
        }

        self.openers = [
            "ngl",
            "tbh",
            "okay but",
            "wild how",
            "lowkey",
            "i feel like",
            "this is the part where",
            "honestly",
            "real question is",
            "the underrated bit is",
        ]

        self.closers_q = [
            "thoughts?",
            "what's the catch?",
            "what are you watching here?",
            "how are you thinking about it?",
            "what's the next step?",
        ]

        self.closers_s = [
            "that’s the real test.",
            "feels like the clean way to frame it.",
            "curious where this lands.",
            "this is the part people miss.",
            "that's a good sign.",
        ]

        self.anti_phrases = set(GENERIC_PHRASES)

    def _pick_bucket(self, topic: str) -> list:
        return self.rng.choice(self.buckets.get(topic, self.buckets["general"]))

    def _pick_voice(self, options: list[str], text: str) -> str:
        # If voice roulette is enabled globally, prefer it; otherwise keep existing
        if ENABLE_VOICE_ROULETTE:
            return choose_voice(text)
        return self.rng.choice(options)

    def _make_comment(self, text: str, voice: str) -> str:
        # Keep existing behavior; only add pro KOL tightening when enabled
        t = (text or "").strip()
        topic = detect_topic(t)

        opener = self.rng.choice(self.openers)

        # Tiny heuristic: question vs statement
        ask = self.rng.random() < 0.45

        if voice in ("Relatable Friend Mode", "curious_friend"):
            core = self._friend_core(t, topic)
        elif voice in ("Human Expertise Mode", "builder"):
            core = self._expert_core(t, topic)
        elif voice in ("First Person Human Perspective",):
            core = self._first_person_core(t, topic)
        elif voice in ("Deep Human Reasoning Mode", "researcher"):
            core = self._reasoning_core(t, topic)
        elif voice in ("skeptic",):
            core = self._skeptic_core(t, topic)
        elif voice in ("deadpan_meme",):
            core = self._meme_core(t, topic)
        elif voice in ("trader",):
            core = self._trader_core(t, topic)
        else:
            core = self._friend_core(t, topic)

        if PRO_KOL_MODE:
            # remove over-hype filler
            core = re.sub(r"\b(wow|huge|exciting|insane|amazing)\b", "", core, flags=re.I).strip()
            core = re.sub(r"\s{2,}", " ", core)

        if ask:
            closer = self.rng.choice(self.closers_q)
            out = f"{opener} {core} {closer}".strip()
            if not out.endswith("?"):
                out += "?"
        else:
            closer = self.rng.choice(self.closers_s)
            out = f"{opener} {core} {closer}".strip()
            if out.endswith("?"):
                out = out[:-1].strip() + "."

        return enforce_word_count_natural(out)

    def _friend_core(self, text: str, topic: str) -> str:
        # short, grounded reaction
        if topic == "chart":
            return "this level matters more than the headline"
        if topic == "defi":
            return "tvl and incentives will tell the real story"
        if topic == "nft":
            return "vibes are there but utility has to follow"
        if topic == "airdrop":
            return "worth checking criteria before getting too excited"
        if topic == "meme":
            return "this is funny but also kinda true"
        return "this framing is actually helpful"

    def _expert_core(self, text: str, topic: str) -> str:
        if topic == "defi":
            return "separate optics from mechanics and trace the incentives"
        if topic == "chart":
            return "risk is defined by invalidation, not the narrative"
        if topic == "airdrop":
            return "verify eligibility and avoid signing random approvals"
        return "the details here matter more than the tweet makes it seem"

    def _first_person_core(self, text: str, topic: str) -> str:
        if topic == "chart":
            return "i’ve learned to wait for confirmation, not vibes"
        if topic == "defi":
            return "i usually start by reading docs and watching tvl"
        return "i’ve been burned before by skipping the fine print"

    def _reasoning_core(self, text: str, topic: str) -> str:
        if topic == "defi":
            return "the question is what breaks first under stress"
        if topic == "airdrop":
            return "the clean move is to treat this like a checklist"
        return "this is where the second-order effects show up"

    def _skeptic_core(self, text: str, topic: str) -> str:
        if topic == "defi":
            return "it lives or dies on incentives and attack surface"
        if topic == "airdrop":
            return "assume nothing until the rules are explicit"
        return "the gap between claims and reality is the risk"

    def _meme_core(self, text: str, topic: str) -> str:
        return "monday is doing more work than the headline"

    def _trader_core(self, text: str, topic: str) -> str:
        return "define the invalidation first, then size the risk"

    def generate_two(self, tweet_text: str, author: Optional[str], handle: Optional[str], lang: Optional[str], url: Optional[str] = None) -> List[Dict[str, Any]]:
        topic = detect_topic(tweet_text)
        bucket = self._pick_bucket(topic)

        # choose two different voices
        v1 = self._pick_voice(bucket[0], tweet_text)
        v2 = self._pick_voice(bucket[1], tweet_text)
        if v1 == v2:
            v2 = "skeptic" if v1 != "skeptic" else "curious_friend"

        c1 = self._make_comment(tweet_text, v1)
        c2 = self._make_comment(tweet_text, v2)

        cands = [c1, c2]
        cands = enforce_unique(cands, tweet_text=tweet_text)
        if len(cands) < 2:
            # small rescue
            cands = (cands + [_rescue_one(tweet_text)])[:2]

        out = []
        for c in cands[:2]:
            out.append({"lang": lang or "en", "text": c})
        return out


generator = OfflineCommentGenerator()

def _rescue_one(tweet_text: str) -> str:
    # very safe fallback
    topic = detect_topic(tweet_text)
    if topic == "chart":
        return "clean if it holds, messy if it breaks."
    if topic == "defi":
        return "incentives will decide if this sticks."
    if topic == "meme":
        return "this is dumb but i laughed."
    return "curious how this plays out."

def _rescue_two(tweet_text: str) -> list[str]:
    a = _rescue_one(tweet_text)
    b = "what’s the catch?" if not a.endswith("?") else "interesting, what’s next?"
    return [a, b]


# ------------------------------------------------------------------------------
# LLM provider calls
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Groq model fallback list (fixes org-blocked model errors)
# ------------------------------------------------------------------------------
GROQ_MODEL_CANDIDATES = [
    os.getenv("GROQ_MODEL", "").strip(),      # allow overriding from env
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]
GROQ_MODEL_CANDIDATES = [m for m in GROQ_MODEL_CANDIDATES if m]


def groq_two_comments(tweet_text: str, author: str | None) -> list[str]:
    if not (USE_GROQ and _groq_client):
        raise RuntimeError("Groq disabled or client not available")

    sys_prompt = (
        "You write short, human comments for a social post.\n"
        "- Output exactly two comments.\n"
        "- Each comment is 6-13 words.\n"
        "- No emojis, no hashtags, no links.\n"
        "- The two comments must be different vibes (supportive + curious).\n"
        "- Keep it natural and not robotic.\n"
        "- Avoid generic phrases like 'thanks for sharing'.\n"
        "- Prefer returning a JSON array of two strings.\n"
    )

    # keep your extra system rules (pro KOL / decimals / cashtags)
    sys_prompt += llm_extra_system_rules()

    ctx_blob = build_context_blob(tweet_text, author, url=None)
    user_prompt = (
        ctx_blob
        + f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        + "Return exactly two distinct comments (JSON array or two lines)."
    )

    last_err: Exception | None = None

    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            resp = _groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
                max_tokens=180,
            )
            raw = (resp.choices[0].message.content or "").strip()
            candidates = parse_two_comments_flex(raw)
            candidates = [enforce_word_count_natural(c) for c in candidates]
            candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
            return enforce_unique(candidates, tweet_text=tweet_text)

        except Exception as e:
            last_err = e
            # log and try next model
            logger.warning("Groq model failed (%s): %s", model_name, e)
            continue

    raise RuntimeError(f"Groq failed all model candidates: {last_err}")


def _llm_sys_prompt() -> str:
    return (
        "You write extremely short, human comments for social posts.\n"
        "- Output exactly two comments.\n"
        "- Each comment must be 6-13 words.\n"
        "- Natural conversational tone, as if you just read the post.\n"
        "- One concise thought per comment (no 'thanks for sharing' add-ons).\n"
        "- Light CT / influencer slang (tbh, rn, ngl) is fine, but use sparingly.\n"
        "- The two comments must have different vibes (e.g., supportive vs curious).\n"
        "- If a comment clearly reads like a question, end it with '?'.\n"
        "- Avoid emojis, hashtags, links, or AI-ish phrases.\n"
        "- Avoid repetitive templates; vary syntax and rhythm.\n"
        "- Prefer returning a pure JSON array of two strings, like: "
        "[\"first comment\", \"second comment\"].\n"
        "- If you cannot return JSON, return two lines separated by a newline.\n"
    ) + llm_extra_system_rules()


def openai_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_OPENAI and _openai_client):
        raise RuntimeError("OpenAI disabled or client not available")

    ctx_blob = build_context_blob(tweet_text, author, url=None)
    user_prompt = (
        ctx_blob +
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Return exactly two distinct comments (JSON array or two lines)."
    )
    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _llm_sys_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=160,
        temperature=0.9,
    )
    raw = (resp.choices[0].message.content or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    return enforce_unique(candidates, tweet_text=tweet_text)


def gemini_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_GEMINI and _gemini_model):
        raise RuntimeError("Gemini disabled or client not available")

    ctx_blob = build_context_blob(tweet_text, author, url=None)
    user_prompt = (
        ctx_blob +
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Return exactly two distinct comments (JSON array or two lines)."
    )
    prompt = _llm_sys_prompt() + "\n\n" + user_prompt
    resp = _gemini_model.generate_content(prompt)
    raw = ""
    try:
        parts = getattr(getattr(resp, "candidates", [None])[0], "content", None)
        if parts and getattr(parts, "parts", None):
            raw = "".join(getattr(p, "text", "") for p in parts.parts)
        elif hasattr(resp, "text"):
            raw = resp.text
        else:
            raw = str(resp)
    except Exception:
        raw = str(resp)
    raw = (raw or "").strip()

    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    return enforce_unique(candidates, tweet_text=tweet_text)


def generate_two_comments_with_providers(
    tweet_text: str,
    author: Optional[str],
    handle: Optional[str],
    lang: Optional[str],
    url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid provider strategy with randomness:

    - For each request, randomize the order of enabled providers (Groq, OpenAI, Gemini).
    - Try them in that random order, accumulating comments.
    - As soon as we have 2 solid comments, stop.
    - If all fail or give < 2, fall back to offline generator.
    """

    # Upgrade add-ons: set per-request voice + research context (opt-in)
    _voice_token = None
    _research_token = None
    if ENABLE_VOICE_ROULETTE:
        try:
            v = choose_voice(tweet_text or "")
            _voice_token = REQUEST_VOICE.set(v)
        except Exception:
            _voice_token = None

    if ENABLE_RESEARCH:
        try:
            thread_ctx = REQUEST_THREAD_CTX.get() or {}
            rc = fetch_research_context(tweet_text or "", thread_ctx)
            _research_token = REQUEST_RESEARCH_CTX.set(rc)
        except Exception:
            _research_token = None

    candidates: list[str] = []

    providers: list[tuple[str, Any]] = []
    if USE_GROQ:
        providers.append(("groq", groq_two_comments))
    if USE_OPENAI:
        providers.append(("openai", openai_two_comments))
    if USE_GEMINI:
        providers.append(("gemini", gemini_two_comments))

    # shuffle for variety across requests
    random.shuffle(providers)

    for name, fn in providers:
        try:
            more = fn(tweet_text, author)
            if more:
                candidates = enforce_unique(candidates + more, tweet_text=tweet_text)
        except Exception as e:
            logger.warning("%s provider failed: %s", name, e)

        if len(candidates) >= 2:
            break

    # If we still don't have 2 comments, offline generator rescues
    if len(candidates) < 2:
        try:
            extra_items = generator.generate_two(
                tweet_text,
                author or None,
                handle,
                lang,
                url=url,
            )
            extra = [i.get("text", "") for i in extra_items if i.get("text")]
            if extra:
                candidates = enforce_unique(candidates + extra, tweet_text=tweet_text)
        except Exception as e:
            logger.exception("Total failure in provider cascade: %s", e)

    # If still nothing, hard fallback to 2 simple offline lines
    if not candidates:
        raw = _rescue_two(tweet_text)
        candidates = enforce_unique(raw, tweet_text=tweet_text) or raw

    # Build response items
    out: List[Dict[str, Any]] = []
    for c in candidates[:2]:
        out.append({
            "lang": lang or "en",
            "text": c,
        })

    # If we somehow got <2 after filtering, offline fill
    if len(out) < 2:
        try:
            extra_items = generator.generate_two(
                tweet_text,
                author or None,
                handle,
                lang,
                url=url,
            )
            for item in extra_items:
                if len(out) >= 2:
                    break
                txt = item.get("text")
                if not txt:
                    continue
                out.append({
                    "lang": item.get("lang") or lang or "en",
                    "text": txt,
                })
        except Exception:
            pass

    # Final hard cap: exactly 2
    try:
        return out[:2]
        # Absolute guarantee: always 2 items
if len(out) < 2:
    rescue = _rescue_two(tweet_text)
    for r in rescue:
        if len(out) >= 2:
            break
        out.append({"lang": lang or "en", "text": sanitize_comment(r)})

    finally:
        try:
            if _research_token is not None:
                REQUEST_RESEARCH_CTX.reset(_research_token)
            if _voice_token is not None:
                REQUEST_VOICE.reset(_voice_token)
        except Exception:
            pass


# ------------------------------------------------------------------------------
# API routes (batching + pacing)
# ------------------------------------------------------------------------------

def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def _canonical_x_url_from_tweet(url: str, t: Any) -> str:
    # keeps your original behavior; if utils has a canonical URL, prefer it
    try:
        if hasattr(t, "canonical_url") and t.canonical_url:
            return t.canonical_url
    except Exception:
        pass

    # normalize x.com vs twitter.com
    try:
        u = urlparse(url)
        if "twitter.com" in (u.netloc or ""):
            return url.replace("twitter.com", "x.com")
    except Exception:
        pass
    return url


def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        # x.com/{handle}/status/{id}
        p = urlparse(url).path.strip("/").split("/")
        if len(p) >= 1:
            return p[0]
    except Exception:
        return None
    return None


@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "crowntalk"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/comment", methods=["POST", "OPTIONS"])
def comment_endpoint():
    if request.method == "OPTIONS":
        return ("", 200)

    """
    Accepts:
      - url: string
      - urls: list of urls (optional)
    Returns:
      - results: list of {url, display_url, items:[{lang,text},{lang,text}]}
    """
    payload = request.get_json(force=True, silent=True) or {}
    url = payload.get("url")
    urls = payload.get("urls") or []

    if url and isinstance(url, str):
        urls = [url] + [u for u in urls if u and u != url]

    urls = [u for u in urls if isinstance(u, str) and u.strip()]
    urls = clean_and_normalize_urls(urls)

    if len(urls) > MAX_URLS_PER_REQUEST:
        urls = urls[:MAX_URLS_PER_REQUEST]

    results = []
    for batch in chunked(urls, BATCH_SIZE):
        for url in batch:
            try:
                t = fetch_tweet_data(url)

                _thread_token = None
                try:
                    if ENABLE_THREAD_CONTEXT:
                        _thread_token = REQUEST_THREAD_CTX.set(extract_thread_context(t))
                except Exception:
                    _thread_token = None

                # Prefer handle from upstream payload, fall back to URL parsing
                handle = t.handle or _extract_handle_from_url(url)

                try:
                    two = generate_two_comments_with_providers(
                    t.text,
                    t.author_name or None,
                    handle,
                    t.lang or None,
                    url=url,
                )

                finally:
                    try:
                        if _thread_token is not None:
                            REQUEST_THREAD_CTX.reset(_thread_token)
                    except Exception:
                        pass

                display_url = _canonical_x_url_from_tweet(url, t)

                results.append(
                    {
                        "url": url,
                        "display_url": display_url,
                        "items": two,
                    }
                )

            except CrownTALKError as e:
                logger.warning("CrownTALKError for %s: %s", url, e)
                results.append({"url": url, "display_url": url, "items": [{"lang": "en", "text": "Couldn’t fetch tweet."}, {"lang": "en", "text": "Try again in a moment?"}]})
            except Exception as e:
                logger.exception("Unhandled error while processing %s", url)
                results.append({"url": url, "display_url": url, "items": [{"lang": "en", "text": "Something broke on our side."}, {"lang": "en", "text": "Try again in a moment?"}]})

            # pacing
            if PER_URL_SLEEP > 0:
                time.sleep(PER_URL_SLEEP)

    return jsonify({"ok": True, "results": results})


@app.route("/comment_single", methods=["POST"])
def comment_single():
    payload = request.get_json(force=True, silent=True) or {}
    url = payload.get("url")
    if not url or not isinstance(url, str):
        return jsonify({"ok": False, "error": "Missing url"}), 400

    try:
        t = fetch_tweet_data(url)

        _thread_token = None
        try:
            if ENABLE_THREAD_CONTEXT:
                _thread_token = REQUEST_THREAD_CTX.set(extract_thread_context(t))
        except Exception:
            _thread_token = None

        handle = t.handle or _extract_handle_from_url(url)

        try:
            two = generate_two_comments_with_providers(
            t.text,
            t.author_name or None,
            handle,
            t.lang or None,
            url=url,
        )
        
        finally:
            try:
                if _thread_token is not None:
                    REQUEST_THREAD_CTX.reset(_thread_token)
            except Exception:
                pass

        display_url = _canonical_x_url_from_tweet(url, t)
        return jsonify({"ok": True, "url": url, "display_url": display_url, "items": two})

    except CrownTALKError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Unhandled error in comment_single")
        return jsonify({"ok": False, "error": "Unhandled error"}), 500


# ------------------------------------------------------------------------------
# Keep-alive ping
# ------------------------------------------------------------------------------
_last_ping = 0

@app.route("/ping", methods=["GET"])
def ping():
    global _last_ping
    _last_ping = now_ts()
    return jsonify({"ok": True, "ts": _last_ping})


def _keep_alive_loop():
    while True:
        try:
            # No-op loop; on Render this prevents idle sleep if you ping /ping periodically
            time.sleep(KEEP_ALIVE_INTERVAL)
        except Exception:
            time.sleep(KEEP_ALIVE_INTERVAL)


# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        th = threading.Thread(target=_keep_alive_loop, daemon=True)
        th.start()
    except Exception:
        pass

    app.run(host="0.0.0.0", port=PORT)
