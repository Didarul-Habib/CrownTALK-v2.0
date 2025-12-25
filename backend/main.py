from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from collections import Counter
from contextvars import ContextVar
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify
import time
from typing import Any, Dict

DEBUG_LAST_LLM: Dict[str, Any] = {
    "groq": None,
    "openai": None,
    "gemini": None,
}


def _debug_store(provider: str, payload: Dict[str, Any]) -> None:
    """
    Store the latest raw LLM output for inspection via /debug.
    Safe to fail silently – this is only for debugging.
    """
    try:
        DEBUG_LAST_LLM[provider] = {
            "ts": time.time(),
            **payload,
        }
    except Exception:
        # never break main flow for debug
        pass


# Helpers from utils.py (already deployed)
from utils import CrownTALKError, fetch_tweet_data, clean_and_normalize_urls

# ------------------------------------------------------------------------------
# App / Logging / Config
# ------------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

PORT = int(os.environ.get("PORT", "10000"))
DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "https://crowntalk.onrender.com")

# Batch & pacing (env-tunable)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))                 # ← process N at a time
PER_URL_SLEEP = float(os.environ.get("PER_URL_SLEEP_SECONDS", "1.0"))  # ← sleep after every URL
MAX_URLS_PER_REQUEST = int(os.environ.get("MAX_URLS_PER_REQUEST", "20"))  # ← hard cap per request

KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))

# ------------------------------------------------------------------------------
# Pro KOL upgrade switches (all three add-ons)
# ------------------------------------------------------------------------------
PRO_KOL_MODE = os.getenv("PRO_KOL_MODE", "0").strip() == "1"


PRO_KOL_POLISH = PRO_KOL_MODE
PRO_KOL_STRICT = PRO_KOL_MODE
PRO_KOL_REWRITE = PRO_KOL_MODE
# If true, offline generator will *not* use KOL buckets/combinators.
# It will only do a tiny rule-based rescue when all LLMs fail.
DISABLE_OFFLINE_TEMPLATES = os.getenv("DISABLE_OFFLINE_TEMPLATES", "1").strip() == "1"


# Rewrite tuning (extra LLM calls; can hit quota)
PRO_KOL_REWRITE_MAX_TRIES = int(os.getenv("PRO_KOL_REWRITE_MAX_TRIES", "2"))
PRO_KOL_REWRITE_TEMPERATURE = float(os.getenv("PRO_KOL_REWRITE_TEMPERATURE", "0.7"))
PRO_KOL_REWRITE_MAX_TOKENS = int(os.getenv("PRO_KOL_REWRITE_MAX_TOKENS", "180"))

# If meme tweet, allow 1 witty line (still no emojis/hashtags)
PRO_KOL_ALLOW_WIT = os.getenv("PRO_KOL_ALLOW_WIT", "1").strip() != "0"

# ------------------------------------------------------------------------------
# Thread / Research / Voice toggles (add-ons)
# ------------------------------------------------------------------------------
ENABLE_THREAD_CONTEXT = os.getenv("ENABLE_THREAD_CONTEXT", "0").strip() == "1"
ENABLE_RESEARCH = os.getenv("ENABLE_RESEARCH", "0").strip() == "1"
ENABLE_COINGECKO = os.getenv("ENABLE_COINGECKO", "0").strip() == "1"
COINGECKO_DEMO_KEY = os.getenv("COINGECKO_DEMO_KEY", "").strip() or None
RESEARCH_CACHE_TTL_SEC = int(os.getenv("RESEARCH_CACHE_TTL_SEC", "900"))  # default 15m

# Request-scoped context (per tweet request)
REQUEST_THREAD_CTX: ContextVar[Optional[dict]] = ContextVar("REQUEST_THREAD_CTX", default=None)
REQUEST_RESEARCH_CTX: ContextVar[Optional[dict]] = ContextVar("REQUEST_RESEARCH_CTX", default=None)
REQUEST_VOICE: ContextVar[Optional[dict]] = ContextVar("REQUEST_VOICE", default=None)

# In-memory caches (process-local)
_RESEARCH_CACHE: dict[str, tuple[float, dict]] = {}
_DEFI_LLAMA_PROTOCOLS: Optional[tuple[float, list[dict]]] = None
_COINGECKO_DISABLED_UNTIL: float = 0.0
_RECENT_VOICES: list[str] = []


DEFI_LLAMA_BASE = "https://api.llama.fi"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _research_cache_get(key: str) -> Optional[dict]:
    rec = _RESEARCH_CACHE.get(key)
    if not rec:
        return None
    ts, data = rec
    if time.time() - ts > RESEARCH_CACHE_TTL_SEC:
        try:
            del _RESEARCH_CACHE[key]
        except KeyError:
            pass
        return None
    return data


def _research_cache_set(key: str, data: dict) -> None:
    try:
        _RESEARCH_CACHE[key] = (time.time(), data)
    except Exception:
        pass

def _load_defillama_protocols() -> list[dict]:
    global _DEFI_LLAMA_PROTOCOLS
    if not ENABLE_RESEARCH:
        return []

    now = time.time()
    if _DEFI_LLAMA_PROTOCOLS:
        ts, data = _DEFI_LLAMA_PROTOCOLS
        if now - ts < RESEARCH_CACHE_TTL_SEC:
            return data

    try:
        resp = requests.get(f"{DEFI_LLAMA_BASE}/protocols", timeout=2)
        if resp.ok:
            data = resp.json()
            if isinstance(data, list):
                _DEFI_LLAMA_PROTOCOLS = (now, data)
                return data
    except Exception:
        pass

    return _DEFI_LLAMA_PROTOCOLS[1] if _DEFI_LLAMA_PROTOCOLS else []

def _resolve_protocol_slug_from_symbol(symbol: str) -> Optional[str]:
    """
    Resolve a cashtag symbol (e.g. 'SOL') into a DefiLlama slug.
    Uses entity_map first, then /protocols list.
    """
    symbol = (symbol or "").strip()
    if not symbol:
        return None

    # 1) fast path via entity_map
    slug = lookup_entity_slug("cashtag", symbol)
    if slug:
        return slug

    # 2) search in protocol list
    protos = _load_defillama_protocols()
    if not protos:
        return None

    low = symbol.lower()
    exact = [p for p in protos if str(p.get("symbol", "")).lower() == low]
    if not exact:
        exact = [p for p in protos if str(p.get("name", "")).lower() == low]
    if not exact:
        fuzzy = [
            p for p in protos
            if low in str(p.get("symbol", "")).lower()
            or low in str(p.get("name", "")).lower()
        ]
        exact = fuzzy

    if not exact:
        return None

    chosen = exact[0]
    slug = (chosen.get("slug") or chosen.get("id") or chosen.get("name") or "").strip()
    if slug:
        upsert_entity_slug("cashtag", symbol, slug)
        name = (chosen.get("name") or "").strip()
        if name:
            upsert_entity_slug("name", name, slug)
    return slug

def _fetch_defillama_for_slug(slug: str) -> dict:
    if not slug:
        return {}

    cache_key = f"llama:{slug}"
    cached = _research_cache_get(cache_key)
    if cached is not None:
        return cached

    out: dict[str, Any] = {}
    try:
        r1 = requests.get(f"{DEFI_LLAMA_BASE}/protocol/{slug}", timeout=2)
        if r1.ok:
            out["protocol"] = r1.json()
    except Exception:
        pass

    try:
        r2 = requests.get(f"{DEFI_LLAMA_BASE}/tvl/{slug}", timeout=2)
        if r2.ok:
            out["tvl"] = r2.json()
    except Exception:
        pass

    if out:
        _research_cache_set(cache_key, out)
    return out

def _coingecko_search(symbol: str) -> Optional[str]:
    global _COINGECKO_DISABLED_UNTIL
    if not ENABLE_COINGECKO or time.time() < _COINGECKO_DISABLED_UNTIL:
        return None

    try:
        params = {"query": symbol}
        headers = {}
        if COINGECKO_DEMO_KEY:
            headers["x-cg-demo-api-key"] = COINGECKO_DEMO_KEY
        r = requests.get(f"{COINGECKO_BASE}/search", params=params, headers=headers, timeout=2)
        if r.status_code == 429:
            _COINGECKO_DISABLED_UNTIL = time.time() + 900  # 15 min
            return None
        if not r.ok:
            return None
        data = r.json() or {}
        coins = data.get("coins") or []
        if not coins:
            return None
        return coins[0].get("id")
    except Exception:
        return None


def _coingecko_price(coin_id: str) -> Optional[dict]:
    global _COINGECKO_DISABLED_UNTIL
    if not ENABLE_COINGECKO or time.time() < _COINGECKO_DISABLED_UNTIL:
        return None

    try:
        params = {"ids": coin_id, "vs_currencies": "usd"}
        headers = {}
        if COINGECKO_DEMO_KEY:
            headers["x-cg-demo-api-key"] = COINGECKO_DEMO_KEY
        r = requests.get(f"{COINGECKO_BASE}/simple/price", params=params, headers=headers, timeout=2)
        if r.status_code == 429:
            _COINGECKO_DISABLED_UNTIL = time.time() + 900
            return None
        if not r.ok:
            return None
        return r.json().get(coin_id)
    except Exception:
        return None


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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ------------------------------------------------------------------------------
# Optional OpenAI
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
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "1").strip() == "1"
USE_GEMINI = bool(GEMINI_API_KEY) and GEMINI_ENABLED
_gemini_model = None
if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception:
        _gemini_model = None
        USE_GEMINI = False


# ------------------------------------------------------------------------------
# Keepalive (Render free – optional)
# ------------------------------------------------------------------------------
def keep_alive() -> None:
    if not BACKEND_PUBLIC_URL:
        return
    while True:
        try:
            requests.get(f"{BACKEND_PUBLIC_URL}/", timeout=5)
        except Exception:
            pass
        time.sleep(KEEP_ALIVE_INTERVAL)

# ------------------------------------------------------------------------------
# DB init (safe across workers)
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
            
            CREATE TABLE IF NOT EXISTS entity_map(
                kind TEXT NOT NULL,
                k TEXT NOT NULL,
                slug TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(kind, k)
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

            CREATE TABLE IF NOT EXISTS comments_frames_seen(
                fhash TEXT PRIMARY KEY,
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
# Light memory / OTP guards (anti-pattern, anti-repeat)
# ------------------------------------------------------------------------------
def now_ts() -> int:
    return int(time.time())

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def upsert_entity_slug(kind: str, key: str, slug: str) -> None:
    """
    Store mapping: (kind, key) -> DefiLlama slug.
    kind: "cashtag" or "name"
    key:  canonical lowercase form (e.g. "sol", "solana")
    """
    key = (key or "").strip().lower()
    slug = (slug or "").strip()
    if not (kind and key and slug):
        return
    try:
        with get_conn() as c:
            c.execute(
                """
                INSERT INTO entity_map(kind, k, slug, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(kind, k) DO UPDATE SET
                    slug = excluded.slug,
                    updated_at = excluded.updated_at
                """,
                (kind, key, slug, now_ts()),
            )
    except Exception:
        pass


def lookup_entity_slug(kind: str, key: str) -> Optional[str]:
    key = (key or "").strip().lower()
    if not (kind and key):
        return None
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT slug FROM entity_map WHERE kind=? AND k=? LIMIT 1",
                (kind, key),
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None

TOKEN_RE = re.compile(
    r"(?:\$\w{2,15}|\d+(?:\.\d+)?|[A-Za-z0-9’']+(?:-[A-Za-z0-9’']+)*)"
)

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
            return c.execute("SELECT 1 FROM comments_seen WHERE hash=? LIMIT 1", (h,)).fetchone() is not None
    except Exception:
        return False

def remember_comment(text: str, url: str = "", lang: Optional[str] = None) -> None:
    try:
        norm = _normalize_for_memory(text)
        if not norm:
            return
        with get_conn() as c:
            c.execute("INSERT OR IGNORE INTO comments_seen(hash, created_at) VALUES(?,?)", (sha256(norm), now_ts()))
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
            c.execute("INSERT OR IGNORE INTO comments_openers_seen(opener, created_at) VALUES (?,?)", (opener, now_ts()))
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
                if c.execute("SELECT 1 FROM comments_ngrams_seen WHERE ngram=? LIMIT 1", (g,)).fetchone():
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

# ------------------------------------------------------------------------------
# Structure fingerprint (kills repeated "same skeleton, new topic" comments)
# ------------------------------------------------------------------------------
STYLE_STOPWORDS = {
    "i","you","we","they","it","this","that","these","those",
    "is","are","was","were","be","been","being",
    "a","an","the","and","or","but","so","because",
    "to","of","in","on","for","with","at","from","by","as",
    "if","then","once","until","unless","when","while",
    "not","no","too","very","more","most","less",
    "what","why","how","where","who",
}

_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

def style_fingerprint(text: str) -> str:
    """
    Converts a comment into a compact structure signature:
    - keeps function words (stopwords)
    - maps content words to W
    - maps numbers to N
    - maps tickers to $T
    This blocks repeating sentence skeletons across different topics.
    """
    t = normalize_ws(text).lower()
    if not t:
        return ""
    toks = TOKEN_RE.findall(t)
    if not toks:
        return ""

    mapped: list[str] = []
    for tok in toks:
        if tok in STYLE_STOPWORDS:
            mapped.append(tok)
        elif tok.startswith("$"):
            mapped.append("$T")
        elif _NUM_RE.match(tok):
            mapped.append("N")
        else:
            mapped.append("W")

    # compress repeated W W W -> W (helps avoid over-specific fingerprints)
    out: list[str] = []
    for m in mapped:
        if out and m == "W" and out[-1] == "W":
            continue
        out.append(m)

    fp = " ".join(out).strip()
    return fp[:140]


def template_burned(tmpl: str) -> bool:
    fp = style_fingerprint(tmpl)
    if not fp:
        return False
    thash = sha256(fp)
    try:
        with get_conn() as c:
            return c.execute(
                "SELECT 1 FROM comments_templates_seen WHERE thash=? LIMIT 1",
                (thash,),
            ).fetchone() is not None
    except Exception:
        return False

def remember_template(tmpl: str) -> None:
    try:
        fp = style_fingerprint(tmpl)
        if not fp:
            return
        thash = sha256(fp)
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_templates_seen(thash, created_at) VALUES (?,?)",
                (thash, now_ts()),
            )
    except Exception:
        pass


def frame_fingerprint(comment: str, tweet_text: str = "") -> str:
    """
    OTP-style 'frame' signature:
    - voice id (trader/builder/researcher/etc)
    - mode (support/question/skeptical/playful from guess_mode)
    - question vs statement
    - structural style fingerprint (W/W/$T pattern)
    """
    base_fp = style_fingerprint(comment)
    if not base_fp:
        return ""

    voice = current_voice_card() or {}
    voice_id = voice.get("id") or "generic"

    mode = guess_mode(tweet_text or "")
    qflag = "Q" if "?" in (comment or "") else "S"

    return f"{voice_id}|{mode}|{qflag}|{base_fp}"


def frame_seen(comment: str, tweet_text: str = "") -> bool:
    fp = frame_fingerprint(comment, tweet_text)
    if not fp:
        return False
    fh = sha256(fp)
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT 1 FROM comments_frames_seen WHERE fhash=? LIMIT 1",
                (fh,),
            ).fetchone()
        return bool(row)
    except Exception:
        return False


def remember_frame(comment: str, tweet_text: str = "") -> None:
    fp = frame_fingerprint(comment, tweet_text)
    if not fp:
        return
    fh = sha256(fp)
    try:
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_frames_seen(fhash, created_at) VALUES (?,?)",
                (fh, now_ts()),
            )
    except Exception:
        pass

def _word_trigrams(s: str) -> set:
    w = TOKEN_RE.findall((s or "").lower())
    return set(" ".join(w[i:i+3]) for i in range(max(0, len(w) - 2)))

def too_similar_to_recent(text: str, threshold: float = 0.62, sample: int = 300) -> bool:
    """Jaccard(word-3grams) vs last N comments to block paraphrase repeats."""
    try:
        with get_conn() as c:
            rows = c.execute("SELECT text FROM comments ORDER BY id DESC LIMIT ?", (sample,)).fetchall()
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

def _pair_too_similar(a: str, b: str, threshold: float = 0.25) -> bool:
    """Pairwise similarity (Jaccard over trigrams) to avoid EN#1 ≈ EN#2."""
    ta = _word_trigrams(a)
    tb = _word_trigrams(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    uni = len(ta | tb)
    if not uni:
        return False
    return (inter / uni) >= threshold

# ------------------------------------------------------------------------------
# CORS + Health
# ------------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "groq": bool(USE_GROQ)}), 200

# ------------------------------------------------------------------------------
# Rules: word count + sanitization + Tokenization
# ------------------------------------------------------------------------------
TOKEN_RE = re.compile(
    r"(?:\$\w{2,15}|\d+(?:\.\d+)?|[A-Za-z0-9’']+(?:-[A-Za-z0-9’']+)*)"
)

def words(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")

def sanitize_comment(raw: str) -> str:
    txt = re.sub(r"https?://\S+", "", raw or "")
    txt = re.sub(r"[@#]\S+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"[.!?;:…]+$", "", txt).strip()
    txt = re.sub(r"[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", "", txt)
    return txt

def enforce_word_count_natural(raw: str, min_w=6, max_w=13) -> str:
    txt = sanitize_comment(raw)
    toks = words(txt)
    if not toks:
        return ""
    if len(toks) > max_w:
        toks = toks[:max_w]
    while len(toks) < min_w:
        for filler in ["honestly","tbh","still","though","right"]:
            if len(toks) >= min_w: break
            toks.append(filler)
        if len(toks) < min_w: break
    return " ".join(toks).strip()

# ------------------------------------------------------------------------------
# Topic / keywords (to keep comments context-aware, not templated)
# ------------------------------------------------------------------------------
EN_STOPWORDS = {
    "the","a","an","and","or","but","to","in","on","of","for","with","at","from","by","about","as",
    "into","like","through","after","over","between","out","against","during","without","before","under",
    "around","among","is","are","be","am","was","were","it","its","that","this","so","very","really"
}

AI_BLOCKLIST = {
    # generic hype / ai slop
    "amazing","awesome","incredible","empowering","game changer","game-changing","transformative",
    "paradigm shift","as an ai","as a language model","in conclusion","in summary","furthermore","moreover",
    "navigate this landscape","ever-evolving landscape","leverage this insight","cutting edge","state of the art",
    "unprecedented","unleash","harness the power","embark on this journey","revolutionize","disruptive",
    "bestie","like and retweet","thoughts?","agree?","who's with me","drop your thoughts","smash that like button",
    "link in bio","in case you missed it","i think","i believe","great point","just saying","according to",
    "to be honest","actually","literally","personally i think","my take","as someone who","at the end of the day",
    "moving forward","synergy","circle back","bandwidth","double down","let that sink in","on so many levels","tbh",
    "this resonates","food for thought","hit different",
    "love that","love this","love the","love your","love the concept","love the direction",
    "love where you're taking this",
    "excited to see","excited for","can't wait to see","can’t wait to see",
    "looking forward to","look forward to",
    "this is huge","this could be huge","this is massive","this is insane",
    "game changing","game-changing","total game changer","what a game changing approach",
    "mind blown","mind-blowing","blows my mind","massive alpha",
    "thanks for sharing","thank you for sharing","thanks for this","appreciate you",
    "appreciate it","appreciate this","proud of you","so proud of this",
    "the vibe around","vibe around","the vibe here is pretty real",
    "this is what we need","exactly what we need","difference between thesis and cope",
    "thesis and cope",
    "thesis stays",
}

GENERIC_PHRASES = {
    "well researched and insightful",
    "very interesting concept",
    "interesting concept",
    "sounds like a game changer",
    "game changer",
    "big step for",
    "that's a big step",
    "i'm still unsure",
    "i'm curious how scalable",
    "can you elaborate more",
    "what's the catch",
    "good daily routine",
    "great daily routine",
    "amazing to see",
    "glad to see a shift",
    "i'm curious to see how",
    "i'm curious to see how rails",
    "i'm curious to see how rails xyz",
    "i'm curious to see how this ecosystem grows",
    "i'm curious to see how this ecosystem grows over time",
    "i'm glad to see",
    "good luck with",
    "good luck to those competing",
    "good luck ",
    "great point about",
    "your patience and persistence",
    "love your take",
    "love the concept of",
    "love that you're emphasizing",
    "love that aligned communities are unbreakable",
    "i'm intrigued by the idea of",
    "congrats ",
    "congrats",
    "i'll be shifting my energy towards",
    "i'm watching ",
    "i'm sending you strength",
    "wishing you strength",
    "wishing you all the strength",
    "wishing you ",
    "been hearing similar chats",
    "that's a great strategy",
    "that's a great example",
     "this is huge",
    "this is huge for",
    "this could be huge",
    "this could be huge for",
    "this could be the change",
    "the change the defi space has been waiting for",
    "this is the kind of innovation",
    "this is the kind of innovation people have been waiting for",
    "this is the kind of innovation people have been waiting for since",
    "this is a major breakthrough",
    "major breakthrough",
    "this is a genius move",
    "genius move",
    "that's amazing",
    "that's actually really impressive",
    "love the way",
    "love the way ",
    "love the direction",
    "love the direction but",
    "glad to hear",
    "glad to hear that",
    "glad to hear that polygon is treating you right",
    "this is exactly what we need",
    "this is exactly what we need in web3",
    "this is exactly what we need in web3 transparency and accountability",
    "this is the gap you're trying to bridge",
    "that's the gap you're trying to bridge",
    "this is a total game changer",
    "sounds like a total game changer",
    "that's a game changer for",
    "that's a game changer for bitcoin holders",
    "game changing approach",
    "what a game changing approach",
    "what a game changing approach to make defi more accessible",
    "breaking free from silos",
    "breaking free from silos is a bold exciting move",
    "sounds like a solid tool",
    "sounds like a solid tool for traders",
    "this could be the change the defi space has been waiting for",
    "this could be the change the defi space has been waiting for since",
    "this could be the change the defi space has been waiting for since defi's",
    "this could be the change the defi space has been waiting for since defi",
    "this is the change the defi space has been waiting for",
    "this could be the change the defi space has been waiting for",
    "your enthusiasm is infectious",
    "your enthusiasm is infectious can't wait to see this for myself",
    "can't wait to see this for myself",
    "can't wait to see this",
    "can't wait to see it",
    "can't wait to see how this plays out",
    "looking forward to trying it out",
    "looking forward to trying this out",
    "looking forward to trying",
    "looking forward to this",
    "this could be huge for logistics and warehouse management",
    "this could be huge for logistics",
    "this is huge for bybit users",
    "this is huge for bybit users with easy on chain access",
    "good luck all on the sixr cricket quiz",
    "good luck all on the sixr cricket quiz and the kudos swap points",
    "this is a major step forward",
    "bold exciting move",
    "this is a bold move",
    "this is the change",
    "this is the change the space has been waiting for",
    "mind blown",
    "mind blown by the idea",
    "idk what's going on but you've been posting tho",
    "you've been posting tho",
    "glad to hear that polygon is treating you right",
    "this could be the change people have been waiting for",
    "this is the kind of thing people have been waiting for",
    "the space has been waiting for this",
}

def contains_generic_phrase(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in GENERIC_PHRASES)

STARTER_BLOCKLIST = {
    "yeah this","honestly this","kind of","nice to","hard to","feels like","this is","short line","funny how",
    "appreciate that","interested to","curious where","nice to see","chill sober","good reminder","yeah that",
    "good to see the boring",
    # extra starters we don't want repeated
    "love that","love the","love your","i'm curious","im curious","curious about","love your take",
}
try:
    EMOJI_PATTERN = re.compile(
        r"[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        flags=re.UNICODE,
    )
except re.error:
    EMOJI_PATTERN = re.compile(r"[\u2600-\u27BF]+", flags=re.UNICODE)

def detect_topic(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("gm ", "gn ", "good morning", "good night")):
        return "greeting"
    if any(k in t for k in ("airdrop", "whitelist", "wl spot", "mint is live")):
        return "giveaway"
    if any(k in t for k in ("chart", "support", "resistance", "ath", "price target", "%", "market cap", "mc")):
        return "chart"
    if any(k in t for k in ("bug", "issue", "broken", "down again", "wtf", "why is", "tired of")):
        return "complaint"
    if any(k in t for k in ("announcing", "announcement", "we're live", "we are live", "launching", "we shipped")):
        return "announcement"
    if any(k in t for k in ("meme", "shitpost", "ratioed", "memeing")) or "lol" in t:
        return "meme"
    if "🧵" in text or len(text) > 220:
        return "thread"
    if len(text) < 80:
        return "one_liner"
    return "generic"

def llm_mode_hint(tweet_text: str) -> str:
    """
    Small, dynamic steering line for the LLM.
    Keeps style consistent across trading/NFT/DeFi/meme/dev threads without buckets.
    """
    t = (tweet_text or "").lower()

    # dev / research / technical threads
    if "tokenization" in t or "embedding" in t or "prompt" in t or "llm" in t or "model" in t:
        return "Mode: builder. Be precise and practical. Ask one sharp technical question."

    # trading / charts
    if any(k in t for k in ("chart", "support", "resistance", "breakout", "break down", "liquidity", "liq", "volume", "open interest", "oi", "funding", "entry", "entries", "tp", "sl", "stop", "r:r", "rr", "ath", "range")):
        return "Mode: trader. Focus on positioning, liquidity/flow, risk, timeframe. No cheerleading."

    # NFT / mint / whitelist
    if any(k in t for k in ("mint", "minting", "wl", "whitelist", "allowlist", "og", "floor", "sweep", "reveal", "collection", "trait", "art", "pfp")):
        return "Mode: NFT. Focus on demand, distribution, team execution, and timing. Avoid fluff."

    # DeFi / protocol / product mechanics
    if any(k in t for k in ("defi", "protocol", "amm", "perps", "lending", "borrow", "staking", "restaking", "yield", "apy", "apr", "vault", "fees", "liquidation", "bridge", "rollup", "mainnet", "testnet", "audit", "exploit")):
        return "Mode: DeFi operator. Focus on incentives, mechanisms, risk surface, UX. Be grounded."

    # meme / shitpost
    if any(k in t for k in ("lol", "lmao", "ngmi", "wagmi", "cope", "rekt", "ratio", "meme", "shitpost")):
        return "Mode: meme. Keep it witty and minimal, still coherent. No cringe hype."

    # default CT pro
    return "Mode: CT pro. Grounded, specific, calm. One observation + one sharp question."


def is_crypto_tweet(text: str) -> bool:
    t = (text or "").lower()
    crypto_keywords = [
        "crypto","defi","nft","airdrop","token","coin","chain","l1","l2","staking","yield",
        "dex","cex","onchain","on-chain","gas fees","btc","eth","sol","arb","layer two","mainnet"
    ]
    return any(k in t for k in crypto_keywords) or bool(re.search(r"\$\w{2,8}", text or ""))

def detect_sentiment(text: str) -> str:
    """Very lightweight bullish / bearish tone detector for CT posts."""
    t = (text or "").lower()
    bull = [
        "bullish","sending","send it","moon","mooning","ath","all time high",
        "pump","pumping","green candles","ape in","apeing in","printing"
    ]
    bear = [
        "worried","concerned","dump","dumping","rug","rugged","rekt","down only",
        "exit liquidity","overvalued","scam","red candles","bagholding","bag holder"
    ]
    if any(k in t for k in bull):
        return "bullish"
    if any(k in t for k in bear):
        return "bearish"
    return "neutral"

def extract_keywords(text: str) -> list[str]:
    cleaned = re.sub(r"https?://\S+", "", text or "")
    cleaned = re.sub(r"[@#]\S+", "", cleaned)
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", cleaned)
    if not tokens:
        return []
    filtered = [t for t in tokens if t.lower() not in EN_STOPWORDS and len(t) > 2] or tokens
    counts = Counter([t.lower() for t in filtered])
    seen, out = set(), []
    for w in sorted(filtered, key=lambda w: (-counts[w.lower()], -len(w))):
        lw = w.lower()
        if lw not in seen:
            seen.add(lw); out.append(w)
    return out[:10]

def pick_focus_token(tokens: List[str]) -> Optional[str]:
    if not tokens:
        return None
    upperish = [t for t in tokens if t.isupper() or t[0].isupper()]
    return random.choice(upperish) if upperish else random.choice(tokens)

# ------------------------------------------------------------------------------
# Variety buckets + combinator (keeps comments varied)
# ------------------------------------------------------------------------------
LEADINS = [
    "short answer:","zooming out,","if you're weighing","plainly,","real talk:","on the math,",
    "from experience,","quick take:","low key,","no fluff:","in practice,","gut check:",
    "signal over noise:","nuts and bolts:","from the builder side,","first principles:",
"ct lens:","flow check:","risk check:","tape says:","liq check:","timeframe matters:",
    "hot take:","cold take:","conviction check:","positioning wise:","alpha is:",
    "narratives aside,","strip it down:","here’s the edge:","what matters most:"
]
CLAIMS = [
    "{focus} is doing more work than the headline",
    "{focus} is the part that moves things","{focus} is the practical hinge",
    "{focus} is the constraint to solve","{focus} tells you the next step",
    "it lives or dies on {focus}","risk mostly hides in {focus}",
    "execution shows up as {focus}","watch how {focus} trends, not the hype",
    "{focus} is the boring piece that decides outcomes","{focus} sets the real ceiling",
    "{focus} is the bit with actual leverage","most errors start before {focus} is clear",
"{focus} is the real driver, everything else is noise",
    "{focus} is where smart money will express the view",
    "{focus} is the lever, not the vibe",
    "{focus} is what makes this tradeable",
    "the market will reward {focus}, not the narrative",
    "you can’t hand-wave {focus} and expect it to work",
]
NUANCE = [
    "separate it from optics","strip the hype and check it","ignore the noise and test it",
    "details beat slogans here","context > theatrics","measure it in weeks, not likes",
    "model it once and the picture clears","ship first, argue later","constraints explain the behavior",
    "once {focus} holds, the plan is simple","touch grass and look at {focus}",
"watch liquidity, then talk",
    "size it like uncertainty is real",
    "timeframe decides who’s right",
    "tape > timelines",
    "let price confirm the story",
    "if it’s obvious, it’s probably crowded",
    "avoid marrying the narrative",
    "respect downside before upside",
]
CLOSERS = [
    "then the plan makes sense","and the whole picture clicks","and entries/exits get cleaner",
    "and you avoid dumb errors","and the convo gets useful","and incentives line up",
    "and the path forward writes itself","and the take stops being vibes-only",
"and the trade becomes cleaner",
    "and you stop chasing vibes",
    "and you can actually size it",
    "and the market tells you if you’re wrong",
]

PRO_KOL_BAD = {
    "wow", "exciting", "so excited", "can't wait", "cant wait", "love this", "love that",
    "amazing", "awesome", "incredible", "huge", "insane", "legendary",
}

PRO_KOL_GOOD = {
    "signal", "execution", "risk", "liquidity", "flow", "incentives",
    "positioning", "distribution", "supply", "demand", "timeline", "context",
    "constraint", "tradeoff", "edge", "verify", "confirm", "pricing", "variance",
    "sizing", "asymmetric", "timeframe",
}

def pro_kol_score(text: str) -> int:
    t = (text or "").strip()
    low = t.lower()

    score = 0

    # penalize hype/fanboy
    if any(p in low for p in PRO_KOL_BAD):
        score -= 3

    # reward grounded "operator" words
    score += sum(1 for w in PRO_KOL_GOOD if w in low)

    # reward "claim + constraint" structure
    if any(x in low for x in ("if ", "once ", "as long as ", "depends on ", "until ", "unless ")):
        score += 1

    # reward questions that are specific (not vague “sounds interesting”)
    if low.endswith("?") and any(x in low for x in ("how", "what", "where", "which", "when")):
        score += 1

    # penalize exclamation / hype punctuation
    if "!" in t:
        score -= 1

    # penalize super generic
    if contains_generic_phrase(t):
        score -= 2

    return score

def pro_kol_ok(text: str, min_score: int = 1) -> bool:
    return pro_kol_score(text) >= min_score

PRO_POLISH_REPLACEMENTS = [
    (r"^(wow|omg|yo|bro)\b[,\s]*", ""),
    (r"\b(exciting|hype)\b", "notable"),
    (r"\b(huge|massive)\b", "meaningful"),
    (r"\b(insane)\b", "wild"),
    (r"\b(amazing|awesome|incredible)\b", "solid"),
    (r"\bsounds interesting\b", "worth watching"),
    (r"[!]+", ""),
]

def pro_kol_polish(text: str, topic: str = "") -> str:
    """
    Light touch polish. Keeps human tone.
    For meme posts, we keep more personality.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # If meme, do less aggressive tone flattening
    is_meme = (topic == "meme")
    for pat, rep in PRO_POLISH_REPLACEMENTS:
        if is_meme and "sounds interesting" in pat:
            continue
        t = re.sub(pat, rep, t, flags=re.I)

    t = re.sub(r"\s+", " ", t).strip()
    return t


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
        b = random.choice(NUANCE + CLOSERS)  # varied joiner
        join = " — " if random.random() < 0.5 else ", "
        s = a + join + b.replace("{focus}", focus)

    out = normalize_ws(prefix + s)
    out = re.sub(r"\s([,.;:?!])", r"\1", out)
    out = re.sub(r"[.!?;:…]+$", "", out)
    return out

# ------------------------------------------------------------------------------
# Offline generator (with OTP guards + 6–13 words enforcement)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Offline generator (with OTP guards + 6–13 words enforcement)
# ------------------------------------------------------------------------------

class OfflineCommentGenerator:
    def __init__(self) -> None:
        self.random = random.Random()

    def _violates_ai_blocklist(self, text: str) -> bool:
        low = (text or "").lower()
        if any(p in low for p in AI_BLOCKLIST):
            return True
        if re.search(r"\b(so|very|really)\s+\1\b", low):
            return True
        if len(re.findall(r"\.\.\.", text or "")) > 1:
            return True
        if low.count("—") > 3:
            return True
        return False

    def _diversity_ok(self, text: str) -> bool:
        if not text:
            return False

        opener = _openers(text)
        if any(opener.startswith(b) for b in STARTER_BLOCKLIST):
            return False
        if opener_seen(opener):
            return False
        if trigram_overlap_bad(text, threshold=2):
            return False
        if too_similar_to_recent(text):
            return False

        toks = re.findall(r"[A-Za-z][A-Za-z0-9']+", text.lower())
        ai_words = {w.lower() for w in AI_BLOCKLIST if " " not in w}
        novel = [t for t in toks if t not in EN_STOPWORDS and t not in ai_words]
        return len(set(novel)) >= 2

    def _tidy_en(self, t: str) -> str:
        # strip emojis for EN
        t = re.sub(r"[^\x00-\x7F]+", "", t or "")
        t = enforce_word_count_natural(t, 6, 13)
        return t

    def _native_buckets(self, script: str) -> List[str]:
        f = "{focus}"
        if script == "bn":
            return [
                f"{f} নিয়ে সরাসরি কথা, বাড়তি হাইপ না",
                f"{f} ঠিক থাকলে বাকিটা মিলেই যায়",
                f"{f} ই ঠিক গেম বদলায়",
            ]
        if script == "hi":
            return [
                f"{f} यहाँ असली काम दिखता है, शोर नहीं",
                f"{f} सही हो तो बाकी अपने आप सेट",
                f"{f} पर टिके रहो, बातें साफ़",
            ]
        if script == "ar":
            return [
                f"{f} هو الجزء العملي بعيداً عن الضجيج",
                f"لو ركّزنا على {f} الصورة توضّح",
                f"{f} هنا يغيّر النتيجة فعلاً",
            ]
        if script == "ja":
            return [
                f"{f} の実務的な部分が要点だよ",
                f"{f} に注目すると全体が見えてくる",
                f"{f} が効いてるから話が進む",
            ]
        if script == "ko":
            return [
                f"{f} 가 핵심이고 나머진 따라와요",
                f"{f} 보고 있으면 그림이 깔끔해져요",
                f"{f} 얘기가 제일 현실적이에요",
            ]
        if script == "zh":
            return [
                f"{f} 才是重點，別被噪音帶偏",
                f"抓住 {f}，其他自然順起來",
                f"{f} 才是實打實的關鍵",
            ]
        # generic non-EN fallback
        return [
            f"{f} is the practical bit here",
            f"keep eyes on {f}, rest follows",
            f"{f} is where it turns real",
        ]

    def _enforce_length_cjk(
        self,
        s: str,
        min_chars: int = 12,
        max_chars: int = 48,
    ) -> str:
        """Length guard for CJK/ja/ko where 'word' counts aren't meaningful."""
        s = re.sub(r"\s+", " ", s or "").strip()
        if len(s) > max_chars:
            s = s[:max_chars].rstrip()
        return s

    def _make_native_comment(self, text: str, ctx: Dict[str, Any]) -> Optional[str]:
        key = extract_keywords(text)
        focus = pick_focus_token(key) or "this"
        script = ctx.get("script", "latn")
        buckets = self._native_buckets(script)
        last = ""

        for _ in range(32):
            out = normalize_ws(random.choice(buckets).format(focus=focus))
            if self._violates_ai_blocklist(out):
                continue
            if not self._diversity_ok(out):
                last = out
                continue
            if comment_seen(out):
                last = out
                continue

            remember_template(re.sub(r"\b\w+\b", "w", out)[:80])
            remember_comment(out)
            remember_opener(_openers(out))
            remember_ngrams(out)

            if script in {"ja", "ko", "zh"}:
                return self._enforce_length_cjk(out) or out
            return enforce_word_count_natural(out, 6, 13)

        if last:
            if script in {"ja", "ko", "zh"}:
                return self._enforce_length_cjk(last) or last
            return enforce_word_count_natural(last, 6, 13)
        return None

    def _fixed_buckets(
        self,
        ctx: Dict[str, Any],
        topic: str,
        is_crypto: bool,
        sentiment: str,
    ) -> Dict[str, List[str]]:
        focus_slot = "{focus}"
        name_pref = ""

        if self.random.random() < 0.30:
            if ctx.get("handle"):
                name_pref = f"@{ctx['handle']} "
            elif ctx.get("author_name"):
                name_pref = f"{ctx['author_name'].split()[0]}, "

        def P(s: str) -> str:
            return f"{name_pref}{s}"

        # base CT / professional buckets
        react = [
            P(f"{focus_slot} take actually feels grounded"),
            P(f"Hard to disagree with this view on {focus_slot}"),
            P(f"Have been nodding along reading about {focus_slot}"),
            P(f"Kinda lines up with my experience of {focus_slot}"),
            P(f"Nice to see someone phrase {focus_slot} this clearly"),
        ]

        convo = [
            P(f"Curious where {focus_slot} goes if this plays out"),
            P(f"Real conversation people have about {focus_slot}"),
            P(f"Been hearing similar chats around {focus_slot} lately"),
            P(f"Low key everyone is thinking this about {focus_slot}"),
            P(f"Interested to hear more stories around {focus_slot}"),
        ]

        calm = [
            P(f"Sensible breakdown of {focus_slot} without drama"),
            P(f"Grounded walk through {focus_slot} step by step"),
            P(f"Helps keep {focus_slot} in perspective over hype"),
            P(f"Good reminder not to overreact to {focus_slot} stuff"),
            P(f"Frames {focus_slot} without the usual noise"),
        ]

        vibe = [
            P(f"{focus_slot} feels very timeline core right now"),
            P(f"The read around {focus_slot} here feels real"),
            P(f"This hits the everyday side of {focus_slot} nicely"),
            P(f"Quietly one of the better posts on {focus_slot}"),
        ]

        nuance = [
            P(f"Nuance around {focus_slot} helps more than takes"),
            P(f"Not pushing an extreme angle on {focus_slot} actually helps"),
            P(f"Good mix of context and restraint around {focus_slot}"),
        ]

        quick = [
            P(f"Honestly this is how {focus_slot} tends to go"),
            P(f"Kind of exactly what {focus_slot} looks like in practice"),
            P(f"Hard not to recognise {focus_slot} in this"),
        ]

        # KOL / CT alpha-ish bucket
        kol = [
            P(f"{focus_slot} is where serious CT eyes are parked rn"),
            P(f"{focus_slot} reads like early narrative, not exit liquidity"),
            P(f"{focus_slot} is what desks actually model risk around"),
            P(f"{focus_slot} feels like the lever, not the headline"),
            P(f"Respecting {focus_slot} flow, not just timeline noise"),
    P(f"{focus_slot} is the kind of edge you only see early"),
    P(f"{focus_slot} is tradable if liquidity stays decent"),
    P(f"{focus_slot} looks like positioning, not retail hype"),
    P(f"{focus_slot} is where the real risk sits rn"),
    P(f"{focus_slot} needs confirmation from price, not quotes"),
    P(f"{focus_slot} is the part market makers will punish"),
    P(f"{focus_slot} is the pivot before sentiment catches up"),
    P(f"{focus_slot} is the only part worth tracking daily"),
    P(f"{focus_slot} feels like accumulation if you zoom out"),
    P(f"{focus_slot} feels crowded if everyone agrees instantly"),
        ]

        kol_q = [
    P(f"Where does {focus_slot} liquidity actually come from?"),
    P(f"What's the catalyst for {focus_slot} besides narrative?"),
    P(f"Is {focus_slot} real demand or just rotation?"),
    P(f"Does {focus_slot} hold when volatility spikes?"),
]
        buckets: Dict[str, List[str]] = {
            "react": react,
            "conversation": convo,
            "calm": calm,
            "vibe": vibe,
            "nuanced": nuance,
            "quick": quick,
            "kol": kol,
            "kol_q": kol_q,
        }

        if topic == "chart":
            buckets["chart"] = [
                P(f"Those levels on {focus_slot} line up with price memory"),
                P(f"Risk/reward around {focus_slot} is laid out cleanly"),
                P(f"Helps frame entries and exits around {focus_slot}"),
            ]
            buckets["chart_risk"] = [
                P(f"Not advice but {focus_slot} risk profile matters more than hype"),
                P(f"Position sizing around {focus_slot} matters more than narratives fr"),
            ]
        elif topic == "meme":
            buckets["meme"] = [
                P(f"This is exactly how {focus_slot} feels some days"),
                P(f"Can not unsee this version of {focus_slot} now"),
                P(f"Joke lands because {focus_slot} is way too real"),
            ]
            buckets["sarcasm"] = [
                P(f"Yeah {focus_slot} totally super healthy behavior obviously"),
                P(f"{focus_slot} speedrun straight to therapist arc lol"),
            ]
        elif topic == "complaint":
            buckets["complaint"] = [
                P(f"Totally fair to be burnt out by {focus_slot}"),
                P(f"Nice to see someone admit {focus_slot} is exhausting"),
                P(f"Feels like no one in charge understands {focus_slot}"),
            ]
        elif topic in ("announcement", "update"):
            buckets["announcement"] = [
                P(f"Ship first talk later energy around {focus_slot} is nice"),
                P(f"Concrete steps on {focus_slot} beat teasers"),
                P(f"Real update on {focus_slot} > vague roadmap"),
            ]
        elif topic == "thread":
            buckets["thread"] = [
                P(f"Thread layers context on {focus_slot} well"),
                P(f"Bookmarking this as a reference for {focus_slot}"),
                P(f"Clean structure makes {focus_slot} easy to follow"),
            ]
        elif topic == "one_liner":
            buckets["one_liner"] = [
                P(f"Blunt but fair on {focus_slot}"),
                P(f"Straightforward way to frame {focus_slot} without fluff"),
            ]

        if is_crypto:
            buckets["crypto"] = [
                P(f"Onchain side of {focus_slot} finally getting discussed honestly"),
                P(f"Nice blend of risk and conviction for {focus_slot} here"),
                P(f"Better than the usual moon talk around {focus_slot}"),
            ]

        # sentiment-aware tweaks
        if sentiment == "bullish":
            buckets["bullish"] = [
                P(f"{focus_slot} looks like early upside, not late fomo"),
                P(f"Respecting {focus_slot} momentum but sizing like an adult"),
            ]
        elif sentiment == "bearish":
            buckets["skeptic"] = [
                P(f"{focus_slot} feels toppy, risk needs real respect rn"),
                P(f"Glad someone is naming {focus_slot} downside cleanly"),
            ]

        if self.random.random() < 0.5 and ctx.get("author_name"):
            first = ctx["author_name"].split()[0]
            buckets["author"] = [
                P(f"{first} keeps a plain language angle on {focus_slot}"),
                P(f"Trust {first} more on {focus_slot} after posts like this"),
            ]

        return buckets

    def _english_candidate(self, text: str, ctx: Dict[str, Any]) -> Optional[str]:
        topic = detect_topic(text)
        crypto = is_crypto_tweet(text)
        key = extract_keywords(text)
        sentiment = detect_sentiment(text)

        if random.random() < 0.7:
            out = _combinator(ctx, key)
        else:
            buckets = self._fixed_buckets(ctx, topic, crypto, sentiment)
            kind = random.choice(list(buckets.keys()))
            tmpl = random.choice(buckets[kind])
            if template_burned(tmpl):
                return None
            focus = pick_focus_token(key) or "this"
            out = tmpl.format(focus=focus)

        out = self._tidy_en(out)
        return out or None

    def _accept(self, line: str) -> bool:
        if self._violates_ai_blocklist(line):
            return False
        # Hard-ban overused template fragments
        if contains_pattern_phrase(line):
            return False
        if not self._diversity_ok(line):
            return False
        if comment_seen(line):
            return False
        # 🚫 Block repeated sentence skeletons across tweets
        if template_burned(line):
            return False
        if not pro_kol_ok(line):
            return False
        return True


    def _commit(self, line: str, url: str = "", lang: str = "en") -> None:
        remember_template(re.sub(r"\b\w+\b", "w", line)[:80])
        remember_comment(line, url=url, lang=lang)
        remember_opener(_openers(line))
        remember_ngrams(line)

    def generate_two(
        self,
        text: str,
        author: Optional[str],
        handle: Optional[str],
        lang_hint: Optional[str],
        url: str = "",
    ) -> List[Dict[str, Any]]:
        ctx = build_context_profile(text, url=url, tweet_author=author, handle=handle)
        out: List[Dict[str, Any]] = []
        non_en = ctx["script"] != "latn"

        # if non-Latin, try to include one native + one EN
        if non_en:
            native = self._make_native_comment(text, ctx)
            if native and self._accept(native):
                if ctx["script"] in {"ja", "ko", "zh"}:
                    native = self._enforce_length_cjk(native)
                else:
                    native = enforce_word_count_natural(native, 6, 13)
                self._commit(native, url=url, lang=ctx["script"])
                out.append({"lang": ctx["script"], "text": native})

        # fill with English until we have 2
        tries = 0
        while len(out) < 2 and tries < 80:
            tries += 1
            cand = self._english_candidate(text, ctx)
            if not cand:
                continue
            cand = enforce_word_count_natural(cand, 6, 13)
            if not cand:
                continue
            if any(cand.strip().lower() == c["text"].strip().lower() for c in out):
                continue
            if self._accept(cand):
                self._commit(cand, url=url, lang="en")
                out.append({"lang": "en", "text": cand})

        # hard guarantee two comments
        if len(out) < 2:
            out += [
                {"lang": "en", "text": enforce_word_count_natural(s, 6, 13)}
                for s in _rescue_two(text)
            ]
            out = [c for c in out if c["text"]][:2]

        # keep EN#1 / EN#2 from being near-duplicates
        if len(out) == 2 and _pair_too_similar(out[0]["text"], out[1]["text"]):
            extras = [_rescue_two(text)[0]]
            extras = [enforce_word_count_natural(e, 6, 13) for e in extras if e]
            for e in extras:
                if not e:
                    continue
                if not self._accept(e):
                    continue
                out[1] = {"lang": "en", "text": e}
                break

        return out[:2]


# Utilities used by the generator
def build_context_profile(raw_text: str, url: Optional[str] = None, tweet_author: Optional[str] = None, handle: Optional[str] = None) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if url and not handle:
        try:
            p = urlparse(url); segs = [s for s in p.path.split("/") if s]
            if segs: handle = segs[0]
        except Exception: pass
    script = "latn"
    # Count script signals (order matters for disambiguation)
    text_no_urls = re.sub(r"https?://\S+", "", text)
    total_letters = len(re.findall(r"[^\W\d_]", text_no_urls, flags=re.UNICODE))

    # Heuristics for scripts
    counts = {
        "ja_hira_kata": len(re.findall(r"[\u3040-\u30FF]", text_no_urls)),   # Hiragana + Katakana
        "ko": len(re.findall(r"[\uAC00-\uD7AF]", text_no_urls)),             # Hangul Syllables
        "cjk": len(re.findall(r"[\u4E00-\u9FFF]", text_no_urls)),            # CJK Unified Ideographs
        "bn": len(re.findall(r"[\u0980-\u09FF]", text_no_urls)),
        "hi": len(re.findall(r"[\u0900-\u097F]", text_no_urls)),
        "ar": len(re.findall(r"[\u0600-\u06FF]", text_no_urls)),
        "ta": len(re.findall(r"[\u0B80-\u0BFF]", text_no_urls)),
        "te": len(re.findall(r"[\u0C00-\u0C7F]", text_no_urls)),
        "ur": len(re.findall(r"[\u0600-\u06FF]", text_no_urls)),
    }
    if total_letters:
        # Prefer Japanese if kana present significantly
        if counts["ja_hira_kata"] / total_letters >= 0.15:
            script = "ja"
        elif counts["ko"] / total_letters >= 0.25:
            script = "ko"
        elif counts["cjk"] / total_letters >= 0.25:
            script = "zh"
        elif counts["bn"] / total_letters >= 0.25:
            script = "bn"
        elif counts["hi"] / total_letters >= 0.25:
            script = "hi"
        elif counts["ar"] / total_letters >= 0.25:
            script = "ar"
        elif counts["ta"] / total_letters >= 0.25:
            script = "ta"
        elif counts["te"] / total_letters >= 0.25:
            script = "te"
        elif counts["ur"] / total_letters >= 0.25:
            script = "ur"

    return {"author_name": (tweet_author or "").strip() or None,
            "handle": (handle or "").strip() or None,
            "script": script}


def _rescue_two(tweet_text: str) -> List[str]:
    base = re.sub(r"https?://\S+|[@#]\S+", "", tweet_text or "").strip()
    kw = (re.findall(r"[A-Za-z]{3,}", base) or ["this"])[0].lower()
    # CT-ish but still single thought, kept short
    a = enforce_word_count_natural(f"Fair angle on {kw}, makes sense rn", 6, 13)
    b = enforce_word_count_natural(f"Watching how {kw} actually plays out rn", 6, 13)
    if not a: a = "Makes sense rn tbh still though right"
    if not b: b = "Watching where this goes rn tbh still"
    return [a, b]

def build_canonical_x_url(original_url: str, t: Any) -> str:
    """
    Build https://x.com/<handle>/status/<tweet_id> when possible.
    Fallback: return original_url.
    """
    try:
        handle = getattr(t, "handle", None)
        tweet_id = getattr(t, "tweet_id", None) or getattr(t, "id", None)

        if handle and tweet_id:
            return f"https://x.com/{handle}/status/{tweet_id}"
    except Exception:
        pass
    return original_url

VX_API_BASE = "https://api.vxtwitter.com"


def fetch_thread_context(url: str, tweet_data: Any | None = None) -> Optional[dict]:
    """
    Best-effort VXTwitter context fetch.
    Extracts:
      - quoted_tweet text
      - parent / reply-to tweet text
    Returns small dict or None.
    """
    if not ENABLE_THREAD_CONTEXT:
        return None

    tweet_id = getattr(tweet_data, "tweet_id", None) or getattr(tweet_data, "id", None)
    if not tweet_id:
        m = re.search(r"status/(\d+)", url)
        if m:
            tweet_id = m.group(1)
    if not tweet_id:
        return None

    try:
        resp = requests.get(f"{VX_API_BASE}/Twitter/status/{tweet_id}", timeout=2)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    quoted_text = None
    parent_text = None

    try:
        root = data.get("tweet") or data

        # quoted tweet, various possible keys
        for key in ("quoted_tweet", "quoted_status", "quote"):
            obj = root.get(key)
            if isinstance(obj, dict):
                quoted_text = obj.get("text") or obj.get("full_text") or quoted_text
                if quoted_text:
                    break

        # parent / reply-to
        for key in ("reply_to", "in_reply_to_tweet", "in_reply_to_status", "parent"):
            obj = root.get(key)
            if isinstance(obj, dict):
                parent_text = obj.get("text") or obj.get("full_text") or parent_text
                if parent_text:
                    break
    except Exception:
        quoted_text = quoted_text or None
        parent_text = parent_text or None

    if not (quoted_text or parent_text):
        return None

    return {
        "tweet_id": str(tweet_id),
        "quoted_text": quoted_text,
        "parent_text": parent_text,
    }

def _build_context_json_snippet() -> str:
    """
    Construct a compact JSON blob with thread + research context.
    Injected into every provider prompt.
    """
    ctx: dict[str, Any] = {}
    thread_ctx = REQUEST_THREAD_CTX.get(None)
    research_ctx = REQUEST_RESEARCH_CTX.get(None)

    if thread_ctx:
        ctx["thread"] = thread_ctx
    if research_ctx:
        ctx["research"] = research_ctx

    if not ctx:
        return ""

    try:
        blob = json.dumps(ctx, ensure_ascii=False)
    except Exception:
        blob = str(ctx)

    # Keep it reasonably small for prompts
    return (
        "\n\nExtra context (JSON, for grounding; "
        "you may quote from it but MUST NOT invent beyond it):\n"
        + blob[:2000]
    )

def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        m = re.search(r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/([^/]+)/status/", url, re.I)
        return m.group(1) if m else None
    except Exception:
        return None

generator = OfflineCommentGenerator()

# --- Minimal helpers used by Groq path ---
def words(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")

def _sanitize_comment(raw: str) -> str:
    txt = re.sub(r"https?://\S+", "", raw or "")
    txt = re.sub(r"[@#]\S+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"[.!?;:…]+$", "", txt).strip()
    txt = EMOJI_PATTERN.sub("", txt)
    return txt

def _strip_second_clause(text: str) -> str:
    """
    Enforce 'one thought only':
    cut trailing polite fluff like 'thanks for sharing', 'appreciate it', etc.
    """
    low = text.lower()
    cut_tokens = [
        " thanks for sharing"," thank you for sharing"," thanks for this",
        " appreciate you"," appreciate it"," appreciate this",
        " thanks fam"," thanks man"," thanks bro"," thank you"," thanks",
        " cheers "," cheers,", " cheers."
    ]
    cut_positions = [low.find(tok) for tok in cut_tokens if low.find(tok) > 0]
    if cut_positions:
        cut_at = min(cut_positions)
        text = text[:cut_at]
    text = re.sub(r"[,\-–]+$", "", text).strip()
    return text

QUESTION_HEADS = ["how","what","why","when","where","who","can","could","do","did","are","is","will","would","should"]
QUESTION_PHRASES = ["any chance", "curious if", "wondering if", "what's the plan", "what is the plan"]

def _ensure_question_punctuation(text: str) -> str:
    """
    If a line clearly *reads* like a question but has no '?', add it.
    """
    t = text.strip()
    low = t.lower()
    if "?" in t:
        return t
    is_question = False
    for h in QUESTION_HEADS:
        if low.startswith(h + " "):
            is_question = True
            break
    if not is_question:
        for ph in QUESTION_PHRASES:
            if ph in low:
                is_question = True
                break
    if is_question:
        return t + "?"
    return t

def enforce_word_count_natural(raw: str, min_w: int = 6, max_w: int = 13) -> str:
    """
    Shared final cleaner for ALL comments (offline + Groq + OpenAI + Gemini).
    - strips links/handles/emojis
    - enforces 6–13 tokens
    - cuts second polite clause ("thanks for sharing" etc)
    - removes some AI-ish fillers
    - adds '?' to clear questions
    """
    txt = _sanitize_comment(raw)
    txt = kol_polish(txt)

    # kill ultra-generic openers like "love that", "excited to see"
    txt_low = txt.lower().lstrip()
    bad_starts = [
        "love that ","love this ","love the ","love your ",
        "excited to see","excited for","can't wait to","cant wait to",
        "glad to see","happy to see","this is huge","this is massive",
        "this could be huge","this is insane",
    ]
    for bs in bad_starts:
        if txt_low.startswith(bs):
            # drop the opener chunk
            txt_low = txt_low[len(bs):].lstrip()
            txt = txt_low
            break

    toks = words(txt)
    if not toks:
        return ""
    if len(toks) > max_w:
        toks = toks[:max_w]

    fillers = ["notably", "overall", "net", "frankly", "basically"]
    i = 0
    while len(toks) < min_w and i < len(fillers):
        toks.append(fillers[i])
        i += 1

    out = " ".join(toks).strip()

    # Enforce single thought: strip second clause like "thanks for sharing"
    out = _strip_second_clause(out)

    # Remove our own filler if now pointless
    low = out.lower()
    if any(b in low for b in AI_BLOCKLIST) or contains_generic_phrase(low):
        out = " ".join(
            t for t in out.split()
            if t.lower() not in {"honestly", "tbh", "still", "though"}
        ) or out
        out = out.strip()

    out = _ensure_question_punctuation(out)

    if PRO_KOL_POLISH:
        out = pro_kol_polish(out, topic=detect_topic(raw or ""))
    return out

def kol_polish(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # remove common hype openers
    t = re.sub(r"^(wow|omg|yo|bro)\b[,\s]*", "", t, flags=re.I)

    # remove exclamation
    t = t.replace("!", "")

    # remove "sounds interesting" vagueness
    t = re.sub(r"\bsounds interesting\b", "worth watching", t, flags=re.I)

    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t

def guess_mode(text: str) -> str:
    """
    Very rough mode guess:
    - 'question' for obvious questions / curiosity
    - 'skeptical' for doubt / concern
    - 'support' for congrats / bullish / positive tone
    - 'playful' for meme-ish language
    """
    t = (text or "").strip().lower()
    if not t:
        return "support"

    if "?" in t or any(
        ph in t
        for ph in (
            "how ", "what ", "why ", "when ", "where ", "can you",
            "do you", "could you", "would you", "any chance",
            "curious", "wondering", "what's the plan", "whats the plan",
            "what is the plan"
        )
    ):
        return "question"

    if any(p in t for p in ("worried", "concerned", "not sure", "unsure", "doubt", "skeptical", "risk here")):
        return "skeptical"

    if any(
        p in t
        for p in (
            "congrats", "congratulations", "glad to", "love this", "love that",
            "bullish", "nice move", "well done", "great work", "clean work",
            "happy to see"
        )
    ):
        return "support"

    if any(p in t for p in ("lol", "lmao", "meme", "kinda wild", "ratio", "cope", "ngmi")):
        return "playful"

    return "support"


VOICE_CARDS = [
    {
        "id": "trader",
        "description": "CT trader: focuses on risk, liquidity, levels, entries/exits, timeframe. No moonboy hype.",
        "boost_topics": {"chart": 2.0},
        "boost_if_crypto": 1.7,
    },
    {
        "id": "builder",
        "description": "Builder / dev: talks about execution, DX, architecture, roadmap realism.",
        "boost_topics": {"thread": 1.8, "announcement": 1.4},
        "boost_if_crypto": 1.3,
    },
    {
        "id": "researcher",
        "description": "On-chain researcher: cares about data, TVL, flows, incentives, mech design.",
        "boost_topics": {"thread": 1.6},
        "boost_if_crypto": 1.8,
    },
    {
        "id": "skeptic",
        "description": "Skeptic: stress-tests assumptions, points out risk and failure modes.",
        "boost_topics": {"complaint": 2.0},
        "boost_if_crypto": 1.4,
    },
    {
        "id": "curious_friend",
        "description": "Curious friend: grounded, non-hype, asking simple but sharp questions.",
        "boost_topics": {"generic": 1.3, "one_liner": 1.5},
        "boost_if_crypto": 1.0,
    },
    {
        "id": "deadpan_meme",
        "description": "Deadpan meme enjoyer: dry, low-energy, mildly ironic but still coherent.",
        "boost_topics": {"meme": 2.4},
        "boost_if_crypto": 1.1,
    },
]


def _pick_voice_card(tweet_text: str) -> dict:
    topic = detect_topic(tweet_text or "")
    if topic not in {"greeting","giveaway","chart","complaint","announcement","meme","thread","one_liner","generic"}:
        topic = "generic"
    crypto = is_crypto_tweet(tweet_text or "")

    weights: list[float] = []
    cards: list[dict] = []

    for card in VOICE_CARDS:
        w = 1.0
        boost_topics = card.get("boost_topics") or {}
        if topic in boost_topics:
            w *= boost_topics[topic]
        if crypto:
            w *= card.get("boost_if_crypto", 1.0)

        # light penalty for very recent voices to avoid repetition
        if _RECENT_VOICES and card["id"] == _RECENT_VOICES[-1]:
            w *= 0.4
        elif card["id"] in _RECENT_VOICES[-3:]:
            w *= 0.7

        weights.append(w)
        cards.append(card)

    # normalize & sample
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    chosen = random.choices(cards, weights=weights, k=1)[0]
    _RECENT_VOICES.append(chosen["id"])
    if len(_RECENT_VOICES) > 16:
        _RECENT_VOICES.pop(0)
    return chosen


def set_request_voice(tweet_text: str) -> None:
    if not tweet_text:
        REQUEST_VOICE.set(None)
        return
    REQUEST_VOICE.set(_pick_voice_card(tweet_text))


def current_voice_card() -> Optional[dict]:
    return REQUEST_VOICE.get(None)


def pick_two_diverse_text(candidates: list[str]) -> list[str]:
    """
    Hybrid selector:
    - prefers two comments with DIFFERENT modes (support vs question etc)
    - also tries to keep trigram overlap low
    - if one is a question and the other is not, order as: [statement, question]
    """
    # clean + dedupe
    uniq: list[str] = []
    for c in candidates:
        c = (c or "").strip()
        if c and c not in uniq:
            uniq.append(c)

    if not uniq:
        return []

    if len(uniq) == 1:
        return uniq

    # if only two, we still might reorder by mode
    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]
        m1, m2 = guess_mode(a), guess_mode(b)
        # statement first, question second
        if m1 == "question" and m2 != "question":
            return [b, a]
        if m2 == "question" and m1 != "question":
            return [a, b]
        return [a, b]

    # more than two: search best pair (different modes + low similarity)
    scored = [(c, guess_mode(c)) for c in uniq]
    best_pair: Optional[tuple[str, str]] = None
    best_score = 999.0  # lower better

    for i, (c1, m1) in enumerate(scored):
        for c2, m2 in scored[i + 1 :]:
            ta = _word_trigrams(c1)
            tb = _word_trigrams(c2)
            if ta and tb:
                inter = len(ta & tb)
                uni = len(ta | tb)
                sim = inter / uni if uni else 0.0
            else:
                sim = 0.0

            # preference: different modes
            mode_penalty = 0.0 if m1 != m2 else 0.4
            score = sim + mode_penalty
            if score < best_score:
                best_score = score
                best_pair = (c1, c2)

    if not best_pair:
        best_pair = (uniq[0], uniq[1])

    a, b = best_pair
    m1, m2 = guess_mode(a), guess_mode(b)

    # reorder so question goes second when we have exactly one
    if m1 == "question" and m2 != "question":
        return [b, a]
    if m2 == "question" and m1 != "question":
        return [a, b]
    return [a, b]

def enforce_unique(candidates: list[str], tweet_text: Optional[str] = None) -> list[str]:
    """
    Enforce uniqueness + non-generic quality for a set of candidate comments.

    What it does:
    - sanitize + enforce 6–13 tokens
    - drop generic phrases
    - skip past repeats / templates / n-gram overlap / near-duplicates
    - prefer two comments with different "modes" (statement + question when possible)
    - commits ONLY the final selected comments to the DB (so we don't burn patterns)
    """
    tweet_text = (tweet_text or "").strip()

    def _prep(x: str) -> str:
        # enforce_word_count_natural already sanitizes links/handles/emojis + cuts 2nd clause
        return (enforce_word_count_natural(x, 6, 13) or "").strip()

    # ----------------------------
    # Pass 1: strict filtering
    # ----------------------------
    pool: list[str] = []
    seen_fps: set[str] = set()
    thesis_seen = False

    for raw in (candidates or []):
        c = _prep(raw)
        if not c:
            continue
        low = c.lower()

        # Soft cap on "thesis" spam (keeps vibe varied)
        if "thesis" in low:
            if thesis_seen:
                continue
            thesis_seen = True

        if contains_generic_phrase(c):
            continue

        # Drop ultra-templatey CT phrases ("risk/reward around X", etc.)
        if contains_pattern_phrase(c):
            continue

        if PRO_KOL_STRICT and not pro_kol_ok(c, tweet_text=tweet_text):
            continue

        if tweet_text and not hallucination_safe(c, tweet_text):
            continue

        # Per-request structural dedupe
        fp = style_fingerprint(c)
        if fp and fp in seen_fps:
            continue

        # Historical repetition guards
        if comment_seen(c):
            continue
        if template_burned(c):
            continue
        if opener_seen(_openers(c)):
            continue
        if trigram_overlap_bad(c, threshold=2):
            continue
        if too_similar_to_recent(c):
            continue

        pool.append(c)
        if fp:
            seen_fps.add(fp)

    picked: list[str] = pick_two_diverse_text(pool)[:2] if pool else []

    # ----------------------------
    # Pass 2: relaxed filtering if we still need output
    # (avoid hard-failing into the same rescue lines)
    # ----------------------------
    if len(picked) < 2:
        relaxed: list[str] = []
        seen_fps2: set[str] = set()
        thesis_seen2 = False

        for raw in (candidates or []):
            c = _prep(raw)
            if not c:
                continue
            low = c.lower()

            if "thesis" in low:
                if thesis_seen2:
                    continue
                thesis_seen2 = True

            if contains_generic_phrase(c):
                continue

            # Still never allow pattern phrases, even in relaxed mode
            if contains_pattern_phrase(c):
                continue

            if PRO_KOL_STRICT and not pro_kol_ok(c, tweet_text=tweet_text):
                continue
            if tweet_text and not hallucination_safe(c, tweet_text):
                continue

            fp = style_fingerprint(c)
            if fp and fp in seen_fps2:
                continue

            if comment_seen(c):
                continue

            # Relax: allow template_burned + small n-gram overlap,
            # but still block openers and near-paraphrases of recent outputs.
            if opener_seen(_openers(c)):
                continue
            if too_similar_to_recent(c, threshold=0.72):
                continue

            relaxed.append(c)
            if fp:
                seen_fps2.add(fp)

        if relaxed and not picked:
            picked = pick_two_diverse_text(relaxed)[:2]
        elif relaxed:
            picked = pick_two_diverse_text(picked + relaxed)[:2]

    # Commit ONLY what we return (prevents burning good patterns prematurely)
    for c in picked:
        try:
            remember_comment(c)
            remember_template(c)
            remember_opener(_openers(c))
            remember_ngrams(c)
        except Exception:
            pass

    return picked[:2]



PRO_BAD_PHRASES = {
    "wow", "exciting", "huge", "insane", "amazing", "awesome",
    "love this", "love that", "can't wait", "cant wait", "sounds interesting",
    "thanks for sharing", "appreciate you","difference between thesis and cope",
    "thesis and cope",
    "thesis stays",
}

PRO_OPERATOR_WORDS = {
    "risk","liquidity","flow","incentives","execution","timeline",
    "positioning","constraints","tradeoffs","demand","supply",
    "mechanics","pricing","distribution","sizing","volatility","edge",
}

# Overused "template-y" fragments we never want in final comments
PATTERN_PHRASES = {
    "signal over noise:",
    "quick take:",
    "nuts and bolts:",

    "risk mostly hides in",
    "risk/reward around",
    "risk reward around",
    "risk profile matters more than hype",
    "risk profile matters more than narratives fr",

    "it lives or dies on",
    "execution shows up as",
    "the market will reward",
    "most errors start before",

    "and incentives line up",
    "nice blend of risk and conviction for",
    "is where the real risk sits rn",
    "hold when volatility spikes",
    "is the practical hinge",
}

def contains_pattern_phrase(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in PATTERN_PHRASES)


def extract_entities(tweet_text: str) -> dict:
    t = tweet_text or ""
    cashtags = re.findall(r"\$\w{2,15}", t)
    handles = re.findall(r"@\w{2,30}", t)
    decimals = re.findall(r"\d+\.\d+", t)
    integers = re.findall(r"\b\d+\b", t)
    return {
        "cashtags": list(dict.fromkeys(cashtags)),
        "handles": list(dict.fromkeys(handles)),
        "numbers": list(dict.fromkeys(decimals + integers)),
    }

def build_research_context_for_tweet(tweet_text: str) -> dict:
    """
    Best-effort DeFi research:
    - use DefiLlama /protocols + /protocol/{slug} + /tvl/{slug}
    - optional CoinGecko for spot price
    - aggressive caching + short timeouts
    Stores:
      {
        "status": "ok" | "empty" | "disabled" | "error",
        "cashtags": [...],
        "protocols": [
          {
            "cashtag": "$SOL",
            "slug": "solana",
            "name": "...",
            "symbol": "SOL",
            "category": "...",
            "chains": [...],
            "url": "...",
            "tvl": 123456789.0,
            "price": {"coin_id": "solana", "usd": 172.3}  # optional
          },
          ...
        ]
      }
    """
    if not ENABLE_RESEARCH:
        return {"status": "disabled"}

    if not is_crypto_tweet(tweet_text or ""):
        return {"status": "empty"}

    ents = extract_entities(tweet_text or "")
    cashtags = ents.get("cashtags") or []
    if not cashtags:
        return {"status": "empty", "cashtags": []}

    key = "tweet:" + "|".join(sorted(cashtags))
    cached = _research_cache_get(key)
    if cached is not None:
        return cached

    protocols: list[dict] = []
    for tag in cashtags[:3]:  # keep it small
        symbol = tag[1:].upper()
        slug = _resolve_protocol_slug_from_symbol(symbol)
        if not slug:
            continue

        llama = _fetch_defillama_for_slug(slug)
        if not llama:
            continue

        proto = llama.get("protocol") or {}
        tvl_data = llama.get("tvl") or []

        entry: dict[str, Any] = {
            "cashtag": tag,
            "slug": slug,
            "name": proto.get("name") or "",
            "symbol": proto.get("symbol") or "",
            "category": proto.get("category") or proto.get("sector") or "",
            "chains": proto.get("chains") or [],
            "url": proto.get("url") or "",
            "tvl": None,
        }

        try:
            if isinstance(tvl_data, list) and tvl_data:
                last = tvl_data[-1]
                entry["tvl"] = float(last.get("totalLiquidityUSD") or 0.0)
        except Exception:
            pass

        # optional price
        price_block = None
        if ENABLE_COINGECKO:
            coin_id = proto.get("gecko_id") or _coingecko_search(symbol)
            if coin_id:
                price = _coingecko_price(coin_id)
                if price:
                    price_block = {"coin_id": coin_id, "usd": price.get("usd")}
        if price_block:
            entry["price"] = price_block

        protocols.append(entry)

    status = "ok" if protocols else "empty"
    ctx = {"status": status, "cashtags": cashtags, "protocols": protocols}
    _research_cache_set(key, ctx)
    return ctx

def hallucination_safe(comment: str, tweet_text: str) -> bool:
    """
    Anti-hallucination guard:
    - Reject new $TICKERs not present in the tweet.
    - Reject large numbers (>10) that are not present in the tweet (string match).
    """
    c = comment or ""
    t = tweet_text or ""
    if not c:
        return False

    # 1) cashtags: comment ⊆ tweet
    comment_tags = set(re.findall(r"\$\w{2,15}", c))
    tweet_ents = extract_entities(t)
    tweet_tags = set(tweet_ents.get("cashtags") or [])
    if comment_tags - tweet_tags:
        return False

    # 2) large numbers
    comment_nums = re.findall(r"\d+(?:\.\d+)?", c)
    tweet_nums = set(re.findall(r"\d+(?:\.\d+)?", t))
    for ns in comment_nums:
        try:
            val = float(ns)
        except ValueError:
            continue
        if val > 10 and ns not in tweet_nums:
            return False

    return True

def has_tweet_anchor(comment: str, tweet_text: str) -> bool:
    """
    Require each comment to hook into the tweet:
    - a cashtag ($SOL)
    - a number (26, 17.99)
    - or one of the top extracted keywords
    """
    if not comment or not tweet_text:
        return False

    ents = extract_entities(tweet_text or "")
    keys = extract_keywords(tweet_text or "")

    anchors = set()

    for tag in ents.get("cashtags") or []:
        anchors.add(tag.lower())

    for num in ents.get("numbers") or []:
        anchors.add(num.lower())

    for k in keys[:8]:
        anchors.add(k.lower())

    low = comment.lower()
    return any(a and a in low for a in anchors)

def pro_kol_ok(comment: str, tweet_text: str = "") -> bool:
    """
    Rejects generic/hypey/off-topic outputs, but still allows meme wit when appropriate.
    """
    c = (comment or "").strip()
    if not c:
        return False
    low = c.lower()

    # Never allow hard-coded template-y phrases (risk/reward around X, incentives line up, etc.)
    if contains_pattern_phrase(c):
        return False

    # ✅ New: comment must anchor to the tweet (ticker/number/keyword)
    if tweet_text and not has_tweet_anchor(c, tweet_text):
        return False

    # must not be pure vague praise
    if len(c.split()) <= 8 and any(x in low for x in ("great", "nice", "good", "cool")) and "?" not in low:
        return False

    # on-topic signal: mention at least ONE entity/keyword/operator angle
    ents = extract_entities(tweet_text or "")
    keys = extract_keywords(tweet_text or "")
    focus_pool = set(
        [k.lower() for k in keys]
        + [x.lower() for x in ents["cashtags"]]
        + [x.lower().lstrip("@") for x in ents["handles"]]
        + [x.lower() for x in ents["numbers"]]
    )

    has_focus = any(fp and fp in low for fp in list(focus_pool)[:12])
    has_operator = any(w in low for w in PRO_OPERATOR_WORDS)

    topic = detect_topic(tweet_text or "")

    # Meme tweets can pass with wit even if no operator words
    if topic == "meme" and PRO_KOL_ALLOW_WIT:
        return True if ("?" in low or len(c.split()) >= 6) else False

    return has_focus or has_operator


def offline_two_comments(text: str, author: Optional[str]) -> list[str]:
    """
    Offline backup:
    - If DISABLE_OFFLINE_TEMPLATES=1 (your default), we do a tiny rule-based
      rescue using _rescue_two and pro_kol filters. No big KOL buckets.
    - Otherwise, use the full OfflineCommentGenerator buckets/combinator.
    """
    # Minimal mode: avoid buckets/combinator templates
    if DISABLE_OFFLINE_TEMPLATES:
        raw = _rescue_two(text)  # returns two short generic-but-safe lines
        # Enforce uniqueness & all the anti-template / anti-hallucination rules
        result = enforce_unique(raw, tweet_text=text)
        return result[:2]

    # Legacy mode: full generator (buckets + combinator)
    items = generator.generate_two(text, author or None, None, None)
    en = [i["text"] for i in items if (i.get("lang") or "en") == "en" and i.get("text")]
    non = [i["text"] for i in items if (i.get("lang") or "en") != "en" and i.get("text")]

    result: list[str] = []
    if en:
        result.append(en[0])
    if len(en) >= 2:
        result.append(en[1])
    elif non:
        result.append(non[0])

    # apply uniqueness + hybrid pairing
    result = enforce_unique(result, tweet_text=text)
    return result[:2]


# ------------------------------------------------------------------------------
# LLM parsing helper shared by providers
# ------------------------------------------------------------------------------
import json
import re
from typing import List

_JSON_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_two_comments_flex(raw: str) -> List[str]:
    """
    Robust parser for Groq/OpenAI/Gemini outputs.

    Accepts:
    - JSON array of strings
    - JSON array of objects with {comment|text|content|en...}
    - JSON object with "comments": [...]
    - JSON object with comment_1/comment_2/en_1/en_2 keys
    - Plain text with EN #1 / EN #2 lines
    - Plain text with one-per-line or "c1 || c2"

    Returns up to 2 *distinct* cleaned strings (no word-count filtering here;
    that still happens later via enforce_word_count_natural etc).
    """
    if not raw:
        return []

    text = raw.strip()
    if not text:
        return []

    # 1. If the model wrapped JSON in ```json ...```, strip the fences first
    m = _JSON_BLOCK_RE.search(text)
    json_candidate = m.group(1).strip() if m else text

    def _try_json(s: str):
        # direct load
        try:
            return json.loads(s)
        except Exception:
            pass
        # try first {...} or [...] segment within text
        m2 = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(1))
            except Exception:
                return None
        return None

    data = _try_json(json_candidate)

    items: List[str] = []

    # 2. JSON array case
    if isinstance(data, list):
        for el in data:
            if isinstance(el, str):
                items.append(el)
            elif isinstance(el, dict):
                for k in ("comment", "text", "content", "en", "en1", "en_1", "en2", "en_2"):
                    v = el.get(k)
                    if isinstance(v, str):
                        items.append(v)
                        break

    # 3. JSON object case
    elif isinstance(data, dict):
        # {"comments": ["..", ".."]} or {"comments": [{text: ...}, ...]}
        comments = data.get("comments")
        if isinstance(comments, list):
            for el in comments:
                if isinstance(el, str):
                    items.append(el)
                elif isinstance(el, dict):
                    for k in ("comment", "text", "content", "en"):
                        v = el.get(k)
                        if isinstance(v, str):
                            items.append(v)
                            break

        # {"comment_1": "...", "comment_2": "..."} / etc.
        for k in (
            "comment_1", "comment1", "first",
            "comment_2", "comment2", "second",
            "en_1", "en1", "en_2", "en2",
        ):
            v = data.get(k)
            if isinstance(v, str):
                items.append(v)

    # 4. EN #1 / EN #2 format in plain text
    if not items:
        m1 = re.search(r"EN\s*#\s*1\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
        m2 = re.search(r"EN\s*#\s*2\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
        if m1:
            items.append(m1.group(1))
        if m2:
            items.append(m2.group(1))

    # 5. Fallback: one-per-line or "c1 || c2"
    if not items:
        # split by lines, drop bullets
        lines = [ln.strip(" -*\t\r\n") for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 2:
            items = lines[:2]
        elif len(lines) == 1:
            parts = [p.strip() for p in re.split(r"\s*\|\|\s*", lines[0]) if p.strip()]
            items = parts[:2]

    # 6. Final cleanup + dedupe
    def _clean(s: str) -> str:
        s = s.strip()
        if (
            len(s) >= 2
            and s[0] in {"'", '"', "“"}
            and s[-1] in {"'", '"', "”"}
        ):
            s = s[1:-1].strip()
        return s

    cleaned: List[str] = []
    seen = set()
    for s in items:
        if not isinstance(s, str):
            continue
        s2 = _clean(s)
        if not s2:
            continue
        if s2 in seen:
            continue
        seen.add(s2)
        cleaned.append(s2)

    return cleaned[:2]


# ------------------------------------------------------------------------------
# Groq generator (exactly 2, 6–13 words, tolerant parsing)
# ------------------------------------------------------------------------------
def groq_two_comments(tweet_text: str, author: str | None) -> list[str]:
    """
    Best-effort Groq path:
    - NEVER raises just because we got <2 comments.
    - Returns 0–2 cleaned comments.
    - Offline generator is responsible for filling gaps.
    """
    if not (USE_GROQ and _groq_client):
        return []

    mode_line = llm_mode_hint(tweet_text)
    sys_prompt = _llm_sys_prompt(mode_line)

    user_prompt = (
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Return exactly two distinct comments (JSON array or two lines)."
    )
    user_prompt += _build_context_json_snippet()

    resp = None
    for attempt in range(3):
        try:
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                n=1,
                max_tokens=160,
                temperature=0.9,
            )
            break
        except Exception as e:
            wait_secs = 0
            try:
                hdrs = getattr(getattr(e, "response", None), "headers", {}) or {}
                ra = hdrs.get("Retry-After")
                if ra:
                    wait_secs = max(1, int(ra))
            except Exception:
                pass
            msg = str(e).lower()
            if not wait_secs and ("429" in msg or "rate" in msg or "quota" in msg or "retry-after" in msg):
                wait_secs = 2 + attempt
            if wait_secs:
                time.sleep(wait_secs)
                continue
            # hard error (network etc) – give up on Groq, let caller try others
            return []

    if resp is None:
        return []

    raw = (resp.choices[0].message.content or "").strip()
    _debug_store(
    "groq",
    {
        "model": GROQ_MODEL,
        "raw": raw,
        "response_id": getattr(resp, "id", None),
    },
)

    candidates = parse_two_comments_flex(raw)
    candidates = [
        restore_decimals_and_tickers(enforce_word_count_natural(c), tweet_text)
        for c in candidates
    ]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)

    # Try to mine extra sentences from raw if we still have <2
    if len(candidates) < 2:
        sents = re.split(r"[.!?]\s+", raw)
        sents = [enforce_word_count_natural(s) for s in sents if s.strip()]
        sents = [s for s in sents if 6 <= len(words(s)) <= 13]
        candidates = enforce_unique(candidates + sents[:2], tweet_text=tweet_text)

    # As a final soft nudge, top up from offline if needed (caller may do this again)
    if len(candidates) < 2:
        offline = offline_two_comments(tweet_text, author)
        candidates = enforce_unique(candidates + offline, tweet_text=tweet_text)

    # Hard cap and light de-dup
    candidates = [c for c in candidates if c][:2]
    if len(candidates) == 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]

    return candidates

# ------------------------------------------------------------------------------
# OpenAI / Gemini generators (same constraints as Groq)
# ------------------------------------------------------------------------------
def _llm_sys_prompt(mode_line: str = "") -> str:
    base = (
        "Human style:\n"
        "- Sound like a smart, grounded CT person (calm, specific, slightly opinionated).\n"
        "- Write each line as an inner reaction, not a public compliment (like a thought you’d type into a chat with a friend).\n"
        "- Avoid hype/fanboy language and vague praise.\n"
        "- Avoid these phrases: wow, exciting, huge, insane, amazing, awesome, love this, can't wait, sounds interesting.\n"
        "- Prefer concrete angles: risk, incentives, liquidity/flow, execution, timeline, tradeoffs, product details.\n"
    )

    # Voice roulette (per-request persona)
    voice = current_voice_card()
    if voice:
        base += (
            "\nVoice profile:\n"
            f"- id: {voice.get('id')}\n"
            f"- style: {voice.get('description')}\n"
            "Write both comments in this voice, but still follow all hard rules.\n"
        )

    # Research realism (anti-hallucination)
    research_ctx = REQUEST_RESEARCH_CTX.get(None)
    if isinstance(research_ctx, dict):
        status = research_ctx.get("status")
        if status in {"empty", "disabled", "error"}:
            base += (
                "\nResearch note:\n"
                "- You either have no reliable research data, or it is incomplete.\n"
                "- If the project is unclear, prefer sharp *questions* over strong claims.\n"
                "- Never invent TVL, yields, valuations, or tokenomics details.\n"
            )
        elif status == "ok":
            base += (
                "\nResearch note:\n"
                "- You have some structured research context for the project.\n"
                "- You may reference it cautiously, but never invent numbers beyond that data.\n"
            )

    mode_line = (mode_line or "").strip()
    if mode_line:
        base += "\n" + mode_line + "\n"
    return base

def openai_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_OPENAI and _openai_client):
        return []

    user_prompt = (
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Return exactly two distinct comments (JSON array or two lines)."
    )
    user_prompt += _build_context_json_snippet()
    mode_line = llm_mode_hint(tweet_text)
    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _llm_sys_prompt(mode_line)},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=160,
            temperature=0.7,
        )
    except Exception:
        return []

    raw = (resp.choices[0].message.content or "").strip()
    _debug_store(
    "openai",
    {
        "model": OPENAI_MODEL,
        "raw": raw,
        "response_id": getattr(resp, "id", None),
    },
)

    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)

    # No hard failures; just return what we have
    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]
    return candidates[:2]

def gemini_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_GEMINI and _gemini_model):
        return []

    user_prompt = (
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Return exactly two distinct comments (JSON array or two lines)."
    )
    user_prompt += _build_context_json_snippet()
    mode_line = llm_mode_hint(tweet_text)
    prompt = _llm_sys_prompt(mode_line) + "\n\n" + user_prompt

    try:
        resp = _gemini_model.generate_content(prompt)
    except Exception:
        return []

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
    _debug_store(
    "gemini",
    {
        "model": GEMINI_MODEL_NAME,
        "raw": raw,
    },
)


    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)

    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]
    return candidates[:2]


def _available_providers() -> list[tuple[str, callable]]:
    """
    Build a list of (name, fn) for all enabled LLM providers.
    Order is randomized per request by the caller.
    """
    providers: list[tuple[str, callable]] = []
    if USE_GROQ and _groq_client:
        providers.append(("groq", groq_two_comments))
    if USE_OPENAI and _openai_client:
        providers.append(("openai", openai_two_comments))
    if USE_GEMINI and _gemini_model:
        providers.append(("gemini", gemini_two_comments))
    return providers

def restore_decimals_and_tickers(comment: str, tweet_text: str) -> str:
    """
    Fix common LLM/tokenization artifacts:
    - if tweet has 17.99 and comment contains '17 99', restore '17.99'
    - if tweet has $SOL and comment contains 'SOL', restore '$SOL' (crypto-only)
    """
    c = comment or ""
    t = tweet_text or ""

    # decimals
    for dec in re.findall(r"\d+\.\d+", t):
        a, b = dec.split(".", 1)
        spaced = f"{a} {b}"
        c = re.sub(rf"\b{re.escape(spaced)}\b", dec, c)

    # tickers (only if tweet looks crypto-ish)
    if is_crypto_tweet(t):
        for cashtag in re.findall(r"\$\w{2,15}", t):
            sym = cashtag[1:]
            # replace standalone symbol if $ version not already present
            if cashtag not in c and re.search(rf"\b{re.escape(sym)}\b", c):
                c = re.sub(rf"\b{re.escape(sym)}\b", cashtag, c, count=1)

    return c

def _pick_rewrite_provider_order() -> list[str]:
    # prefer fast/cheap first
    order = ["groq", "openai", "gemini"]
    random.shuffle(order)
    return order

def _rewrite_sys_prompt(topic: str, sentiment: str) -> str:
    witty = (topic == "meme" and PRO_KOL_ALLOW_WIT)
    return (
        "You rewrite or regenerate two tweet replies.\n"
        "\n"
        "Hard rules:\n"
        "- Output exactly 2 comments.\n"
        "- Each comment must be 6–13 tokens.\n"
        "- One thought per comment (no second clause like 'thanks for sharing').\n"
        "- No emojis, no hashtags, no links.\n"
        "- Do NOT invent facts not present in the tweet.\n"
        "- Preserve numbers and tickers exactly (17.99 stays 17.99, $SOL stays $SOL).\n"
        "\n"
        "Human quality:\n"
        "- Sound like a real person on CT: grounded, specific, slightly opinionated.\n"
        "- Avoid hype/fanboy language and generic praise.\n"
        "- Avoid these phrases: wow, exciting, huge, insane, amazing, awesome, love this, can't wait, sounds interesting.\n"
        "- If the tweet is funny, allow ONE witty/deadpan line.\n"
        "\n"
        "Variety rules:\n"
        "- Comment #1: observation/claim.\n"
        "- Comment #2: sharp question OR risk/constraint note.\n"
        "- Do not reuse the same sentence skeleton (avoid template feel).\n"
        "- Do not start both comments with the same first word.\n"
        "\n"
        f"Context: topic={topic}, sentiment={sentiment}, witty_allowed={str(witty).lower()}.\n"
        "\n"
        "Return a JSON array of two strings: [\"...\", \"...\"].\n"
    )

def pro_kol_rewrite_pair(tweet_text: str, author: Optional[str], seed: list[str]) -> Optional[list[str]]:
    topic = detect_topic(tweet_text or "")
    sentiment = detect_sentiment(tweet_text or "")
    ents = extract_entities(tweet_text or "")
    keys = extract_keywords(tweet_text or "")
    focus = pick_focus_token(keys) or ""

    # recent openers to avoid repeating (helps kill “pattern vibe”)
    recent_openers: list[str] = []
    try:
        with get_conn() as c:
            rows = c.execute("SELECT text FROM comments ORDER BY id DESC LIMIT 30").fetchall()
        recent_openers = list(dict.fromkeys([_openers(t or "") for (t,) in rows if t]))[:20]
    except Exception:
        recent_openers = []

    user_payload = {
        "author": author or "",
        "tweet": tweet_text or "",
        "entities": ents,
        "keywords": keys[:8],
        "focus": focus,
        "seed_comments": seed[:2],
        "avoid_openers": recent_openers[:12],
        "banned_phrases": [
            "it lives or dies on",
            "most errors start before",
            "signal over noise",
            "first principles",
            "quick take:",
            "nuts and bolts:",
            "risk mostly hides in",
            "risk/reward around",
            "risk reward around",
            "execution shows up as",
            "the market will reward",
            "and incentives line up",
            "nice blend of risk and conviction for",
            "is where the real risk sits rn",
            "hold when volatility spikes",
        ],
    }

    sys_prompt = _rewrite_sys_prompt(topic, sentiment)
    user_prompt = (
        "Rewrite/regenerate two better comments based on this JSON.\n"
        "Use the tweet context. If a project is unclear, ask a good question.\n"
        "JSON:\n" + json.dumps(user_payload, ensure_ascii=False)
    )

    for _ in range(PRO_KOL_REWRITE_MAX_TRIES):
        for provider in _pick_rewrite_provider_order():
            try:
                raw = ""
                if provider == "groq" and USE_GROQ and _groq_client:
                    resp = _groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "system", "content": sys_prompt},
                                  {"role": "user", "content": user_prompt}],
                        max_tokens=PRO_KOL_REWRITE_MAX_TOKENS,
                        temperature=PRO_KOL_REWRITE_TEMPERATURE,
                    )
                    raw = (resp.choices[0].message.content or "").strip()

                elif provider == "openai" and USE_OPENAI and _openai_client:
                    resp = _openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "system", "content": sys_prompt},
                                  {"role": "user", "content": user_prompt}],
                        max_tokens=PRO_KOL_REWRITE_MAX_TOKENS,
                        temperature=PRO_KOL_REWRITE_TEMPERATURE,
                    )
                    raw = (resp.choices[0].message.content or "").strip()

                elif provider == "gemini" and USE_GEMINI and _gemini_model:
                    prompt = sys_prompt + "\n\n" + user_prompt
                    resp = _gemini_model.generate_content(prompt)
                    if hasattr(resp, "text"):
                        raw = (resp.text or "").strip()
                    else:
                        raw = str(resp).strip()

                if not raw:
                    continue

                cand = parse_two_comments_flex(raw)
                cand = [restore_decimals_and_tickers(enforce_word_count_natural(x), tweet_text) for x in cand]
                cand = [x for x in cand if 6 <= len(words(x)) <= 13]

                # strict + variety pass
                cand = enforce_unique(cand, tweet_text=tweet_text)

                if len(cand) >= 2:
                    # final pro strict check
                    if PRO_KOL_STRICT and (not pro_kol_ok(cand[0], tweet_text) or not pro_kol_ok(cand[1], tweet_text)):
                        continue
                    if cand[0].split()[0].lower() == cand[1].split()[0].lower():
                        continue
                    if _pair_too_similar(cand[0], cand[1]):
                        continue
                    return cand[:2]

            except Exception as e:
                logger.warning("pro rewrite provider %s failed: %s", provider, e)

    return None

def generate_two_comments_with_providers(
    tweet_text: str,
    author: Optional[str],
    handle: Optional[str],
    lang: Optional[str],
    url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid provider strategy with randomness:

    - If any LLM providers are enabled (Groq/OpenAI/Gemini):
        * try them in randomized order
        * only use LLM-origin comments (no bucket/template generation)
    - If no providers are enabled:
        * fall back to offline generator (buckets/templates)
    """
    candidates: list[str] = []

    providers = _available_providers()
    have_providers = bool(providers)

    # ----------------------------
    # LLM providers path
    # ----------------------------
    if have_providers:
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

    # ----------------------------
    # Offline fallback ONLY if no providers
    # ----------------------------
    if not have_providers and len(candidates) < 2:
        try:
            offline = offline_two_comments(tweet_text, author)
            if offline:
                candidates = enforce_unique(candidates + offline, tweet_text=tweet_text)
        except Exception as e:
            logger.warning("Offline generator failed: %s", e)

    # Last-chance offline generator directly (no providers mode only)
    if not have_providers and len(candidates) < 2:
        try:
            extra_items = generator.generate_two(
                tweet_text,
                author or None,
                handle,
                lang,
                url=url,
            )
            extra: list[str] = []
            for i in (extra_items or []):
                if isinstance(i, dict):
                    txt = i.get("text")
                else:
                    txt = i
                txt = (str(txt).strip() if txt is not None else "")
                if txt:
                    extra.append(txt)

            if extra:
                candidates = enforce_unique(candidates + extra, tweet_text=tweet_text)
        except Exception as e:
            logger.exception("Total failure in provider cascade: %s", e)

    # If still nothing and no providers, use simple rescue lines
    if not have_providers and not candidates:
        raw = _rescue_two(tweet_text)
        candidates = raw

    # Limit to at most 2 raw strings
    candidates = [c for c in candidates if c][:2]

    # ------------------------------------------------------------------
    # Pro KOL rewrite/regenerate pass (LLM-based; keeps rule system)
    # ------------------------------------------------------------------
    if PRO_KOL_REWRITE and candidates:
        needs = (
            len(candidates) < 2
            or (PRO_KOL_STRICT and any(not pro_kol_ok(c, tweet_text) for c in candidates))
            or (len(candidates) == 2 and _pair_too_similar(candidates[0], candidates[1]))
            or (len(candidates) == 2 and candidates[0].split()[0].lower() == candidates[1].split()[0].lower())
            or any(template_burned(c) for c in candidates)
        )
        if needs and have_providers:
            improved = pro_kol_rewrite_pair(tweet_text, author, candidates)
            if improved and len(improved) >= 2:
                candidates = improved[:2]

    out: List[Dict[str, Any]] = []
    for c in candidates:
        out.append({"lang": lang or "en", "text": c})

    # If we somehow have <2 dicts:
    # - In provider mode: just return what we have (LLM-only, no templates).
    # - In offline-only mode: try one more offline fill.
    if len(out) < 2 and not have_providers:
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
                if isinstance(item, dict):
                    txt = item.get("text")
                    ilang = item.get("lang")
                else:
                    txt = item
                    ilang = None
                if not txt:
                    continue
                out.append({
                    "lang": ilang or lang or "en",
                    "text": txt,
                })
        except Exception:
            pass

    return out[:2]

# ------------------------------------------------------------------------------
# API routes (batching + pacing)
# ------------------------------------------------------------------------------

def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def _canonical_x_url_from_tweet(original_url: str, t: TweetData) -> str:
    """
    Build a clean x.com URL with handle + status id when we have them.

    - If upstream payload gives us both handle and tweet_id:
        https://x.com/{handle}/status/{tweet_id}
    - Otherwise fall back to whatever url we already normalized.
    """
    if t.handle and t.tweet_id:
        return f"https://x.com/{t.handle}/status/{t.tweet_id}"
    return original_url


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "ok",
        "groq": bool(USE_GROQ),
        "ts": int(time.time()),
    }), 200


@app.route("/comment", methods=["POST", "OPTIONS"])
def comment_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body", "code": "invalid_json"}), 400

    urls = payload.get("urls")
    if not isinstance(urls, list) or not urls:
        return jsonify({
            "error": "Body must contain non-empty 'urls' array",
            "code": "bad_request",
        }), 400

    try:
        cleaned = clean_and_normalize_urls(urls)
    except CrownTALKError as e:
        return jsonify({"error": str(e), "code": e.code}), 400
    except Exception:
        return jsonify({"error": "url_clean_error", "code": "url_clean_error"}), 400

    # Hard cap per request
    if len(cleaned) > MAX_URLS_PER_REQUEST:
        return jsonify({
            "error": f"Too many URLs in one request; send at most {MAX_URLS_PER_REQUEST} links at a time.",
            "code": "too_many_urls",
            "max_urls_per_request": MAX_URLS_PER_REQUEST,
            "hint": "For best results, chunk your list into batches of around 20–25 links.",
        }), 400

    results: list[dict] = []
    failed: list[dict] = []

    for batch in chunked(cleaned, BATCH_SIZE):
        for url in batch:
            try:
                t = fetch_tweet_data(url)

                # Per-request context: thread, research, voice
                try:
                    thread_ctx = fetch_thread_context(url, t) if ENABLE_THREAD_CONTEXT else None
                except Exception:
                    thread_ctx = None
                REQUEST_THREAD_CTX.set(thread_ctx)

                try:
                    research_ctx = build_research_context_for_tweet(t.text or "") if ENABLE_RESEARCH else {"status": "disabled"}
                except Exception:
                    research_ctx = {"status": "error"}
                REQUEST_RESEARCH_CTX.set(research_ctx)

                set_request_voice(t.text or "")

                # Prefer handle from upstream payload, fall back to URL parsing
                handle = t.handle or _extract_handle_from_url(url)

                two = generate_two_comments_with_providers(
                    t.text,
                    t.author_name or None,
                    handle,
                    t.lang or None,
                    url=url,
                )

                display_url = _canonical_x_url_from_tweet(url, t)

                results.append({
                    "url": display_url,
                    "comments": two,
                })
            except CrownTALKError as e:
                failed.append({
                    "url": url,
                    "reason": str(e),
                    "code": e.code,
                })
            except Exception:
                logger.exception("Unhandled error while processing %s", url)
                failed.append({
                    "url": url,
                    "reason": "internal_error",
                    "code": "internal_error",
                })
            time.sleep(PER_URL_SLEEP)

    return jsonify({"results": results, "failed": failed}), 200


@app.route("/reroll", methods=["POST", "OPTIONS"])
def reroll_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get("url") or ""
        if not url:
            return jsonify({
                "error": "Missing 'url' field",
                "comments": [],
                "code": "bad_request",
            }), 400

        # Fetch tweet info
        t = fetch_tweet_data(url)
        handle = t.handle or _extract_handle_from_url(url)

        # Per-request context (same as /comment)
        try:
            thread_ctx = fetch_thread_context(url, t) if ENABLE_THREAD_CONTEXT else None
        except Exception:
            thread_ctx = None
        REQUEST_THREAD_CTX.set(thread_ctx)

        try:
            research_ctx = build_research_context_for_tweet(t.text or "") if ENABLE_RESEARCH else {"status": "disabled"}
        except Exception:
            research_ctx = {"status": "error"}
        REQUEST_RESEARCH_CTX.set(research_ctx)

        set_request_voice(t.text or "")

        two = generate_two_comments_with_providers(
            t.text,
            t.author_name or None,
            handle,
            t.lang or None,
            url=url,
        )

        display_url = _canonical_x_url_from_tweet(url, t)

        return jsonify({
            "url": display_url,
            "comments": two,
        }), 200

    except CrownTALKError as e:
        app.logger.error("CrownTALK error during reroll for %s", url, exc_info=True)
        return jsonify({
            "url": url,
            "error": str(e),
            "comments": [],
            "code": getattr(e, "code", "crown_error"),
        }), 500

    except Exception:
        app.logger.error("Unhandled error during reroll for %s", url, exc_info=True)
        return jsonify({
            "url": url,
            "error": "internal_error",
            "comments": [],
            "code": "internal_error",
        }), 500

from flask import Flask, request, jsonify

@app.get("/debug")
def debug_llm():
    """
    Debug endpoint: returns the last raw outputs per provider.

    Optional query:
      /debug?provider=groq   -> only Groq
      /debug?provider=openai -> only OpenAI
      /debug?provider=gemini -> only Gemini
    """
    provider = request.args.get("provider")
    if provider:
        data = DEBUG_LAST_LLM.get(provider)
        return jsonify({"provider": provider, "data": data}), 200

    return jsonify(DEBUG_LAST_LLM), 200

# ------------------------------------------------------------------------------
# Boot
# ------------------------------------------------------------------------------

def main() -> None:
    init_db()
    # threading.Thread(target=keep_alive, daemon=True).start()  # optional keep-alive
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
