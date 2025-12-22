from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from contextvars import ContextVar

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional providers
USE_GROQ = os.getenv("USE_GROQ", "1").strip() != "0"
USE_OPENAI = os.getenv("USE_OPENAI", "0").strip() == "1"
USE_GEMINI = os.getenv("USE_GEMINI", "0").strip() == "1"

# Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Models
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Server settings
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")

# DB
DB_PATH = os.getenv("DB_PATH", "comments.db").strip()
DB_LOCK = threading.Lock()

# Behavior
MAX_RECENT_COMMENTS = int(os.getenv("MAX_RECENT_COMMENTS", "200"))
MAX_RECENT_OPENERS = int(os.getenv("MAX_RECENT_OPENERS", "300"))
MAX_RECENT_NGRAMS = int(os.getenv("MAX_RECENT_NGRAMS", "600"))
MAX_RECENT_TEMPLATES = int(os.getenv("MAX_RECENT_TEMPLATES", "250"))
KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))

# ------------------------------------------------------------------------------
# Pro KOL / Research / Context upgrades (safe, best-effort)
# ------------------------------------------------------------------------------
# Turn on/off without breaking main generator:
# - PRO_KOL_MODE=1 enables stricter "KOL quality" gating + optional rewrite
# - ENABLE_THREAD_CONTEXT=1 fetches quote/reply context (best-effort)
# - ENABLE_RESEARCH=1 fetches project context (DefiLlama free; CoinGecko optional)
#
# All network calls are best-effort with short timeouts; failures never block comments.
PRO_KOL_MODE = os.getenv("PRO_KOL_MODE", "0").strip() == "1"
PRO_KOL_REWRITE = os.getenv("PRO_KOL_REWRITE", "1").strip() != "0"
PRO_KOL_ALLOW_WIT = os.getenv("PRO_KOL_ALLOW_WIT", "1").strip() != "0"

ENABLE_THREAD_CONTEXT = os.getenv("ENABLE_THREAD_CONTEXT", "1").strip() != "0"
ENABLE_RESEARCH = os.getenv("ENABLE_RESEARCH", "1").strip() != "0"

# CoinGecko is optional (rate-limited). Prefer DefiLlama by default.
ENABLE_COINGECKO = os.getenv("ENABLE_COINGECKO", "0").strip() == "1"
COINGECKO_DEMO_KEY = os.getenv("COINGECKO_DEMO_KEY", "").strip()

# Caching to avoid rate limits + speed up repeats
RESEARCH_CACHE_TTL_SEC = int(os.getenv("RESEARCH_CACHE_TTL_SEC", "900"))  # 15 min
LLAMA_PROTOCOLS_CACHE_TTL_SEC = int(os.getenv("LLAMA_PROTOCOLS_CACHE_TTL_SEC", "21600"))  # 6 hours

# Anti-hallucination: prevent introducing new $TICKERs / weird numbers
ANTI_HALLUCINATION = os.getenv("ANTI_HALLUCINATION", "1").strip() != "0"

# Request-scoped context (set once per tweet request; read inside provider calls)
REQUEST_RESEARCH_CTX: ContextVar[dict] = ContextVar("REQUEST_RESEARCH_CTX", default={})
REQUEST_THREAD_CTX: ContextVar[dict] = ContextVar("REQUEST_THREAD_CTX", default={})
REQUEST_VOICE: ContextVar[str] = ContextVar("REQUEST_VOICE", default="")

logger = logging.getLogger("crowntalk")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Provider clients (initialized lazily)
# ------------------------------------------------------------------------------
_groq_client = None
_openai_client = None
_gemini_model = None

def init_providers() -> None:
    global _groq_client, _openai_client, _gemini_model

    if USE_GROQ and GROQ_API_KEY and _groq_client is None:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            logger.warning("Groq client init failed: %s", e)

    if USE_OPENAI and OPENAI_API_KEY and _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.warning("OpenAI client init failed: %s", e)

    if USE_GEMINI and GEMINI_API_KEY and _gemini_model is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)

# ------------------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def now_ts() -> int:
    return int(time.time())

def _do_init() -> None:
    with DB_LOCK:
        with get_conn() as c:
            c.executescript(
                """
            CREATE TABLE IF NOT EXISTS comments_seen(
                chash TEXT PRIMARY KEY,
                created_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS comments_openers_seen(
                ohash TEXT PRIMARY KEY,
                created_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS comments_ngrams_seen(
                nhash TEXT PRIMARY KEY,
                created_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS comments_templates_seen(
                thash TEXT PRIMARY KEY,
                created_at INTEGER
            );

            -- Entity memory (project resolution)
            CREATE TABLE IF NOT EXISTS entity_map(
                kind TEXT NOT NULL,
                k TEXT NOT NULL,
                slug TEXT NOT NULL,
                updated_at INTEGER,
                PRIMARY KEY(kind, k)
            );
            """
            )

def init_db() -> None:
    _do_init()

def sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def prune_table(table: str, max_rows: int) -> None:
    with DB_LOCK:
        with get_conn() as c:
            # delete older rows beyond max_rows
            row = c.execute(f"SELECT COUNT(1) FROM {table}").fetchone()
            count = int(row[0]) if row else 0
            if count <= max_rows:
                return
            # keep most recent max_rows
            c.execute(
                f"""
                DELETE FROM {table}
                WHERE rowid NOT IN (
                    SELECT rowid FROM {table}
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (max_rows,),
            )

def remember_comment(text: str) -> None:
    if not text:
        return
    h = sha256(text.strip().lower())
    with DB_LOCK:
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_seen(chash, created_at) VALUES (?, ?)",
                (h, now_ts()),
            )
    prune_table("comments_seen", MAX_RECENT_COMMENTS)

def comment_seen(text: str) -> bool:
    if not text:
        return False
    h = sha256(text.strip().lower())
    with DB_LOCK:
        with get_conn() as c:
            row = c.execute("SELECT 1 FROM comments_seen WHERE chash=? LIMIT 1", (h,)).fetchone()
            return bool(row)

def remember_opener(openers: List[str]) -> None:
    if not openers:
        return
    with DB_LOCK:
        with get_conn() as c:
            ts = now_ts()
            for op in openers[:2]:
                h = sha256(op)
                c.execute(
                    "INSERT OR IGNORE INTO comments_openers_seen(ohash, created_at) VALUES (?, ?)",
                    (h, ts),
                )
    prune_table("comments_openers_seen", MAX_RECENT_OPENERS)

def opener_seen(openers: List[str]) -> bool:
    if not openers:
        return False
    with DB_LOCK:
        with get_conn() as c:
            for op in openers[:2]:
                h = sha256(op)
                row = c.execute(
                    "SELECT 1 FROM comments_openers_seen WHERE ohash=? LIMIT 1", (h,)
                ).fetchone()
                if row:
                    return True
    return False


def remember_ngrams(text: str) -> None:
    if not text:
        return
    # store hashed trigrams
    trigs = _trigrams(text)
    if not trigs:
        return
    with DB_LOCK:
        with get_conn() as c:
            ts = now_ts()
            for t in trigs[:12]:
                h = sha256(t)
                c.execute(
                    "INSERT OR IGNORE INTO comments_ngrams_seen(nhash, created_at) VALUES (?, ?)",
                    (h, ts),
                )
    prune_table("comments_ngrams_seen", MAX_RECENT_NGRAMS)

def trigram_seen(trig: str) -> bool:
    if not trig:
        return False
    h = sha256(trig)
    with DB_LOCK:
        with get_conn() as c:
            row = c.execute("SELECT 1 FROM comments_ngrams_seen WHERE nhash=? LIMIT 1", (h,)).fetchone()
            return bool(row)

def remember_template(text: str) -> None:
    fp = style_fingerprint(text)
    if not fp:
        return
    h = sha256(fp)
    with DB_LOCK:
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_templates_seen(thash, created_at) VALUES (?, ?)",
                (h, now_ts()),
            )
    prune_table("comments_templates_seen", MAX_RECENT_TEMPLATES)

def template_seen(text: str) -> bool:
    fp = style_fingerprint(text)
    if not fp:
        return False
    h = sha256(fp)
    with DB_LOCK:
        with get_conn() as c:
            row = c.execute(
                "SELECT 1 FROM comments_templates_seen WHERE thash=? LIMIT 1", (h,)
            ).fetchone()
            return bool(row)

# ------------------------------------------------------------------------------
# Utilities: normalization, topic detection, etc
# ------------------------------------------------------------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def strip_links(s: str) -> str:
    return re.sub(r"https?://\S+", "", s or "")

def strip_hashtags(s: str) -> str:
    return re.sub(r"#\w+", "", s or "")

def strip_emojis(s: str) -> str:
    # simple emoji strip (best-effort)
    return re.sub(r"[\U00010000-\U0010ffff]", "", s or "")

def clean_comment(s: str) -> str:
    s = normalize_ws(s)
    s = strip_emojis(s)
    s = s.replace("…", "...")
    return normalize_ws(s)

def detect_sentiment(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("love", "great", "amazing", "bull", "exciting", "nice", "🔥", "good")):
        return "positive"
    if any(k in t for k in ("hate", "bad", "bear", "dump", "scam", "rug", "wtf")):
        return "negative"
    return "neutral"

def is_crypto_tweet(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("sol", "eth", "btc", "token", "airdrop", "defi", "nft", "meme", "dex", "perps", "chain", "$"))

def detect_topic(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("chart", "support", "resistance", "breakout", "levels", "liquidation")):
        return "chart"
    if any(k in t for k in ("airdrop", "points", "claim", "eligibility")):
        return "airdrop"
    if any(k in t for k in ("nft", "mint", "floor", "collection")):
        return "nft"
    if any(k in t for k in ("meme", "wen", "gm", "ngmi", "cope", "lol")):
        return "meme"
    if any(k in t for k in ("thread", "🧵", "breakdown", "deep dive", "analysis")):
        return "thread"
    if any(k in t for k in ("defi", "dex", "perps", "amm", "lending", "borrow", "stake")):
        return "defi"
    if any(k in t for k in ("token", "vesting", "unlock", "supply", "fdv", "market cap")):
        return "tokenomics"
    return "general"

def extract_keywords(text: str) -> List[str]:
    t = re.sub(r"[^\w\s$@.-]", " ", (text or "").lower())
    toks = [x for x in t.split() if x and len(x) > 2]
    # keep cashtags and handles
    cashtags = [x for x in toks if x.startswith("$") and len(x) > 2]
    handles = [x for x in toks if x.startswith("@") and len(x) > 2]
    # basic stopwords filter
    stop = set(["the","and","for","with","that","this","from","your","you","are","was","were","have","has","had","just","like","what","when","how","why","who","its","it's","im","i'm","rt"])
    plain = [x for x in toks if (x not in stop and not x.startswith("$") and not x.startswith("@"))]
    # unique preserve order
    out = []
    for x in cashtags + handles + plain:
        if x not in out:
            out.append(x)
    return out[:12]

def pick_focus_token(keywords: List[str]) -> str:
    for k in keywords:
        if k.startswith("$") and len(k) > 2:
            return k
    for k in keywords:
        if not k.startswith("@") and len(k) > 3:
            return k
    return ""

def build_canonical_x_url(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        u = url.strip()
        # accept x.com and twitter.com
        u = u.replace("twitter.com", "x.com")
        # remove tracking/query
        pu = urlparse(u)
        base = f"{pu.scheme}://{pu.netloc}{pu.path}"
        return base
    except Exception:
        return url

def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        pu = urlparse(url)
        path = (pu.path or "").strip("/")
        parts = path.split("/")
        if len(parts) >= 1 and parts[0] and parts[0] != "i":
            return parts[0]
    except Exception:
        pass
    return None


generator = OfflineCommentGenerator()

# ------------------------------------------------------------------------------
# Quote / thread context (VXTwitter best-effort)
# ------------------------------------------------------------------------------
_VX_BASE = "https://api.vxtwitter.com"
_HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "2.2"))

def _extract_tweet_id_from_url(url: str) -> Optional[str]:
    try:
        m = re.search(r"/status/(\d+)", url or "")
        return m.group(1) if m else None
    except Exception:
        return None

def fetch_thread_context(url: Optional[str]) -> dict:
    """
    Best-effort: fetch quote-tweet text + parent/reply-to text using vxtwitter.
    Never raises; returns {} on failure.
    """
    if not (ENABLE_THREAD_CONTEXT and url):
        return {}
    tid = _extract_tweet_id_from_url(url)
    if not tid:
        return {}
    handle = _extract_handle_from_url(url) or "i"
    api_url = f"{_VX_BASE}/{handle}/status/{tid}" if handle != "i" else f"{_VX_BASE}/i/status/{tid}"

    try:
        r = requests.get(api_url, timeout=_HTTP_TIMEOUT_SEC)
        if r.status_code != 200:
            return {}
        data = r.json()
    except Exception:
        return {}

    def _dig_text(obj: Any) -> str:
        if isinstance(obj, dict):
            for k in ("text", "full_text", "content"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    t = _dig_text(v)
                    if t:
                        return t
        if isinstance(obj, list):
            for it in obj:
                t = _dig_text(it)
                if t:
                    return t
        return ""

    def _get_branch(d: dict, keys: tuple[str, ...]) -> Any:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur

    quoted_obj = None
    for path in (
        ("quoted_tweet",),
        ("quoted_status",),
        ("quotedStatus",),
        ("tweet", "quoted_tweet"),
        ("tweet", "quoted_status"),
        ("tweet", "quotedStatus"),
    ):
        quoted_obj = _get_branch(data, path)
        if quoted_obj:
            break

    parent_obj = None
    for path in (
        ("in_reply_to",),
        ("in_reply_to_status",),
        ("reply_to",),
        ("tweet", "in_reply_to"),
        ("tweet", "in_reply_to_status"),
        ("tweet", "reply_to"),
    ):
        parent_obj = _get_branch(data, path)
        if parent_obj:
            break

    ctx: dict = {}
    qtxt = _dig_text(quoted_obj) if quoted_obj else ""
    ptxt = _dig_text(parent_obj) if parent_obj else ""
    if qtxt:
        ctx["quoted_text"] = qtxt
    if ptxt:
        ctx["parent_text"] = ptxt
    return ctx

# ------------------------------------------------------------------------------
# Entity memory (ticker/name -> DefiLlama slug) + research context
# ------------------------------------------------------------------------------
_RESEARCH_CACHE: dict[str, tuple[float, dict]] = {}
_LLAMA_PROTOCOLS_CACHE: tuple[float, list[dict]] | None = None
_COINGECKO_BACKOFF_UNTIL: float = 0.0

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
                "INSERT INTO entity_map(kind,k,slug,updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(kind,k) DO UPDATE SET slug=excluded.slug, updated_at=excluded.updated_at",
                (kind, k, slug, now_ts()),
            )
    except Exception:
        pass

def _cache_get(key: str) -> dict | None:
    item = _RESEARCH_CACHE.get(key)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > RESEARCH_CACHE_TTL_SEC:
        return None
    return val

def _cache_set(key: str, val: dict) -> None:
    _RESEARCH_CACHE[key] = (time.time(), val)

def _http_get_json(url: str, headers: Optional[dict] = None, timeout: float = 2.2) -> Any:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# DefiLlama (free)
LLAMA_BASE = "https://api.llama.fi"
COINS_LLAMA_BASE = "https://coins.llama.fi"

def llama_protocols() -> list[dict]:
    global _LLAMA_PROTOCOLS_CACHE
    now = time.time()
    if _LLAMA_PROTOCOLS_CACHE and (now - _LLAMA_PROTOCOLS_CACHE[0]) < LLAMA_PROTOCOLS_CACHE_TTL_SEC:
        return _LLAMA_PROTOCOLS_CACHE[1]
    data = _http_get_json(f"{LLAMA_BASE}/protocols", timeout=_HTTP_TIMEOUT_SEC)
    if isinstance(data, list):
        _LLAMA_PROTOCOLS_CACHE = (now, data)
        return data
    return _LLAMA_PROTOCOLS_CACHE[1] if _LLAMA_PROTOCOLS_CACHE else []

def llama_find_protocol(query: str) -> Optional[dict]:
    q = (query or "").strip().lower()
    if not q:
        return None
    prots = llama_protocols()
    if not prots:
        return None
    for p in prots:
        if str(p.get("slug","")).lower() == q:
            return p
    for p in prots:
        if str(p.get("symbol","")).lower() == q:
            return p
    for p in prots:
        name = str(p.get("name","")).lower()
        if q in name:
            return p
    return None

def llama_protocol_detail(slug: str) -> Optional[dict]:
    return _http_get_json(f"{LLAMA_BASE}/protocol/{slug}", timeout=_HTTP_TIMEOUT_SEC)

def llama_tvl(slug: str) -> Optional[float]:
    data = _http_get_json(f"{LLAMA_BASE}/tvl/{slug}", timeout=_HTTP_TIMEOUT_SEC)
    if isinstance(data, (int, float)):
        return float(data)
    return None

# CoinGecko (optional, rate-limited)
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def _coingecko_headers() -> dict:
    return {"x-cg-demo-api-key": COINGECKO_DEMO_KEY} if COINGECKO_DEMO_KEY else {}

def coingecko_search(query: str) -> Optional[dict]:
    global _COINGECKO_BACKOFF_UNTIL
    if not ENABLE_COINGECKO:
        return None
    if time.time() < _COINGECKO_BACKOFF_UNTIL:
        return None
    q = (query or "").strip()
    if not q:
        return None
    try:
        url = f"{COINGECKO_BASE}/search?query={requests.utils.quote(q)}"
        r = requests.get(url, headers=_coingecko_headers(), timeout=_HTTP_TIMEOUT_SEC)
        if r.status_code == 429:
            _COINGECKO_BACKOFF_UNTIL = time.time() + 15 * 60
            return None
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def coingecko_simple_price(coin_id: str) -> Optional[dict]:
    global _COINGECKO_BACKOFF_UNTIL
    if not ENABLE_COINGECKO:
        return None
    if time.time() < _COINGECKO_BACKOFF_UNTIL:
        return None
    cid = (coin_id or "").strip()
    if not cid:
        return None
    try:
        url = (
            f"{COINGECKO_BASE}/simple/price?"
            f"ids={requests.utils.quote(cid)}&vs_currencies=usd&include_market_cap=true&include_24hr_change=true"
        )
        r = requests.get(url, headers=_coingecko_headers(), timeout=_HTTP_TIMEOUT_SEC)
        if r.status_code == 429:
            _COINGECKO_BACKOFF_UNTIL = time.time() + 15 * 60
            return None
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def extract_entities(tweet_text: str) -> dict:
    t = tweet_text or ""
    cashtags = re.findall(r"\$\w{2,15}", t)
    handles = re.findall(r"@\w{2,30}", t)
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    return {
        "cashtags": list(dict.fromkeys(cashtags)),
        "handles": list(dict.fromkeys(handles)),
        "numbers": list(dict.fromkeys(nums)),
    }

def fetch_research_context(tweet_text: str, thread_ctx: Optional[dict] = None) -> dict:
    """
    Best-effort project context:
    - DefiLlama (protocol slug/name/symbol, tvl, chains, short description)
    - Optional CoinGecko (price/mcap/24h)
    Caches aggressively. Never raises.
    """
    if not ENABLE_RESEARCH:
        return {}

    base = (tweet_text or "").strip()
    extra = ""
    if thread_ctx:
        extra = " " + (thread_ctx.get("quoted_text") or "") + " " + (thread_ctx.get("parent_text") or "")
    cache_key = "rc:" + sha256((base + extra)[:900])
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ctx: dict = {"source": [], "project": {}, "market": {}, "note": ""}

    ents = extract_entities(base + extra)
    cashtags = ents.get("cashtags") or []
    keywords = extract_keywords(base + extra)

    slug = None
    if cashtags:
        slug = entity_map_get("ticker", cashtags[0].lower())
    if not slug and keywords:
        slug = entity_map_get("name", keywords[0].lower())

    proto = None
    if slug:
        proto = llama_find_protocol(slug)
    else:
        q = (cashtags[0][1:] if cashtags else (keywords[0] if keywords else ""))
        if q:
            proto = llama_find_protocol(q)

    if proto:
        slug = str(proto.get("slug") or "").strip()
        if slug:
            ctx["project"].update({
                "name": proto.get("name"),
                "symbol": proto.get("symbol"),
                "slug": slug,
                "category": proto.get("category"),
                "chain": proto.get("chain"),
            })
            ctx["source"].append("defillama")

            sym = str(proto.get("symbol") or "").strip()
            nm = str(proto.get("name") or "").strip()
            if sym:
                entity_map_set("ticker", f"${sym}".lower(), slug)
                entity_map_set("ticker", sym.lower(), slug)
            if nm:
                entity_map_set("name", nm.lower(), slug)

            tvl = llama_tvl(slug)
            if tvl is not None:
                ctx["project"]["tvl_usd"] = tvl

            detail = llama_protocol_detail(slug)
            if isinstance(detail, dict):
                chains = detail.get("chains") or []
                if isinstance(chains, list):
                    ctx["project"]["chains"] = chains[:8]
                desc = detail.get("description")
                if isinstance(desc, str) and desc.strip():
                    ctx["project"]["description"] = desc.strip()[:260]

    if ENABLE_COINGECKO and is_crypto_tweet(base):
        q = None
        if cashtags:
            q = cashtags[0][1:]
        elif ctx["project"].get("name"):
            q = ctx["project"]["name"]
        if q:
            ss = coingecko_search(q)
            if isinstance(ss, dict):
                coins = ss.get("coins") or []
                if coins:
                    top = coins[0]
                    coin_id = top.get("id")
                    ctx["market"]["coingecko_id"] = coin_id
                    ctx["market"]["symbol"] = top.get("symbol")
                    ctx["market"]["name"] = top.get("name")
                    ctx["source"].append("coingecko")
                    if coin_id:
                        p = coingecko_simple_price(coin_id)
                        if isinstance(p, dict) and coin_id in p and isinstance(p[coin_id], dict):
                            m = p[coin_id]
                            for k in ("usd", "usd_market_cap", "usd_24h_change"):
                                if k in m:
                                    ctx["market"][k] = m[k]

    if not ctx["source"]:
        ctx["note"] = "No reliable project match; ask a question rather than asserting."
    _cache_set(cache_key, ctx)
    return ctx

# ------------------------------------------------------------------------------
# Voice roulette (one voice per request; prevents same style every time)
# ------------------------------------------------------------------------------
VOICE_CARDS: dict[str, str] = {
    "trader": "Voice: trader. Focus on levels/flow, positioning, risk, timeframe. Calm, not hype.",
    "builder": "Voice: builder. Practical mechanisms, constraints, UX, execution details. Slightly nerdy.",
    "researcher": "Voice: researcher. Grounded, precise. Point out assumptions, ask one sharp question.",
    "skeptic": "Voice: skeptic. Identify what could break, hidden risks, incentives, second-order effects.",
    "curious_friend": "Voice: curious friend. Conversational, human, but still specific and grounded.",
    "deadpan_meme": "Voice: deadpan. One witty or dry line allowed if tweet is funny. Still coherent.",
}

def choose_voice(tweet_text: str, thread_ctx: Optional[dict] = None) -> str:
    t = (tweet_text or "").lower()
    topic = detect_topic(tweet_text or "")
    if topic == "meme" and PRO_KOL_ALLOW_WIT:
        return "deadpan_meme"

    weights = {
        "trader": 1.0,
        "builder": 1.0,
        "researcher": 1.0,
        "skeptic": 1.0,
        "curious_friend": 1.0,
    }
    if topic == "chart" or any(k in t for k in ("chart","support","resistance","oi","funding","liq","liquidity","entry","tp","sl")):
        weights["trader"] += 2.3
    if any(k in t for k in ("protocol","amm","perps","lending","audit","exploit","bridge","fees","mainnet","testnet")):
        weights["builder"] += 1.8
        weights["skeptic"] += 1.2
    if any(k in t for k in ("thread","🧵","tokenization","research","paper","mechanics","breakdown")) or len(tweet_text or "") > 220:
        weights["researcher"] += 2.0
    if any(k in t for k in ("rug","risk","concern","worry","scam","ponzi","exit liquidity","down only")):
        weights["skeptic"] += 2.2

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

# ------------------------------------------------------------------------------
# Tokenization + style fingerprint (prevents 17 99 / keeps $SOL; blocks template skeleton repeats)
# ------------------------------------------------------------------------------
WORD_RE = re.compile(r"(?:\$\w{2,15}|\d+(?:\.\d+)?|[A-Za-z0-9’']+(?:-[A-Za-z0-9’']+)*)")

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
    - stopwords kept
    - content words -> W
    - numbers -> N
    - cashtags -> $T
    Used for template-burn prevention (avoid same skeleton with new topic).
    """
    t = normalize_ws(text).lower()
    if not t:
        return ""
    toks = WORD_RE.findall(t)
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

    # compress consecutive W tokens
    out: list[str] = []
    for m in mapped:
        if out and m == "W" and out[-1] == "W":
            continue
        out.append(m)

    fp = " ".join(out).strip()
    return fp[:140]

def template_burned(tmpl: str) -> bool:
    """True if this template structure was used recently."""
    return template_seen(tmpl)

# ------------------------------------------------------------------------------
# Rules: word count + sanitization
# ------------------------------------------------------------------------------
# (WORD_RE is defined earlier to keep decimals/tickers)

def words(t: str) -> list[str]:
    return WORD_RE.findall(t or "")

def enforce_word_count_natural(text: str, min_words: int = 6, max_words: int = 13) -> str:
    """
    Clamp to 6–13 words but keep punctuation and decimals.
    """
    t = clean_comment(text)
    if not t:
        return ""

    toks = words(t)
    if len(toks) < min_words:
        # pad with a tiny natural suffix
        for suffix in ("right now", "this week", "in practice", "tbh"):
            tt = clean_comment(t + " " + suffix)
            if min_words <= len(words(tt)) <= max_words:
                return tt
        return t
    if len(toks) > max_words:
        # trim by token count but keep original spacing/punctuation roughly
        # safest: rebuild from tokens
        clipped = " ".join(toks[:max_words])
        # preserve '?' if original looked like question
        if t.strip().endswith("?") and not clipped.endswith("?"):
            clipped += "?"
        return clean_comment(clipped)
    return t

GENERIC_PHRASES = [
    "thanks for sharing",
    "great post",
    "love this",
    "so true",
    "interesting",
    "nice thread",
    "awesome",
    "amazing",
    "good stuff",
]

def contains_generic_phrase(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in GENERIC_PHRASES)

def _openers(text: str) -> List[str]:
    """
    Returns a small opener fingerprint: first 3 tokens and first 5 tokens.
    Used to prevent repeating the same 'voice' opening.
    """
    w = WORD_RE.findall((text or "").lower())
    if not w:
        return []
    o3 = " ".join(w[:3])
    o5 = " ".join(w[:5])
    return [o3, o5]

def _trigrams(text: str) -> List[str]:
    w = WORD_RE.findall((text or "").lower())
    if len(w) < 3:
        return []
    out = []
    for i in range(len(w) - 2):
        out.append(" ".join(w[i:i+3]))
    return out

def trigram_overlap_bad(text: str, threshold: int = 2) -> bool:
    """
    True if this comment overlaps too much with recently seen trigrams.
    """
    trigs = _trigrams(text)
    if not trigs:
        return False
    seen = 0
    for t in trigs:
        if trigram_seen(t):
            seen += 1
            if seen >= threshold:
                return True
    return False

def _word_trigrams(s: str) -> set:
    w = WORD_RE.findall((s or "").lower())
    return set(" ".join(w[i:i+3]) for i in range(len(w)-2)) if len(w) >= 3 else set()

def too_similar(a: str, b: str, overlap_thresh: int = 2) -> bool:
    A = _word_trigrams(a)
    B = _word_trigrams(b)
    if not A or not B:
        return False
    return len(A & B) >= overlap_thresh

def too_similar_to_recent(text: str) -> bool:
    """
    Heuristic: if many trigrams are already seen, it's probably a repeat.
    """
    return trigram_overlap_bad(text, threshold=2)

def _pair_too_similar(a: str, b: str) -> bool:
    if too_similar(a, b, overlap_thresh=2):
        return True
    # also avoid identical opener
    return bool(set(_openers(a)) & set(_openers(b)))

def pick_two_diverse_text(candidates: list[str]) -> list[str]:
    """
    Pick two with least overlap, prefer different modes.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates
    # brute force best pair
    best = None
    best_score = 10**9
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            a, b = candidates[i], candidates[j]
            score = 0
            if _pair_too_similar(a, b):
                score += 100
            if contains_generic_phrase(a):
                score += 10
            if contains_generic_phrase(b):
                score += 10
            score += len(_word_trigrams(a) & _word_trigrams(b))
            if score < best_score:
                best_score = score
                best = (a, b)
    if best:
        return [best[0], best[1]]
    # fallback first two
    return candidates[:2]

def _mentions_new_cashtag_or_number(comment: str, tweet_text: str) -> bool:
    """Return True if comment introduces new $TICKER or large numbers not in tweet."""
    if not ANTI_HALLUCINATION:
        return False
    c = comment or ""
    t = tweet_text or ""
    c_tags = set(re.findall(r"\$\w{2,15}", c))
    t_tags = set(re.findall(r"\$\w{2,15}", t))
    if c_tags - t_tags:
        return True

    # numbers: allow small 1-10 and allow if present in tweet
    c_nums = re.findall(r"\d+(?:\.\d+)?", c)
    t_nums = set(re.findall(r"\d+(?:\.\d+)?", t))
    for n in c_nums:
        if n in t_nums:
            continue
        try:
            val = float(n)
        except Exception:
            continue
        if val > 10:
            return True
    return False

def enforce_unique(candidates: list[str], tweet_text: Optional[str] = None) -> list[str]:
    """
    - sanitize + enforce word count
    - drop generic phrases
    - skip past repeats / templates / trigram overlaps
    - add small tweak if previously seen
    - finally: pick two diverse comments
    """
    tweet_text = tweet_text or ""
    out: list[str] = []

    for c in candidates:
        c = enforce_word_count_natural(c)
        c = clean_comment(c)
        if not c:
            continue

        # kill obvious generic
        if contains_generic_phrase(c):
            continue

        # structural repetition guards
        if opener_seen(_openers(c)) or trigram_overlap_bad(c, threshold=2) or too_similar_to_recent(c):
            continue

        # block repeated sentence skeletons
        if template_burned(c):
            continue

        # anti-hallucination: no new $TICKER / big numbers
        if tweet_text and _mentions_new_cashtag_or_number(c, tweet_text):
            continue

        if not comment_seen(c):
            remember_comment(c)
            remember_template(c)
            remember_opener(_openers(c))
            remember_ngrams(c)
            out.append(c)
        else:
            # small tweak path to rescue near-duplicate if it's short
            toks = words(c)
            if len(toks) < 13:
                alt = enforce_word_count_natural(c + " today")
                if alt and not comment_seen(alt) and not contains_generic_phrase(alt):
                    if template_burned(alt):
                        continue
                    if tweet_text and _mentions_new_cashtag_or_number(alt, tweet_text):
                        continue
                    remember_comment(alt)
                    remember_template(alt)
                    remember_opener(_openers(alt))
                    remember_ngrams(alt)
                    out.append(alt)

        if len(out) >= 6:
            break

    # final hybrid pairing: maximize vibe diversity
    if len(out) > 2:
        out = pick_two_diverse_text(out)

    return out[:2]

def offline_two_comments(text: str) -> list[str]:
    """
    Offline fallback (human-ish).
    """
    topic = detect_topic(text)
    sent = detect_sentiment(text)
    kws = extract_keywords(text)
    focus = pick_focus_token(kws)

    # create several candidates with variety
    cands = []
    if sent == "positive":
        cands += [
            f"That’s a solid point on {focus} tbh.",
            f"Yeah this feels like real signal, not noise.",
            f"Love the clarity here — what’s your next step?",
        ]
    elif sent == "negative":
        cands += [
            "This is exactly where people overfit the narrative.",
            "Feels fragile — what breaks it first?",
            "Hard agree, hype blinds risk faster than anything.",
        ]
    else:
        cands += [
            f"Curious how you’re thinking about {focus} here.",
            "What’s the strongest counter-argument to this?",
            "This is the kind of nuance most threads miss.",
        ]

    # topic flavored
    if topic == "chart":
        cands += [
            "Clean levels — are you watching that next liquidity pocket?",
            "Where’s invalidation for this setup, realistically?",
            "Looks good, but I’d respect the chop until confirmation.",
        ]
    if topic in ("defi", "tokenomics"):
        cands += [
            "Incentives matter — who’s the natural seller in this?",
            "What’s the risk surface here: exploit, governance, or liquidity?",
            "How does this hold up when emissions cool off?",
        ]
    if topic == "meme":
        cands += [
            "Monday carrying the entire ecosystem again, classic.",
            "This is either genius or a future screenshot, no in-between.",
        ]

    cands = [clean_comment(x) for x in cands]
    cands = enforce_unique(cands, tweet_text=text)
    if len(cands) < 2:
        # hard fallback
        cands = ["Interesting angle, what’s the biggest risk here?", "This feels more grounded than most takes rn."][:2]
    return cands[:2]

# ------------------------------------------------------------------------------
# Provider prompt context builder (inject thread context + research + voice)
# ------------------------------------------------------------------------------
def _get_request_context(tweet_text: str, url: Optional[str] = None) -> dict:
    thread_ctx = REQUEST_THREAD_CTX.get() or {}
    research_ctx = REQUEST_RESEARCH_CTX.get() or {}
    voice = REQUEST_VOICE.get() or ""

    keys = extract_keywords(tweet_text or "")
    focus = pick_focus_token(keys) or ""
    topic = detect_topic(tweet_text or "")
    sentiment = detect_sentiment(tweet_text or "")

    return {
        "topic": topic,
        "sentiment": sentiment,
        "is_crypto": bool(is_crypto_tweet(tweet_text or "")),
        "keywords": keys[:10],
        "focus": focus,
        "voice": voice,
        "voice_card": VOICE_CARDS.get(voice, ""),
        "thread_context": thread_ctx,
        "research": research_ctx,
        "url": url or "",
    }

def _llm_context_text(tweet_text: str, author: Optional[str], url: Optional[str] = None) -> str:
    ctx = _get_request_context(tweet_text, url=url)
    return (
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Context JSON (may be empty; do not hallucinate beyond this):\n"
        + json.dumps(ctx, ensure_ascii=False)
        + "\n\n"
        "Return exactly two distinct comments (JSON array or two lines)."
    )

# ------------------------------------------------------------------------------
# Groq generator (exactly 2, 6–13 words, tolerant parsing)
# ------------------------------------------------------------------------------
def parse_two_comments_flex(raw_text: str) -> list[str]:
    raw = (raw_text or "").strip()
    if not raw:
        return []
    # try json array first
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            out = []
            for x in obj:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
            return out[:2]
        if isinstance(obj, dict):
            for k in ("comments", "result", "data"):
                v = obj.get(k)
                if isinstance(v, list):
                    out = [str(i).strip() for i in v if str(i).strip()]
                    return out[:2]
    except Exception:
        pass

    # fallback: split lines, remove bullets
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", x).strip() for x in raw.splitlines() if x.strip()]
    # sometimes returns "1) ... 2) ..."
    if len(lines) >= 2:
        return [lines[0], lines[1]]

    # fallback regex for bracketed JSON-ish
    m = re.search(r"\[[\s\S]*\]", raw_text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                out = [str(i).strip() for i in obj if str(i).strip()]
                return out[:2]
        except Exception:
            pass

    # last fallback: split on ' / ' or '||'
    for sep in ("||", " / ", " | "):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if len(parts) >= 2:
                return [parts[0], parts[1]]
    return lines[:2]

def groq_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not _groq_client:
        return []

    sys_prompt = (
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
        "- Do NOT invent facts not present in the tweet or context JSON.\n"
        "- Preserve numbers and cashtags exactly (17.99 stays 17.99, $SOL stays $SOL).\n"
        "- If context JSON has research or quote/parent text, you may reference it lightly.\n"
        "- If research is empty/unclear, ask a good question instead of asserting.\n"
        "- Prefer returning a pure JSON array of two strings.\n"
        "- If you cannot return JSON, return two lines separated by a newline.\n"
    )

    user_prompt = _llm_context_text(tweet_text, author, url=None)

    resp = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        n=1,
        temperature=0.7,
        max_tokens=160,
    )
    raw = (resp.choices[0].message.content or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(x) for x in candidates]
    candidates = [x for x in candidates if 6 <= len(words(x)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    return candidates[:2]

# ------------------------------------------------------------------------------
# LLM sys prompt shared (OpenAI/Gemini)
# ------------------------------------------------------------------------------
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
        "- Do NOT invent facts not present in the tweet or context JSON.\n"
        "- Preserve numbers and cashtags exactly (17.99 stays 17.99, $SOL stays $SOL).\n"
        "- If context JSON has research or quote/parent text, you may reference it lightly.\n"
        "- If research is empty/unclear, ask a good question instead of asserting.\n"
        "- Prefer returning a pure JSON array of two strings.\n"
        "- If you cannot return JSON, return two lines separated by a newline.\n"
    )

def openai_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not _openai_client:
        return []
    sys_prompt = _llm_sys_prompt()
    user_prompt = _llm_context_text(tweet_text, author, url=None)

    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    raw = (resp.choices[0].message.content or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(x) for x in candidates]
    candidates = [x for x in candidates if 6 <= len(words(x)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    return candidates[:2]

def gemini_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not _gemini_model:
        return []
    sys_prompt = _llm_sys_prompt()
    user_prompt = _llm_context_text(tweet_text, author, url=None)
    prompt = sys_prompt + "\n\n" + user_prompt

    resp = _gemini_model.generate_content(prompt)
    raw = getattr(resp, "text", "") or str(resp)
    raw = (raw or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(x) for x in candidates]
    candidates = [x for x in candidates if 6 <= len(words(x)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    return candidates[:2]


# ------------------------------------------------------------------------------
# Optional pro rewrite/regenerate (only when PRO_KOL_MODE=1; never blocks main flow)
# ------------------------------------------------------------------------------
def _needs_pro_rewrite(cands: list[str], tweet_text: str) -> bool:
    if len(cands) < 2:
        return True
    a, b = cands[0], cands[1]
    if _pair_too_similar(a, b):
        return True
    if template_burned(a) or template_burned(b):
        return True
    if contains_generic_phrase(a) or contains_generic_phrase(b):
        return True
    if ANTI_HALLUCINATION and (_mentions_new_cashtag_or_number(a, tweet_text) or _mentions_new_cashtag_or_number(b, tweet_text)):
        return True
    return False

def pro_rewrite_two_comments(tweet_text: str, author: Optional[str], seed: list[str]) -> Optional[list[str]]:
    providers = _available_providers()
    if not providers:
        return None
    random.shuffle(providers)

    ctx = _get_request_context(tweet_text, url=None)
    voice = ctx.get("voice") or ""
    voice_line = VOICE_CARDS.get(voice, "")

    sys_prompt = (
        "You rewrite two tweet replies to be more human and context-aware.\n"
        "- Output exactly 2 comments.\n"
        "- Each comment must be 6-13 words.\n"
        "- One thought only per comment.\n"
        "- No emojis, hashtags, links.\n"
        "- Preserve decimals and cashtags exactly (17.99, $SOL).\n"
        "- Use the context JSON; do not invent facts.\n"
        "- If research is empty/unclear, ask a sharp question.\n"
        "- Avoid repeating the same sentence skeleton in both comments.\n"
        + (voice_line + "\n" if voice_line else "")
        + "Return a JSON array of two strings.\n"
    )

    user_prompt = (
        _llm_context_text(tweet_text, author, url=None)
        + "\n\nSeed comments (improve these, do not copy):\n"
        + json.dumps(seed[:2], ensure_ascii=False)
    )

    for name, _fn in providers:
        try:
            raw = ""
            if name == "groq" and USE_GROQ and _groq_client:
                resp = _groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                    n=1,
                    max_tokens=200,
                    temperature=0.7,
                )
                raw = (resp.choices[0].message.content or "").strip()
            elif name == "openai" and USE_OPENAI and _openai_client:
                resp = _openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                    max_tokens=200,
                    temperature=0.7,
                )
                raw = (resp.choices[0].message.content or "").strip()
            elif name == "gemini" and USE_GEMINI and _gemini_model:
                prompt = sys_prompt + "\n\n" + user_prompt
                resp = _gemini_model.generate_content(prompt)
                raw = getattr(resp, "text", "") or str(resp)
                raw = (raw or "").strip()

            if not raw:
                continue
            cand = parse_two_comments_flex(raw)
            cand = [enforce_word_count_natural(x) for x in cand]
            cand = [x for x in cand if 6 <= len(words(x)) <= 13]
            cand = enforce_unique(cand, tweet_text=tweet_text)
            if len(cand) >= 2 and not _pair_too_similar(cand[0], cand[1]):
                return cand[:2]
        except Exception as e:
            logger.warning("pro rewrite via %s failed: %s", name, e)
    return None


# ------------------------------------------------------------------------------
# Provider selection / orchestration
# ------------------------------------------------------------------------------
def _available_providers() -> List[Tuple[str, Any]]:
    providers: List[Tuple[str, Any]] = []
    if USE_GROQ and _groq_client:
        providers.append(("groq", groq_two_comments))
    if USE_OPENAI and _openai_client:
        providers.append(("openai", openai_two_comments))
    if USE_GEMINI and _gemini_model:
        providers.append(("gemini", gemini_two_comments))
    return providers

def _rescue_two(tweet_text: str) -> list[str]:
    return offline_two_comments(tweet_text)

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
    candidates: list[str] = []

    # --------------------------------------------------------------
    # Request-scoped context (thread context + research + voice)
    # Fetch once per tweet request, then all providers can use it.
    # --------------------------------------------------------------
    thread_ctx = fetch_thread_context(url) if url else {}
    research_ctx = fetch_research_context(tweet_text, thread_ctx=thread_ctx) if ENABLE_RESEARCH else {}
    voice = choose_voice(tweet_text, thread_ctx=thread_ctx)

    tok_thread = REQUEST_THREAD_CTX.set(thread_ctx or {})
    tok_research = REQUEST_RESEARCH_CTX.set(research_ctx or {})
    tok_voice = REQUEST_VOICE.set(voice or "")

    try:
        providers = _available_providers()
        if providers:
            random.shuffle(providers)
            for name, fn in providers:
                try:
                    got = fn(tweet_text, author)
                    if got:
                        candidates = enforce_unique(candidates + got, tweet_text=tweet_text) or candidates + got
                    if len(candidates) >= 2:
                        break
                except Exception as e:
                    logger.warning("%s provider failed: %s", name, e)

        # If still < 2, ask offline generator directly
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
                    candidates = enforce_unique(candidates + extra, tweet_text=tweet_text) or (candidates + extra)
            except Exception as e:
                logger.exception("Total failure in provider cascade: %s", e)

        # If still nothing, hard fallback to 2 simple offline lines
        if not candidates:
            raw = _rescue_two(tweet_text)
            candidates = enforce_unique(raw, tweet_text=tweet_text) or raw

        # Limit to exactly 2 text comments
        candidates = [c for c in candidates if c][:2]

        # Optional pro rewrite (KOL mode) — improves variety + context use
        if PRO_KOL_MODE and PRO_KOL_REWRITE and _needs_pro_rewrite(candidates, tweet_text):
            try:
                pr = pro_rewrite_two_comments(tweet_text, author, candidates)
                if pr and len(pr) >= 2:
                    candidates = pr[:2]
            except Exception:
                pass

        out: List[Dict[str, Any]] = []
        for c in candidates:
            out.append({"lang": lang or "en", "text": c})

        # If somehow we still ended up with < 2 dicts, ask offline generator directly
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
        return out[:2]

    finally:
        REQUEST_THREAD_CTX.reset(tok_thread)
        REQUEST_RESEARCH_CTX.reset(tok_research)
        REQUEST_VOICE.reset(tok_voice)


# ------------------------------------------------------------------------------
# API routes
# ------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "crowntalk"})

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.post("/comment")
def comment_endpoint():
    init_providers()
    init_db()

    payload = request.get_json(force=True, silent=True) or {}
    url = payload.get("url") or payload.get("link") or ""
    lang = payload.get("lang") or "en"
    tweet_text = payload.get("text") or payload.get("tweet_text") or ""
    author = payload.get("author") or payload.get("author_name")
    handle = payload.get("handle") or payload.get("author_handle")

    tweet_text = clean_comment(tweet_text)
    url = build_canonical_x_url(url) if url else url

    # if no tweet_text provided, attempt to fetch via vxtwitter minimal (best-effort)
    if not tweet_text and url:
        try:
            tid = _extract_tweet_id_from_url(url)
            h = _extract_handle_from_url(url) or "i"
            api_url = f"{_VX_BASE}/{h}/status/{tid}" if (tid and h != "i") else (f"{_VX_BASE}/i/status/{tid}" if tid else "")
            if api_url:
                r = requests.get(api_url, timeout=_HTTP_TIMEOUT_SEC)
                if r.status_code == 200:
                    j = r.json()
                    # best-effort: find a text field
                    txt = ""
                    if isinstance(j, dict):
                        for k in ("text","full_text"):
                            if isinstance(j.get(k), str):
                                txt = j.get(k)
                                break
                        if not txt and isinstance(j.get("tweet"), dict):
                            for k in ("text","full_text"):
                                if isinstance(j["tweet"].get(k), str):
                                    txt = j["tweet"].get(k)
                                    break
                    tweet_text = clean_comment(txt)
        except Exception:
            pass

    items = generate_two_comments_with_providers(
        tweet_text=tweet_text,
        author=author,
        handle=handle,
        lang=lang,
        url=url,
    )

    # safe output structure
    return jsonify({"items": items})

# Keep-alive to avoid cold starts (optional)
def _keep_alive_loop():
    while True:
        try:
            time.sleep(KEEP_ALIVE_INTERVAL)
            logger.info("keep-alive tick")
        except Exception:
            time.sleep(KEEP_ALIVE_INTERVAL)

if os.getenv("ENABLE_KEEP_ALIVE", "0").strip() == "1":
    t = threading.Thread(target=_keep_alive_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    init_providers()
    init_db()
    app.run(host=HOST, port=PORT)