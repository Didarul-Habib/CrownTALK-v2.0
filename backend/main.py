# PART 1/4
# Based on user's working file (see uploaded): :contentReference[oaicite:0]{index=0}
from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from collections import Counter
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify

# Optional providers (imported lazily / guarded by env)
try:
    import cohere
except Exception:
    cohere = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from mistralai import Mistral
except Exception:
    Mistral = None

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
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))                 # ← process N at a time
PER_URL_SLEEP = float(os.environ.get("PER_URL_SLEEP_SECONDS", "0.1"))  # ← sleep after every URL
MAX_URLS_PER_REQUEST = int(os.environ.get("MAX_URLS_PER_REQUEST", "25"))  # ← hard cap per request

KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))

# ------------------------------------------------------------------------------
# Optional Groq (free-tier)
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
# OpenAI (optional)
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
# Gemini (optional)
# ------------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USE_GEMINI = bool(GEMINI_API_KEY)
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
# Mistral (optional)
# ------------------------------------------------------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
USE_MISTRAL = bool(MISTRAL_API_KEY) and (Mistral is not None)
_mistral_client = None
if USE_MISTRAL:
    try:
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception:
        _mistral_client = None
        USE_MISTRAL = False
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# ------------------------------------------------------------------------------
# Cohere (optional)
# ------------------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
USE_COHERE = bool(COHERE_API_KEY) and cohere is not None
_cohere_client = None
if USE_COHERE:
    try:
        _cohere_client = cohere.Client(COHERE_API_KEY)
    except Exception:
        _cohere_client = None
        USE_COHERE = False
COHERE_MODEL = os.getenv("COHERE_MODEL", "command")

# ------------------------------------------------------------------------------
# HuggingFace (transformers pipeline) optional local model (e.g., gpt2 for free user)
# ------------------------------------------------------------------------------
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL")
USE_HF = bool(HUGGINGFACE_MODEL) and pipeline is not None
_hf_pipeline = None
if USE_HF:
    try:
        _hf_pipeline = pipeline("text-generation", model=HUGGINGFACE_MODEL)
    except Exception:
        _hf_pipeline = None
        USE_HF = False

# ------------------------------------------------------------------------------
# Keepalive
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
# Light memory / OTP guards (anti-pattern, anti-repeat)
# ------------------------------------------------------------------------------
def now_ts() -> int:
    return int(time.time())

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

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

def style_fingerprint(tmpl: str) -> str:
    # simple but stable fingerprint for templates
    try:
        s = re.sub(r"\W+", " ", (tmpl or "").lower()).strip()
        return re.sub(r"\s+", " ", s)
    except Exception:
        return tmpl or ""

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

def _word_trigrams(s: str) -> set:
    w = re.findall(r"[A-Za-z0-9']+", s.lower())
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

# PART 2/4
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
# Rules: word count + sanitization
# ------------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z0-9’']+(-[A-Za-z0-9’']+)?")

def words(t: str) -> list[str]:
    return WORD_RE.findall(t or "")

def sanitize_comment(raw: str) -> str:
    txt = re.sub(r"https?://\S+", "", raw or "")
    txt = re.sub(r"[@#]\S+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"[.!?;:…]+$", "", txt).strip()
    try:
        txt = re.sub(r"[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", "", txt)
    except re.error:
        txt = re.sub(r"[\u2600-\u27BF]+", "", txt)
    return txt

def _ensure_question_punctuation(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.endswith("?"):
        return s
    if re.match(r"^(how|what|why|when|where|can|could|would|do|does|did)\b", s.lower()):
        return s.rstrip(".!") + "?"
    return s

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
    out = " ".join(toks).strip()

    # kill ultra-generic openers like "love that", "excited to see"
    txt_low = out.lower().lstrip()
    bad_starts = [
        "love that ","love this ","love the ","love your ",
        "excited to see","excited for","can't wait to","cant wait to",
        "glad to see","happy to see","this is huge","this is massive",
        "this could be huge","this is insane",
    ]
    for bs in bad_starts:
        if txt_low.startswith(bs):
            txt_low = txt_low[len(bs):].lstrip()
            out = txt_low
            break

    out = _ensure_question_punctuation(out)
    return out

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
    "excited to see","excited for","can't wait to see","can\u2019t wait to see",
    "looking forward to","look forward to",
    "this is huge","this could be huge","this is massive","this is insane",
    "game changing","game-changing","total game changer","what a game changing approach",
    "mind blown","mind-blowing","blows my mind","massive alpha",
    "thanks for sharing","thank you for sharing","thanks for this","appreciate you",
    "appreciate it","appreciate this","proud of you","so proud of this",
    "the vibe around","vibe around","the vibe here is pretty real",
    "this is what we need","exactly what we need",
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

# PART 3/4
def is_crypto_tweet(text: str) -> bool:
    t = (text or "").lower()
    crypto_keywords = [
        "crypto","defi","nft","airdrop","token","coin","chain","l1","l2","staking","yield",
        "dex","cex","onchain","on-chain","gas fees","btc","eth","sol","arb","layer two","mainnet"
    ]
    return any(k in t for k in crypto_keywords) or bool(re.search(r"\$\w{2,8}", text or ""))

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
    "signal over noise:","nuts and bolts:","from the builder side,","first principles:"
]
CLAIMS = [
    "{focus} is doing more work than the headline","{focus} is where the thesis tightens",
    "{focus} is the part that moves things","{focus} is the practical hinge",
    "{focus} is the constraint to solve","{focus} tells you the next step",
    "it lives or dies on {focus}","risk mostly hides in {focus}",
    "execution shows up as {focus}","watch how {focus} trends, not the hype",
    "{focus} is the boring piece that decides outcomes","{focus} sets the real ceiling",
    "{focus} is the bit with actual leverage","most errors start before {focus} is clear"
]
NUANCE = [
    "separate it from optics","strip the hype and check it","ignore the noise and test it",
    "details beat slogans here","context > theatrics","measure it in weeks, not likes",
    "model it once and the picture clears","ship first, argue later","constraints explain the behavior",
    "once {focus} holds, the plan is simple","touch grass and look at {focus}"
]
CLOSERS = [
    "then the plan makes sense","and the whole picture clicks","and entries/exits get cleaner",
    "and you avoid dumb errors","and the convo gets useful","and incentives line up",
    "and the path forward writes itself","and the take stops being vibes-only"
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
            P(f"The vibe around {focus_slot} here is pretty real"),
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
        ]

        buckets: Dict[str, List[str]] = {
            "react": react,
            "conversation": convo,
            "calm": calm,
            "vibe": vibe,
            "nuanced": nuance,
            "quick": quick,
            "kol": kol,
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
        if not self._diversity_ok(line):
            return False
        if comment_seen(line):
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

# PART 4/4
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

def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        m = re.search(r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/([^/]+)/status/", url, re.I)
        return m.group(1) if m else None
    except Exception:
        return None

generator = OfflineCommentGenerator()

# --- Minimal helpers used by Groq path ---
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

    fillers = ["honestly", "still", "though", "right"]
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
    return out

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
    - sanitize + enforce 6–13 words
    - drop generic phrases
    - skip past repeats / templates / trigram overlaps
    - small chance to tweak if previously seen
    - finally: pick two diverse comments (Hybrid: statement + question if possible)
    """
    out: list[str] = []

    for c in candidates:
        c = enforce_word_count_natural(c)
        if not c:
            continue

        # kill very generic / overused phrases
        if contains_generic_phrase(c):
            continue

        # structural repetition guards
        if opener_seen(_openers(c)) or trigram_overlap_bad(c, threshold=2) or too_similar_to_recent(c):
            continue

        if not comment_seen(c):
            remember_comment(c)
            remember_opener(_openers(c))
            remember_ngrams(c)
            out.append(c)
        else:
            # small tweak path to rescue near-duplicate if it's short
            toks = words(c)
            if len(toks) < 13:
                alt = enforce_word_count_natural(c + " today")
                if alt and not comment_seen(alt) and not contains_generic_phrase(alt):
                    remember_comment(alt)
                    remember_opener(_openers(alt))
                    remember_ngrams(alt)
                    out.append(alt)

    # final hybrid pairing: maximize vibe diversity
    if len(out) >= 2:
        out = pick_two_diverse_text(out)

    return out[:2]

def offline_two_comments(text: str, author: Optional[str]) -> list[str]:
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
def parse_two_comments_flex(raw_text: str) -> list[str]:
    out: list[str] = []
    try:
        m = re.search(r"\[[\s\S]*\]", raw_text)
        candidate = m.group(0) if m else raw_text
        data = json.loads(candidate)
        if isinstance(data, dict):
            data = data.get("comments") or data.get("items") or data.get("data")
        if isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        out = []
    if len(out) >= 2:
        return out[:2]
    quoted = re.findall(r'["“](.+?)["”]', raw_text)
    if len(quoted) >= 2:
        return [q.strip() for q in quoted[:2]]
    parts = re.split(r"(?:^|\n)\s*(?:\d+[\).\:-]|[-•*])\s*", raw_text)
    parts = [p.strip() for p in parts if p and not p.isspace()]
    parts = [p for p in parts if len(p.split()) >= 3]
    if len(parts) >= 2:
        return parts[:2]
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[:2]
    m2 = re.split(r"\s*[;|/\\]+\s*", raw_text)
    if len(m2) >= 2:
        return [m2[0].strip(), m2[1].strip()]
    return []

# ------------------------------------------------------------------------------
# Groq generator (exactly 2, 6–13 words, tolerant parsing)
def groq_two_comments(tweet_text: str, author: str | None) -> list[str]:
    if not (USE_GROQ and _groq_client):
        raise RuntimeError("Groq disabled or client not available")

    sys_prompt = (
        "You write extremely short, human comments for social posts.\n"
        "- Output exactly two comments.\n"
        "- Each comment must be 6-13 words.\n"
        "- Natural conversational tone, as if you just read the post.\n"
        "- The two comments must have different vibes (e.g., supportive vs curious).\n"
        "- Avoid emojis, hashtags, links, or AI-ish phrases.\n"
        "- Avoid repetitive templates; vary syntax and rhythm.\n"
        "- Prefer returning a pure JSON array of two strings, like: "
        "[\"first comment\", \"second comment\"].\n"
        "- If you cannot return JSON, return two lines separated by a newline.\n"
    )

    user_prompt = (
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Return exactly two distinct comments (JSON array or two lines)."
    )

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
                temperature=0.8,
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
                time.sleep(wait_secs); continue
            raise

    if resp is None:
        raise RuntimeError("Groq call failed after retries")

    raw = (resp.choices[0].message.content or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates)

    if len(candidates) < 2:
        sents = re.split(r"[.!?]\s+", raw)
        sents = [enforce_word_count_natural(s) for s in sents if s.strip()]
        sents = [s for s in sents if 6 <= len(words(s)) <= 13]
        candidates = enforce_unique(candidates + sents[:2])

    tries = 0
    while len(candidates) < 2 and tries < 2:
        tries += 1
        candidates = enforce_unique(candidates + offline_two_comments(tweet_text, author))

    if len(candidates) < 2:
        raise RuntimeError("Could not produce two valid comments")

    return candidates[:2]

# ------------------------------------------------------------------------------
# OpenAI / Gemini / Mistral / Cohere / HF generators (same constraints as Groq)
def _llm_sys_prompt() -> str:
    return (
        "You write extremely short, human comments for social posts.\n"
        "- Output exactly two comments.\n"
        "- Each comment must be 6-13 words.\n"
        "- Natural conversational tone, as if you just read the post.\n"
        "- The two comments must have different vibes (e.g., supportive vs curious).\n"
        "- Avoid emojis, hashtags, links, or AI-ish phrases.\n"
        "- Avoid repetitive templates; vary syntax and rhythm.\n"
        "- Prefer returning a pure JSON array of two strings, like: "
        "[\"first comment\", \"second comment\"].\n"
        "- If you cannot return JSON, return two lines separated by a newline.\n"
    )

def openai_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_OPENAI and _openai_client):
        raise RuntimeError("OpenAI disabled or client not available")

    user_prompt = (
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
        temperature=0.8,
    )
    raw = (resp.choices[0].message.content or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates)
    if len(candidates) < 2:
        raise RuntimeError("OpenAI did not produce two valid comments")
    return candidates[:2]

def gemini_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_GEMINI and _gemini_model):
        raise RuntimeError("Gemini disabled or client not available")

    user_prompt = (
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
    candidates = enforce_unique(candidates)
    if len(candidates) < 2:
        raise RuntimeError("Gemini did not produce two valid comments")
    return candidates[:2]

def mistral_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_MISTRAL and _mistral_client):
        raise RuntimeError("Mistral disabled or client not available")

    prompt = _llm_sys_prompt() + "\n\n" + (f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\nReturn exactly two distinct comments.")
    raw = ""
    try:
        try:
            resp = _mistral_client.generate(model=MISTRAL_MODEL, input=prompt, max_tokens=160, temperature=0.9)
            if hasattr(resp, "generations") and resp.generations:
                gen = resp.generations[0]
                if isinstance(gen, (list, tuple)) and gen:
                    raw = getattr(gen[0], "text", str(gen[0])) or ""
                else:
                    raw = getattr(gen, "text", str(gen)) or ""
            elif hasattr(resp, "output"):
                raw = getattr(resp, "output", "") or str(resp)
            else:
                raw = str(resp)
        except Exception:
            resp = _mistral_client.create(prompt=prompt, model=MISTRAL_MODEL, max_tokens=160, temperature=0.9)
            raw = str(resp)
    except Exception as e:
        raise RuntimeError(f"Mistral call failed: {e}")

    raw = (raw or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates)
    if len(candidates) < 2:
        sents = re.split(r"[.!?]\s+", raw)
        sents = [enforce_word_count_natural(s) for s in sents if s.strip()]
        sents = [s for s in sents if 6 <= len(words(s)) <= 13]
        candidates = enforce_unique(candidates + sents[:2], tweet_text=tweet_text)

    tries = 0
    while len(candidates) < 2 and tries < 2:
        tries += 1
        candidates = enforce_unique(candidates + offline_two_comments(tweet_text, author), tweet_text=tweet_text)

    if len(candidates) < 2:
        raise RuntimeError("Could not produce two valid comments from Mistral")

    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra)
        if len(merged) >= 2:
            candidates = merged[:2]

    return candidates[:2]

def cohere_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_COHERE and _cohere_client):
        raise RuntimeError("Cohere disabled or client not available")

    prompt = _llm_sys_prompt() + "\n\n" + (f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\nReturn exactly two distinct comments.")
    try:
        resp = _cohere_client.generate(model=COHERE_MODEL, prompt=prompt, max_tokens=120, temperature=0.8, k=0)
        raw = getattr(resp, 'text', '') or str(resp)
    except Exception as e:
        raise RuntimeError(f"Cohere call failed: {e}")

    raw = (raw or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    if len(candidates) < 2:
        candidates = enforce_unique(candidates + offline_two_comments(tweet_text, author), tweet_text=tweet_text)
    if len(candidates) < 2:
        raise RuntimeError("Cohere did not produce two valid comments")
    return candidates[:2]

def hf_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_HF and _hf_pipeline):
        raise RuntimeError("HuggingFace pipeline not available")

    prompt = _llm_sys_prompt() + "\n\n" + (f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\nReturn exactly two distinct comments.")
    try:
        outs = _hf_pipeline(prompt, max_new_tokens=120, do_sample=True, temperature=0.8)
        if isinstance(outs, list) and outs:
            raw = outs[0].get('generated_text') or outs[0].get('text') or str(outs[0])
        else:
            raw = str(outs)
    except Exception as e:
        raise RuntimeError(f"HuggingFace model failed: {e}")

    raw = (raw or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    if len(candidates) < 2:
        candidates = enforce_unique(candidates + offline_two_comments(tweet_text, author), tweet_text=tweet_text)
    if len(candidates) < 2:
        raise RuntimeError("HuggingFace did not produce two valid comments")
    return candidates[:2]

# ------------------------------------------------------------------------------
def _available_providers() -> list[tuple[str, callable]]:
    providers: list[tuple[str, callable]] = []
    if USE_GROQ and _groq_client:
        providers.append(("groq", groq_two_comments))
    if USE_OPENAI and _openai_client:
        providers.append(("openai", openai_two_comments))
    if USE_GEMINI and _gemini_model:
        providers.append(("gemini", gemini_two_comments))
    if USE_MISTRAL and _mistral_client:
        providers.append(("mistral", mistral_two_comments))
    if USE_COHERE and _cohere_client:
        providers.append(("cohere", cohere_two_comments))
    if USE_HF and _hf_pipeline:
        providers.append(("hf", hf_two_comments))
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

    - For each request, randomize the order of enabled providers (Groq, OpenAI, Gemini, Mistral, Cohere, HF).
    - Try them in that random order, accumulating comments.
    - As soon as we have 2 solid comments, stop.
    - If all fail or give < 2, fall back to offline generator.
    """
    candidates: list[str] = []

    providers = _available_providers()
    if providers:
        # randomize call order each request
        random.shuffle(providers)

        for name, fn in providers:
            try:
                more = fn(tweet_text, author)
                if more:
                    # merge + dedupe / anti-pattern logic
                    candidates = enforce_unique(candidates + more, tweet_text=tweet_text)
            except Exception as e:
                logger.warning("%s provider failed: %s", name, e)

            if len(candidates) >= 2:
                break

    # If we still don't have 2 comments, offline generator rescues
    if len(candidates) < 2:
        try:
            offline = offline_two_comments(tweet_text, author)
            if offline:
                candidates = enforce_unique(candidates + offline, tweet_text=tweet_text)
        except Exception as e:
            logger.warning("Offline generator failed: %s", e)

    # Last-chance fallback: call OfflineCommentGenerator directly
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

    # Limit to exactly 2 text comments
    candidates = [c for c in candidates if c][:2]

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

# ------------------------------------------------------------------------------
# API routes (batching + pacing)
# ------------------------------------------------------------------------------

def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def _canonical_x_url_from_tweet(original_url: str, t: Any) -> str:
    if getattr(t, 'handle', None) and getattr(t, 'tweet_id', None):
        return f"https://x.com/{t.handle}/status/{t.tweet_id}"
    return original_url

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

        t = fetch_tweet_data(url)
        handle = t.handle or _extract_handle_from_url(url)

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
        return jsonify({
            "url": url,
            "error": str(e),
            "comments": [],
            "code": e.code,
        }), 502
    except Exception:
        logger.exception("Unhandled error during reroll for %s", url)
        return jsonify({
            "url": url,
            "error": "internal_error",
            "comments": [],
            "code": "internal_error",
        }), 500

# ------------------------------------------------------------------------------

def main() -> None:
    init_db()
    # threading.Thread(target=keep_alive, daemon=True).start()  # optional keep-alive
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
