from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from collections import Counter
from typing import List
import requests  # for optional keep-alive

from flask import Flask, request, jsonify
from utils import CrownTALKError, fetch_tweet_data, clean_and_normalize_urls

# ----------------- Groq setup -----------------
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

# ----------------- App & logging --------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

# ----------------- Config ---------------------
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "https://crowntalk.onrender.com")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
PER_URL_SLEEP = float(os.environ.get("PER_URL_SLEEP_SECONDS", "0.1"))
DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")
KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))

# ----------------- DB helpers -----------------
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False

def get_conn():
    return sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)

def _locked_init(fn):
    if not _HAS_FCNTL:
        return fn()
    lock_path = "/tmp/crowntalk.db.lock"
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)

def _do_init():
    with get_conn() as conn:
        conn.execute("PRAGMA busy_timeout=5000;")
        try: conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError: pass
        conn.executescript("""
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
        """)

def init_db():
    def _safe():
        try: _do_init()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(1.0); _do_init()
            else: raise
    _locked_init(_safe) if _HAS_FCNTL else _safe()

# ----------------- CORS -----------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ----------------- Health ---------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "groq": bool(USE_GROQ)}), 200

# ----------------- Memory ---------------------
def _normalize_for_memory(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
def _hash_comment(text: str) -> str:
    import hashlib as _h; return _h.sha256(_normalize_for_memory(text).encode("utf-8")).hexdigest()
def comment_seen(text: str) -> bool:
    try:
        with get_conn() as conn:
            return conn.execute("SELECT 1 FROM comments_seen WHERE hash=? LIMIT 1", (_hash_comment(text),)).fetchone() is not None
    except Exception: return False
def remember_comment(text: str) -> None:
    try:
        with get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO comments_seen(hash, created_at) VALUES(?,?)",
                         (_hash_comment(text), int(time.time())))
    except Exception: pass

# ----------------- Rules ----------------------
WORD_RE = re.compile(r"[A-Za-z0-9’']+(-[A-Za-z0-9’']+)?")
def words(t: str) -> list[str]: return WORD_RE.findall(t)
def sanitize_comment(raw: str) -> str:
    txt = re.sub(r"https?://\S+", "", raw)
    txt = re.sub(r"[@#]\S+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"[.!?;:…]+$", "", txt).strip()
    txt = re.sub(r"[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", "", txt)
    return txt
def enforce_word_count_natural(raw: str, min_w=6, max_w=13) -> str:
    txt = sanitize_comment(raw); toks = words(txt)
    if not toks: return ""
    if len(toks) > max_w: toks = toks[:max_w]
    while len(toks) < min_w:
        for filler in ["honestly","tbh","still","though","right"]:
            if len(toks) >= min_w: break
            toks.append(filler)
        if len(toks) < min_w: break
    return " ".join(toks).strip()

def enforce_unique(candidates: list[str]) -> list[str]:
    out=[]
    for c in candidates:
        c = enforce_word_count_natural(c)
        if not c: continue
        if not comment_seen(c):
            remember_comment(c); out.append(c); continue
        toks = words(c)
        if len(toks) < 13:
            tweak = random.choice(["today","lately","right now","for real"])
            alt = enforce_word_count_natural(" ".join(toks+[tweak]))
            if alt and not comment_seen(alt):
                remember_comment(alt); out.append(alt)
    return out

# ----------------- Offline generator ----------
EN_STOPWORDS = {"the","a","an","and","or","but","if","when","where","how","this","that","those","these",
"it","its","is","are","was","were","be","been","being","for","to","of","in","on","at","with","by","from",
"as","about","into","over","after","before","your","my","our","their","his","her","you","we","they","i",
"just","so","very","too","up","down","out","off","again"}
AI_BLOCKLIST = {"as an ai","as a language model","in conclusion","in summary","navigate this landscape",
"cutting edge","state of the art","unprecedented","empower","revolutionize","paradigm shift",
"smash that like button","link in bio","thoughts?","agree?","bestie","queen"}

def extract_keywords(text: str) -> list[str]:
    cleaned = re.sub(r"https?://\S+|[@#]\S+","", text)
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", cleaned)
    if not tokens: return []
    lower = [t.lower() for t in tokens]
    filtered = [t for t,l in zip(tokens,lower) if l not in EN_STOPWORDS and len(l)>2] or tokens
    from collections import Counter as _C
    counts = _C([t.lower() for t in filtered])
    seen=set(); result=[]
    for w in sorted(filtered, key=lambda w:(-counts[w.lower()], -len(w))):
        lw=w.lower()
        if lw not in seen:
            seen.add(lw); result.append(w)
    return result[:12]

def _rescue_two(tweet_text: str) -> list[str]:
    base = re.sub(r"https?://\S+|[@#]\S+", "", tweet_text or "").strip()
    kw = (re.findall(r"[A-Za-z]{3,}", base) or ["this"])[0].lower()
    a = enforce_word_count_natural(f"Fair point on {kw}, makes sense right now")
    b = enforce_word_count_natural(f"Curious where {kw} goes next, watching closely")
    if not a: a = "Makes sense right now honestly tbh still though right"
    if not b: b = "Curious where this goes honestly tbh still though right"
    return [a, b]

def offline_two_comments(tweet_text: str, author: str | None) -> list[str]:
    rnd = random.Random(time.time_ns() ^ hash(tweet_text))
    kws = extract_keywords(tweet_text) or ["this"]
    def build(tone: str) -> str:
        focus = rnd.choice(kws)
        starters = {
            "supportive":[
                f"Yeah this on {focus} makes solid sense",
                f"Glad someone said this about {focus}",
                f"This read on {focus} feels pretty accurate",
                f"Quietly agree with the take on {focus}",
            ],
            "curious":[
                f"Interesting angle on {focus}, what comes next",
                f"Curious where {focus} goes after this",
                f"Makes me wonder how {focus} plays out",
                f"Not sure yet, but {focus} point is strong",
            ],
            "skeptical":[
                f"I get it, but {focus} still seems messy",
                f"Half agree, {focus} has tradeoffs we ignore",
                f"Good point, though {focus} might be tricky",
                f"Reasonable take, yet {focus} keeps changing",
            ],
            "observational":[
                f"This is what {focus} looks like in practice",
                f"Everyday reality for {focus} is basically this",
                f"Seen similar patterns with {focus} lately",
                f"Kinda nails the vibe around {focus}",
            ],
        }
        raw = random.choice(starters.get(tone, starters["observational"]))
        out = enforce_word_count_natural(raw)
        if any(b in out.lower() for b in AI_BLOCKLIST):
            out = enforce_word_count_natural(f"{focus} take feels grounded, not overhyped")
        return out

    tone_a = rnd.choice(["supportive","observational","curious","skeptical"])
    tone_b = rnd.choice([t for t in ["supportive","observational","curious","skeptical"] if t != tone_a])

    uniq = enforce_unique([build(tone_a), build(tone_b)])
    tries_more = 0
    while len(uniq) < 2 and tries_more < 6:
        tries_more += 1
        uniq += [c for c in enforce_unique([build(rnd.choice(["supportive","observational","curious","skeptical"]))]) if c]
    if len(uniq) < 2:
        uniq += _rescue_two(tweet_text)
    uniq = [enforce_word_count_natural(c) for c in uniq if c][:2]
    return uniq

# ----------------- Groq generator -------------
def groq_two_comments(tweet_text: str, author: str | None) -> list[str]:
    if not (USE_GROQ and _groq_client):
        raise RuntimeError("Groq disabled or client not available")

    sys_prompt = (
        "You write extremely short, human comments for social posts.\n"
        "- Output exactly two comments.\n"
        "- Each comment must be 6–13 words.\n"
        "- Natural conversational tone, as if you just read the post.\n"
        "- The two comments must have different vibes (e.g., supportive vs curious).\n"
        "- Avoid emojis, hashtags, links, or AI-ish phrases.\n"
        "- Avoid repetitive templates; vary syntax and rhythm.\n"
        "- Return ONLY JSON array of two strings, no extra text."
    )
    user_prompt = (
        f"Post (author: {author or 'unknown'}):\n{tweet_text}\n\n"
        "Return JSON array of two distinct comments."
    )

    resp = None
    for attempt in range(3):
        try:
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_prompt}],
                n=1, max_tokens=120, temperature=0.8,
            )
            break
        except Exception as e:
            wait_secs = 0
            try:
                hdrs = getattr(getattr(e, "response", None), "headers", {}) or {}
                ra = hdrs.get("Retry-After")
                if ra: wait_secs = max(1, int(ra))
            except Exception: pass
            msg = str(e).lower()
            if not wait_secs and ("429" in msg or "rate" in msg or "quota" in msg or "retry-after" in msg):
                wait_secs = 2 + attempt
            if wait_secs:
                time.sleep(wait_secs); continue
            raise
    if resp is None:
        raise RuntimeError("Groq call failed after retries")

    raw = resp.choices[0].message.content.strip()
    comments: list[str] = []
    try:
        m = re.search(r"\[[\s\S]*\]", raw)
        data = json.loads(m.group(0) if m else raw)
        if isinstance(data, list):
            comments = [str(x) for x in data][:2]
    except Exception:
        parts = [p.strip("-• ").strip() for p in raw.splitlines() if p.strip()]
        comments = parts[:2]

    comments = [c for c in enforce_unique(comments) if c]
    tries = 0
    while len(comments) < 2 and tries < 4:
        tries += 1
        comments = [c for c in enforce_unique(comments + offline_two_comments(tweet_text, author)) if c]

    if len(comments) < 2:
        comments += _rescue_two(tweet_text)
        comments = [enforce_word_count_natural(c) for c in comments][:2]
        comments = [c for c in comments if c]

    if len(comments) < 2:
        raise RuntimeError("Could not produce two valid comments")
    return comments[:2]

# ----------------- Chunking -------------------
def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# ----------------- Keep-alive (optional) -----
def keep_alive():
    if not BACKEND_PUBLIC_URL: return
    while True:
        try: requests.get(f"{BACKEND_PUBLIC_URL}/", timeout=5)
        except Exception: pass
        time.sleep(KEEP_ALIVE_INTERVAL)

# ----------------- Routes --------------------
@app.route("/comment", methods=["POST", "OPTIONS"])
def comment_endpoint():
    if request.method == "OPTIONS": return ("", 204)
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body", "code": "invalid_json"}), 400

    urls = payload.get("urls")
    if not isinstance(urls, list) or not urls:
        return jsonify({"error": "Body must contain non-empty 'urls' array", "code": "bad_request"}), 400

    try:
        cleaned_urls = clean_and_normalize_urls(urls)
    except CrownTALKError as e:
        return jsonify({"error": str(e), "code": e.code}), 400
    except Exception:
        return jsonify({"error": "Failed to clean URLs", "code": "url_clean_error"}), 400

    results, failed = [], []
    for batch in chunked(cleaned_urls, BATCH_SIZE):
        for url in batch:
            try:
                t = fetch_tweet_data(url)
                if USE_GROQ and _groq_client:
                    try: comments = groq_two_comments(t.text, t.author_name or None)
                    except Exception as sub_err:
                        logger.warning("Groq failed for %s: %s — falling back to offline", url, sub_err)
                        comments = offline_two_comments(t.text, t.author_name or None)
                else:
                    comments = offline_two_comments(t.text, t.author_name or None)

                try:
                    with get_conn() as conn:
                        for c in comments:
                            conn.execute("INSERT INTO comments(url, lang, text) VALUES(?,?,?)", (url, "en", c))
                except Exception: pass

                results.append({"url": url, "comments": [{"lang": "en", "text": c} for c in comments]})
            except CrownTALKError as e:
                failed.append({"url": url, "reason": str(e), "code": e.code})
            except Exception:
                logger.exception("Unhandled error while processing %s", url)
                failed.append({"url": url, "reason": "internal_error", "code": "internal_error"})
            time.sleep(PER_URL_SLEEP)

    return jsonify({"results": results, "failed": failed}), 200

@app.route("/reroll", methods=["POST", "OPTIONS"])
def reroll_endpoint():
    if request.method == "OPTIONS": return ("", 204)
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body", "comments": [], "code": "invalid_json"}), 400

    url = data.get("url") or ""
    if not url:
        return jsonify({"error": "Missing 'url' field", "comments": [], "code": "bad_request"}), 400

    try:
        t = fetch_tweet_data(url)
        if USE_GROQ and _groq_client:
            try: comments = groq_two_comments(t.text, t.author_name or None)
            except Exception as sub_err:
                logger.warning("Groq reroll failed for %s: %s — falling back", url, sub_err)
                comments = offline_two_comments(t.text, t.author_name or None)
        else:
            comments = offline_two_comments(t.text, t.author_name or None)

        return jsonify({"url": url, "comments": [{"lang": "en", "text": c} for c in comments]}), 200
    except CrownTALKError as e:
        return jsonify({"url": url, "error": str(e), "comments": [], "code": e.code}), 502
    except Exception:
        logger.exception("Unhandled error during reroll for %s", url)
        return jsonify({"url": url, "error": "internal_error", "comments": [], "code": "internal_error"}), 500

# ----------------- Entrypoint -----------------
def _boot():
    init_db()
    # threading.Thread(target=keep_alive, daemon=True).start()  # optional
_boot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
