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
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

# ---- your helpers -----------------------------------------------------------
from utils import (
    CrownTALKError,
    fetch_tweet_data,          # your VX/FX fetcher with retry/backoff
    clean_and_normalize_urls,  # your URL cleaner
)

# ----------------------------------------------------------------------------
# APP / CONFIG
# ----------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

PORT = int(os.environ.get("PORT", "10000"))
DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")
MEM_DB_PATH = os.environ.get("MEM_DB_PATH", DB_PATH)  # reuse same sqlite file
MAX_URLS_PER_REQUEST = 8
BATCH_SIZE = 2
KEEP_ALIVE_INTERVAL = 600  # seconds
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "")

# ----------------------------------------------------------------------------
# KEEPALIVE (Render free)
# ----------------------------------------------------------------------------
def keep_alive():
    if not BACKEND_PUBLIC_URL:
        return
    while True:
        try:
            import requests  # already in your requirements
            requests.get(f"{BACKEND_PUBLIC_URL}/", timeout=5)
        except Exception:
            pass
        time.sleep(KEEP_ALIVE_INTERVAL)

# ----------------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------------
def now_ts() -> int:
    return int(time.time())

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text or "").strip()

def only_ascii(s: str) -> str:
    return "".join(ch for ch in (s or "") if ord(ch) < 128)

def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        m = re.search(
            r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/([^/]+)/status/",
            url,
            re.I,
        )
        if m:
            return m.group(1)
    except Exception:
        return None
    return None

# ----------------------------------------------------------------------------
# DB INIT (safe multi-proc)
# ----------------------------------------------------------------------------
try:
    import fcntl  # Linux on Render ✅
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False

def get_conn():
    return sqlite3.connect(
        DB_PATH, timeout=30, isolation_level=None, check_same_thread=False
    )

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

            -- OTP-style guards
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

def init_db():
    def _safe():
        try:
            _do_init()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(1.0)
                _do_init()
            else:
                raise
    if _HAS_FCNTL:
        _locked_init(_safe)
    else:
        _safe()

# ----------------------------------------------------------------------------
# LIGHTWEIGHT MEMORY (exact dupes + style memory)
# ----------------------------------------------------------------------------
def _normalize_for_memory(text: str) -> str:
    t = normalize_ws(text).lower()
    t = re.sub(r"[^\w\s']+", " ", t)         # keep word/apostrophes
    t = re.sub(r"\s+", " ", t).strip()
    return t

def comment_seen(text: str) -> bool:
    norm = _normalize_for_memory(text)
    if not norm:
        return False
    h = sha256(norm)
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT 1 FROM comments_seen WHERE hash=? LIMIT 1", (h,)).fetchone()
            return row is not None
    except Exception:
        return False

def remember_comment(text: str, url: str = "", lang: Optional[str] = None) -> None:
    try:
        norm = _normalize_for_memory(text)
        if not norm:
            return
        h = sha256(norm)
        with get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO comments_seen(hash, created_at) VALUES(?,?)", (h, now_ts()))
            # also store in comments for auditing
            conn.execute("INSERT INTO comments(url, lang, text) VALUES(?,?,?)", (url, lang, text))
    except Exception:
        pass
    # additional memory
    try:
        remember_ngrams(text)
        remember_opener(_openers(text))
    except Exception:
        pass

# ---- OTP helpers: openers, n-grams, and template burn ----------------------
def _openers(text: str) -> str:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return " ".join(w[:3])

def _trigrams(text: str) -> List[str]:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [" ".join(w[i:i+3]) for i in range(len(w)-2)]

def opener_seen(opener: str) -> bool:
    try:
        with get_conn() as conn:
            return conn.execute("SELECT 1 FROM comments_openers_seen WHERE opener=? LIMIT 1", (opener,)).fetchone() is not None
    except Exception:
        return False

def remember_opener(opener: str):
    try:
        with get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO comments_openers_seen(opener, created_at) VALUES (?,?)", (opener, now_ts()))
    except Exception:
        pass

def trigram_overlap_bad(text: str, threshold: int = 2) -> bool:
    grams = _trigrams(text)
    if not grams:
        return False
    hits = 0
    try:
        with get_conn() as conn:
            for g in grams:
                if conn.execute("SELECT 1 FROM comments_ngrams_seen WHERE ngram=? LIMIT 1", (g,)).fetchone():
                    hits += 1
                    if hits >= threshold:
                        return True
    except Exception:
        return False
    return False

def remember_ngrams(text: str):
    grams = _trigrams(text)
    if not grams:
        return
    now = now_ts()
    try:
        with get_conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO comments_ngrams_seen(ngram, created_at) VALUES (?,?)",
                [(g, now) for g in grams],
            )
    except Exception:
        pass

def template_burned(tmpl: str) -> bool:
    thash = sha256(tmpl)
    try:
        with get_conn() as conn:
            return conn.execute("SELECT 1 FROM comments_templates_seen WHERE thash=? LIMIT 1", (thash,)).fetchone() is not None
    except Exception:
        return False

def remember_template(tmpl: str):
    thash = sha256(tmpl)
    try:
        with get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO comments_templates_seen(thash, created_at) VALUES (?,?)", (thash, now_ts()))
    except Exception:
        pass

# ----------------------------------------------------------------------------
# CHUNKER
# ----------------------------------------------------------------------------
def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# ----------------------------------------------------------------------------
# GENERATOR
# ----------------------------------------------------------------------------
EN_STOPWORDS = {
    "the","a","an","and","or","but","to","in","on","of","for","with","at","from",
    "by","about","as","into","like","through","after","over","between","out",
    "against","during","without","before","under","around","among","is","are","be",
    "am","was","were","it","its","that","this"
}

AI_BLOCKLIST = {
    # original + expanded
    "amazing","awesome","incredible","empowering","game changer","game-changing","transformative",
    "paradigm shift","in this digital age","as an ai","as a language model","in conclusion",
    "in summary","furthermore","moreover","navigate this landscape","ever-evolving landscape",
    "leverage this insight","cutting edge","state of the art","unprecedented","unleash","unleashing",
    "unlock the power","harness the power","embark on this journey","empower","revolutionize",
    "disruptive","bestie","like and retweet","thoughts?","agree?","who's with me",
    "drop your thoughts","smash that like button","link in bio","in case you missed it","i think",
    "i believe","great point","just saying","according to","to be honest",
    # new stricter bits
    "actually","literally","personally i think","my take","as someone who",
    "at the end of the day","moving forward","synergy","circle back","bandwidth",
    "double down","let that sink in","on so many levels","tbh","tracks",
}

EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re.UNICODE)

def build_context_profile(
    raw_text: str,
    url: Optional[str] = None,
    tweet_author: Optional[str] = None,
    handle: Optional[str] = None,
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

def pick_focus_token(key_tokens: List[str]) -> Optional[str]:
    toks = [t for t in key_tokens if t and len(t) > 1 and t.lower() not in EN_STOPWORDS]
    return random.choice(toks) if toks else None

class OfflineCommentGenerator:
    def __init__(self):
        self.random = random.Random()

    STARTER_BLOCKLIST = {
        "yeah this","honestly this","kind of","nice to","hard to","feels like",
        "this is","short line","funny how","appreciate that","interested to",
        "curious where","nice to see","chill sober","good reminder","yeah that",
    }

    def _violates_ai_blocklist(self, text: str) -> bool:
        low = text.lower()
        for phrase in AI_BLOCKLIST:
            if phrase in low:
                return True
        # pattern filters
        if re.search(r"\b(so|very|really)\s+\1\b", low):  # repeated adverbs
            return True
        if len(re.findall(r"\.\.\.", text)) > 1:          # ellipses abuse
            return True
        if re.search(r"[🔥🚀✨]{2,}", text):               # emoji clusters
            return True
        if low.count("—") > 1:                            # em-dash spam
            return True
        if re.search(r"\btracks\b.*\btbh\b", low):        # “tracks … tbh”
            return True
        return False

    def _diversity_ok(self, text: str) -> bool:
        opener = _openers(text)
        if opener in self.STARTER_BLOCKLIST:
            return False
        if opener_seen(opener):
            return False
        if trigram_overlap_bad(text, threshold=2):
            return False
        toks = re.findall(r"[A-Za-z][A-Za-z0-9']+", text.lower())
        novel = [t for t in toks if t not in EN_STOPWORDS and t not in {w.lower() for w in AI_BLOCKLIST}]
        return len(set(novel)) >= 2

    def _length_ok(self, text: str, mode: str) -> bool:
        wc = len(text.split())
        return (mode == "short" and 6 <= wc <= 9) or \
               (mode == "medium" and 10 <= wc <= 14) or \
               (mode == "long" and 15 <= wc <= 22)

    def _tidy_comment(self, text: str, english: bool = True) -> str:
        t = normalize_ws(text)
        t = re.sub(r"\s([,.;:?!])", r"\1", t)
        t = t.strip()
        if english:
            t = re.sub(r"[^\x00-\x7F]+", "", t)  # English: strip emoji
        if len(t) < 4:
            return ""
        return t

    def _english_buckets(self, ctx: Dict[str, Any]) -> Dict[str, List[str]]:
        name_pref = ""
        if self.random.random() < 0.30:
            if ctx.get("handle"):
                name_pref = f"@{ctx['handle']} "
            elif ctx.get("author_name"):
                first = ctx["author_name"].split()[0]
                name_pref = f"{first}, "

        focus_slot = "{focus}"
        return {
            "answerish": [
                f"{name_pref}short answer: that {focus_slot} detail matters most",
                f"{name_pref}if you’re weighing {focus_slot}, the metric trend is the tell",
                f"{name_pref}the follow-up here is how {focus_slot} changes the next step",
            ],
            "specificity": [
                f"{name_pref}{focus_slot} reads like execution, not hype",
                f"{name_pref}if {focus_slot} holds, the thesis tightens",
                f"{name_pref}numbers aside, {focus_slot} is the practical bit here",
            ],
            "nuance": [
                f"{name_pref}good to see the boring parts of {focus_slot} get airtime",
                f"{name_pref}this frames {focus_slot} without the usual noise",
                f"{name_pref}{focus_slot} matters more than the headline take",
            ],
            "pushback": [
                f"{name_pref}only caveat: {focus_slot} doesn’t solve incentives alone",
                f"{name_pref}worth separating {focus_slot} from pure optics",
                f"{name_pref}i’d stress {focus_slot} is the constraint, not the story",
            ],
        }

    def _native_buckets(self, script: str, ctx: Dict[str, Any]) -> List[str]:
        focus_slot = "{focus}"
        if script == "bn":  # Bengali
            return [
                f"{focus_slot} নিয়ে ব্যবহারিক কথা—হাইপ না, কাজ কী হচ্ছে",
                f"{focus_slot} ঠিক কোন জায়গায় বদল আনবে সেটাই পয়েন্ট",
                f"{focus_slot} টা থাকলে কথাটা আরও ক্লিয়ার",
            ]
        if script == "hi":  # Hindi
            return [
                f"{focus_slot} की असल बात यही है—शोर कम, काम ज़्यादा",
                f"{focus_slot} यहाँ गेम बदलता है, बस उतना ही",
                f"{focus_slot} पर ध्यान दो, बचे सब अपने आप सेट हो जाएगा",
            ]
        if script == "ar":  # Arabic (plain)
            return [
                f"النقطة العملية هنا هي {focus_slot}، بعيداً عن الضجيج",
                f"{focus_slot} هو التفصيل اللي يوضح الصورة",
                f"لو ركّزنا على {focus_slot} تتّضح الفكرة",
            ]
        # default neutral
        return [
            f"{focus_slot} here is the practical hinge",
            f"Focus on {focus_slot} and the rest aligns",
            f"{focus_slot} is the part that moves the needle",
        ]

    def _detect_topic(self, text: str) -> Tuple[str, bool]:
        lo = (text or "").lower()
        crypto = any(tok in lo for tok in ["eth", "sol", "btc", "chain", "dex", "liquidity", "token", "airdrop"])
        topic = "crypto" if crypto else "general"
        return topic, crypto

    def _keywords(self, text: str, ctx: Dict[str, Any]) -> List[str]:
        tokens = []
        tokens += ctx.get("projects", [])
        tokens += ctx.get("keywords", [])
        tokens += ctx.get("numbers", [])
        tokens = [t for t in tokens if t]
        seen = set(); out = []
        for t in tokens:
            if t not in seen:
                seen.add(t); out.append(t)
        return out[:8]

    def _make_english_comment(self, text: str, author: Optional[str], ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        topic, crypto = self._detect_topic(text)
        key_tokens = self._keywords(text, ctx)
        focus = pick_focus_token(key_tokens) or ""
        buckets = self._english_buckets(ctx)
        kinds = list(buckets.keys())
        kind = self.random.choice(kinds)
        author_ref = ctx.get("handle") or (author.split()[0] if author else "")

        length_mode = self.random.choice(["short","medium","long"])
        last_candidate = ""

        for attempt in range(40):
            if attempt and attempt % 8 == 0:
                length_mode = self.random.choice(["short","medium","long"])
                alt_focus = pick_focus_token(key_tokens)
                if alt_focus:
                    focus = alt_focus
                kind = random.choice(kinds)

            tmpl = random.choice(buckets[kind])
            if template_burned(tmpl):
                continue

            out = tmpl.format(author=author_ref or "", focus=focus)
            out = self._tidy_comment(out, english=True)
            if not out:
                continue
            if self._violates_ai_blocklist(out):
                continue
            if not self._length_ok(out, length_mode):
                continue
            if not self._diversity_ok(out):
                last_candidate = out
                continue
            if comment_seen(out):
                last_candidate = out
                continue
            remember_template(tmpl)
            remember_comment(out)
            return {"kind": kind, "text": out}

        if last_candidate:
            return {"kind": "fallback", "text": last_candidate}
        return None

    def _make_native_comment(self, text: str, ctx: Dict[str, Any]) -> Optional[str]:
        script = ctx.get("script", "latn")
        buckets = self._native_buckets(script, ctx)
        key_tokens = self._keywords(text, ctx)
        focus = pick_focus_token(key_tokens) or ""
        last_candidate = ""

        for _ in range(32):
            tmpl = random.choice(buckets)
            candidate = tmpl.format(focus=focus)
            candidate = normalize_ws(candidate)
            if self._violates_ai_blocklist(candidate):
                continue
            if not self._diversity_ok(candidate):
                continue
            if comment_seen(candidate):
                continue
            remember_comment(candidate)
            return candidate
        return last_candidate or None

    def generate_comments(
        self,
        text: str,
        author: Optional[str],
        handle: Optional[str] = None,
        lang_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ctx = build_context_profile(text, url=None, tweet_author=author, handle=handle)
        is_non_english = ctx["script"] != "latn"

        comments: List[Dict[str, Any]] = []

        if is_non_english:
            # 1) Native
            native = self._make_native_comment(text, ctx)
            if native:
                comments.append({"lang": ctx["script"], "text": native})

            # 2) English
            en = self._make_english_comment(text, author, ctx)
            if en:
                comments.append({"lang": "en", "text": en["text"]})
            return comments

        # English tweet: return TWO different English lines
        en1 = self._make_english_comment(text, author, ctx)
        en2 = None
        for _ in range(6):
            cand = self._make_english_comment(text, author, ctx)
            if not cand or not en1:
                continue
            if cand["text"].strip().lower() != en1["text"].strip().lower():
                en2 = cand
                break

        if en1:
            comments.append({"lang": "en", "text": en1["text"]})
        if en2:
            comments.append({"lang": "en", "text": en2["text"]})

        if len(comments) == 1:
            cand = self._make_english_comment(text, author, ctx)
            if cand and cand["text"].strip().lower() != comments[0]["text"].strip().lower():
                comments.append({"lang": "en", "text": cand["text"]})

        return comments

generator = OfflineCommentGenerator()

# ----------------------------------------------------------------------------
# CORS + HEALTH
# ----------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.route("/comment", methods=["POST", "OPTIONS"])
def comment_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        payload = request.get_json(force=True, silent=True) or {}
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
                    t = fetch_tweet_data(url)  # your robust fetcher
                    comments = generator.generate_comments(
                        text=t.text,
                        author=t.author_name or None,
                        handle=_extract_handle_from_url(url),
                        lang_hint=t.lang or None,
                    )
                    results.append({"url": url, "comments": comments})
                except CrownTALKError as e:
                    failed.append({"url": url, "reason": str(e), "code": e.code})
                except Exception:
                    logger.exception("Unhandled error while processing %s", url)
                    failed.append({"url": url, "reason": "internal_error", "code": "internal_error"})
                time.sleep(0.05)  # tiny yield
        return jsonify({"results": results, "failed": failed}), 200
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        return jsonify({"error": "internal_error", "code": "internal_error"}), 500

@app.route("/reroll", methods=["POST", "OPTIONS"])
def reroll_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get("url") or ""
        if not url:
            return jsonify({"error": "Missing 'url' field", "comments": [], "code": "bad_request"}), 400
        try:
            t = fetch_tweet_data(url)
            comments = generator.generate_comments(
                text=t.text,
                author=t.author_name or None,
                handle=_extract_handle_from_url(url),
                lang_hint=t.lang or None,
            )
            return jsonify({"url": url, "comments": comments}), 200
        except CrownTALKError as e:
            return jsonify({"url": url, "error": str(e), "comments": [], "code": e.code}), 502
        except Exception:
            logger.exception("Unhandled error during reroll for %s", url)
            return jsonify({"url": url, "error": "internal_error", "comments": [], "code": "internal_error"}), 500
    except Exception:
        logger.exception("Unhandled error during reroll")
        return jsonify({"error": "internal_error", "comments": [], "code": "internal_error"}), 500

# ----------------------------------------------------------------------------
# BOOT
# ----------------------------------------------------------------------------
def main():
    init_db()
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
