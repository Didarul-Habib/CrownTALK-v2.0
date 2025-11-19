# main.py
from flask import Flask, request, jsonify
import threading
import time
import os
import re
import json
import random
import hashlib
import sqlite3
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crown")

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
PORT = int(os.environ.get("PORT", "10000"))
MEM_DB_PATH = os.environ.get("MEM_DB_PATH", "comments_mem.sqlite3")
MAX_URLS_PER_REQUEST = 8
REQUEST_TIMEOUT_SECS = 10

# ---------------------------------------------------------
# LIGHTWEIGHT KEEPALIVE (Render free)
# ---------------------------------------------------------
def keep_alive():
    while True:
        try:
            time.sleep(25)
        except Exception:
            pass

# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------
def now_ts() -> int:
    return int(time.time())

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text or "").strip()

def only_ascii(s: str) -> str:
    return "".join(ch for ch in (s or "") if ord(ch) < 128)

def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        m = re.search(r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/([^/]+)/status/", url, re.I)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None

# ---------------------------------------------------------
# DB INIT
# ---------------------------------------------------------
def init_db():
    def _safe():
        try:
            os.makedirs(os.path.dirname(MEM_DB_PATH) or ".", exist_ok=True)
        except Exception:
            pass
        try:
            conn = sqlite3.connect(MEM_DB_PATH, timeout=5.0)
            try:
                conn.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS comments(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT,
                        lang TEXT,
                        text TEXT,
                        created_at INTEGER DEFAULT (strftime('%s','now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_comments_url ON comments(url);
                    CREATE TABLE IF NOT EXISTS comments_seen(
                        hash TEXT PRIMARY KEY,
                        created_at INTEGER
                    );
                    -- prevent style and paraphrase repeats ("OTP" feeling)
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
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.exception("DB init failed: %s", e)

    # fcntl may not be available on Windows; Render is Linux so you're fine
    try:
        import fcntl  # noqa: F401
        _HAS_FCNTL = True
    except Exception:
        _HAS_FCNTL = False

    def _locked_init(target):
        lock_path = MEM_DB_PATH + ".init.lock"
        with open(lock_path, "a+") as f:
            f.seek(0)
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                target()
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass

    if _HAS_FCNTL:
        _locked_init(_safe)
    else:
        _safe()

# ---------------------------------------------------------
# MEMORY HELPERS
# ---------------------------------------------------------
def _normalize_for_memory(text: str) -> str:
    t = normalize_ws(text).lower()
    # remove punctuation except apostrophes inside words
    t = re.sub(r"[^\w\s']+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def comment_seen(text: str) -> bool:
    try:
        norm = _normalize_for_memory(text)
        key = sha256(norm)
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            return conn.execute("SELECT 1 FROM comments_seen WHERE hash=? LIMIT 1", (key,)).fetchone() is not None
        finally:
            conn.close()
    except Exception:
        return False

def remember_comment(text: str):
    try:
        norm = _normalize_for_memory(text)
        key = sha256(norm)
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            conn.execute("INSERT OR IGNORE INTO comments_seen(hash, created_at) VALUES (?,?)", (key, now_ts()))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
    # extra: remember trigrams + opener + store in comments table (url/lan unknown here)
    try:
        remember_ngrams(text)
        remember_opener(_openers(text))
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            conn.execute("INSERT INTO comments(url, lang, text) VALUES(?,?,?)", ("", None, text))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

# ---- OTP style memory helpers (openers + trigrams + template burn) ---------
def _openers(text: str) -> str:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return " ".join(w[:3])

def _trigrams(text: str) -> List[str]:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [" ".join(w[i:i+3]) for i in range(len(w)-2)]

def opener_seen(opener: str) -> bool:
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            row = conn.execute("SELECT 1 FROM comments_openers_seen WHERE opener=? LIMIT 1", (opener,)).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False

def remember_opener(opener: str):
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            conn.execute("INSERT OR IGNORE INTO comments_openers_seen(opener, created_at) VALUES (?,?)", (opener, now_ts()))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

def trigram_overlap_bad(text: str, threshold: int = 2) -> bool:
    grams = _trigrams(text)
    if not grams:
        return False
    hits = 0
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            for g in grams:
                if conn.execute("SELECT 1 FROM comments_ngrams_seen WHERE ngram=? LIMIT 1", (g,)).fetchone():
                    hits += 1
                    if hits >= threshold:
                        return True
        finally:
            conn.close()
    except Exception:
        return False
    return False

def remember_ngrams(text: str):
    grams = _trigrams(text)
    if not grams:
        return
    now = now_ts()
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            conn.executemany("INSERT OR IGNORE INTO comments_ngrams_seen(ngram, created_at) VALUES (?,?)", [(g, now) for g in grams])
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

def template_burned(tmpl: str) -> bool:
    thash = sha256(tmpl)
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            row = conn.execute("SELECT 1 FROM comments_templates_seen WHERE thash=? LIMIT 1", (thash,)).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False

def remember_template(tmpl: str):
    thash = sha256(tmpl)
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            conn.execute("INSERT OR IGNORE INTO comments_templates_seen(thash, created_at) VALUES (?,?)", (thash, now_ts()))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

# ---------------------------------------------------------
# TWEET FETCH PLACEHOLDER (you can swap with your utils)
# ---------------------------------------------------------
@dataclass
class TweetData:
    url: str
    text: str
    author_name: Optional[str]
    lang: Optional[str]

def fetch_tweet(url: str) -> TweetData:
    """
    Replace with your existing VX/FX fetcher. We keep the signature.
    """
    # placeholder: derive author from URL, no network calls (offline)
    handle = _extract_handle_from_url(url)
    # Use handle as author if nothing else
    return TweetData(url=url, text=url, author_name=handle, lang=None)

# ---------------------------------------------------------
# CHUNKER
# ---------------------------------------------------------
def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# ---------------------------------------------------------
# OFFLINE COMMENT GENERATOR
# ---------------------------------------------------------
EN_STOPWORDS = {
    "the","a","an","and","or","but","to","in","on","of","for","with","at","from",
    "by","about","as","into","like","through","after","over","between","out",
    "against","during","without","before","under","around","among","is","are","be",
    "am","was","were","it","its","that","this"
}

AI_BLOCKLIST = {
    "in case you missed it","i think","i believe","great point","amazing",
    "just saying","according to","in summary","to be honest"
}
# expand
AI_BLOCKLIST.update({
    "actually","literally","personally i think","my take","as someone who",
    "at the end of the day","moving forward","synergy","circle back","bandwidth",
    "double down","let that sink in","on so many levels","tbh","tracks"
})

def build_context_profile(raw_text: str, url: Optional[str] = None, tweet_author: Optional[str] = None, handle: Optional[str] = None) -> Dict[str, Any]:
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
        if re.search(r"\b(so|very|really)\s+\1\b", low):
            return True
        if len(re.findall(r"\.\.\.", text)) > 1:
            return True
        if re.search(r"[🔥🚀✨]{2,}", text):
            return True
        if low.count("—") > 1:
            return True
        if re.search(r"\btracks\b.*\btbh\b", low):
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
        blk = {w.lower() for w in AI_BLOCKLIST}
        novel = [t for t in toks if t not in blk and t not in EN_STOPWORDS]
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
            # English: zero emoji
            t = re.sub(r"[^\x00-\x7F]+", "", t)
        if len(t) < 4:
            return ""
        return t

    def _english_buckets(self, ctx: Dict[str, Any]) -> Dict[str, List[str]]:
        # minimal, you can expand with more variants
        name_pref = ""
        if self.random.random() < 0.30:
            if ctx.get("handle"):
                name_pref = f"@{ctx['handle']} "
            elif ctx.get("author_name"):
                first = ctx["author_name"].split()[0]
                name_pref = f"{first}, "

        focus_slot = "{focus}"  # may be empty
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
        # lightweight native lines; extend as you like per language
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
        # default: just echo English-like neutral (still counted as native path)
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
        # keep cashtags/hashtags as-is
        tokens = [t for t in tokens if t]
        # avoid duplicates
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

        # fallback: minimal specific line
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

    def generate_comments(self, text: str, author: Optional[str], handle: Optional[str] = None, lang_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        ctx = build_context_profile(text, url=None, tweet_author=author, handle=handle)
        is_non_english = ctx["script"] != "latn"

        comments = []
        if is_non_english:
            native = self._make_native_comment(text, ctx)
            if native:
                comments.append({"lang": ctx["script"], "text": native})
            en = self._make_english_comment(text, author, ctx)
            if en:
                comments.append({"lang": "en", "text": en["text"]})
        else:
            en = self._make_english_comment(text, author, ctx)
            if en:
                comments.append({"lang": "en", "text": en["text"]})
        return comments

generator = OfflineCommentGenerator()

# ---------------------------------------------------------
# API
# ---------------------------------------------------------
@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json(force=True, silent=True) or {}
        urls = data.get("urls") or []
        if not isinstance(urls, list):
            return jsonify({"error": "bad_request", "code": "urls_must_be_list"}), 400
        urls = urls[:MAX_URLS_PER_REQUEST]

        results = []
        for batch in chunked(urls, 4):
            for url in batch:
                try:
                    t = fetch_tweet(url)
                    comments = generator.generate_comments(
                        text=t.text,
                        author=t.author_name or None,
                        handle=_extract_handle_from_url(url),
                        lang_hint=t.lang or None,
                    )
                    results.append({"url": url, "comments": [{"text": c["text"], "lang": c["lang"]} for c in comments]})
                except Exception as e:
                    logger.exception("Error generating for %s: %s", url, e)
                    results.append({"url": url, "comments": [], "error": "internal_error", "code": "internal_error"})
        return jsonify({"results": results})
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        return jsonify({"error": "internal_error", "code": "internal_error"}), 500

@app.route("/reroll", methods=["POST"])
def reroll():
    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get("url")
        if not url:
            return jsonify({"error": "bad_request", "code": "missing_url"}), 400

        try:
            t = fetch_tweet(url)
            comments = generator.generate_comments(
                text=t.text,
                author=t.author_name or None,
                handle=_extract_handle_from_url(url),
                lang_hint=t.lang or None,
            )
            return jsonify({"url": url, "comments": [{"text": c["text"], "lang": c["lang"]} for c in comments]})
        except Exception as e:
            logger.exception("Error during reroll: %s", e)
            return jsonify({"url": url, "error": "internal_error", "comments": [], "code": "internal_error"}), 500
    except Exception:
        logger.exception("Unhandled error during reroll for %s", url)
        return jsonify({"url": url, "error": "internal_error", "comments": [], "code": "internal_error"}), 500

# ---------------------------------------------------------
# CHUNKER
# ---------------------------------------------------------
def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# ---------------------------------------------------------
# OFFLINE COMMENT GENERATOR
# ---------------------------------------------------------
EN_STOPWORDS = {
    "the","a","an","and","or","but","to","in","on","of","for","with","at","from",
    "by","about","as","into","like","through","after","over","between","out",
    "against","during","without","before","under","around","among","is","are","be",
    "am","was","were","it","its","that","this"
}

# (The generator and helpers are already defined above.)

# ---------------------------------------------------------
# BOOT
# ---------------------------------------------------------
def _do_init():
    init_db()

def _safe_boot():
    try:
        _do_init()
    except Exception as e:
        logger.error("Boot init failed: %s", e)

def main():
    init_db()
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)

# WSGI / local
if __name__ == "__main__":
    main()
else:
    _safe_boot()
