# main.py
from flask import Flask, request, jsonify
import threading
import requests
import time
import re
import random
from collections import Counter
import sqlite3
import hashlib
import logging
import os

def _extract_handle_from_url(url: str) -> str | None:
    try:
        m = re.search(r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/([^/]+)/status/", url, re.I)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


# ---------------------------------------------------------
# DB location (used by lightweight comment memory below)
# ---------------------------------------------------------
DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")

# File lock for safe one-time init on multi-proc hosts (Render)
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False

def get_conn():
    return sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None,   # autocommit
        check_same_thread=False,
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
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            pass
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
    """)

# ---------------------------------------------------------
# Shared helpers from utils.py
# ---------------------------------------------------------
from utils import (
    CrownTALKError,
    fetch_tweet_data,          # now rate-limited + retry + fallback
    clean_and_normalize_urls,  # url cleaner
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

BACKEND_PUBLIC_URL = "https://crowntalk.onrender.com"

BATCH_SIZE = 2
KEEP_ALIVE_INTERVAL = 600  # seconds

# ---------------------------------------------------------
# Lightweight global memory for duplicates (best-effort)
# ---------------------------------------------------------
MEM_DB_PATH = DB_PATH  # reuse same file

def _normalize_for_memory(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _hash_comment(norm_text: str) -> str:
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

def comment_seen(text: str) -> bool:
    norm = _normalize_for_memory(text)
    if not norm:
        return False
    h = _hash_comment(norm)
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            cur = conn.execute("SELECT 1 FROM comments_seen WHERE hash=? LIMIT 1", (h,))
            row = cur.fetchone()
        finally:
            conn.close()
        return row is not None
    except Exception:
        return False

def remember_comment(text: str) -> None:
    norm = _normalize_for_memory(text)
    if not norm:
        return
    h = _hash_comment(norm)
    now = int(time.time())
    try:
        conn = sqlite3.connect(MEM_DB_PATH, timeout=3.0)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO comments_seen(hash, created_at) VALUES(?,?)",
                (h, now),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ---------------------------------------------------------
# COMMENT ENDPOINT
# ---------------------------------------------------------
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
        return jsonify({"error": "Body must contain non-empty 'urls' array", "code": "bad_request"}), 400

    try:
        cleaned_urls = clean_and_normalize_urls(urls)
    except CrownTALKError as e:
        return jsonify({"error": str(e), "code": e.code}), 400
    except Exception:
        return jsonify({"error": "Failed to clean URLs", "code": "url_clean_error"}), 400

    results, failed = [], []
    generator = OfflineCommentGenerator()

    for batch in chunked(cleaned_urls, BATCH_SIZE):
        for url in batch:
            try:
                t = fetch_tweet_data(url)
                comments = generator.generate_comments(text=t.text, author=t.author_name or None, handle=_extract_handle_from_url(url), lang_hint=t.lang or None)
                results.append({"url": url, "comments": comments})
            except CrownTALKError as e:
                failed.append({"url": url, "reason": str(e), "code": e.code})
            except Exception:
                logger.exception("Unhandled error while processing %s", url)
                failed.append({"url": url, "reason": "internal_error", "code": "internal_error"})

            # tiny yield to avoid burst spikes against upstream API
            time.sleep(0.05)

    return jsonify({"results": results, "failed": failed}), 200

# ---------------------------------------------------------
# REROLL ENDPOINT
# ---------------------------------------------------------
@app.route("/reroll", methods=["POST", "OPTIONS"])
def reroll_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body", "comments": [], "code": "invalid_json"}), 400

    url = data.get("url") or ""
    if not url:
        return jsonify({"error": "Missing 'url' field", "comments": [], "code": "bad_request"}), 400

    try:
        t = fetch_tweet_data(url)
        generator = OfflineCommentGenerator()
        comments = generator.generate_comments(text=t.text, author=t.author_name or None, handle=_extract_handle_from_url(url), lang_hint=t.lang or None)
        return jsonify({"url": url, "comments": comments}), 200
    except CrownTALKError as e:
        return jsonify({"url": url, "error": str(e), "comments": [], "code": e.code}), 502
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
    "the","a","an","and","or","but","if","when","where","how","this","that","those","these",
    "it","its","is","are","was","were","be","been","being","for","to","of","in","on","at",
    "with","by","from","as","about","into","over","after","before","your","my","our","their",
    "his","her","you","we","they","i","just","so","very","too","up","down","out","off","again",
}

AI_BLOCKLIST = {
    "amazing","awesome","incredible","empowering","game changer","game-changing","transformative",
    "paradigm shift","in this digital age","as an ai","as a language model","in conclusion",
    "in summary","furthermore","moreover","navigate this landscape","ever-evolving landscape",
    "leverage this insight","cutting edge","state of the art","unprecedented","unleash","unleashing",
    "unlock the power","harness the power","embark on this journey","empower","revolutionize",
    "disruptive","slay","yass","yas","queen","bestie","like and retweet","thoughts?","thoughts ?",
    "agree?","agree ?","who's with me","drop your thoughts","smash that like button","link in bio",
}


# expand with additional phrases commonly associated with AI-y outputs
AI_BLOCKLIST.update({
    "actually","literally","personally i think","my take","as someone who",
    "at the end of the day","moving forward","synergy","circle back","bandwidth",
    "double down","let that sink in","on so many levels","tbh","tracks"
})
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re.UNICODE
)

class OfflineCommentGenerator:
    def __init__(self):

    STARTER_BLOCKLIST = {
        "yeah this","honestly this","kind of","nice to","hard to","feels like",
        "this is","short line","funny how","appreciate that","interested to",
        "curious where","nice to see","chill sober","good reminder","yeah that",
    }

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
        novel = [t for t in toks if t not in blk]
        return len(set(novel)) >= 2

    def _length_ok(self, text: str, mode: str) -> bool:
        wc = len(text.split())
        return (mode == "short" and 6 <= wc <= 9) or                (mode == "medium" and 10 <= wc <= 14) or                (mode == "long" and 15 <= wc <= 22)
        self.random = random.Random()

    def _is_probably_english(self, text: str, lang_hint: str | None) -> bool:
        stripped = re.sub(r"https?://\S+", "", text)
        stripped = re.sub(r"[@#]\S+", "", stripped)
        chars = [c for c in stripped if not c.isspace()]

        cjk_chars = [
            c for c in chars
            if ("\u4e00" <= c <= "\u9fff")
            or ("\u3040" <= c <= "\u30ff")
            or ("\uac00" <= c <= "\ud7af")
        ]
        ascii_letters = [c for c in chars if c.isascii() and c.isalpha()]
        total = max(len(chars), 1)
        ratio_ascii_letters = len(ascii_letters) / total

        if lang_hint:
            lh = lang_hint.lower()
            if lh.startswith("en"):
                if len(cjk_chars) >= 2 and ratio_ascii_letters < 0.7:
                    return False
                return True
            return False

        if len(cjk_chars) >= 2 and ratio_ascii_letters < 0.7:
            return False
        if ratio_ascii_letters > 0.75:
            return True
        return ratio_ascii_letters > 0.55

    def _make_native_comment(self, text: str, key_tokens: list[str]) -> str:
        cleaned = re.sub(r"https?://\S+", "", text)
        cleaned = re.sub(r"[@#]\S+", "", cleaned).strip()

        has_cjk = (
            any("\u4e00" <= c <= "\u9fff" for c in cleaned)
            or any("\u3040" <= c <= "\u30ff" for c in cleaned)
            or any("\uac00" <= c <= "\ud7af" for c in cleaned)
        )

        last_candidate = ""

        for _ in range(20):
            if has_cjk:
                segments = []
                for sep in ["。", "！", "？", "!", "?", "\n"]:
                    parts = [p.strip() for p in cleaned.split(sep) if p.strip()]
                    if parts:
                        segments.extend(parts)
                if not segments:
                    segments = [cleaned]

                snippet = self.random.choice(segments)
                if len(snippet) > 24:
                    start = 0 if len(snippet) <= 24 else self.random.randint(0, len(snippet) - 24)
                    snippet = snippet[start:start+24]

                candidate = EMOJI_PATTERN.sub("", snippet).strip()
            else:
                words = [w for w in cleaned.split() if not w.startswith("@")]
                if not words:
                    focus = pick_focus_token(key_tokens)
                    candidate = focus or "不错"
                else:
                    if len(words) < 5:
                        while len(words) < 5:
                            words.extend(words)
                        words = words[:5]

                    focus_token = pick_focus_token(key_tokens) if key_tokens else None
                    if focus_token and focus_token in words:
                        center_idx = words.index(focus_token)
                    else:
                        center_idx = len(words) // 2

                    window_size = min(max(5, len(words)), 12)
                    start = max(0, min(center_idx - window_size // 2, len(words) - window_size))
                    snippet_words = words[start:start+window_size]
                    candidate = " ".join(snippet_words)

                candidate = self._tidy_comment(candidate)

            if not candidate:
                continue
            last_candidate = candidate
            if self._violates_ai_blocklist(candidate):
                continue
            if not self._diversity_ok(candidate):
                continue
            if comment_seen(candidate):
                continue
            remember_comment(candidate)
            return candidate

        if not last_candidate:
            last_candidate = "不错"
        remember_comment(last_candidate)
        return last_candidate

    def generate_comments(self, text: str, author: str | None, handle: str | None = None, lang_hint: str | None = None):
        is_english = self._is_probably_english(text, lang_hint)
        topic = detect_topic(text)
        crypto = is_crypto_tweet(text)
        key_tokens = extract_keywords(text)

        if is_english:
            c1 = self._make_english_comment(text, author, topic, crypto, key_tokens, used_kinds=set())
            c2 = self._make_english_comment(text, author, topic, crypto, key_tokens, used_kinds={c1["kind"]})
            return [{"lang": "en", "text": c1["text"]}, {"lang": "en", "text": c2["text"]}]

        native = self._make_native_comment(text, key_tokens)
        en = self._make_english_comment(text, author, topic, crypto, key_tokens, used_kinds=set())
        return [{"lang": "native", "text": native}, {"lang": "en", "text": en["text"]}]

    def _tidy_comment(self, text: str) -> str:
        if not text:
            return ""
        text = EMOJI_PATTERN.sub("", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[.!?;:…]+$", "", text).strip()

        words = text.split()
        if not words:
            return ""

        if len(words) < 5:
            fillers = ["right", "honestly", "tbh", "still", "though"]
            while len(words) < 5 and fillers:
                words.append(fillers.pop(0))
        elif len(words) > 12:
            words = words[:12]

        final = " ".join(words).strip()
        final = re.sub(r"[.!?;:…]+$", "", final).strip()
        return final

    
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

    def def _make_english_comment(
        self, text: str, author: str | None, topic: str, is_crypto: bool,
        key_tokens: list[str], used_kinds: set,
    ) -> dict:
        focus = pick_focus_token(key_tokens) or "this"
        author_ref = None
        if author and random.random() < 0.6:
            parts = author.split()
            author_ref = parts[0] if parts else author

        buckets = self._get_template_buckets(topic, is_crypto)

        available_kinds = [k for k in buckets if k not in used_kinds]
        if not available_kinds:
            available_kinds = list(buckets.keys())
        kind = random.choice(available_kinds)

        last_candidate = ""

        length_mode = self.random.choice(["short","medium","long"])
        for attempt in range(40):
            if attempt and attempt % 8 == 0:
                length_mode = self.random.choice(["short","medium","long"])
                alt_focus = pick_focus_token(key_tokens)
                if alt_focus:
                    focus = alt_focus
                kind = random.choice(list(buckets.keys()))

            tmpl = random.choice(buckets[kind])
            if template_burned(tmpl):
                continue

            out = tmpl.format(author=author_ref or "", focus=focus)
            out = self._tidy_comment(out)
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

        fallback = self._tidy_comment(f"Pretty solid points on {focus}") or "Pretty solid points on this"
        remember_comment(fallback)
        return {"kind": kind, "text": fallback}

    def _get_template_buckets(self, topic: str, is_crypto: bool) -> dict:
        base_react = [
            "{focus} take actually feels pretty grounded",
            "Hard to disagree with this view on {focus}",
            "Have been nodding along reading about {focus}",
            "Kinda lines up with my experience of {focus}",
            "Nice to see someone phrase {focus} this clearly",
        ]
        base_convo = [
            "Curious where {focus} goes if this plays out",
            "Feels like a real conversation people have about {focus}",
            "Been having similar chats around {focus} lately",
            "Low key everyone is thinking this about {focus}",
            "Interested to hear more stories around {focus}",
        ]
        base_calm = [
            "Chill sober take on {focus} which I like",
            "Sensible breakdown of {focus} without extra drama",
            "Grounded way of walking through {focus} step by step",
            "Helps keep {focus} in perspective instead of hype",
            "Good reminder not to overreact to {focus} stuff",
        ]
        vibe_flavor = [
            "{focus} feels very timeline core right now",
            "The vibe around {focus} here is pretty real",
            "This hits the everyday side of {focus} nicely",
            "Quietly one of the better posts on {focus}",
        ]
        nuance_flavor = [
            "Appreciate that {focus} is handled without yelling",
            "Nice to see some nuance instead of pure takes on {focus}",
            "Not pushing an extreme angle on {focus} actually helps",
            "Good mix of context and restraint around {focus}",
        ]
        quick_react = [
            "Yeah this tracks for {focus} tbh",
            "Honestly this is how {focus} tends to go",
            "Kind of exactly what {focus} looks like in practice",
            "Hard not to recognise {focus} in this",
        ]
        author_flavor = [
            "{author} always finds a plain language angle on {focus}",
            "Feels like {author} actually lived through this {focus} mess",
            "{author} explaining {focus} hits different from the usual threads",
            "Trust {author} more on {focus} after posts like this",
        ]
        chart_flavor = [
            "This is how most traders quietly look at {focus}",
            "Those levels on {focus} line up with price memory",
            "Risk reward on {focus} is laid out really cleanly",
            "Helps frame entries and exits around {focus}",
        ]
        meme_flavor = [
            "This is exactly how {focus} feels some days",
            "Can not unsee this version of {focus} now",
            "Joke lands because {focus} is way too real",
            "Every timeline has at least one {focus} meme now",
        ]
        complaint_flavor = [
            "Very normal to be burnt out by {focus}",
            "Everyone pretending {focus} is fine is kinda wild",
            "Nice to see someone admit {focus} is exhausting",
            "Feels like no one in charge understands {focus}",
        ]
        announcement_flavor = [
            "Ship first talk later energy around {focus} is nice",
            "Cool seeing concrete stuff for {focus} instead of teasers",
            "Real update on {focus} beats vague roadmaps every time",
            "Interested to see if they keep shipping on {focus}",
        ]
        thread_flavor = [
            "Thread does a good job layering context on {focus}",
            "Bookmarking this as a reference for {focus} later",
            "Clean structure here makes {focus} easy to follow",
            "Skimming this gives a solid overview of {focus}",
        ]
        one_liner_flavor = [
            "Short line but pretty accurate read on {focus}",
            "Funny how one sentence sums up {focus} so well",
            "This is blunt but fair about {focus}",
            "Straightforward way of framing {focus} without fluff",
        ]
        crypto_extra = [
            "Onchain side of {focus} is finally getting discussed honestly",
            "Nice blend of risk and conviction for {focus} here",
            "People trading {focus} will recognise this feeling instantly",
            "Better than the usual moon talk around {focus}",
        ]

        buckets = {
            "react": base_react,
            "conversation": base_convo,
            "calm": base_calm,
            "vibe": vibe_flavor,
            "nuanced": nuance_flavor,
            "quick": quick_react,
        }

        if topic == "chart":
            buckets["chart"] = chart_flavor
        elif topic == "meme":
            buckets["meme"] = meme_flavor
        elif topic == "complaint":
            buckets["complaint"] = complaint_flavor
        elif topic in ("announcement", "update"):
            buckets["announcement"] = announcement_flavor
        elif topic == "thread":
            buckets["thread"] = thread_flavor
        elif topic == "one_liner":
            buckets["one_liner"] = one_liner_flavor

        if is_crypto:
            buckets["crypto"] = crypto_extra

        if random.random() < 0.5:
            buckets["author"] = author_flavor

        return buckets

# ---------------------------------------------------------
# Topic / keyword helpers
# ---------------------------------------------------------
def detect_topic(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("gm ", "gn ", "good morning", "good night")):
        return "greeting"
    if any(k in t for k in ("airdrop", "whitelist", "wl spot", "mint is live")):
        return "giveaway"
    if any(k in t for k in ("chart","support","resistance","ath","price target","%","market cap","mc")):
        return "chart"
    if any(k in t for k in ("bug","issue","broken","down again","wtf","why is","tired of")):
        return "complaint"
    if any(k in t for k in ("announcing","announcement","we're live","we are live","launching","we shipped")):
        return "announcement"
    if any(k in t for k in ("meme","shitpost","ratioed","memeing")) or "lol" in t:
        return "meme"
    if "🧵" in text or len(text) > 220:
        return "thread"
    if len(text) < 80:
        return "one_liner"
    return "generic"

def is_crypto_tweet(text: str) -> bool:
    t = text.lower()
    crypto_keywords = [
        "crypto","defi","nft","airdrop","token","coin","chain","l1","l2","staking","yield",
        "dex","cex","onchain","on-chain","gas fees","btc","eth","sol","arb","layer two","mainnet",
    ]
    if any(k in t for k in crypto_keywords):
        return True
    if re.search(r"\$\w{2,8}", text):
        return True
    return False

def extract_keywords(text: str) -> list[str]:
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"[@#]\S+", "", cleaned)
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", cleaned)
    if not tokens:
        return []
    tokens_lower = [tok.lower() for tok in tokens]
    filtered = [tok for tok, low in zip(tokens, tokens_lower) if low not in EN_STOPWORDS and len(low) > 2]
    if not filtered:
        filtered = tokens
    counts = Counter([t.lower() for t in filtered])
    sorted_tokens = sorted(filtered, key=lambda w: (-counts[w.lower()], -len(w)))
    seen, result = set(), []
    for w in sorted_tokens:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            result.append(w)
    return result[:10]

def pick_focus_token(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    upperish = [t for t in tokens if t.isupper() or t[0].isupper()]
    if upperish:
        return random.choice(upperish)
    return random.choice(tokens)

# ---------------------------------------------------------
# Keep-alive pinger (optional)
# ---------------------------------------------------------
def keep_alive():
    if not BACKEND_PUBLIC_URL:
        return
    while True:
        try:
            requests.get(f"{BACKEND_PUBLIC_URL}/", timeout=5)
        except Exception:
            pass
        time.sleep(KEEP_ALIVE_INTERVAL)

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
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

if __name__ == "__main__":
    init_db()
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
else:
    init_db()
