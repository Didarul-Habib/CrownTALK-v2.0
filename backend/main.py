from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from collections import Counter
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify

# Helpers from utils.py (already deployed)
from utils import CrownTALKError, fetch_tweet_data, clean_and_normalize_urls, style_fingerprint

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
PER_URL_SLEEP = float(os.environ.get("PER_URL_SLEEP_SECONDS", "1.0"))  # ← sleep after every URL
MAX_URLS_PER_REQUEST = int(os.environ.get("MAX_URLS_PER_REQUEST", "25"))  # ← hard cap per request

KEEPALIVE = os.environ.get("KEEPALIVE", "false").lower() == "true"

# ------------------------------------------------------------------------------
# DB helpers
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


def now_ts() -> int:
    return int(time.time())


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()


INIT_SQL = """
CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                lang TEXT,
                text TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_comments_url ON comments(url);
            CREATE INDEX IF NOT EXISTS idx_comments_created_at ON comments(created_at);

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


def _do_init() -> None:
    with get_conn() as c:
        c.executescript(INIT_SQL)


def init_db() -> None:
    def _safe():
        try:
            _locked_init(_do_init)
        except sqlite3.OperationalError as e:
            logger.warning("DB init error: %s", e)
    t = threading.Thread(target=_safe, daemon=True)
    t.start()
    t.join(timeout=3)


# ------------------------------------------------------------------------------
# Rules: word count + sanitization
# ------------------------------------------------------------------------------

WORD_RE = re.compile(r"\b[\w’']+(?:-[\w’']+)*\b", re.UNICODE)

def words(t: str) -> list[str]:
    return WORD_RE.findall(t or "")


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

    # Hard cap first
    if len(toks) > max_w:
        toks = toks[:max_w]

    # Softly pad short lines with casual fillers
    while len(toks) < min_w:
        for filler in ["honestly", "tbh", "still", "though", "right"]:
            if len(toks) >= min_w:
                break
            toks.append(filler)
        if len(toks) < min_w:
            break

    return " ".join(toks).strip()


# ------------------------------------------------------------------------------
# Topic / keywords (to keep comments context-aware, not templated)
# ------------------------------------------------------------------------------

EN_STOPWORDS = {
    "the","a","an","and","or","but","to","in","on","of","for","with","at","from","by","about","as",
    "into","like","through","after","over","between","out","against","during","without","before","under",
    "around","among","is","are","be","am","was","were","it","its","that","this","so","very","really","curious"
}

AI_BLOCKLIST = {
    "as an ai","as a language model","as an assistant",
    "i am an ai","i am just an ai","i am just a bot",
    "cannot provide","cannot answer","not financial advice",
    "lorem ipsum","placeholder text",
}

GENERIC_PHRASES = {
    # Old generic stuff
    "love that","love this","love the","love your",
    "this is huge","this is massive","this is insane","curious","curious to see","wow","wonder","love to see","love","love to watch",
    "nice thread","great thread","nice one","great one",
    "bullish on this","bearish on this","this goes hard",
    "gm fam","gm anon","gn fam","gn anon",
    "good reminder","needed this","facts only",

    # New repetitive templates we want to kill
    "love to see it","love to see this","love to see builders",
    "love seeing this","love seeing builders",
    "sounds like a game changer","sounds like a game-changer",
    "that’s a clever play","that's a clever play",
    "that’s a clever angle","that's a clever angle",
    "what a time to be alive",
    "what a move","what a call",
    "how do you see this","how do you see it",
    "excited to see where this goes",
    "this could be huge","this could be big",
}

def contains_generic_phrase(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in GENERIC_PHRASES)

STARTER_BLOCKLIST = {
    # Hard ban openers – we never want comments starting like this
    "love that","love this","love to see","love seeing",
    "nice thread","great thread",
    "gm ","gn ",
    "bullish on ","bearish on ",
    "yeah this","honestly this","kind of","nice to","hard to","wow","curious",
    "feels like","this is","short line","funny how",
    "appreciate that","interested to","curious where","nice to see",
    "chill sober","good reminder","yeah that",
    "good to see the boring",
}

STARTER_SOFT_PENALTY = {
    "that's ", "that’s ",
    "sounds like ", "looks like ", "seems like ",
    "what a ", "how do ", "how does ",
}

AI_JARGON_PENALTY = {
    "redefining what it means",
    "game changer",
    "game-changer",
    "transformative",
    "ecosystem",
    "catalyst for crypto growth",
    "catalyst for growth",
    "marks the beginning of the end",
    "responsible launchpad ecosystem",
}

BAD_END_WORDS = {
    "of", "the", "to", "for", "in", "on", "at", "by",
    "with", "than", "then",
    "this", "that", "these", "those",
    "and", "or", "but",
    "if", "when", "while", "where", "who", "which",
    "maybe", "probably", "just", "only",
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
    if any(k in t for k in ("chart", "support", "resistance", "ath", "price target", "%", "market cap")):
        return "markets"
    if any(k in t for k in ("nft", "pfp", "mint", "floor", "collection")):
        return "nft"
    return "generic"


def extract_keywords(text: str) -> list[str]:
    t = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", t)
    toks = [x for x in tokens if x not in EN_STOPWORDS and len(x) > 2]
    return toks[:12]


def pick_focus_token(tokens: list[str]) -> str:
    if not tokens:
        return ""
    counts = Counter(tokens)
    return sorted(counts.items(), key=lambda x: (-x[1], len(x[0])))[0][0]

def tweet_keywords_for_scoring(tweet_text: str | None) -> set[str]:
    """
    Extract a richer set of tokens from the tweet:
    - cashtags ($TOKEN)
    - @handles
    - Capitalized words (project names)
    - regular keyword tokens
    """
    if not tweet_text:
        return set()

    text = tweet_text or ""
    cashtags = re.findall(r"\$[A-Za-z0-9]{2,12}", text)
    handles = re.findall(r"@[A-Za-z0-9_]{2,15}", text)
    caps = re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", text)

    base_keywords = extract_keywords(text)
    all_tokens = []
    all_tokens.extend(cashtags)
    all_tokens.extend(handles)
    all_tokens.extend(caps)
    all_tokens.extend(base_keywords)

    return {t.lower() for t in all_tokens if t}

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# ------------------------------------------------------------------------------
# Memory + similarity helpers
# ------------------------------------------------------------------------------

def _normalize_for_memory(text: str) -> str:
    t = text or ""
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.!?;:…]+$", "", t)
    return t


def remember_ngrams(text: str) -> None:
    try:
        grams = _trigrams(text)
        ts = now_ts()
        with get_conn() as c:
            for g in grams:
                c.execute(
                    "INSERT OR IGNORE INTO comments_ngrams_seen(ngram, created_at) VALUES (?,?)",
                    (g, ts),
                )
    except Exception:
        pass


def remember_opener(opener: str) -> None:
    if not opener:
        return
    try:
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_openers_seen(opener, created_at) VALUES (?,?)",
                (opener, now_ts()),
            )
    except Exception:
        pass


def remember_comment(text: str, url: str = "", lang: Optional[str] = None) -> None:
    """
    Store a final approved comment into the DB + memory tables.

    This is called only after we have chosen comments to return to the user.
    """
    try:
        norm = _normalize_for_memory(text)
        if not norm:
            return
        with get_conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO comments_seen(hash, created_at) VALUES(?,?)",
                (sha256(norm), now_ts()),
            )
            c.execute(
                "INSERT INTO comments(url, lang, text) VALUES (?,?,?)",
                (url, lang, text),
            )
    except Exception:
        # don't break the request if logging fails
        return
    try:
        remember_ngrams(text)
        remember_opener(_openers(text))
    except Exception:
        # secondary memory failures are non-fatal
        return


def _openers(text: str) -> str:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return " ".join(w[:3])


def _trigrams(text: str) -> List[str]:
    w = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [" ".join(w[i:i+3]) for i in range(len(w) - 2)]


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


def opener_seen(opener: str) -> bool:
    try:
        with get_conn() as c:
            return c.execute("SELECT 1 FROM comments_openers_seen WHERE opener=? LIMIT 1", (opener,)).fetchone() is not None
    except Exception:
        return False


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


def _word_trigrams(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return {" ".join(toks[i:i+3]) for i in range(len(toks) - 2)}

# ------------------------------------------------------------------------------
# LLM system prompt (tuned for KOL-ish, non-templated comments)
# ------------------------------------------------------------------------------
def _llm_sys_prompt() -> str:
    return (
        "You write ultra-short reply comments for Twitter/X.\n"
        "\n"
        "ROLE\n"
        "- You are an experienced web3 / crypto KOL.\n"
        "- You talk like a real CT user, not a corporate account and not an AI.\n"
        "\n"
        "TASK\n"
        "- Write exactly TWO different reply comments to the tweet.\n"
        "- Each comment must be ONE sentence, 6–13 words long.\n"
        "- No numbering, no bullets, no labels, no explanations.\n"
        "- Either respond as a JSON array of two strings, or as two plain lines.\n"
        "\n"
        "STYLE\n"
        "- Use natural, modern CT language. Light slang is OK: words like "
        "'ngl', 'low-key', 'alpha', 'degen', 'anon', 'fr', "
        "IF they genuinely fit the tweet.\n"
        "- Speak in first person or direct address when it makes sense "
        "(\"ngl I'm watching this\", \"curious how you scale this anon\").\n"
        "- Do NOT start both comments with the same first word.\n"
        "- Each comment should be a single clear thought, not a paragraph.\n"
        "- Prefer reacting to one concrete detail: project name, token symbol "
        "($TOKEN), product, number, mechanism.\n"
        "- One comment can be more supportive/bullish, the other more curious "
        "or slightly skeptical.\n"
        "- End sentences naturally with a period or question mark.\n"
        "\n"
        "AVOID (HARD)\n"
        "- No emojis, no hashtags, no links, no 'follow me' or 'check this out'.\n"
        "- Do NOT say you are an AI or language model.\n"
        "- Avoid generic corporate phrases like: 'redefining what it means', "
        "'game changer', 'game-changer', 'ecosystem', 'catalyst for growth', "
        "'transformative', 'marks the beginning of the end'.\n"
        "- Avoid generic templates like: 'love to see it', 'this is huge', "
        "'sounds like a game-changer', 'nice thread', 'great thread'.\n"
        "- Do not write trading advice or disclaimers like 'not financial advice'.\n"
        "\n"
        "GOOD EXAMPLES (STYLE ONLY)\n"
        "- 'Low-key bullish on AlignerZ after this, real builder vibes ngl.'\n"
        "- 'Prediction markets plus TryLimitless suddenly make way more sense fr.'\n"
        "- 'Curious how Genome actually tracks attention without nuking UX anon.'\n"
        "\n"
        "BAD EXAMPLES (DO NOT COPY)\n"
        "- 'AlignerZ is redefining what it means to be a responsible launchpad ecosystem.'\n"
        "- 'Decentralized claim verification is a game-changer for institutional trust.'\n"
        "- 'Scalable infrastructure will be the real catalyst for crypto growth next year.'\n"
    )
 

from typing import Optional  # you already import this at top; just make sure it's there


def build_user_prompt(tweet_text: str, author: Optional[str]) -> str:
    """
    Common user prompt used by all providers so they follow the same rules.
    """
    return (
        f"Original tweet (author: {author or 'unknown'}):\n"
        f"{tweet_text}\n\n"
        "Write two different reply comments that follow the style rules above.\n"
        "Make them feel like spontaneous reactions from a human KOL who just read the tweet.\n"
        "Focus on one specific idea or reaction in each comment.\n"
        "Write two different reply comments that follow the style rules above.\n"
        "React like a crypto-native KOL who just read this in their timeline.\n"
        "Focus on what would actually matter to CT degens and builders.\n"
    )
# ------------------------------------------------------------------------------
# Offline generator
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

    def _topic_buckets_generic(self) -> list[str]:
        # mix of reflective, builder, skeptical, and curious tones
        return [
            # reflective / interpretive
            "The angle on {focus} here is actually pretty sharp.",
            "You frame {focus} in a way most people totally miss.",
            "{focus} is exactly where the next real edge probably sits.",
            "The nuance around {focus} here hits harder than people think.",
            # builder / strategy
            "From a builder lens, {focus} is the part that matters most.",
            "If you execute {focus} well, the rest almost takes care of itself.",
            "Quietly stacking {focus} like this is how you win long term.",
            "This is how serious people should be thinking about {focus}.",
            # skeptical / grounded
            "Big promises on {focus}, but execution will expose who’s real.",
            "Everyone talks {focus}, very few are actually shipping it.",
            "If {focus} flops, the whole narrative unwinds quick.",
            "Nice thread, but {focus} still has some open questions.",
            # curious / conversational
            "Lowkey curious how {focus} plays out once the hype fades.",
            "Would love to see more concrete examples around {focus}.",
            "Interesting take on {focus}, makes me rethink a couple assumptions.",
            "Hard not to keep watching {focus} after reading this.",
        ]

    def _topic_buckets_markets(self) -> list[str]:
        # Short market comments: positioning, structure, risk/reward
        return [
            "Risk reward on {focus} still looks asymmetric for patient people.",
            "If {focus} holds this level, the whole structure flips quickly.",
            "Most are staring at candles while {focus} quietly tells the story.",
            "Once {focus} reclaims this zone, positioning probably shifts fast.",
            "Market keeps mispricing {focus}; flow data points the other way.",
            "If you're modeling {focus} right, the risk is very clear.",
        ]

    def _topic_buckets_nft(self) -> list[str]:
        # NFT angle: long term, design, collectors
        return [
            "Beyond the art, {focus} gives this collection real staying power.",
            "Long term, {focus} decides whether this project actually survives.",
            "The way they handle {focus} feels much more deliberate here.",
            "If they execute on {focus}, floor price becomes a side effect.",
            "{focus} is what separates this from another hype cycle mint.",
            "Serious collectors are going to care a lot about {focus}.",
        ]

    def _topic_buckets_giveaway(self) -> list[str]:
        # Giveaways / WL: filter quality, retention, incentives
        return [
            "Structuring the drop around {focus} is actually a smarter filter.",
            "{focus}-based access tends to attract people who stick around.",
            "Tying rewards to {focus} makes this feel less like pure farming.",
            "Curious how {focus} will shape retention after the first wave.",
            "Giveaways that center {focus} usually convert much better long term.",
        ]

    def _native_buckets(self, script: str) -> list[str]:
        # Very lightweight; front-end cares more about vibe than perfect grammar
        if script == "latn":
            return self._topic_buckets_generic()
        # Could be extended: separate templates per script
        return self._topic_buckets_generic()

    def _diversity_ok(self, line: str) -> bool:
        if not line:
            return False
        low = line.lower().strip()
        start = " ".join(low.split()[:3])
        if any(start.startswith(b) for b in STARTER_BLOCKLIST):
            return False
        if contains_generic_phrase(low):
            return False
        if self._violates_ai_blocklist(low):
            return False
        if comment_seen(low):
            return False
        if trigram_overlap_bad(low, threshold=2) or too_similar_to_recent(low):
            return False
        return True

    def _enforce_length_cjk(self, s: str, max_chars: int = 40) -> str:
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

            if script in {"ja", "ko", "zh"}:
                return self._enforce_length_cjk(out) or out
            return enforce_word_count_natural(out, 6, 13)

        if last:
            if script in {"ja", "ko", "zh"}:
                return self._enforce_length_cjk(last) or last
            return enforce_word_count_natural(last, 6, 13)

        return None

    def _accept(self, line: str) -> bool:
        if not line:
            return False
        if self._violates_ai_blocklist(line):
            return False
        if comment_seen(line):
            return False
        if trigram_overlap_bad(line, threshold=2) or too_similar_to_recent(line):
            return False
        if contains_generic_phrase(line):
            return False
        return True

    def _commit(self, line: str, url: str = "", lang: str = "en") -> None:
        # Commit of final choices is handled at the API layer.
        # Offline generator no longer writes directly to memory.
        return

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
                out.append({"lang": ctx.get("lang") or "unknown", "text": native})

          # Always generate at least one EN comment
        key = extract_keywords(text)
        focus = pick_focus_token(key) or "this"
        buckets = self._topic_buckets_generic()
        topic = detect_topic(text)
        if topic == "markets":
            buckets += self._topic_buckets_markets()
        elif topic == "nft":
            buckets += self._topic_buckets_nft()
        elif topic == "giveaway":
            buckets += self._topic_buckets_giveaway()

        tried = set()
        last_good = None

        for _ in range(48):
            tmpl = random.choice(buckets)
            if tmpl in tried:
                continue
            tried.add(tmpl)
            line = normalize_ws(tmpl.format(focus=focus))
            line = enforce_word_count_natural(line, 6, 13)
            if not line:
                continue
            if not self._accept(line):
                continue
            last_good = line
            out.append({"lang": "en", "text": line})
            if len(out) >= 2:
                break

        if not out and last_good:
            out.append({"lang": "en", "text": last_good})

        # Local diversity guard between the two comments
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


def build_context_profile(text: str, url: str = "", tweet_author: Optional[str] = None, handle: Optional[str] = None) -> Dict[str, Any]:
    t = (text or "").strip()
    lang = "en"  # rough default; could be improved
    script = "latn"
    if re.search(r"[ぁ-ゟ゠-ヿ一-龯]", t):
        script = "ja"
    elif re.search(r"[가-힣]", t):
        script = "ko"
    elif re.search(r"[\u4e00-\u9fff]", t):
        script = "zh"

    return {
        "url": url,
        "tweet_author": tweet_author,
        "handle": handle,
        "lang": lang,
        "script": script,
    }


def _rescue_two(text: str) -> list[str]:
    toks = extract_keywords(text)
    focus = pick_focus_token(toks) or "this"
    base = [
        f"Honestly, {focus} here feels super underpriced.",
        f"Curious how {focus} plays out over the next few weeks.",
    ]
    return [enforce_word_count_natural(x, 6, 13) for x in base]


def _pair_too_similar(a: str, b: str) -> bool:
    ta = set(_word_trigrams(a))
    tb = set(_word_trigrams(b))
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    uni = len(ta | tb)
    return bool(uni and (inter / uni) >= 0.7)


def _extract_handle_from_url(url: str) -> Optional[str]:
    try:
        m = re.search(r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/([^/]+)/status/", url, re.I)
        return m.group(1) if m else None
    except Exception:
        return None


generator = OfflineCommentGenerator()

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
        " thanks fam"," thanks man"," thanks anon"," thanks bro",
    ]
    for tok in cut_tokens:
        idx = low.find(tok)
        if idx != -1:
            return text[:idx].rstrip()
    return text


def _ensure_question_punctuation(text: str) -> str:
    low = text.strip().lower()
    if low.endswith("?"):
        return text
    if low.startswith(("why ","what ","how ","where ","when ","do you","did you","are we","is this","can we","could we")):
        return text.rstrip(".! ") + "?"
    return text


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
        return [quoted[0].strip(), quoted[1].strip()]
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
# Groq generator
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
# Optional Hugging Face Inference API
# ------------------------------------------------------------------------------
HF_API_KEY = os.getenv("HF_API_KEY")
USE_HF = bool(HF_API_KEY)
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# ------------------------------------------------------------------------------
# Optional Cohere
# ------------------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
USE_COHERE = bool(COHERE_API_KEY)
_cohere_client = None
if USE_COHERE:
    try:
        import cohere
        _cohere_client = cohere.Client(COHERE_API_KEY)
    except Exception:
        _cohere_client = None
        USE_COHERE = False

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
        time.sleep(60)


def groq_two_comments(tweet_text: str, author: str | None) -> list[str]:
    if not (USE_GROQ and _groq_client):
        raise RuntimeError("Groq disabled or client not available")

    sys_prompt = _llm_sys_prompt()
    user_prompt = build_user_prompt(tweet_text, author)

    resp = None
    for attempt in range(1, 4):
        try:
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
            raise

    if resp is None:
        raise RuntimeError("Groq call failed after retries")

    raw = (resp.choices[0].message.content or "").strip()
    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)

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
        raise RuntimeError("Could not produce two valid comments")

    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]
    return candidates[:2]
    

def openai_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_OPENAI and _openai_client):
        raise RuntimeError("OpenAI disabled or client not available")

    user_prompt = build_user_prompt(tweet_text, author)

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
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    if len(candidates) < 2:
        raise RuntimeError("OpenAI did not produce two valid comments")
    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]
    return candidates[:2]
    

def gemini_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    if not (USE_GEMINI and _gemini_model):
        raise RuntimeError("Gemini disabled or client not available")

    user_prompt = build_user_prompt(tweet_text, author)
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
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    if len(candidates) < 2:
        raise RuntimeError("Gemini did not produce two valid comments")
    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]
    return candidates[:2]
    
           
def hf_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    """
    Use Hugging Face Inference API to generate two comments.
    """
    if not (USE_HF and HF_API_KEY):
        raise RuntimeError("Hugging Face disabled or API key not configured")

    user_prompt = build_user_prompt(tweet_text, author)
    prompt = _llm_sys_prompt() + "\n\n" + user_prompt

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 160,
            "temperature": 0.9,
        },
    }

    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face error {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    raw = ""
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            raw = first.get("generated_text") or ""
        else:
            raw = str(first)
    elif isinstance(data, dict) and "generated_text" in data:
        raw = data.get("generated_text") or ""
    else:
        raw = str(data)

    raw = (raw or "").strip()

    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    if len(candidates) < 2:
        raise RuntimeError("Hugging Face did not produce two valid comments")
    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]
    return candidates[:2]
    
def cohere_two_comments(tweet_text: str, author: Optional[str]) -> list[str]:
    """
    Use Cohere generate() to produce two comments.
    """
    if not (USE_COHERE and _cohere_client):
        raise RuntimeError("Cohere disabled or client not available")

    full_prompt = _llm_sys_prompt() + "\n\n" + build_user_prompt(tweet_text, author)

    try:
        resp = _cohere_client.generate(
            model=os.getenv("COHERE_MODEL", "command-r-plus"),
            prompt=full_prompt,
            max_tokens=160,
            temperature=0.9,
        )
        if not getattr(resp, "generations", None):
            raise RuntimeError("Cohere returned no generations")
        raw = (resp.generations[0].text or "").strip()
    except Exception as e:
        raise RuntimeError(f"Cohere error: {e}") from e

    candidates = parse_two_comments_flex(raw)
    candidates = [enforce_word_count_natural(c) for c in candidates]
    candidates = [c for c in candidates if 6 <= len(words(c)) <= 13]
    candidates = enforce_unique(candidates, tweet_text=tweet_text)
    if len(candidates) < 2:
        raise RuntimeError("Cohere did not produce two valid comments")
    if len(candidates) >= 2 and _pair_too_similar(candidates[0], candidates[1]):
        extra = offline_two_comments(tweet_text, author)
        merged = enforce_unique(candidates + extra, tweet_text=tweet_text)
        if len(merged) >= 2:
            candidates = merged[:2]
    return candidates[:2]

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

    result = enforce_unique(result, tweet_text=text)
    return result[:2]


def pick_two_diverse_text(candidates: list[str]) -> list[str]:
    """
    Try to pick two comments with different vibes / structure:
    ideally one more declarative and one more question-like.
    """
    if len(candidates) <= 2:
        return candidates[:2]

    scored = []
    for c in candidates:
        low = c.lower().strip()
        is_question = low.endswith("?") or low.startswith((
            "why ","what ","how ","where ","when ",
            "do you","did you","are we","is this",
            "can we","could we"
        ))
        scored.append((c, is_question))

    questions = [c for c, q in scored if q]
    statements = [c for c, q in scored if not q]

    if questions and statements:
        return [statements[0], questions[0]]

    # fallback: just first two
    return candidates[:2]


def enforce_unique(candidates: list[str], tweet_text: Optional[str] = None) -> list[str]:
    """
    - sanitize + enforce 6–13 words
    - drop generic phrases & hard-banned starters
    - skip past repeats / templates / trigram overlaps
    - score comments against tweet keywords
    - pick best scored ones, then pick two diverse ones
    """
    kw_set = tweet_keywords_for_scoring(tweet_text) if tweet_text else set()
    cleaned: list[tuple[str, float]] = []
    seen_openers_local: set[str] = set()

    for c in candidates:
        c = enforce_word_count_natural(c)
        if not c:
            continue

        low = c.lower().strip()

        if contains_generic_phrase(low):
            continue

        op = _openers(low)
        if op and op in seen_openers_local:
            continue

        if comment_seen(low):
            continue
        if trigram_overlap_bad(low, threshold=2) or too_similar_to_recent(low):
            continue

        tmpl_fp = re.sub(r"\b\w+\b", "w", low)[:80]
        if template_burned(tmpl_fp):
            continue

        score = score_comment_for_post(c, kw_set)
        if score <= -1e8:
            continue

        if op:
            seen_openers_local.add(op)
        cleaned.append((c, score))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: x[1], reverse=True)
    texts_ordered = [c for (c, _) in cleaned]

    if len(texts_ordered) >= 2:
        texts_ordered = pick_two_diverse_text(texts_ordered)

    return texts_ordered[:2]
        


def score_comment_for_post(comment: str, kw_set: set[str]) -> float:
    """
    Score a comment relative to a tweet:

    - Hard reject: wrong length, generic phrases, bad starters, weird cut-offs.
    - Penalties: AI / corporate jargon, over-generic, boring structure.
    - Bonuses: uses tweet keywords, first-person voice, question when appropriate.
    """
    if not comment:
        return -1e9

    comment = comment.strip()
    low = comment.lower()
    wcount = len(words(comment))

    # Hard length constraint
    if wcount < 6 or wcount > 13:
        return -1e9

    # Hard generic / starter filters
    if contains_generic_phrase(low):
        return -1e9

    for s in STARTER_BLOCKLIST:
        if low.startswith(s):
            return -1e9

    # Obvious cut-off endings like "up for grabs might not be the only"
    last_word = re.findall(r"\w+", low)
    last_word = last_word[-1] if last_word else ""
    if last_word in BAD_END_WORDS:
        return -1e9

    score = 1.0

    # Soft penalty for boring openers like "That's", "Sounds like"
    for s in STARTER_SOFT_PENALTY:
        if low.startswith(s):
            score -= 0.4
            break

    # Penalize AI / corporate jargon if present
    for p in AI_JARGON_PENALTY:
        if p in low:
            score -= 0.5

    # Reward using tweet keywords (project names, tickers, etc.)
    if kw_set:
        c_kw = set(extract_keywords(low))
        overlap = len(c_kw & kw_set)
        if overlap == 0:
            score -= 0.6  # totally generic
        else:
            score += 0.2 * min(overlap, 3)

    # Bonus for first-person or direct involvement
    if re.search(r"\b(i|im|i'm|me|my)\b", low):
        score += 0.25

    # Tiny bonus if it's a clean question
    if low.endswith("?"):
        score += 0.15

    # Penalize ellipses and multiple clauses (too rambly)
    if "..." in low:
        score -= 0.3
    if comment.count(".") > 1:
        score -= 0.3

    return score


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
    if USE_HF and HF_API_KEY:
        providers.append(("huggingface", hf_two_comments))
    if USE_COHERE and _cohere_client:
        providers.append(("cohere", cohere_two_comments))
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
            offline = offline_two_comments(tweet_text, author)
            candidates = enforce_unique(candidates + offline, tweet_text=tweet_text)
        except Exception as e:
            logger.warning("offline generator failed: %s", e)

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
                txt = (item.get("text") or "").strip()
                if txt:
                    out.append({"lang": item.get("lang") or lang or "en", "text": txt})
        except Exception as e:
            logger.exception("Total failure in provider cascade: %s", e)

    # Final hard cap: exactly 2
    return out[:2]


# ------------------------------------------------------------------------------
# API routes (batching + pacing)
# ------------------------------------------------------------------------------

def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


def _normalize_comment_items(
    items: list[Any],
    tweet_lang: str | None,
    author: str | None,
    handle: str | None,
    tweet_text: str,
    url: str,
) -> list[dict]:
    """
    Take whatever generate_two_comments_with_providers() gave us and make sure we end up with:

        [ { "lang": "...", "text": "..." }, { ... } ]

    with 2 items max, 6–13 words, and no empties.
    """

    lang = tweet_lang or "en"
    normalized: list[dict] = []

    # 1) If provider already returned dicts: normalize them
    for it in items or []:
        if isinstance(it, dict):
            raw = str(it.get("text", "")).strip()
            if not raw:
                continue
            txt = enforce_word_count_natural(raw, 6, 13)
            if not txt:
                continue
            normalized.append({
                "lang": (it.get("lang") or lang),
                "text": txt,
            })
        elif isinstance(it, str):
            raw = it.strip()
            if not raw:
                continue
            txt = enforce_word_count_natural(raw, 6, 13)
            if not txt:
                continue
            normalized.append({
                "lang": lang,
                "text": txt,
            })

    # 2) If we still don’t have 2, fall back to offline generator
    if len(normalized) < 2:
        try:
            offline = offline_two_comments(tweet_text, author)
            for s in offline:
                if len(normalized) >= 2:
                    break
                if not s:
                    continue
                txt = enforce_word_count_natural(s, 6, 13)
                if not txt:
                    continue
                normalized.append({
                    "lang": lang,
                    "text": txt,
                })
        except Exception as e:
            logger.warning("offline generator failed while normalizing: %s", e)

    # 3) Final uniqueness + pair diversity
    #    (we only care about 'text' for uniqueness)
    texts = [n["text"] for n in normalized]
    texts = enforce_unique(texts, tweet_text=tweet_text)
    texts = texts[:2]

    final: list[dict] = []
    for t in texts:
        t_txt = enforce_word_count_natural(t, 6, 13)
        if not t_txt:
            continue
        final.append({
            "lang": lang,
            "text": t_txt,
        })

    # Last safety net: still < 2 → make a simple manual pair
    if len(final) < 2:
        rescue = _rescue_two(tweet_text)
        for s in rescue:
            if len(final) >= 2:
                break
            if not s:
                continue
            t_txt = enforce_word_count_natural(s, 6, 13)
            if not t_txt:
                continue
            final.append({"lang": lang, "text": t_txt})

    return final[:2]

def _canonical_x_url_from_tweet(original_url: str, t: Any) -> str:
    """
    Prefer the canonical URL from VX/FX if present.
    Fallback: build https://x.com/{handle}/status/{id} when we know handle+id.
    Otherwise, return the original normalized URL.
    """
    canonical = getattr(t, "canonical_url", None)
    if canonical and isinstance(canonical, str):
        # make sure it’s x.com not vx/fx
        if "x.com" in canonical or "twitter.com" in canonical:
            return canonical

    handle = getattr(t, "handle", None)
    tweet_id = getattr(t, "tweet_id", None)
    if handle and tweet_id:
        return f"https://x.com/{handle}/status/{tweet_id}"

    return original_url

@app.route("/comment", methods=["POST", "OPTIONS"])
def comment_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    # Robust JSON parsing
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

    # Hard cap to protect backend
    if len(urls) > MAX_URLS_PER_REQUEST:
        urls = urls[:MAX_URLS_PER_REQUEST]

    # Clean/normalize any messy input (extra text, no scheme, etc.)
    try:
        norm_urls = clean_and_normalize_urls(urls)
    except Exception as e:
        logger.exception("Failed to normalize urls: %s", e)
        return jsonify({
            "error": "Could not normalize URLs",
            "code": "normalize_failed",
        }), 400

    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for url in norm_urls:
        try:
            t = fetch_tweet_data(url)
            handle = t.handle or _extract_handle_from_url(url)
            if not t.text:
                raise CrownTALKError("Empty tweet text", code="empty_tweet")

            raw_items = generate_two_comments_with_providers(
                t.text,
                t.author_name or None,
                handle,
                t.lang or None,
                url=url,
            )

            comment_items = _normalize_comment_items(
                raw_items,
                tweet_lang=t.lang,
                author=t.author_name,
                handle=handle,
                tweet_text=t.text,
                url=url,
            )

            # Persist final comments into memory DB
            lang = t.lang or "en"
            for item in comment_items:
                txt = item.get("text") or ""
                if not txt:
                    continue
                try:
                    remember_comment(txt, url=url, lang=lang)
                    remember_template(txt)
                except Exception:
                    # non-fatal
                    pass

            display_url = _canonical_x_url_from_tweet(url, t)
            results.append({
                "url": display_url,
                "comments": comment_items,
            })

        except CrownTALKError as e:
            failed.append({
                "url": url,
                "reason": str(e),
                "code": getattr(e, "code", "upstream_error"),
            })
        except Exception as e:
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
    except Exception:
        return jsonify({
            "error": "Invalid JSON body",
            "comments": [],
            "code": "invalid_json",
        }), 400

    url_in = (data.get("url") or "").strip()
    if not url_in:
        return jsonify({
            "error": "Missing 'url' field",
            "comments": [],
            "code": "bad_request",
        }), 400

    try:
        # Normalize URL the same way as /comment
        norm_list = clean_and_normalize_urls([url_in])
        if not norm_list:
            raise CrownTALKError("Bad tweet URL", code="bad_tweet_url")
        url = norm_list[0]

        t = fetch_tweet_data(url)
        handle = t.handle or _extract_handle_from_url(url)
        if not t.text:
            raise CrownTALKError("Empty tweet text", code="empty_tweet")

        raw_items = generate_two_comments_with_providers(
            t.text,
            t.author_name or None,
            handle,
            t.lang or None,
            url=url,
        )

        comment_items = _normalize_comment_items(
            raw_items,
            tweet_lang=t.lang,
            author=t.author_name,
            handle=handle,
            tweet_text=t.text,
            url=url,
        )

        lang = t.lang or "en"
        for item in comment_items:
            txt = item.get("text") or ""
            if not txt:
                continue
            try:
                remember_comment(txt, url=url, lang=lang)
                remember_template(txt)
            except Exception:
                pass

        display_url = _canonical_x_url_from_tweet(url, t)
        return jsonify({
            "url": display_url,
            "comments": comment_items,
            "code": "ok",
        }), 200

    except CrownTALKError as e:
        return jsonify({
            "url": url_in,
            "error": str(e),
            "comments": [],
            "code": getattr(e, "code", "upstream_error"),
        }), 502
    except Exception:
        logger.exception("Unhandled error during reroll for %s", url_in)
        return jsonify({
            "url": url_in,
            "error": "internal_error",
            "comments": [],
            "code": "internal_error",
        }), 500


def main() -> None:
    init_db()
    if KEEPALIVE:
        threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
