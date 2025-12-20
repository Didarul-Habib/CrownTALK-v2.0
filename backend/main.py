from __future__ import annotations

import json, os, re, time, random, hashlib, logging, sqlite3, threading
from collections import Counter
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify

# Helpers from utils.py (already deployed)
from utils import CrownTALKError, fetch_tweet_data, clean_and_normalize_urls

# ------------------------------------------------------------------------------
# App / Logging / Config
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

app = Flask(__name__)

PORT = int(os.environ.get("PORT", "10000"))
DB_PATH = os.environ.get("DB_PATH", "crowntalk.db")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))                 # ← process N at a time
PER_URL_SLEEP = float(os.environ.get("PER_URL_SLEEP_SECONDS", "0.1"))  # ← sleep after every URL
MAX_URLS_PER_REQUEST = int(os.environ.get("MAX_URLS_PER_REQUEST", "25"))  # ← hard cap per request

KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))

# ------------------------------------------------------------------------------
# Optional Groq (free-tier). If not set, we run fully offline.
# ------------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)
if USE_GROQ:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.warning("Groq import failed, running offline only: %s", e)
        USE_GROQ = False
        _groq_client = None
else:
    _groq_client = None

# ------------------------------------------------------------------------------
# DB: memory for templates / comments / openers / ngrams
# ------------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT UNIQUE,
            url TEXT,
            lang TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS openers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opener TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trigrams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigram TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Memory helpers
# ------------------------------------------------------------------------------

def _db_execute(query: str, params: tuple = ()) -> None:
    conn = _get_db()
    try:
        conn.execute(query, params)
        conn.commit()
    finally:
        conn.close()


def remember_template(tmpl: str) -> None:
    if not tmpl:
        return
    _db_execute("INSERT OR IGNORE INTO templates(template) VALUES (?)", (tmpl,))


def remember_comment(text: str, url: str = "", lang: str = "en") -> None:
    if not text:
        return
    _db_execute(
        "INSERT OR IGNORE INTO comments(text, url, lang) VALUES (?, ?, ?)",
        (text, url or "", lang or "en"),
    )


def remember_opener(opener: str) -> None:
    if not opener:
        return
    _db_execute(
        "INSERT OR IGNORE INTO openers(opener) VALUES (?)",
        (opener,),
    )


def remember_ngrams(text: str) -> None:
    if not text:
        return
    words = re.findall(r"\w+", text.lower())
    for a, b, c in zip(words, words[1:], words[2:]):
        tri = f"{a} {b} {c}"
        _db_execute(
            "INSERT OR IGNORE INTO trigrams(trigram) VALUES (?)",
            (tri,),
        )


def _fetch_recent_comments(limit: int = 200) -> List[str]:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM comments ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [r["text"] for r in rows]


def comment_seen(text: str) -> bool:
    if not text:
        return False
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM comments WHERE text = ? LIMIT 1", (text,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def opener_seen(opener: str) -> bool:
    if not opener:
        return False
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM openers WHERE opener = ? LIMIT 1", (opener,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def trigram_overlap_bad(text: str, threshold: int = 3) -> bool:
    # basic trigram overlap vs stored trigrams
    if not text:
        return False
    words = re.findall(r"\w+", text.lower())
    trigrams = [f"{a} {b} {c}" for a, b, c in zip(words, words[1:], words[2:])]
    if not trigrams:
        return False

    conn = _get_db()
    cur = conn.cursor()
    hits = 0
    for tri in trigrams:
        cur.execute("SELECT 1 FROM trigrams WHERE trigram = ? LIMIT 1", (tri,))
        if cur.fetchone():
            hits += 1
            if hits >= threshold:
                conn.close()
                return True
    conn.close()
    return False


def too_similar_to_recent(text: str, limit: int = 50) -> bool:
    # approximate similarity using normalised text + shared trigrams
    if not text:
        return False

    text = normalize_ws(text)
    words = re.findall(r"\w+", text.lower())
    trigrams = {f"{a} {b} {c}" for a, b, c in zip(words, words[1:], words[2:])}

    recent = _fetch_recent_comments(limit=limit)
    for c in recent:
        cw = re.findall(r"\w+", normalize_ws(c).lower())
        ctri = {f"{a} {b} {c}" for a, b, c in zip(cw, cw[1:], cw[2:])}
        if not ctri:
            continue
        overlap = len(trigrams & ctri)
        if overlap >= 4:
            return True
    return False


def template_burned(tmpl: str) -> bool:
    # template is a "shape" of a comment (words replaced by "w")
    if not tmpl:
        return False
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM templates WHERE template = ? LIMIT 1", (tmpl,))
    row = cur.fetchone()
    conn.close()
    return row is not None


# ------------------------------------------------------------------------------
# Text utils
# ------------------------------------------------------------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def enforce_word_count_natural(
    text: str,
    min_words: int = 6,
    max_words: int = 13,
) -> str:
    """
    Cut or lightly extend to keep comment between [min_words, max_words].
    Avoids brutal truncation in the middle of a phrase.
    """
    if not text:
        return ""

    text = normalize_ws(text)
    words = text.split(" ")
    if len(words) < min_words:
        return text

    if len(words) > max_words:
        words = words[:max_words]
        # avoid chopping mid token like "%", ".", ","
        while words and words[-1] in {",", ".", "and", "or", "but"}:
            words.pop()
        text = " ".join(words)

    return text


def soft_truncate(text: str, max_chars: int = 220) -> str:
    if not text:
        return ""
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    # backtrack to last space if possible
    last_space = cut.rfind(" ")
    if last_space > 20:
        cut = cut[:last_space]
    return cut


def _openers(text: str) -> str:
    if not text:
        return ""
    first = text.split(" ", 1)[0]
    return first.lower()


def strip_trailing_period_or_excl(text: str) -> str:
    """
    Remove trailing '.' or '!' but keep '?' when needed.
    Also strip duplicated punctuation like '??' → '?'.
    """
    if not text:
        return ""
    text = text.rstrip()

    # collapse repeated ? or ! or . at the end
    text = re.sub(r"[\?\!\.]{2,}$", lambda m: m.group(0)[0], text)

    if text.endswith(".") or text.endswith("!"):
        text = text[:-1]
    return text


def normalize_punctuation(text: str) -> str:
    """
    - No trailing '.' or '!'
    - Single trailing '?' when it is a question
    - Normalize weird spacing before punctuation
    """
    if not text:
        return ""
    text = re.sub(r"\s+([?!.,])", r"\1", text)
    text = strip_trailing_period_or_excl(text)
    return text


def extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    # keep $, % and decimals as tokens
    tokens = re.findall(r"\$[A-Za-z0-9_]+|\d+\.\d+%?|\d+%|\d+|\w+", text)
    # dedupe but keep order
    seen = set()
    out = []
    for t in tokens:
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
    return out


def pick_focus_token(tokens: List[str]) -> Optional[str]:
    """
    Prefer:
    - $TICKER
    - tokens with % or numbers
    - otherwise any mid-length token
    """
    if not tokens:
        return None
    # 1) $TICKER
    tickers = [t for t in tokens if t.startswith("$") and len(t) <= 8]
    if tickers:
        return tickers[0]
    # 2) numeric / % tokens
    numeric = [t for t in tokens if any(ch.isdigit() for ch in t)]
    if numeric:
        return numeric[0]
    # 3) fallback mid-length
    mids = [t for t in tokens if 3 <= len(t) <= 12]
    if mids:
        return mids[0]
    return tokens[0]


def detect_language_hint(text: str) -> str:
    if not text:
        return "en"
    t = text.lower()
    # ultra cheap heuristics
    if re.search(r"[অ-হ]", t):
        return "bn"
    if re.search(r"[क-ह]", t):
        return "hi"
    if re.search(r"[أ-ي]", t):
        return "ar"
    if re.search(r"[ぁ-んァ-ン一-龯]", t):
        return "ja"
    return "en"


def detect_script(text: str) -> str:
    if not text:
        return "latn"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "hani"
    if re.search(r"[ぁ-んァ-ン]", text):
        return "jpan"
    if re.search(r"[가-힣]", text):
        return "hang"
    if re.search(r"[أ-ي]", text):
        return "arab"
    if re.search(r"[অ-হ]", text):
        return "beng"
    if re.search(r"[क-ह]", text):
        return "deva"
    return "latn"


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


def topic_focus_words(text: str) -> List[str]:
    tokens = extract_keywords(text)
    if not tokens:
        return []
    topic = detect_topic(text)
    if topic in ("chart", "markets"):
        # prefer tickers or numbers
        cand = [t for t in tokens if t.startswith("$") or any(ch.isdigit() for ch in t)]
        return cand or tokens
    return tokens


# ------------------------------------------------------------------------------
# Anti-AI phrase blocklist and diversity controls
# ------------------------------------------------------------------------------

AI_BLOCKLIST = [
    "low-key bullish",
    "low key bullish",
    "real builder vibes ngl",
    "very few are actually shipping",
    "exactly where the next real edge probably sits",
    "the angle on",
    "everyone talks",
    "hard not to keep watching",
    "after this",
    "ngl",
]


def looks_too_ai_like(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if any(p in low for p in AI_BLOCKLIST):
        return True
    # templatey openers
    if re.match(r"^(low[- ]key|ngl|honestly|tbh)\b", low):
        return True
    # repeated "Low-key bullish X after this"
    if re.search(r"low[- ]key .* after this", low):
        return True
    return False


def diverse_enough(a: str, b: str) -> bool:
    if not a or not b:
        return True
    aw = re.findall(r"\w+", a.lower())
    bw = re.findall(r"\w+", b.lower())
    if not aw or not bw:
        return True
    ca = Counter(aw)
    cb = Counter(bw)
    shared = sum(min(ca[w], cb[w]) for w in ca.keys() & cb.keys())
    return shared <= max(3, min(len(aw), len(bw)) // 3)


# ------------------------------------------------------------------------------
# Offline generator (fallback + for non-Latin scripts)
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

    def _native_buckets(self, script: str) -> List[str]:
        # {focus} will be replaced by a keyword from tweet
        f = "{focus}"

        if script in ("hani", "jpan", "hang"):
            return [
                f"{f} 这点讲得挺到位的",
                f"{f} 要是兜得住，后面空间还挺大",
                f"看{f} 的节奏就知道谁是认真在做事",
            ]
        if script == "beng":
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
                f"{f} هنا يفرّق بين الضجيج والعمل الحقيقي",
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
            if last and not diverse_enough(last, out):
                continue
            last = out
            if script in ("hani", "jpan", "hang"):
                out = self._enforce_length_cjk(out)
            if len(out) < 4:
                continue
            return out
        return None

    def _english_style_buckets(self, focus_slot: str) -> Dict[str, List[str]]:
        P = lambda s: normalize_ws(s)

        markets = [
            P(f"Risk reward on {focus_slot} still looks asymmetric for patient people"),
            P(f"If {focus_slot} holds this level, the whole structure flips quickly"),
            P(f"Most stare at candles while {focus_slot} quietly tells the story"),
            P(f"Once {focus_slot} reclaims this zone, positioning probably shifts fast"),
            P(f"Market keeps mispricing {focus_slot}, flow data says otherwise"),
            P(f"If you're modeling {focus_slot} right, the risk is very clear"),
        ]

        nft = [
            P(f"Beyond the art, {focus_slot} gives this collection real staying power"),
            P(f"Long term, {focus_slot} decides whether this project actually survives"),
            P(f"The way they handle {focus_slot} feels much more deliberate here"),
            P(f"If they execute on {focus_slot}, floor price becomes a side effect"),
            P(f"{focus_slot} is what separates this from another hype cycle mint"),
            P(f"Serious collectors are going to care a lot about {focus_slot}"),
        ]

        giveaway = [
            P(f"Structuring the drop around {focus_slot} is actually a smarter filter"),
            P(f"{focus_slot}-based access tends to attract people who stick around"),
            P(f"Tying rewards to {focus_slot} makes this feel less like pure farming"),
            P(f"Curious how {focus_slot} will shape retention after the first wave"),
            P(f"Giveaways that center {focus_slot} usually convert better long term"),
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

        return {
            "markets": markets,
            "nft": nft,
            "giveaway": giveaway,
            "convo": convo,
            "calm": calm,
        }

    def _english_candidate(self, text: str, ctx: Dict[str, Any]) -> Optional[str]:
        topic = detect_topic(text)
        focus_list = topic_focus_words(text)
        focus = pick_focus_token(focus_list) or "this"
        styles = self._english_style_buckets(focus_slot=focus)
        pool: List[str] = []

        # mix some buckets depending on topic
        if topic in ("chart", "markets"):
            pool.extend(styles["markets"])
            pool.extend(styles["calm"])
        elif topic == "giveaway":
            pool.extend(styles["giveaway"])
            pool.extend(styles["convo"])
        elif topic in ("thread", "announcement"):
            pool.extend(styles["calm"])
            pool.extend(styles["convo"])
        else:
            pool.extend(styles["convo"])
            pool.extend(styles["calm"])

        if not pool:
            pool = styles["convo"]

        for _ in range(32):
            cand = normalize_ws(random.choice(pool))
            if self._violates_ai_blocklist(cand):
                continue
            return cand
        return None

    def _quality_gate(self, line: str, text: str) -> bool:
        if not line:
            return False
        line = normalize_ws(line)
        if len(line.split()) < 4:
            return False
        if looks_too_ai_like(line):
            return False
        if trigram_overlap_bad(line):
            return False
        if too_similar_to_recent(line):
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

        # first try native language if not Latin
        if non_en:
            for _ in range(12):
                native = self._make_native_comment(text, ctx)
                if not native:
                    continue
                if not self._quality_gate(native, text):
                    continue
                if ctx["script"] in ("hani", "jpan", "hang"):
                    native = self._enforce_length_cjk(native)
                else:
                    native = enforce_word_count_natural(native, 6, 13)
                self._commit(native, url=url, lang=ctx["script"])
                out.append({"lang": ctx["script"], "text": native})
                break

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
            if not self._quality_gate(cand, text):
                continue
            # avoid duplicates vs existing
            if any(cand.strip().lower() == c["text"].strip().lower() for c in out):
                continue
            self._commit(cand, url=url, lang="en")
            out.append({"lang": "en", "text": cand})

        return out[:2]


# Global offline generator
generator = OfflineCommentGenerator()


# ------------------------------------------------------------------------------
# Groq provider (chat model)
# ------------------------------------------------------------------------------

def _groq_prompt(tweet_text: str, author: Optional[str]) -> List[Dict[str, str]]:
    """
    System prompt tuned for:
    - 2 comments
    - 6–13 words
    - no list/structure, just raw lines
    - casual KOL tone for web3/crypto
    """
    author_display = author or "the author"
    system = (
        "You write ultra-human, web3-native quote tweets and replies.\n"
        "- Tone: experienced KOL/influencer, casual but sharp, not cringe.\n"
        "- Use modern crypto slang naturally (degen, bag, meta, zk, L2), "
        "but not in every line.\n"
        "- No generic AI phrases like 'low-key bullish after this', "
        "'real builder vibes ngl', 'everyone talks X, few ship it'.\n"
        "- Each output must be a single comment, 6-13 words.\n"
        "- No bullets, no numbering, no quotes, no emojis unless clearly natural.\n"
        "- Do NOT mention you're an AI.\n"
        "- Keep punctuation light. Avoid ending with '.' if not needed.\n"
        "- Respect numbers and tickers exactly as in the tweet (20% stays 20%).\n"
        "- Respect $TICKER format (never change $HLS to HLS or $$HLS).\n"
    )

    user = (
        f"Tweet by {author_display}:\n"
        f"\"{tweet_text}\"\n\n"
        "Write TWO different human comments that feel like instant, honest reactions.\n"
        "Each comment:\n"
        "- 6 to 13 words\n"
        "- No list format, just plain sentences separated by newline\n"
        "- No trailing period unless it's really needed\n"
        "- Use the exact numbers and $tickers from the tweet\n"
        "Return ONLY the two comments, each on its own line."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _call_groq_two_comments(tweet_text: str, author: Optional[str]) -> List[str]:
    if not USE_GROQ or not _groq_client:
        return []

    messages = _groq_prompt(tweet_text, author)

    resp = _groq_client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile"),
        messages=messages,
        max_tokens=160,
        temperature=0.9,
        top_p=0.9,
        n=1,
    )

    text = resp.choices[0].message.content or ""
    # try to split into two lines
    lines = [normalize_ws(l) for l in text.splitlines() if normalize_ws(l)]
    if len(lines) >= 2:
        c1, c2 = lines[0], lines[1]
    elif len(lines) == 1:
        # attempt to split by punctuation if model returned one long string
        parts = re.split(r"[.!?]\s+", lines[0])
        parts = [normalize_ws(p) for p in parts if normalize_ws(p)]
        if len(parts) >= 2:
            c1, c2 = parts[0], parts[1]
        else:
            c1 = lines[0]
            c2 = ""
    else:
        return []

    out = []
    for c in (c1, c2):
        c = enforce_word_count_natural(c, 6, 13)
        c = normalize_punctuation(c)
        if c and not looks_too_ai_like(c):
            out.append(c)
    return out


def _available_providers():
    providers = []
    if USE_GROQ:
        providers.append(("groq", _call_groq_two_comments))
    # Could add OpenAI / Gemini later in same style
    return providers


def enforce_unique(candidates: List[str], tweet_text: str) -> List[str]:
    """
    Enforce:
    - non-empty
    - passes AI guards
    - not too similar to recent memory
    - no duplicates
    """
    out: List[str] = []
    seen_norm = set()
    for c in candidates:
        if not c:
            continue
        c = normalize_ws(c)
        c = normalize_punctuation(c)
        if not c:
            continue
        low = c.lower()
        if low in seen_norm:
            continue
        if looks_too_ai_like(c):
            continue
        if trigram_overlap_bad(c):
            continue
        if too_similar_to_recent(c):
            continue
        if comment_seen(c):
            continue

        seen_norm.add(low)
        out.append(c)
        if len(out) >= 4:
            break
    return out


def _rescue_two(tweet_text: str) -> List[str]:
    # very small offline rescue if everything else fails hard
    key = extract_keywords(tweet_text)
    focus = pick_focus_token(key) or "this"
    base = [
        f"{focus} is where this really gets interesting",
        f"long term {focus} probably matters way more than people admit",
        f"watch {focus}, everything else is just noise",
        f"hard to ignore {focus} after reading this",
    ]
    random.shuffle(base)
    out: List[str] = []
    for b in base:
        b = enforce_word_count_natural(b, 6, 13)
        b = normalize_punctuation(b)
        if not looks_too_ai_like(b):
            out.append(b)
        if len(out) == 2:
            break
    return out


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
    candidates: List[str] = []

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
            offline_items = generator.generate_two(
                tweet_text,
                author,
                handle,
                lang,
                url=url or "",
            )
            for item in offline_items:
                txt = (item.get("text") or "").strip()
                if txt:
                    candidates.append(txt)
            candidates = enforce_unique(candidates, tweet_text=tweet_text)
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
                url=url or "",
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
# URL / tweet parsing helpers (server side)
# ------------------------------------------------------------------------------

class TweetData:
    def __init__(
        self,
        url: str,
        text: str,
        author: Optional[str],
        handle: Optional[str],
        lang: Optional[str],
    ) -> None:
        self.url = url
        self.text = text
        self.author = author
        self.handle = handle
        self.lang = lang

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "text": self.text,
            "author": self.author,
            "handle": self.handle,
            "lang": self.lang,
        }


def _normalize_single_url(raw: str) -> str:
    """
    Remove extra text around URL if user pasted e.g.:

    'x.com/xxx/status/123 ... Please follow'

    → we only care about the first x.com link.
    """
    if not raw:
        return ""
    candidate = clean_and_normalize_urls(raw)
    return candidate or ""


def _extract_first_url_like(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"https?://\S+|x\.com/\S+", s)
    return m.group(0) if m else None


def clean_input_url_field(raw: str) -> str:
    """
    Frontend sends whatever user typed into the box.
    We want just the real X/Twitter URL, no extra text.
    """
    if not raw:
        return ""
    u = _extract_first_url_like(raw)
    if not u:
        return ""
    return _normalize_single_url(u)


def _canonical_x_url_from_tweet(original_url: str, t: TweetData) -> str:
    """
    Ensure the URL we send back is clean:

    - Always https://x.com/{handle}/status/{id}     if we know handle and id
    - Otherwise we fall back to original_url
    """
    try:
        if not original_url:
            return original_url
        parsed = urlparse(original_url)
        path = parsed.path or ""
        m = re.search(r"/status/(\d+)", path)
        if not m:
            return original_url
        tweet_id = m.group(1)
        if t.handle:
            return f"https://x.com/{t.handle}/status/{tweet_id}"
        return f"https://x.com/i/status/{tweet_id}"
    except Exception:
        return original_url


# ------------------------------------------------------------------------------
# Context building from tweet
# ------------------------------------------------------------------------------

def build_context_profile(
    tweet_text: str,
    url: str = "",
    tweet_author: Optional[str] = None,
    handle: Optional[str] = None,
) -> Dict[str, Any]:
    lang_hint = detect_language_hint(tweet_text)
    script = detect_script(tweet_text)
    topic = detect_topic(tweet_text)
    kws = extract_keywords(tweet_text)
    focus = pick_focus_token(kws) or "this"

    return {
        "url": url,
        "tweet_author": tweet_author,
        "handle": handle,
        "lang_hint": lang_hint,
        "script": script,
        "topic": topic,
        "focus": focus,
        "keywords": kws,
    }


# ------------------------------------------------------------------------------
# Diversity selection for final comments
# ------------------------------------------------------------------------------

def pick_two_diverse_text(candidates: list[str]) -> list[str]:
    """
    Given N candidate strings, pick 2 that are diverse and high-quality.
    """
    cleaned = []
    for c in candidates:
        c = normalize_ws(c)
        c = normalize_punctuation(c)
        if not c:
            continue
        if looks_too_ai_like(c):
            continue
        cleaned.append(c)

    # de-dup
    uniq = []
    seen = set()
    for c in cleaned:
        low = c.lower()
        if low in seen:
            continue
        seen.add(low)
        uniq.append(c)

    if not uniq:
        return []

    if len(uniq) == 1:
        return [uniq[0]]

    # Randomly shuffle but reward diversity
    random.shuffle(uniq)

    best_pair = None
    best_score = None

    def diversity_score(a: str, b: str) -> float:
        aw = set(re.findall(r"\w+", a.lower()))
        bw = set(re.findall(r"\w+", b.lower()))
        if not aw or not bw:
            return 0.0
        overlap = len(aw & bw)
        return -overlap  # lower overlap = better score (more negative)

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            score = diversity_score(a, b)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair:
        return [best_pair[0], best_pair[1]]
    return uniq[:2]

# ------------------------------------------------------------------------------
# Keep-alive (optional; disabled in main() by default)
# ------------------------------------------------------------------------------

def keep_alive():
    while True:
        try:
            logger.info("keep_alive ping")
            time.sleep(KEEP_ALIVE_INTERVAL)
        except Exception:
            time.sleep(KEEP_ALIVE_INTERVAL)


# ------------------------------------------------------------------------------
# Flask API
# ------------------------------------------------------------------------------

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
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # The frontend sends something like:
    # { "urls": ["https://x.com/...","..."] }
    urls = data.get("urls") or data.get("url") or []
    if isinstance(urls, str):
        urls = [urls]

    # Hard cap to avoid abuse
    urls = urls[:MAX_URLS_PER_REQUEST]

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    results = []
    for raw_url in urls:
        url = clean_input_url_field(str(raw_url or ""))
        if not url:
            results.append({
                "url": raw_url,
                "comments": [],
                "error": "No valid X/Twitter URL found",
            })
            continue

        try:
            # fetch Tweet data (via utils + VXTwitter under the hood)
            tweet_info = fetch_tweet_data(url)
            t = TweetData(
                url=url,
                text=tweet_info.text,
                author=tweet_info.author_name,
                handle=tweet_info.author_handle,
                lang=tweet_info.lang,
            )

            # Build comments via providers + offline
            two = generate_two_comments_with_providers(
                t.text,
                t.author,
                t.handle,
                t.lang or None,
                url=url,
            )

            display_url = _canonical_x_url_from_tweet(url, t)

            results.append({
                "url": display_url,
                "comments": two,
            })

            time.sleep(PER_URL_SLEEP)

        except CrownTALKError as e:
            results.append({
                "url": url,
                "comments": [],
                "error": str(e),
            })
        except Exception as e:
            logger.exception("Unexpected error processing url %s: %s", url, e)
            results.append({
                "url": url,
                "comments": [],
                "error": "internal_error",
            })

    return jsonify({
        "results": results,
    }), 200


# ------------------------------------------------------------------------------
# Legacy single-URL endpoint (if your frontend still expects /comment with one url)
# ------------------------------------------------------------------------------

@app.route("/comment_single", methods=["POST"])
def comment_single():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    raw_url = data.get("url") or ""
    url = clean_input_url_field(str(raw_url or ""))

    if not url:
        return jsonify({"error": "No valid X/Twitter URL found", "comments": []}), 400

    try:
        tweet_info = fetch_tweet_data(url)
        t = TweetData(
            url=url,
            text=tweet_info.text,
            author=tweet_info.author_name,
            handle=tweet_info.author_handle,
            lang=tweet_info.lang,
        )

        two = generate_two_comments_with_providers(
            t.text,
            t.author,
            t.handle,
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
        }), 400
    except Exception as e:
        logger.exception("Unexpected error in /comment_single for %s: %s", url, e)
        return jsonify({
            "url": url,
            "error": "internal_error",
            "comments": [],
        }), 500


# ------------------------------------------------------------------------------
# Boot
# ------------------------------------------------------------------------------

def main() -> None:
    init_db()
    # threading.Thread(target=keep_alive, daemon=True).start()  # optional keep-alive
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
