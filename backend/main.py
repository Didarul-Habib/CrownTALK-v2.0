from flask import Flask, request, jsonify
import threading
import requests
import time
import re
import random
from collections import Counter
from urllib.parse import urlparse, urlunparse

app = Flask(__name__)

# Public URL of this backend (for keep-alive)
BACKEND_PUBLIC_URL = "https://crowntalk-v2-0.onrender.com"
VX_API_BASE = "https://api.vxtwitter.com"

BATCH_SIZE = 2
KEEP_ALIVE_INTERVAL = 600  # seconds


# ---------------------------------------------------------
# Manual CORS
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
        return jsonify({"error": "Invalid JSON body"}), 400

    urls = payload.get("urls") or []
    if not isinstance(urls, list) or not urls:
        return jsonify({"error": "Body must contain non-empty 'urls' array"}), 400

    cleaned_urls = [clean_url(u) for u in urls]
    cleaned_urls = [u for u in cleaned_urls if u]

    results = []
    failed = []

    generator = OfflineCommentGenerator()

    # process in batches of 2 internally
    for batch_index, batch in enumerate(chunked(cleaned_urls, BATCH_SIZE), start=1):
        for url in batch:
            try:
                tweet_data = fetch_tweet(url)
                if not tweet_data or not tweet_data.get("text"):
                    failed.append(
                        {
                            "url": url,
                            "reason": "Missing tweet text from VXTwitter",
                        }
                    )
                    continue

                text = tweet_data.get("text", "")
                author = tweet_data.get("author") or None
                lang_hint = tweet_data.get("lang") or None

                comments = generator.generate_comments(
                    text=text,
                    author=author,
                    lang_hint=lang_hint,
                )

                results.append(
                    {
                        "url": url,
                        "comments": comments,
                    }
                )
            except Exception as e:
                failed.append(
                    {
                        "url": url,
                        "reason": str(e),
                    }
                )

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
        return jsonify({"error": "Invalid JSON body", "comments": []}), 400

    url = data.get("url") or ""
    if not url:
        return jsonify({"error": "Missing 'url' field", "comments": []}), 400

    url = clean_url(url)
    if not url:
        return jsonify({"error": "Invalid URL", "comments": []}), 400

    generator = OfflineCommentGenerator()

    try:
        tweet_data = fetch_tweet(url)
        if not tweet_data or not tweet_data.get("text"):
            return jsonify(
                {
                    "url": url,
                    "error": "Missing tweet text from VXTwitter",
                    "comments": [],
                }
            ), 502

        text = tweet_data.get("text", "")
        author = tweet_data.get("author") or None
        lang_hint = tweet_data.get("lang") or None

        comments = generator.generate_comments(
            text=text,
            author=author,
            lang_hint=lang_hint,
        )
        return jsonify({"url": url, "comments": comments}), 200

    except Exception as e:
        return jsonify({"url": url, "error": str(e), "comments": []}), 500


# ---------------------------------------------------------
# URL CLEANING / TWEET FETCH
# ---------------------------------------------------------
NUMBERING_RE = re.compile(r"^\s*\d+[\.\)]\s*")


def clean_url(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    raw = NUMBERING_RE.sub("", raw).strip()

    match = re.search(r"https?://\S+", raw)
    if match:
        raw = match.group(0)

    try:
        parsed = urlparse(raw)
    except Exception:
        return ""

    netloc = (parsed.netloc or "").lower()

    if "x.com" in netloc:
        netloc = "twitter.com"
    elif "mobile.twitter.com" in netloc:
        netloc = "twitter.com"

    cleaned = parsed._replace(netloc=netloc, query="", fragment="")
    return urlunparse(cleaned)


def chunked(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def fetch_tweet(url: str) -> dict:
    parsed = urlparse(url)
    path = (parsed.path or "").lstrip("/")

    if not path:
        raise ValueError("Cannot derive tweet path from URL")

    api_url = f"{VX_API_BASE}/{path}"

    resp = requests.get(api_url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    tweet = data.get("tweet") or data

    text = (
        tweet.get("text")
        or tweet.get("full_text")
        or data.get("text")
        or ""
    )

    author = None
    if isinstance(tweet.get("author"), dict):
        author = tweet["author"].get("name")
    elif isinstance(tweet.get("user"), dict):
        author = tweet["user"].get("name")

    lang = tweet.get("language") or tweet.get("lang") or None

    return {
        "text": text or "",
        "author": author,
        "lang": lang,
    }


# ---------------------------------------------------------
# OFFLINE COMMENT GENERATOR
# ---------------------------------------------------------
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "when", "where", "how",
    "this", "that", "those", "these", "it", "its", "is", "are", "was",
    "were", "be", "been", "being", "for", "to", "of", "in", "on", "at",
    "with", "by", "from", "as", "about", "into", "over", "after", "before",
    "your", "my", "our", "their", "his", "her", "you", "we", "they", "i",
    "just", "so", "very", "too", "up", "down", "out", "off", "again",
}

AI_BLOCKLIST = {
    "amazing",
    "awesome",
    "incredible",
    "empowering",
    "game changer",
    "game-changing",
    "transformative",
    "paradigm shift",
    "in this digital age",
    "as an ai",
    "as a language model",
    "in conclusion",
    "in summary",
    "furthermore",
    "moreover",
    "navigate this landscape",
    "ever-evolving landscape",
    "leverage this insight",
    "cutting edge",
    "state of the art",
    "unprecedented",
    "unleash",
    "unleashing",
    "unlock the power",
    "harness the power",
    "embark on this journey",
    "embark on a journey",
    "our journey",
    "empower",
    "revolutionize",
    "disruptive",
    "slay",
    "yass",
    "yas",
    "queen",
    "bestie",
    "like and retweet",
    "thoughts?",
    "thoughts ?",
    "agree?",
    "agree ?",
    "who's with me",
    "drop your thoughts",
    "smash that like button",
    "link in bio",
}

EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)


class OfflineCommentGenerator:
    def __init__(self):
        self.random = random.Random()

    # ---------- internals ----------

    def _is_probably_english(self, text: str, lang_hint: str | None) -> bool:
        """
        Stronger language detection.
        """

        stripped = re.sub(r"https?://\S+", "", text)
        stripped = re.sub(r"[@#]\S+", "", stripped)
        chars = [c for c in stripped if not c.isspace()]

        # Detect CJK
        cjk = [
            c for c in chars
            if '\u4e00' <= c <= '\u9fff'
            or '\u3040' <= c <= '\u30ff'
            or '\uac00' <= c <= '\ud7af'
        ]

        if len(cjk) >= 2:
            return False

        ascii_letters = [c for c in chars if c.isascii() and c.isalpha()]
        ratio_ascii = len(ascii_letters) / max(len(chars), 1)

        if ratio_ascii > 0.7:
            return True

        if lang_hint:
            lang_hint = lang_hint.lower()
            if not lang_hint.startswith("en"):
                return False
            return ratio_ascii > 0.3

        return ratio_ascii > 0.5

    def _make_native_comment(self, text: str, key_tokens: list[str]) -> str:
        cleaned = re.sub(r"https?://\S+", "", text)
        cleaned = re.sub(r"[@#]\S+", "", cleaned).strip()

        # CJK mode
        if any('\u4e00' <= c <= '\u9fff' for c in cleaned) \
           or any('\u3040' <= c <= '\u30ff' for c in cleaned) \
           or any('\uac00' <= c <= '\ud7af' for c in cleaned):

            return cleaned[:20].strip()

        # normal space-split fallback
        words = cleaned.split()
        words = [w for w in words if not w.startswith("@")]

        if not words:
            focus = pick_focus_token(key_tokens)
            return focus or "不错"

        if len(words) < 5:
            words += words * 2
        if len(words) > 12:
            words = words[:12]

        return " ".join(words)

    # PUBLIC GENERATOR ENTRY
    def generate_comments(self, text: str, author: str | None, lang_hint: str | None = None):
        is_english = self._is_probably_english(text, lang_hint)

        topic = detect_topic(text)
        crypto = is_crypto_tweet(text)
        key_tokens = extract_keywords(text)

        if is_english:
            c1 = self._make_english_comment(
                text=text, author=author, topic=topic,
                is_crypto=crypto, key_tokens=key_tokens,
                used_kinds=set(),
            )
            c2 = self._make_english_comment(
                text=text, author=author, topic=topic,
                is_crypto=crypto, key_tokens=key_tokens,
                used_kinds={c1["kind"]},
            )
            return [
                {"lang": "en", "text": c1["text"]},
                {"lang": "en", "text": c2["text"]},
            ]

        native = self._make_native_comment(text, key_tokens)
        en = self._make_english_comment(
            text=text, author=author,
            topic=topic, is_crypto=crypto,
            key_tokens=key_tokens,
            used_kinds=set(),
        )

        return [
            {"lang": "native", "text": native},
            {"lang": "en", "text": en["text"]},
        ]

    # -----------------------------------------
    # ENGLISH COMMENT GENERATION (unchanged)
    # -----------------------------------------
    def _tidy_comment(self, text: str) -> str:
        text = EMOJI_PATTERN.sub("", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[.!?;:…]+$", "", text).strip()

        words = text.split()
        if len(words) < 5:
            fillers = ["right", "honestly", "tbh", "still", "though"]
            while len(words) < 5 and fillers:
                words.append(fillers.pop(0))
        elif len(words) > 12:
            words = words[:12]

        final = " ".join(words)
        return re.sub(r"[.!?;:…]+$", "", final).strip()

    def _violates_ai_blocklist(self, text: str) -> bool:
        low = text.lower()
        return any(phrase in low for phrase in AI_BLOCKLIST)

    def _make_english_comment(
        self, text, author, topic, is_crypto, key_tokens, used_kinds
    ):
        focus = pick_focus_token(key_tokens) or "this"

        author_ref = None
        if author and random.random() < 0.6:
            parts = author.split()
            author_ref = parts[0] if parts else author

        buckets = self._get_template_buckets(topic, is_crypto)

        available = [k for k in buckets if k not in used_kinds]
        if not available:
            available = list(buckets.keys())
        kind = random.choice(available)

        for _ in range(10):
            tmpl = random.choice(buckets[kind])
            out = tmpl.format(author=author_ref or "", focus=focus)
            out = self._tidy_comment(out)

            if out and not self._violates_ai_blocklist(out):
                return {"kind": kind, "text": out}

        fallback = self._tidy_comment(f"Pretty solid points on {focus}")
        return {"kind": kind, "text": fallback}

    def _get_template_buckets(self, topic, is_crypto):
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
            "This keeps {focus} in perspective instead of hype",
            "Good reminder not to overreact to {focus} stuff",
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
# TOPIC / KEYWORD HELPERS
# ---------------------------------------------------------
def detect_topic(text: str) -> str:
    t = text.lower()

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
    if any(k in t for k in ("meme", "shitpost", "ratioed", "memeing"))) or "lol" in t:
        return "meme"
    if "🧵" in text or len(text) > 220:
        return "thread"
    if len(text) < 80:
        return "one_liner"
    return "generic"


def is_crypto_tweet(text: str) -> bool:
    t = text.lower()
    crypto_keywords = [
        "crypto", "defi", "nft", "airdrop", "token", "coin", "chain",
        "l1", "l2", "staking", "yield", "dex", "cex", "onchain",
        "on-chain", "gas fees", "btc", "eth", "sol", "arb",
        "layer two", "mainnet",
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

    filtered = [
        tok
        for tok, low in zip(tokens, tokens_lower)
        if low not in EN_STOPWORDS and len(low) > 2
    ]
    if not filtered:
        filtered = tokens

    counts = Counter([t.lower() for t in filtered])

    sorted_tokens = sorted(
        filtered,
        key=lambda w: (-counts[w.lower()], -len(w)),
    )
    seen = set()
    result = []
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
    return random.choice(upperish) if upperish else random.choice(tokens)


# ---------------------------------------------------------
# KEEP ALIVE THREAD
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
# MAIN ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
