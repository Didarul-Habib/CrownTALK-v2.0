import os
import re
import time
import threading
import random
from collections import deque

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------------------------------
# Flask app setup
# -------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# -------------------------------------------------
# URL cleaning & VXTwitter fetch
# -------------------------------------------------

CLEAN_URL_RE = re.compile(r"^\s*\d+\.\s*")


def clean_url(url: str) -> str:
    """
    Remove '1. https://...' numbering, params, trim spaces.
    """
    if not url:
        return ""
    url = url.strip()
    url = CLEAN_URL_RE.sub("", url).strip()

    # Basic param strip
    if "?" in url:
        url = url.split("?", 1)[0]

    # force https
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def fetch_tweet_from_vx(url: str, max_retries: int = 3, timeout: int = 8):
    """
    Fetch tweet JSON from VXTwitter.
    Returns dict with keys: text, lang, display_name, username.
    Raises RuntimeError on error.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise RuntimeError("Invalid tweet URL")

    host = parsed.netloc
    path = parsed.path

    api_url = f"https://api.vxtwitter.com/{host}{path}"

    last_err = None
    for _ in range(max_retries):
        try:
            resp = requests.get(api_url, timeout=timeout)
            if resp.status_code != 200:
                last_err = RuntimeError(f"VXTwitter status {resp.status_code}")
                time.sleep(0.6)
                continue
            data = resp.json()

            # VXTwitter formats can vary; try several paths.
            tweet_obj = data.get("tweet") or data
            text = (
                tweet_obj.get("full_text")
                or tweet_obj.get("text")
                or tweet_obj.get("content")
                or ""
            )
            lang = tweet_obj.get("lang") or tweet_obj.get("language") or ""

            # user / author info
            user_obj = (
                tweet_obj.get("author")
                or tweet_obj.get("user")
                or data.get("user")
                or {}
            )
            display_name = user_obj.get("name") or ""
            username = (
                user_obj.get("screen_name")
                or user_obj.get("username")
                or tweet_obj.get("username")
                or ""
            )

            text = (text or "").strip()
            if not text:
                raise RuntimeError("Tweet text missing")

            return {
                "text": text,
                "lang": lang,
                "display_name": display_name,
                "username": username,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.8)

    raise RuntimeError(str(last_err) if last_err else "Unknown VXTwitter error")


# -------------------------------------------------
# Offline comment generator
# -------------------------------------------------

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "this",
    "that",
    "with",
    "for",
    "from",
    "into",
    "been",
    "are",
    "is",
    "was",
    "were",
    "on",
    "to",
    "in",
    "of",
    "at",
    "by",
    "it",
    "its",
    "as",
    "about",
    "just",
    "still",
    "kind",
    "kinda",
    "very",
    "really",
}

BANNED_WORDS = {
    "amazing",
    "awesome",
    "incredible",
    "empowering",
    "gamechanger",
    "game-changer",
    "transformative",
    "finally",
    "excited",
    "love this",
    "slay",
    "yass",
    "bestie",
    "queen",
    "thoughts",
    "agree",
    "who’s",
    "who's",
}

AI_GIVEAWAY = {
    "as an ai",
    "in this digital age",
}

# slang / tone words – we will rotate through them, not spam one
SLANG_WORDS = [
    "tbh",
    "ngl",
    "lowkey",
    "fr",
    "no cap",
    "deadass",
    "on god",
    "for real",
]

REACTION_WORDS = [
    "wild",
    "clean",
    "solid",
    "spicy",
    "messy",
    "serious",
    "degen",
    "heavy",
    "interesting",
    "rare",
]

CLOSERS = [
    "still watching",
    "still processing it",
    "timeline not ready",
    "cant ignore this",
    "need to see how it plays",
    "curious where this goes",
]

SUPPORTIVE_PHRASES = [
    "respect the grind",
    "respect the work here",
    "respect for sticking with it",
    "respect for building this",
]

SKEPTICAL_PHRASES = [
    "hope they execute for real",
    "curious if this actually ships",
    "want to see real delivery",
    "lets see if numbers back it",
]

NEUTRAL_PHRASES = [
    "feels like a key pivot",
    "keeps popping up on my feed",
    "keeps coming back in convo",
    "def worth keeping on radar",
]

# keep some cross-request history to avoid exact repeats
COMMENT_HISTORY = deque(maxlen=3000)
COMMENT_HISTORY_SET = set()


def _normalize_comment(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    # strip ending punctuation
    text = re.sub(r"[.!?…~]+$", "", text)

    words = text.split()
    if not words:
        return ""

    # enforce 5–12 words
    if len(words) < 5:
        # simple padding – but keep on topic-ish
        pads = ["for real", "no lie", "not even joking", "for now"]
        while len(words) < 5 and pads:
            extra = pads.pop(0)
            words.extend(extra.split())
    elif len(words) > 12:
        words = words[:12]

    text = " ".join(words)
    text = re.sub(r"[.!?…~]+$", "", text)
    return text


def _contains_banned(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in AI_GIVEAWAY):
        return True
    if any(phrase in lowered for phrase in BANNED_WORDS):
        return True
    return False


def _detect_language(tweet_text: str, lang_hint: str) -> str:
    if (lang_hint or "").lower().startswith("en"):
        return "en"
    # crude detection: lots of non-latin characters -> non-English
    non_ascii = sum(1 for ch in tweet_text if ord(ch) > 127)
    if non_ascii > max(5, len(tweet_text) * 0.25):
        return "non-en"
    return "en"


def _extract_keywords(tweet_text: str):
    """
    Grab potential project names / key tokens from tweet.
    """
    # kill URLs and @mentions / hashtags
    cleaned = re.sub(r"https?://\S+", " ", tweet_text)
    cleaned = re.sub(r"[@#]\S+", " ", cleaned)

    words = re.findall(r"[A-Za-z0-9$][A-Za-z0-9_\-]{2,}", cleaned)
    keywords = []
    project_like = []
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS:
            continue
        if wl.isdigit():
            continue
        if "$" in w or w.isupper() or wl.endswith("coin") or wl.endswith("dao"):
            project_like.append(w)
        else:
            keywords.append(w)

    # dedupe but keep order
    seen = set()
    kw_final = []
    for lst in [project_like, keywords]:
        for w in lst:
            if w.lower() not in seen:
                seen.add(w.lower())
                kw_final.append(w)

    return kw_final, project_like


def _first_name(display_name: str) -> str:
    if not display_name:
        return ""
    parts = display_name.split()
    if not parts:
        return ""
    first = re.sub(r"[^A-Za-z]", "", parts[0])
    return first if first else ""


def _choose_slang(used_slang: set) -> str:
    candidates = [s for s in SLANG_WORDS if s not in used_slang]
    if not candidates:
        candidates = SLANG_WORDS
        used_slang.clear()
    choice = random.choice(candidates)
    used_slang.add(choice)
    return choice


def _build_english_comment(tweet_text: str, meta: dict, style: str, used_slang: set) -> str:
    keywords, project_like = _extract_keywords(tweet_text)
    first_name = _first_name(meta.get("display_name", ""))
    main_kw = keywords[0] if keywords else ""
    project = project_like[0] if project_like else main_kw

    parts = []

    # optional slang at start
    if random.random() < 0.8:
        parts.append(_choose_slang(used_slang))

    # subject phrase
    if project:
        subject_word = project
    elif main_kw:
        subject_word = main_kw
    else:
        subject_word = "this"

    reaction = random.choice(REACTION_WORDS)

    if style == "supportive":
        if first_name and random.random() < 0.7:
            parts.append(f"{subject_word} from {first_name} looking {reaction}")
        else:
            parts.append(f"{subject_word} looking {reaction}")
        parts.append(random.choice(SUPPORTIVE_PHRASES))

    elif style == "skeptical":
        parts.append(f"{subject_word} narrative kinda {reaction}")
        parts.append(random.choice(SKEPTICAL_PHRASES))

    elif style == "degen":
        parts.append(f"{subject_word} setup feels {reaction}")
        parts.append("might be one to gamble on")

    elif style == "serious":
        parts.append(f"{subject_word} details actually {reaction}")
        parts.append(random.choice(NEUTRAL_PHRASES))

    else:  # neutral / observer
        parts.append(f"{subject_word} story pretty {reaction}")
        parts.append(random.choice(NEUTRAL_PHRASES))

    comment = " ".join(parts)
    return _normalize_comment(comment)


def _wrap_for_non_english(tweet_text: str, english_comment: str) -> str:
    """
    For non-english tweets, keep a short native snippet + english in brackets.
    """
    # take first ~16 characters of original as "native"
    native = tweet_text.strip()
    native = re.sub(r"\s+", " ", native)
    if len(native) > 16:
        native = native[:16].rstrip()

    combined = f"{native} ({english_comment})"
    return _normalize_comment(combined)


def generate_comments_for_tweet(tweet_text: str, meta: dict, global_history: set):
    """
    Generate two distinct comments for a tweet, with global de-duplication.
    Respects all rules: 5–12 words, no emojis/hashtags, no hype words,
    minimal repetition, sometimes mentions names/projects.
    """
    lang_mode = _detect_language(tweet_text, meta.get("lang", ""))

    used_slang_local: set = set()
    styles_pool = ["supportive", "skeptical", "degen", "serious", "neutral"]

    comments = []
    attempts = 0

    while len(comments) < 2 and attempts < 20:
        attempts += 1
        style = styles_pool[attempts % len(styles_pool)]
        english = _build_english_comment(tweet_text, meta, style, used_slang_local)

        if not english or _contains_banned(english):
            continue

        if lang_mode == "non-en":
            candidate = _wrap_for_non_english(tweet_text, english)
        else:
            candidate = english

        lowered = candidate.lower()
        if lowered in global_history:
            continue
        if lowered in (c.lower() for c in comments):
            continue

        comments.append(candidate)
        global_history.add(lowered)
        COMMENT_HISTORY.append(lowered)
        COMMENT_HISTORY_SET.add(lowered)

    # if we failed somehow, at least return something tiny but valid
    while len(comments) < 2:
        comments.append(_normalize_comment("tbh still watching this one play out"))

    return comments


# -------------------------------------------------
# API endpoints
# -------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/comment", methods=["OPTIONS", "POST"])
def comment():
    if request.method == "OPTIONS":
        # CORS preflight
        return ("", 200)

    payload = request.get_json(silent=True) or {}
    urls = payload.get("urls") or []
    if not isinstance(urls, list) or not urls:
        return jsonify({"batches": []})

    cleaned_urls = [clean_url(u) for u in urls if clean_url(u)]
    if not cleaned_urls:
        return jsonify({"batches": []})

    batches = []
    global_history = set(COMMENT_HISTORY_SET)  # seed from past but local copy

    batch_size = 2
    batch_index = 0

    for i in range(0, len(cleaned_urls), batch_size):
        batch_urls = cleaned_urls[i : i + batch_size]
        batch_index += 1

        batch_results = []
        batch_failed = []

        for url in batch_urls:
            try:
                info = fetch_tweet_from_vx(url)
                comments = generate_comments_for_tweet(info["text"], info, global_history)
                batch_results.append({"url": url, "comments": comments})
            except Exception as e:  # noqa: BLE001
                batch_failed.append({"url": url, "reason": str(e) or "Tweet text missing"})

        batches.append(
            {
                "batch": batch_index,
                "results": batch_results,
                "failed": batch_failed,
            }
        )

    return jsonify({"batches": batches})


@app.route("/reroll", methods=["OPTIONS", "POST"])
def reroll():
    if request.method == "OPTIONS":
        return ("", 200)

    payload = request.get_json(silent=True) or {}
    url = clean_url(payload.get("url", ""))
    if not url:
        return jsonify({"error": "Invalid url"}), 400

    try:
        info = fetch_tweet_from_vx(url)
        global_history = set(COMMENT_HISTORY_SET)
        comments = generate_comments_for_tweet(info["text"], info, global_history)
        return jsonify({"url": url, "comments": comments})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e) or "Failed to reroll"}), 500


# -------------------------------------------------
# Keep-alive ping thread
# -------------------------------------------------

def keep_alive():
    """
    Periodically hit the health endpoint so Render doesn't freeze the dyno.
    """
    while True:
        try:
            url = os.environ.get("RENDER_EXTERNAL_URL")
            if url:
                try:
                    requests.get(url.rstrip("/") + "/", timeout=5)
                except requests.RequestException:
                    pass
        except Exception:
            pass
        time.sleep(600)  # 10 minutes


if __name__ == "__main__":
    t = threading.Thread(target=keep_alive, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
