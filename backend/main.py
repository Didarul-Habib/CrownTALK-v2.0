from flask import Flask, request, jsonify
import threading
import requests
import time
import re
import random
from collections import Counter

app = Flask(__name__)

# ---------------------------------------------------------
# Manual CORS
# ---------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------
# URL Cleaner
# ---------------------------------------------------------
def clean_url(url):
    if not isinstance(url, str):
        return ""

    url = url.strip()

    # Remove "1. https://..." numbering
    url = re.sub(r"^\d+\.\s*", "", url)

    # Strip query params
    url = url.split("?")[0]

    return url


# ---------------------------------------------------------
# Offline Comment Generator (context-aware)
# ---------------------------------------------------------

# words/phrases we don't want
banned_phrases = {
    "amazing", "awesome", "incredible", "finally", "excited",
    "love this", "empowering", "game changer", "transformative",
    "as an ai", "in this digital age",
    "slay", "yass", "bestie", "queen",
    "thoughts", "agree", "whos with me", "who's with me",
    "love", "lovely"
}

# stopwords to ignore when extracting keywords
stopwords = {
    "the","and","for","that","with","this","from","have","just","been","are",
    "was","were","you","your","they","them","but","about","into","over","under",
    "http","https","www","com","x","t","co","amp","will","cant","can't","its",
    "it's","rt","on","in","to","of","at","is","a","an","be","by","or","it",
    "we","our","us","me","my","so","if","as","up","out"
}

# tiny sentiment word lists
positive_words = {
    "great","good","solid","bullish","up","win","strong","clean","growth",
    "progress","nice","cool"
}
negative_words = {
    "bad","down","bearish","rug","scam","problem","issue","risk","dump",
    "crash","hate","angry","annoying"
}

# filler bits we can use if comment too short
filler_tokens = ["tbh", "fr", "lowkey", "honestly", "really"]

# keep a global history to avoid repeating exact comments
comment_history = set()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_keywords(text, max_keywords=10):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w and w not in stopwords]

    if not words:
        return []

    counts = Counter(words)
    # most common first
    ordered = [w for (w, _) in counts.most_common(max_keywords)]
    return ordered


def simple_sentiment(text):
    text_l = text.lower()
    score = 0
    for w in positive_words:
        if w in text_l:
            score += 1
    for w in negative_words:
        if w in text_l:
            score -= 1
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def build_comment_from_text(text):
    keywords = extract_keywords(text)
    sentiment = simple_sentiment(text)

    if keywords:
        kw = random.choice(keywords)
    else:
        kw = "this"

    templates_neutral = [
        "lowkey {kw} been everywhere lately",
        "tbh {kw} still on my mind",
        "seeing {kw} pop up more lately",
        "cant ignore {kw} right now",
        "ngl {kw} kinda interesting fr",
        "real talk {kw} got people talking",
        "still trying to process {kw} fr",
    ]

    templates_positive = [
        "{kw} actually looking solid ngl",
        "lowkey think {kw} might work out",
        "ngl {kw} direction looks pretty good",
        "tbh {kw} feels like progress fr",
    ]

    templates_negative = [
        "ngl {kw} giving me weird vibes",
        "tbh {kw} still feels risky fr",
        "cant shake the feeling {kw} off",
        "lowkey worried where {kw} goes next",
    ]

    if sentiment == "positive":
        templates = templates_positive + templates_neutral
    elif sentiment == "negative":
        templates = templates_negative + templates_neutral
    else:
        templates = templates_neutral

    template = random.choice(templates)
    comment = template.format(kw=kw)
    return comment


def post_process_comment(comment):
    # remove banned phrases (substring basis)
    c_low = comment.lower()
    for bad in banned_phrases:
        if bad in c_low:
            c_low = c_low.replace(bad, "")
    comment = c_low

    # collapse whitespace
    comment = re.sub(r"\s+", " ", comment).strip()

    # split for length control
    words = comment.split()

    # enforce 5–12 words
    if len(words) < 5:
        while len(words) < 5:
            words.append(random.choice(filler_tokens))
    elif len(words) > 12:
        words = words[:12]

    comment = " ".join(words)

    # remove trailing punctuation
    comment = comment.rstrip(".,!?:;…-")

    # final safety: no emojis / hashtags
    # (we never add them but just in case)
    words = []
    for w in comment.split():
        if "#" in w:
            continue
        # crude emoji filter: drop clearly non-ascii
        if any(ord(ch) > 126 for ch in w):
            continue
        words.append(w)
    comment = " ".join(words).strip()

    # fallback safety
    if not comment or len(comment.split()) < 3:
        comment = "lowkey trying to process all this"

    return comment


def generate_unique_comment(text):
    # try multiple times to avoid history collisions
    for _ in range(10):
        raw = build_comment_from_text(text)
        processed = post_process_comment(raw)
        norm = normalize_text(processed)
        if norm not in comment_history and 5 <= len(processed.split()) <= 12:
            comment_history.add(norm)
            return processed

    # if everything repeats, just return the last processed
    return processed


def generate_two_comments(text):
    c1 = generate_unique_comment(text)
    c2 = generate_unique_comment(text)

    # ensure they differ
    tries = 0
    while normalize_text(c2) == normalize_text(c1) and tries < 5:
        c2 = generate_unique_comment(text)
        tries += 1

    return [c1, c2]


# ---------------------------------------------------------
# VXTwitter Fetcher (patched)
# ---------------------------------------------------------
def fetch_tweet_text(url):
    try:
        match = re.search(r"https?://([^/]+)(/.*)", url)
        if not match:
            return None, "Invalid URL"

        host, path = match.groups()
        api_url = f"https://api.vxtwitter.com/{host}{path}"

        for _ in range(3):
            try:
                r = requests.get(api_url, timeout=10)
                if r.status_code != 200:
                    time.sleep(1)
                    continue

                data = r.json()

                # { "text": "..." }
                if "text" in data and isinstance(data["text"], str):
                    return data["text"], None

                # { "full_text": "..." }
                if "full_text" in data and isinstance(data["full_text"], str):
                    return data["full_text"], None

                # { "tweet": { ... } }
                if "tweet" in data:
                    tweet_obj = data["tweet"]
                    if "text" in tweet_obj:
                        return tweet_obj["text"], None
                    if "full_text" in tweet_obj:
                        return tweet_obj["full_text"], None

                if "error" in data:
                    return None, data["error"]

            except Exception:
                pass

            time.sleep(1)

        return None, "Tweet text not found"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Keep Alive (Render)
# ---------------------------------------------------------
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except Exception:
            pass
        time.sleep(600)


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/")
def home():
    return jsonify({"status": "ok"})


@app.post("/comment")
def comment():
    try:
        data = request.get_json(silent=True)

        if not data or "urls" not in data:
            return jsonify({"error": "Invalid request"}), 400

        urls = data["urls"]
        cleaned = [clean_url(u) for u in urls if isinstance(u, str) and u.strip()]

        # group into batches of 2
        batches = [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]

        out = []

        for i, batch in enumerate(batches):
            batch_info = {
                "batch": i + 1,
                "results": [],
                "failed": []
            }

            for url in batch:
                text, err = fetch_tweet_text(url)

                if err:
                    batch_info["failed"].append({
                        "url": url,
                        "reason": err
                    })
                    continue

                comments = generate_two_comments(text)

                batch_info["results"].append({
                    "url": url,
                    "comments": comments
                })

            out.append(batch_info)

        return jsonify({"batches": out})

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
