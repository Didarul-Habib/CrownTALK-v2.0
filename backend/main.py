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
    url = re.sub(r"^\d+\.\s*", "", url)  # remove "1. https://..."
    url = url.split("?")[0]  # strip query params
    return url


# ---------------------------------------------------------
# Offline Comment Generator (crypto-aware + multilingual)
# ---------------------------------------------------------

banned_phrases = {
    "amazing", "awesome", "incredible", "finally", "excited",
    "love this", "empowering", "game changer", "transformative",
    "as an ai", "in this digital age",
    "slay", "yass", "bestie", "queen",
    "thoughts", "agree", "whos with me", "who's with me",
    "love", "lovely", "like this", "like that"
}

stopwords = {
    "the", "and", "for", "that", "with", "this", "from", "have", "just",
    "been", "are", "was", "were", "you", "your", "they", "them", "but",
    "about", "into", "over", "under", "http", "https", "www", "com",
    "x", "t", "co", "amp", "will", "cant", "can't", "its", "it's",
    "rt", "on", "in", "to", "of", "at", "is", "a", "an", "be", "by",
    "or", "it", "we", "our", "us", "me", "my", "so", "if", "as",
    "up", "out", "at", "im", "i'm"
}

positive_words = {
    "great", "good", "solid", "bullish", "up", "win", "strong", "clean",
    "growth", "progress", "nice", "cool", "pump", "moon", "mooning"
}
negative_words = {
    "bad", "down", "bearish", "rug", "scam", "problem", "issue", "risk",
    "dump", "crash", "angry", "annoying", "rekt", "liquidation"
}

crypto_keywords = {
    "btc", "eth", "sol", "avax", "bnb", "arb", "op", "base", "layer 2", "l2",
    "chain", "nft", "token", "airdrop", "alpha", "defi", "dex", "cex",
    "memecoin", "meme coin", "presale", "ido", "ico", "staking", "lp",
    "yield", "bridge", "wallet"
}

filler_tokens = ["tbh", "fr", "lowkey", "honestly", "really", "ngl"]

comment_history = set()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_keywords(text, max_keywords=12):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w and w not in stopwords]
    if not words:
        return []
    counts = Counter(words)
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


def detect_category(text):
    t = text.lower()
    if any(k in t for k in ("giveaway", "give away", "tag 3", "tag three",
                            "retweet to enter", "like and retweet")):
        return "giveaway"
    if any(k in t for k in ("chart", "support", "resistance", "ath",
                            "price target", "%", "percent",
                            "market cap", "mc", "pump", "dump")):
        return "chart"
    if "🧵" in text or len(text) > 220:
        return "thread"
    if len(text) < 80:
        return "one_liner"
    return "generic"


def is_crypto_tweet(text):
    t = text.lower()
    return any(k in t for k in crypto_keywords)


def build_comment_from_text_en(text):
    keywords = extract_keywords(text)
    sentiment = simple_sentiment(text)
    category = detect_category(text)
    crypto = is_crypto_tweet(text)

    kw = "this"
    if keywords:
        kw = random.choice(keywords)

    # avoid comments like "lowkey 1 been everywhere lately"
    if kw.isdigit():
        kw = "setup"

    neutral_templates = [
        "lowkey {kw} been everywhere lately",
        "tbh {kw} still on my mind",
        "cant ignore {kw} right now",
        "ngl {kw} got people talking",
        "still trying to process {kw} fr",
        "real talk {kw} kinda interesting fr",
        "lowkey watching how {kw} plays out",
        "timeline cant stop circling {kw}",
        "tbh {kw} keeps coming back up",
    ]

    positive_templates = [
        "{kw} actually looking solid ngl",
        "lowkey think {kw} might work out",
        "tbh {kw} feels like progress fr",
        "ngl direction around {kw} looks clean",
        "lowkey {kw} momentum still there",
    ]

    negative_templates = [
        "ngl {kw} giving weird vibes rn",
        "tbh {kw} still feels risky fr",
        "cant shake the worry around {kw}",
        "lowkey nervous where {kw} goes next",
        "ngl {kw} setup feels fragile rn",
    ]

    crypto_neutral = [
        "real talk {kw} got the timeline watching",
        "lowkey curious how {kw} trades next",
        "tbh {kw} volume been catching my eye",
        "ngl {kw} narrative still not priced in",
        "people quietly rotating into {kw} fr",
    ]

    crypto_positive = [
        "ngl {kw} setup looking kinda clean fr",
        "lowkey think {kw} might send later",
        "tbh {kw} risk reward looking decent",
        "chart on {kw} not looking bad ngl",
    ]

    crypto_negative = [
        "ngl {kw} vibes feel like exit liquidity",
        "lowkey worried {kw} ends ugly",
        "tbh {kw} entries already look cooked",
        "hard not to see {kw} as late entry",
    ]

    giveaway_templates = [
        "lowkey hope {kw} picker actually fair",
        "ngl these {kw} giveaways always feel rigged",
        "tbh {kw} giveaway meta still going strong",
        "real talk {kw} farms never really stop",
    ]

    chart_templates = [
        "ngl this {kw} chart kinda wild",
        "tbh {kw} levels actually make some sense",
        "lowkey watching {kw} support zone rn",
        "real talk {kw} price action feels fragile",
        "everyone staring at same {kw} levels fr",
    ]

    thread_templates = [
        "lowkey saving this {kw} thread for later",
        "tbh {kw} breakdown pretty helpful ngl",
        "ngl this {kw} thread goes deeper than expected",
        "real talk {kw} thread explaining a lot here",
        "lot of small details on {kw} in here",
    ]

    oneliner_templates = [
        "ngl short but {kw} message lands",
        "lowkey simple {kw} line but it works",
        "tbh that {kw} bar kinda hits",
        "quick line but {kw} said a lot",
    ]

    if sentiment == "positive":
        base_pool = positive_templates + neutral_templates
    elif sentiment == "negative":
        base_pool = negative_templates + neutral_templates
    else:
        base_pool = neutral_templates

    if category == "giveaway":
        base_pool = giveaway_templates
    elif category == "chart":
        base_pool = chart_templates + base_pool
    elif category == "thread":
        base_pool = thread_templates + base_pool
    elif category == "one_liner":
        base_pool = oneliner_templates + base_pool

    if crypto:
        if sentiment == "positive":
            base_pool += crypto_positive + crypto_neutral
        elif sentiment == "negative":
            base_pool += crypto_negative + crypto_neutral
        else:
            base_pool += crypto_neutral

    template = random.choice(base_pool)
    comment = template.format(kw=kw)
    return comment


# --- language detection: very rough but fully offline ---
def detect_language(text):
    lat = hi = cjk = 0
    for ch in text:
        code = ord(ch)
        if "a" <= ch <= "z" or "A" <= ch <= "Z":
            lat += 1
        elif 0x0900 <= code <= 0x097F:  # Devanagari (Hindi-ish)
            hi += 1
        elif 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:  # JP/CN
            cjk += 1

    if hi > lat and hi >= 5 and hi > cjk:
        return "hi"
    if cjk > lat and cjk >= 5:
        return "zh"
    return "en"


def build_multilang_comment(text, lang):
    # base english comment first
    base_en_raw = build_comment_from_text_en(text)
    base_en = post_process_comment_en(base_en_raw)

    keywords = extract_keywords(text)
    kw = keywords[0] if keywords else ""
    # we’ll normalize per language so no ugly digits
    if kw.isdigit():
        kw = ""

    if lang == "hi":
        hi_templates = [
            "ये बात {kw} पर सही लग रही",
            "सच में {kw} वाली बात सोचने लायक",
            "{kw} वाली बात दिमाग में घूम रही",
            "आजकल {kw} वाला सीन काफी दिख रहा",
            "धीरे धीरे {kw} वाली बातें बढ़ रही",
            "{kw} वाला पॉइंट हल्का भारी लग रहा",
        ]
        kw_local = kw or "ये चीज"
        tmpl = random.choice(hi_templates)
        native = tmpl.format(kw=kw_local)

    elif lang == "zh":
        zh_templates = [
            "{kw} 这事 确实 有点 东西",
            "说实话 {kw} 这点 挺有意思",
            "最近 {kw} 相关 声音 有点多",
            "{kw} 这波 操作 挺让人 关注",
            "{kw} 这块 细节 还挺关键",
            "老实讲 {kw} 后面 走向 值得看",
        ]
        kw_local = kw or "这个"
        tmpl = random.choice(zh_templates)
        native = tmpl.format(kw=kw_local)

    else:
        # safety: english only
        return base_en

    combined = f"{native} ({base_en})"
    words = combined.split()
    if len(words) < 5:
        while len(words) < 5:
            words.append(random.choice(filler_tokens))
    elif len(words) > 12:
        words = words[:12]
    combined = " ".join(words)
    combined = combined.rstrip(".,!?:;…-")

    return combined

# ---------------------------------------------------------
# VXTwitter Fetcher
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

                if "text" in data and isinstance(data["text"], str):
                    return data["text"], None
                if "full_text" in data and isinstance(data["full_text"], str):
                    return data["full_text"], None
                if "tweet" in data:
                    t = data["tweet"]
                    if "text" in t:
                        return t["text"], None
                    if "full_text" in t:
                        return t["full_text"], None
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
        batches = [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]

        out = []

        for i, batch in enumerate(batches):
            batch_info = {"batch": i + 1, "results": [], "failed": []}

            for url in batch:
                text, err = fetch_tweet_text(url)
                if err:
                    batch_info["failed"].append({"url": url, "reason": err})
                    continue

                comments = generate_two_comments(text)
                batch_info["results"].append({"url": url, "comments": comments})

            out.append(batch_info)

        return jsonify({"batches": out})

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


@app.post("/reroll")
def reroll():
    """Per-tweet re-roll endpoint."""
    try:
        data = request.get_json(silent=True)
        if not data or "url" not in data:
            return jsonify({"error": "Invalid request"}), 400

        url = clean_url(data["url"])
        text, err = fetch_tweet_text(url)
        if err:
            return jsonify({"url": url, "error": err, "comments": []})

        comments = generate_two_comments(text)
        return jsonify({"url": url, "error": None, "comments": comments})

    except Exception as e:
        return jsonify({"url": data.get("url", ""), "error": str(e), "comments": []}), 500


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
