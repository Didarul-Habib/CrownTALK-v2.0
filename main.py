import os
import re
import time
import requests
from flask import Flask, request, jsonify, send_from_directory
from langdetect import detect
from openai import OpenAI

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = Flask(__name__, static_folder="static")

# OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI()

# -------------------------------------------------
# Load Comment Style Guide
# -------------------------------------------------
STYLE_GUIDE_PATH = "comment_style_guide.txt"
if os.path.exists(STYLE_GUIDE_PATH):
    with open(STYLE_GUIDE_PATH, "r", encoding="utf-8") as f:
        STYLE_GUIDE = f.read()
else:
    STYLE_GUIDE = ""


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_tweet_url(url: str) -> str:
    """
    - Strip whitespace
    - Remove tracking/query params (?s=20 etc)
    - Normalize x.com -> twitter.com for the VX API
    """
    url = (url or "").strip()
    if not url:
        return ""

    # Drop everything after '?'
    if "?" in url:
        url = url.split("?", 1)[0]

    # x.com -> twitter.com
    url = url.replace("https://x.com/", "https://twitter.com/").replace(
        "http://x.com/", "https://twitter.com/"
    )

    return url


def fetch_tweet_text(url: str):
    """
    TweetAPI: C / VX style.
    Uses vxtwitter to fetch tweet content.
    """
    norm = normalize_tweet_url(url)
    if not norm or "twitter.com/" not in norm:
        return None

    try:
        path = norm.split("twitter.com/")[-1]
        api_url = "https://api.vxtwitter.com/" + path

        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()

        # Handle common VX formats
        if isinstance(data, dict):
            if "tweet" in data and isinstance(data["tweet"], dict):
                if "text" in data["tweet"]:
                    return data["tweet"]["text"]
            if "text" in data:
                return data["text"]

        return None
    except Exception:
        return None


def translate_to_english(text: str) -> str:
    """
    Only call OpenAI if language != English.
    Fail-safe: returns original text on any error.
    """
    try:
        lang = detect(text)
    except Exception:
        return text

    if lang == "en":
        return text

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Translate this text to English. Only output the translation.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text


def generate_comments(tweet_text: str):
    """
    Generate two short, humanlike comments.
    Returns list of 2 strings.
    """
    prompt = f"""
You are CrownTALK 👑 — a humanlike Twitter comment generator.

Follow these strict rules:
- Read the tweet context and write two natural comments.
- 5–12 words each.
- No punctuation at the end.
- No emojis, no hashtags.
- Use natural slang where it fits (tbh, fr, ngl, lowkey, btw, kinda, rn, asap, w man, etc).
- Avoid ALL blacklist words/phrases from the style guide.
- Avoid overused stuff like "finally", "curious", "excited", "love to", "hit different".
- Comments must sound like two different humans.
- No identical openings.
- Must be based on the actual tweet meaning.
- Output EXACTLY two lines, each line = one comment, no labels.

STYLE GUIDE:
{STYLE_GUIDE}

Tweet content:
"{tweet_text}"

Return ONLY the two comments, each on its own line.
"""

    for _ in range(3):  # retry loop
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            raw = resp.choices[0].message.content.strip()

            # Split, strip punctuation at end, enforce word count
            lines = [l.strip() for l in raw.splitlines() if l.strip()]

            cleaned = []
            for l in lines:
                l = re.sub(r"[.,!?;:]+$", "", l).strip()
                words = l.split()
                if 5 <= len(words) <= 12:
                    cleaned.append(l)

            # De-duplicate (case-insensitive)
            uniq = []
            for l in cleaned:
                if l.lower() not in [u.lower() for u in uniq]:
                    uniq.append(l)

            if len(uniq) >= 2:
                return uniq[:2]
        except Exception:
            time.sleep(1)

    # fallback if everything fails
    return ["could not generate reply", "generator failed"]


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def index():
    """Serve the frontend (static index.html)."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/comment", methods=["POST"])
def comment_api():
    """
    Accepts JSON: { "tweets": ["url1", "url2", ...] }
    Returns: { "results": [ {comments:[...]} or {error:"..."} ] }
    """
    data = request.get_json(silent=True) or {}
    tweets = data.get("tweets")

    if not isinstance(tweets, list) or not tweets:
        return jsonify({"error": "No tweets provided", "results": []}), 400

    results = []

    for url in tweets:
        url = (url or "").strip()
        if not url:
            results.append({"error": "Empty URL"})
            continue

        tweet_text = fetch_tweet_text(url)
        if not tweet_text:
            results.append(
                {"error": "Could not fetch this tweet (private or deleted)"}
            )
            continue

        tweet_text_en = translate_to_english(tweet_text)
        comments = generate_comments(tweet_text_en)
        results.append({"comments": comments})

    return jsonify({"results": results})


@app.errorhandler(404)
def not_found(_e):
    """Send index.html for any unknown route (for safety)."""
    return send_from_directory(app.static_folder, "index.html")


# -------------------------------------------------
# Local dev
# -------------------------------------------------
if __name__ == "__main__":
    # For local testing only. On Koyeb, gunicorn will run this.
    app.run(host="0.0.0.0", port=8000)
