import os
import requests
import json
import time
import threading
from flask import Flask, request, jsonify
from urllib.parse import urlparse

app = Flask(__name__)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ---------------------------------------------
# KEEP-ALIVE THREAD (prevents Render sleep)
# ---------------------------------------------
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/")
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()


# ---------------------------------------------
# URL CLEANER
# ---------------------------------------------
def clean_url(url):
    if not url:
        return None

    url = url.strip()

    # Remove numbering like: 1. https://....
    if ". " in url[:4]:
        url = url.split(". ", 1)[1]

    # Remove query params (?ref=xyz)
    if "?" in url:
        url = url.split("?", 1)[0]

    # Enforce twitter/x.com format
    if "twitter.com" in url or "x.com" in url:
        return url

    return None


# ---------------------------------------------
# FETCH TWEET FROM VX API
# ---------------------------------------------
def fetch_tweet_text(url):
    try:
        parsed = urlparse(url)
        path = parsed.path
        user = path.split("/")[1]
        status = path.split("/")[3]

        vx_url = f"https://api.vxtwitter.com/{user}/status/{status}"
        r = requests.get(vx_url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()
        return data.get("text", None)

    except Exception:
        return None


# ---------------------------------------------
# CLEAN GENERATED COMMENT
# ---------------------------------------------
def clean_comment(text):
    if not text:
        return ""

    text = text.strip()

    # Remove punctuation at end
    while len(text) > 0 and text[-1] in "!?.,":
        text = text[:-1]

    # Force boundaries 5–12 words
    words = text.split()
    if len(words) < 5 or len(words) > 12:
        return ""

    return " ".join(words)


# ---------------------------------------------
# GENERATE COMMENTS USING OPENAI
# ---------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
Generate two humanlike comments.
Rules:
- 5–12 words each
- no punctuation at end
- no emojis or hashtags
- natural slang allowed (tbh, fr, ngl, lowkey)
- comments must be different
- based on the tweet
- exactly 2 lines

Tweet:
{tweet_text}
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 50
    }

    max_retries = 2

    for attempt in range(max_retries):
        r = requests.post(OPENAI_URL, headers=headers, json=payload)

        # success
        if r.status_code == 200:
            response = r.json()
            output = response["choices"][0]["message"]["content"].strip().split("\n")
            comments = []

            for line in output:
                cleaned = clean_comment(line)
                if cleaned:
                    comments.append(cleaned)

            # ensure exactly 2
            if len(comments) >= 2:
                return comments[:2]

            # fallback if bad formatting
            break

        # rate limit → wait and retry
        if r.status_code == 429:
            time.sleep(2)
            continue

    # final fallback
    return ["generation failed", "try again later"]


# ---------------------------------------------
# MAIN COMMENT ENDPOINT
# ---------------------------------------------
@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json()
        urls = data.get("urls", [])

        cleaned = []
        for u in urls:
            cu = clean_url(u)
            if cu:
                cleaned.append(cu)

        cleaned = list(dict.fromkeys(cleaned))  # remove duplicates

        results = []
        failed = []

        # Batch of 2
        for i in range(0, len(cleaned), 2):
            batch = cleaned[i:i+2]

            for url in batch:
                tweet_text = fetch_tweet_text(url)

                if not tweet_text:
                    failed.append(url)
                    continue

                comments = generate_comments(tweet_text)

                results.append({
                    "url": url,
                    "comments": comments
                })

            time.sleep(2)  # prevent OpenAI overload

        return jsonify({
            "results": results,
            "failed": failed
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": "server failure"}), 500


@app.route("/")
def home():
    return "CrownTALK backend alive", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
