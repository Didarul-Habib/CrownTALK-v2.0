import re
import time
import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect
from openai import OpenAI

app = Flask(__name__)

client = OpenAI()

# ---------------------------------------------
# Load Comment Style Guide
# ---------------------------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read()


# ---------------------------------------------
# Clean Tweet URL (remove params + normalize)
# ---------------------------------------------
def clean_url(url):
    url = url.strip()
    url = url.replace("mobile.twitter.com", "twitter.com")
    url = url.replace("x.com", "twitter.com")
    url = url.split("?")[0]
    return url


# ---------------------------------------------
# Multi-Source Tweet Fetcher (4-layer fallback)
# ---------------------------------------------
def fetch_tweet_text(url):

    url = clean_url(url)

    # Extract tweet ID
    try:
        tid = url.split("/status/")[1].split("/")[0]
    except:
        return None

    # Multi-source API attempts
    sources = [
        f"https://twxt.nest.rip/api/v2/tweet/{tid}",
        f"https://nitter.lucabased.xyz/{tid}.json",
        f"https://nitter.mint.lgbt/{tid}.json",
        f"https://api.vxtwitter.com/Twitter/status/{tid}"
    ]

    for api_url in sources:
        try:
            r = requests.get(api_url, timeout=10)

            if r.status_code != 200:
                continue

            data = r.json()

            # Twxt format
            if "text" in data:
                return data["text"]

            # Nitter format
            if "tweet" in data and "text" in data["tweet"]:
                return data["tweet"]["text"]

            # VXTwitter format
            if "tweet" in data and "text" in data["tweet"]:
                return data["tweet"]["text"]

        except:
            continue

    return None  # if all sources fail


# ---------------------------------------------
# Auto-Translate to English
# ---------------------------------------------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate this to English. Only output translation."},
                {"role": "user", "content": text}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except:
        return text


# ---------------------------------------------
# Comment Generator (with retry x3)
# ---------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — a humanlike comment generator.

Strict Rules:
- Read the tweet and write two natural comments.
- 5–12 words each.
- No punctuation at the end.
- No emojis, no hashtags.
- No repeated patterns or hype words (finally, excited, hit different, etc).
- Use natural slang (tbh, fr, ngl, lowkey, btw, kinda).
- Comments must be different from each other.
- Must be based on the tweet meaning.
- Output EXACTLY two lines, nothing else.

STYLE GUIDE:
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Return ONLY the two comments.
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65,
            )

            output = response.choices[0].message.content.strip()
            lines = [l.strip() for l in output.split("\n") if l.strip()]

            # Clean trailing punctuation
            lines = [re.sub(r"[.,!?;:]+$", "", l) for l in lines]

            # Filter valid comments
            valid = [l for l in lines if 5 <= len(l.split()) <= 12]

            if len(valid) >= 2:
                return valid[:2]

        except:
            time.sleep(1)

    return ["could not generate reply", "generator failed"]


# ---------------------------------------------
# ROUTES
# ---------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/comment", methods=["POST"])
def comment_api():

    data = request.json

    if "tweets" not in data or not isinstance(data["tweets"], list):
        return jsonify({"error": "Invalid request format"}), 400

    results = []

    for url in data["tweets"]:
        cleaned = clean_url(url)
        tweet_text = fetch_tweet_text(cleaned)

        if not tweet_text:
            results.append({"error": "Could not fetch this tweet (private or deleted)"})
            continue

        tweet_text_en = translate_to_english(tweet_text)
        comments = generate_comments(tweet_text_en)

        results.append({"comments": comments})

    return jsonify({"results": results})


# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
