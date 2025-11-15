import re
import time
import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

# ---------------------------------------------------------
# Load Style Guide
# ---------------------------------------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read()

# ---------------------------------------------------------
# Fastest Tweet Fetchers (Balanced Mode)
# ---------------------------------------------------------
TIMEOUT = 5  # HARD LIMIT for every tweet source

def try_fetch(url):
    """Protected request with timeout + auto JSON handling"""
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

def fetch_tweet_text(tweet_url):
    """
    Balanced Mode:
    1) Twxt first — fastest, lightweight
    2) VXTwitter fallback — reliable
    """

    # Clean the URL first
    cleaned = (
        tweet_url.replace("https://", "")
                 .replace("http://", "")
                 .replace("x.com/", "")
                 .replace("twitter.com/", "")
    )
    cleaned = cleaned.split("?")[0]  # remove ?s=123

    # Source 1 — twxt API
    twxt_api = f"https://api.twxt.dev/v1/tweet?url=https://x.com/{cleaned}"
    data = try_fetch(twxt_api)
    if data and "text" in data:
        return data["text"]

    # Source 2 — vxtwitter
    vx_api = f"https://api.vxtwitter.com/{cleaned}"
    data = try_fetch(vx_api)
    try:
        if data and "tweet" in data and "text" in data["tweet"]:
            return data["tweet"]["text"]
    except:
        pass

    return None  # final failure

# ---------------------------------------------------------
# Translation (Balanced Mode – only if needed)
# ---------------------------------------------------------
def translate_to_english(text):
    try:
        lang = detect(text)

        # if English, skip translation call
        if lang == "en":
            return text

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Translate this text into English only. Nothing else."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()
    except:
        return text  # fallback


# ---------------------------------------------------------
# Comment Generator (Balanced Mode)
# ---------------------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — generate HUMANLIKE Twitter-style comments.

Rules:
- EXACTLY 2 comments.
- 5–12 words each.
- No punctuation.
- No emojis, no hashtags.
- No repeated phrases.
- Must match the tweet context.
- Use casual slang (tbh, fr, lowkey, kinda).
- Avoid blacklisted words.
- Each line is one comment.

STYLE GUIDE:
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Return ONLY the two comments.
"""

    for _ in range(2):  # balanced retry count (2)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65
            )

            output = response.choices[0].message.content.strip()
            comments = [re.sub(r"[.,!?;:]+$", "", x).strip() for x in output.split("\n")]
            comments = [c for c in comments if 5 <= len(c.split()) <= 12]

            if len(comments) >= 2:
                return comments[:2]

        except:
            time.sleep(0.3)

    return ["could not generate reply", "generator failed"]


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/comment", methods=["POST"])
def comment_api():
    data = request.json

    if "tweets" not in data or not isinstance(data["tweets"], list):
        return jsonify({"error": "Invalid format"}), 400

    results = []

    for url in data["tweets"]:
        text = fetch_tweet_text(url)

        if not text:
            results.append({"error": "Could not fetch this tweet (private or deleted)"})
            continue

        english = translate_to_english(text)
        comments = generate_comments(english)
        results.append({"comments": comments})

    return jsonify({"results": results})


# ---------------------------------------------------------
# RUN (Koyeb)
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
