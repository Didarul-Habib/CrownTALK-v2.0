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
# Tweet Fetcher (TweetAPI: C Mode - VX Style)
# ---------------------------------------------
def fetch_tweet_text(url):
    api_url = "https://api.vxtwitter.com/" + url.split("twitter.com/")[-1].split("x.com/")[-1]

    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        if "tweet" in data and "text" in data["tweet"]:
            return data["tweet"]["text"]

        return None
    except:
        return None


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
                {
                    "role": "system",
                    "content": "Translate this text to English. Only output the translation."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except:
        return text  # fallback


# ---------------------------------------------
# Comment Generator
# ---------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — a humanlike comment generator.

Follow these strict rules:
- Read the tweet context and write two natural comments.
- 5–12 words each.
- No punctuation at the end.
- No emojis, hashtags, or repeated patterns.
- Use natural slang (tbh, fr, ngl, lowkey, btw, kinda, etc).
- Avoid ALL blacklist words and phrases from the style guide.
- Comments must sound like different humans.
- No identical openings.
- No hype words like “finally”, “excited”, “love to”, “hit different”, etc.
- Must be based on the actual tweet meaning.
- Output EXACTLY two lines of text with no labels.

STYLE GUIDE:
{STYLE_GUIDE}

Tweet content:
"{tweet_text}"

Return ONLY the two comments. No extra text.
"""

    for _ in range(3):  # retry loop
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65,
            )

            output = response.choices[0].message.content.strip()
            lines = output.split("\n")

            # clean
            lines = [re.sub(r"[.,!?;:]+$", "", l).strip() for l in lines]
            lines = [l for l in lines if len(l.split()) >= 5 and len(l.split()) <= 12]

            if len(lines) >= 2:
                return lines[:2]

        except:
            time.sleep(1)

    return ["could not generate reply", "generator failed to produce output"]


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
        tweet_text = fetch_tweet_text(url)

        if not tweet_text:
            results.append({"error": "Could not fetch this tweet (private or deleted)"})
            continue

        # auto-translate
        tweet_text_en = translate_to_english(tweet_text)

        # generate comments
        comments = generate_comments(tweet_text_en)
        results.append({"comments": comments})

    return jsonify({"results": results})


# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
