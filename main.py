import re
import time
import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

# --------------------------------------------------------------------
# Load Comment Style Guide
# --------------------------------------------------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read()


# --------------------------------------------------------------------
# URL NORMALIZER
# --------------------------------------------------------------------
def normalize_url(url):
    if not url.startswith("http"):
        url = "https://" + url

    url = url.replace("mobile.twitter.com", "twitter.com")
    url = url.replace("x.com", "twitter.com")
    url = url.replace("fxtwitter.com", "twitter.com")
    url = url.replace("vx.com", "twitter.com")

    return url


# --------------------------------------------------------------------
# Robust Tweet Fetcher (VX → FX fallback)
# --------------------------------------------------------------------
def fetch_tweet_text(raw_url):
    url = normalize_url(raw_url)

    # Extract path after /twitter.com/
    try:
        path = url.split("twitter.com/")[1]
    except:
        return None

    api_endpoints = [
        "https://api.vxtwitter.com/",
        "https://api.fxtwitter.com/",
    ]

    for endpoint in api_endpoints:
        api_url = endpoint + path
        try:
            r = requests.get(api_url, timeout=10)
            if r.status_code == 200:
                data = r.json()

                # VX & FX structure
                if "tweet" in data and "text" in data["tweet"]:
                    return data["tweet"]["text"]

                if "text" in data:
                    return data["text"]

        except:
            pass

    # Final direct HTML fallback (very high success rate)
    try:
        html = requests.get(url, timeout=10).text
        match = re.search(r'<meta property="og:description" content="(.*?)"', html)
        if match:
            return match.group(1)
    except:
        pass

    return None


# --------------------------------------------------------------------
# Auto-Translate to English
# --------------------------------------------------------------------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate this text to English only."},
                {"role": "user", "content": text}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except:
        return text


# --------------------------------------------------------------------
# Comment Generator
# --------------------------------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — a humanlike comment generator.

Strict rules:
- Two comments only.
- 5–12 words each.
- No punctuation.
- No emojis, no hashtags.
- No repeated patterns or blacklist words.
- Use natural slang (tbh, fr, lowkey, ngl, rn, kinda, btw).
- Must be based on the tweet meaning.
- Each comment must sound like a different human.

STYLE GUIDE:
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Write ONLY two newline-separated comments.
"""

    for _ in range(3):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65
            )
            out = res.choices[0].message.content.strip()
            lines = [l.strip() for l in out.split("\n") if l.strip()]
            lines = [re.sub(r"[.,!?;:]+$", "", l) for l in lines]
            lines = [l for l in lines if 5 <= len(l.split()) <= 12]

            if len(lines) >= 2:
                return lines[:2]
        except:
            time.sleep(1)

    return ["could not generate reply", "generator failed"]


# --------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/comment", methods=["POST"])
def comment_api():
    data = request.json
    if "tweets" not in data:
        return jsonify({"error": "Invalid request"}), 400

    results = []

    for url in data["tweets"]:
        tweet_text = fetch_tweet_text(url)

        if not tweet_text:
            results.append({"error": "Tweet not accessible or URL invalid"})
            continue

        tweet_text_en = translate_to_english(tweet_text)
        comments = generate_comments(tweet_text_en)

        results.append({"comments": comments})

    return jsonify({"results": results})


# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
