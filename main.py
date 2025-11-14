import re
import time
import threading
import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect
from openai import OpenAI

# ---------------------------------------------
# Flask App
# ---------------------------------------------
app = Flask(__name__)
client = OpenAI()

# ---------------------------------------------
# Keep-Alive System (prevents Koyeb sleeping)
# ---------------------------------------------
APP_URL = "https://<YOUR-KOYEB-APP>.koyeb.app"   # <-- CHANGE THIS!

def keep_alive():
    while True:
        try:
            requests.get(APP_URL, timeout=5)
        except:
            pass
        time.sleep(240)  # every 4 minutes

threading.Thread(target=keep_alive, daemon=True).start()


# ---------------------------------------------
# Load Style Guide
# ---------------------------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read()[:2000]  # trimmed for safety


# ---------------------------------------------
# FIXED Tweet Fetcher with 3 fallbacks
# ---------------------------------------------
def fetch_tweet_text(url):

    # normalize link
    url = url.replace("x.com", "twitter.com").split("?")[0]

    # extract something like: /username/status/123
    tail = "/".join(url.split("twitter.com/")[-1].split("/")[:3])

    api_urls = [
        f"https://api.vxtwitter.com/{tail}",          # main
        f"https://api.fxtwitter.com/{tail}",          # fallback
        f"https://api.tweety.ai/{tail}"               # backup
    ]

    for api in api_urls:
        try:
            r = requests.get(api, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "tweet" in data and "text" in data["tweet"]:
                    return data["tweet"]["text"]
        except:
            continue

    return None


# ---------------------------------------------
# Auto-Translate
# ---------------------------------------------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English. Output only translation."},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except:
        return text


# ---------------------------------------------
# Comment Generator (Fully Stabilized)
# ---------------------------------------------
def generate_comments(tweet_text):

    prompt = f"""
Generate EXACTLY two short humanlike comments.

Rules:
- 5–12 words each
- No punctuation at the end
- No emojis or hashtags
- No repeated patterns
- Must follow slang tone: tbh, fr, ngl, lowkey, kinda, btw, rn etc
- Must sound like two different humans
- Must be based ONLY on the tweet’s meaning
- Avoid ALL blacklist phrases from the style guide
- Avoid hype words: finally, excited, love to, hit different

STYLE GUIDE (trimmed):
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Return ONLY two lines, no labels, no numbering.
"""

    for _ in range(3):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65,
            )

            out = res.choices[0].message.content.strip()
            lines = [l.strip() for l in out.split("\n") if l.strip()]

            cleaned = []
            for l in lines:
                l = re.sub(r"[.,!?;:]+$", "", l).strip()
                if 5 <= len(l.split()) <= 12:
                    cleaned.append(l)

            if len(cleaned) >= 2:
                return cleaned[:2]

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

    if "tweets" not in data:
        return jsonify({"error": "Invalid request"}), 400

    results = []

    for url in data["tweets"]:
        txt = fetch_tweet_text(url)

        if not txt:
            results.append({"error": "Could not fetch this tweet (private or deleted)"})
            continue

        txt_en = translate_to_english(txt)
        comments = generate_comments(txt_en)

        results.append({"comments": comments})

    return jsonify({"results": results})


# ---------------------------------------------
# RUN
# ---------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
