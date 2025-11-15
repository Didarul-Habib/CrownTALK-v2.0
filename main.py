import re
import time
import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()

# ---------------------------------------------
# Style guide load
# ---------------------------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read()


# ---------------------------------------------
# Fetch tweet using VX-Twitter (API C)
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
# Auto translate if not English
# ---------------------------------------------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English only."},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content.strip()

    except:
        return text


# ---------------------------------------------
# Comment generation with retry (3x)
# ---------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — a humanlike comment generator.

Rules:
- 5–12 words each
- No punctuation
- No emojis, no hashtags
- No repeated patterns
- Must feel human and natural
- Use slang lightly (ngl, fr, kinda, tbh, rn)
- Avoid banned phrases (finally, hit different, excited, love to, etc)
- Two comments only, separate lines

STYLE GUIDE:
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Return EXACTLY two lines.
"""

    for _ in range(3):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65,
            )

            out = res.choices[0].message.content.strip().split("\n")
            clean = [re.sub(r"[.,!?;:]+$", "", x).strip() for x in out]
            clean = [x for x in clean if 5 <= len(x.split()) <= 12]

            if len(clean) >= 2:
                return clean[:2]

        except:
            time.sleep(1.5)

    return ["could not generate reply", "generator failed"]


# ---------------------------------------------
# Routes
# ---------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/comment", methods=["POST"])
def comment_api():
    data = request.json
    links = data.get("tweets", [])

    results = []

    for url in links:
        t = fetch_tweet_text(url)

        if not t:
            results.append({"error": "Could not fetch this tweet (private or deleted)"})
            continue

        t_eng = translate_to_english(t)
        comments = generate_comments(t_eng)

        results.append({"comments": comments})

    return jsonify({"results": results})


# ---------------------------------------------
# Run app
# ---------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
