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
# CLEAN URL FUNCTION
# Removes tracking params (?s=20 etc)
# ---------------------------------------------
def clean_url(url):
    try:
        url = url.strip()
        url = url.split("?")[0]
        url = url.replace("mobile.", "")
        return url
    except:
        return url


# ---------------------------------------------
# Twxt API • Ultra-stable Tweet Fetcher
# ---------------------------------------------
def fetch_tweet_text(url):

    url = clean_url(url)

    # extract tweet id
    try:
        tid = url.split("/status/")[1].split("/")[0]
    except:
        return None

    api_url = f"https://twxt.nest.rip/api/v2/tweet/{tid}"

    # retry 3×
    for _ in range(3):
        try:
            r = requests.get(api_url, timeout=10)

            if r.status_code != 200:
                time.sleep(1)
                continue

            data = r.json()

            if "text" in data:
                return data["text"]

        except:
            time.sleep(1)

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
                {"role": "system", "content": "Translate this to English only."},
                {"role": "user", "content": text}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except:
        return text



# ---------------------------------------------
# Comment Generator
# ---------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — a humanlike comment generator.

Rules:
• EXACTLY 2 comments.
• 5–12 words each.
• No punctuation.
• No emojis or hashtags.
• No repeated structure.
• Natural slang allowed (fr, ngl, tbh, btw, kinda, lowkey).
• Must understand tweet context.
• Avoid all blacklist words in style guide.
• Two different human voices.

STYLE GUIDE:
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Respond with ONLY two lines. No numbering.
"""

    # retry 3× for stability
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            output = response.choices[0].message.content.strip()
            lines = output.split("\n")

            # cleanup
            cleaned = []
            for l in lines:
                l = re.sub(r"[.,!?;:]+$", "", l).strip()
                if 5 <= len(l.split()) <= 12:
                    cleaned.append(l)

            if len(cleaned) >= 2:
                return cleaned[:2]

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
        return jsonify({"error": "Invalid request"}), 400

    results = []

    for url in data["tweets"]:
        url = clean_url(url)

        tweet_text = fetch_tweet_text(url)

        if not tweet_text:
            results.append({"error": "Could not fetch this tweet (private or deleted)"})
            continue

        # translate
        text_en = translate_to_english(tweet_text)

        # generate comments
        comments = generate_comments(text_en)
        results.append({"comments": comments})

    return jsonify({"results": results})



# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
