import re
import time
import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect
from openai import OpenAI

app = Flask(__name__, template_folder="templates", static_folder="static")

client = OpenAI()

# ----------------------------------------------------
# Load Style Guide
# ----------------------------------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read().strip()


# ----------------------------------------------------
# TweetAPI — VX style fetcher (Mode C)
# More reliable + stricter fail handling
# ----------------------------------------------------
def fetch_tweet_text(url):
    try:
        # Normalize
        clean = url.replace("x.com", "twitter.com")

        if "twitter.com" not in clean:
            return None

        # Extract ID & build VX API
        path = clean.split("twitter.com/")[-1]
        api_url = f"https://api.vxtwitter.com/{path}"

        r = requests.get(api_url, timeout=12)

        if r.status_code != 200:
            return None

        data = r.json()

        # New VX format
        if "tweet" in data and "text" in data["tweet"]:
            text = data["tweet"]["text"]
            return text.strip()

        return None

    except Exception:
        return None


# ----------------------------------------------------
# Auto Translation (only if needed)
# ----------------------------------------------------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text
    except:
        return text  # if language detection fails

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate to English only. No commentary."},
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return r.choices[0].message.content.strip()
    except:
        return text


# ----------------------------------------------------
# Comment Generator — now safer + cleaner + stricter
# ----------------------------------------------------
def generate_comments(tweet_text):

    prompt = f"""
You are CrownTALK 👑 — a humanlike comment generator.

STRICT RULES:
• Read tweet context and write TWO comments.
• 5–12 words each.
• No punctuation at end.
• No emojis or hashtags.
• No repeated structures.
• No hype words: excited, finally, love to, curious, hit different, etc.
• Use natural slang only: tbh, fr, ngl, lowkey, btw, kinda, rn, etc.
• Each comment must sound like a different human.
• MUST follow the style guide strictly.
• Comments must be based on the tweet meaning.
• No filler lines. Output ONLY two lines.

STYLE GUIDE:
{STYLE_GUIDE}

Tweet text:
"{tweet_text}"
"""

    # Retry logic for stability
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65
            )

            out = r.choices[0].message.content.strip()
            lines = [l.strip() for l in out.split("\n") if l.strip()]

            # Clean punctuation automatically
            cleaned = [re.sub(r"[.,!?;:]+$", "", c).strip() for c in lines]

            # Enforce length rule
            valid = [c for c in cleaned if 5 <= len(c.split()) <= 12]

            if len(valid) >= 2:
                return valid[:2]

        except:
            time.sleep(1)

    # If everything fails → safe fallback
    return [
        "could not craft a natural response rn",
        "generator struggled to process this text"
    ]


# ----------------------------------------------------
# Flask Routes
# ----------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/comment", methods=["POST"])
def comment_api():
    try:
        data = request.json
        tweets = data.get("tweets", [])

        if not isinstance(tweets, list) or len(tweets) == 0:
            return jsonify({"error": "Invalid request format"}), 400

    except:
        return jsonify({"error": "Malformed request"}), 400

    results = []

    for url in tweets:
        # Step 1: Fetch tweet
        text = fetch_tweet_text(url)

        if not text:
            results.append({
                "error": "Could not fetch this tweet (private or deleted)"
            })
            continue

        # Step 2: Translate if needed
        english = translate_to_english(text)

        # Step 3: Generate comments
        comments = generate_comments(english)

        results.append({"comments": comments})

    return jsonify({"results": results})


# ----------------------------------------------------
# Gunicorn-friendly startup
# ----------------------------------------------------
if __name__ == "__main__":
    # Local debug only — Koyeb uses gunicorn
    app.run(host="0.0.0.0", port=8000)
