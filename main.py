import requests
import threading
import time
import re
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__, template_folder="templates", static_folder="static")
client = OpenAI()

# ---------------------------------------------------------
# 🔥 KEEP SERVER ALIVE (Render-friendly version)
# ---------------------------------------------------------
def keep_alive():
    while True:
        try:
            requests.get("https://flask-twitter-api.onrender.com/")
        except:
            pass
        time.sleep(600)  # every 10 minutes

threading.Thread(target=keep_alive, daemon=True).start()


# ---------------------------------------------------------
# 🔥 LOAD STYLE GUIDE
# ---------------------------------------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read()


# ---------------------------------------------------------
# 🔥 FETCH TWEET TEXT (VX API)
# ---------------------------------------------------------
def fetch_tweet_text(url):
    clean = url.replace("https://", "").replace("http://", "")
    api = f"https://api.vxtwitter.com/{clean}"

    try:
        r = requests.get(api, timeout=10)
        data = r.json()

        if "tweet" in data and "text" in data["tweet"]:
            return data["tweet"]["text"]

        if "text" in data:
            return data["text"]

        return None
    except:
        return None


# ---------------------------------------------------------
# 🔥 GENERATE COMMENTS (WITH 429-PROOF RETRY)
# ---------------------------------------------------------
def crown_comment_gen(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — generate **TWO** human-like X/Twitter comments.

RULES:
- 5–12 words each
- No punctuation at the end
- No emojis, no hashtags
- No repeated patterns
- No hype words like: finally, excited, hit different, love to, curious, amazing, game changer
- No AI tone — must feel human
- Use slang naturally (tbh, fr, ngl, lowkey, kinda, btw, rn)
- Two DIFFERENT tones
- Both lines plain text only

STYLE GUIDE:
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Return ONLY two lines, no labels.
"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=80
            )

            raw = resp.choices[0].message.content.strip()
            lines = raw.split("\n")
            lines = [l.strip() for l in lines if l.strip()]

            # cleanup
            final = []
            for line in lines:
                line = re.sub(r"[.,!?;:]+$", "", line)
                if 5 <= len(line.split()) <= 12:
                    final.append(line)

            if len(final) >= 2:
                return final[:2]

        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(4)
                continue

            return None

    return None


# ---------------------------------------------------------
# 🔥 HOME PAGE
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------
# 🔥 COMMENT API
# ---------------------------------------------------------
@app.route("/comment", methods=["POST"])
def comment_api():
    try:
        urls = request.json.get("tweets", [])

        if not urls:
            return jsonify({"error": "No tweet URLs provided."}), 400

        # Clean URLs
        cleaned = []
        for u in urls:
            u = u.strip()
            u = re.sub(r"\?.*$", "", u)
            if u and u not in cleaned:
                cleaned.append(u)

        results = []
        failed = []

        batch_size = 2
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]

            for url in batch:
                text = fetch_tweet_text(url)
                if not text:
                    failed.append(url)
                    continue

                comments = crown_comment_gen(text)
                if not comments:
                    failed.append(url)
                    continue

                results.append({
                    "url": url,
                    "comments": comments
                })

            # small spacing delay (safe for free tier)
            time.sleep(4)

        return jsonify({
            "success": True,
            "results": results,
            "failed": failed
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------
# 🔥 RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
