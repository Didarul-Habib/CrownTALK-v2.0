import os
import time
import re
import requests
import threading
from flask import Flask, request, jsonify
from openai import OpenAI


# ----------------------------------------------------
# 🔥 FIX RENDER PROXY BUG (DO NOT REMOVE)
# ----------------------------------------------------
for p in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(p, None)


# ----------------------------------------------------
# Init
# ----------------------------------------------------
app = Flask(__name__)
client = OpenAI()


# ----------------------------------------------------
# Keep-alive Ping (to prevent Render sleep)
# ----------------------------------------------------
def keep_awake():
    while True:
        try:
            requests.get("https://your-render-url.onrender.com")
        except:
            pass
        time.sleep(60 * 5)  # ping every 5 min


threading.Thread(target=keep_awake, daemon=True).start()


# ----------------------------------------------------
# Fetch tweet text (VX API)
# ----------------------------------------------------
def get_tweet_text(url):
    try:
        clean = url.replace("https://", "").replace("http://", "")
        api = f"https://api.vxtwitter.com/{clean}"

        r = requests.get(api, timeout=10)
        data = r.json()

        if "tweet" in data and "text" in data["tweet"]:
            return data["tweet"]["text"]

        return None
    except:
        return None


# ----------------------------------------------------
# Generate comments (with retry)
# ----------------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — generate TWO short human-like comments.

Rules:
- Based on the tweet context
- 5–12 words
- NO punctuation at the end
- NO emojis, no hashtags
- Avoid repetitive patterns
- Use natural slang: tbh, fr, ngl, lowkey, btw, kinda, rn, etc
- Each line = one comment
- EXACTLY 2 lines

Tweet:
{tweet_text}
"""

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.65,
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )

            out = resp.choices[0].message.content.strip().split("\n")
            out = [re.sub(r"[.,!?;:]+$", "", c).strip() for c in out]
            out = [c for c in out if 5 <= len(c.split()) <= 12]

            if len(out) >= 2:
                return out[:2]

        except Exception as e:
            print("AI error:", e)
            time.sleep(2)

    return ["comment generation failed", "please retry"]


# ----------------------------------------------------
# Homepage
# ----------------------------------------------------
@app.route("/")
def home():
    return jsonify({"status": "CrownTALK backend running"})


# ----------------------------------------------------
# Comment API
# ----------------------------------------------------
@app.route("/comment", methods=["POST"])
def comment():
    body = request.json
    urls = body.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    # Clean URLs
    clean_urls = []
    for u in urls:
        u = u.strip()
        u = re.sub(r"\?.*$", "", u)
        if u not in clean_urls:
            clean_urls.append(u)

    results = []
    failed = []

    # batch process — 2 per batch
    for i in range(0, len(clean_urls), 2):
        batch = clean_urls[i:i + 2]

        for url in batch:
            txt = get_tweet_text(url)
            if not txt:
                failed.append(url)
                continue

            comments = generate_comments(txt)
            results.append({
                "url": url,
                "comments": comments
            })

        time.sleep(3)  # safe spacing

    return jsonify({
        "results": results,
        "failed": failed
    })


# ----------------------------------------------------
# Run
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
