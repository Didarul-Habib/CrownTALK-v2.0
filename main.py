import os
import time
import re
import threading
import requests
from flask import Flask, request, jsonify
from openai import OpenAI


# ----------------------------------------------------
# REMOVE RENDER PROXY VARIABLES (fixes "proxies" error)
# ----------------------------------------------------
for p in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    if p in os.environ:
        del os.environ[p]


# ----------------------------------------------------
# Render service URL (YOUR backend URL)
# ----------------------------------------------------
RENDER_URL = "https://crowntalk-v2-0.onrender.com"


# ----------------------------------------------------
# Init Flask + OpenAI
# ----------------------------------------------------
app = Flask(__name__)
client = OpenAI()


# ----------------------------------------------------
# KEEP SERVER AWAKE (important on Render free tier)
# ----------------------------------------------------
def keep_alive():
    while True:
        try:
            requests.get(RENDER_URL, timeout=8)
            print("Ping sent to keep server awake.")
        except Exception as e:
            print("Ping failed:", e)
        time.sleep(600)  # every 10 minutes


threading.Thread(target=keep_alive, daemon=True).start()


# ----------------------------------------------------
# Fetch tweet text using vxtwitter API
# ----------------------------------------------------
def get_tweet_text(url):
    try:
        clean = url.replace("https://", "").replace("http://", "")
        api = f"https://api.vxtwitter.com/{clean}"

        res = requests.get(api, timeout=10)
        js = res.json()

        # New VX format: {"tweet": {"text": "...."}}
        if "tweet" in js and "text" in js["tweet"]:
            return js["tweet"]["text"]

        # Older format fallback
        if "text" in js:
            return js["text"]

        return None
    except:
        return None


# ----------------------------------------------------
# Generate 2 comments using OpenAI (with retry)
# ----------------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
Generate two natural human comments based on this tweet.

Rules:
- 5–12 words each
- no emojis
- no hashtags
- no punctuation at end
- casual human tone, but not cringe
- each comment on a new line
- comments must be different

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

            out_raw = resp.choices[0].message.content.strip()
            lines = [l.strip() for l in out_raw.split("\n") if l.strip()]

            # cleanup: remove ending punctuation
            cleaned = []
            for c in lines:
                c = re.sub(r"[.,!?;:]+$", "", c).strip()
                if 5 <= len(c.split()) <= 12:
                    cleaned.append(c)

            if len(cleaned) >= 2:
                return cleaned[:2]

        except Exception as e:
            print(f"AI error attempt {attempt+1}:", e)
            time.sleep(1.5)

    return ["generation failed", "please retry"]


# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.route("/")
def home():
    return jsonify({"status": "CrownTALK backend running"})


@app.route("/comment", methods=["POST"])
def comment_api():
    body = request.json
    urls = body.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    # Clean up URLs
    clean_urls = []
    for u in urls:
        u = u.strip()
        u = re.sub(r"\?.*$", "", u)  # remove ? & query
        if u not in clean_urls:
            clean_urls.append(u)

    results = []
    failed = []

    # Batch of 2
    for i in range(0, len(clean_urls), 2):
        batch = clean_urls[i:i + 2]
        print("Processing batch:", batch)

        for url in batch:
            txt = get_tweet_text(url)
            if not txt:
                failed.append(url)
                continue

            comments = generate_comments(txt)
            results.append({
                "url": url,
                "comments": comments,
            })

        time.sleep(2)  # small delay to avoid API spam

    return jsonify({
        "results": results,
        "failed": failed
    })


# ----------------------------------------------------
# Run server
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
