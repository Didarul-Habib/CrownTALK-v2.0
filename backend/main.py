import os
import json
import time
import threading
import requests
from flask import Flask, request, jsonify

# ================================
# FIX RENDER PROXY ISSUE
# ================================
# Render injects proxy env vars → must REMOVE them
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if key in os.environ:
        del os.environ[key]

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY missing!")

# ================================
# CLEAN TWEET URL
# ================================
def clean_url(url: str) -> str:
    return url.split("?")[0].strip()

# ================================
# CALL VX TWEET API
# ================================
def fetch_tweet_text(tweet_url):
    try:
        res = requests.get(f"https://api.vxtwitter.com/{tweet_url}", timeout=10)
        if res.status_code != 200:
            return None

        data = res.json()
        return data.get("text") or data.get("tweet", {}).get("text")
    except:
        return None

# ================================
# OPENAI RAW REST CALL
# ================================
def generate_comment(tweet_text):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You generate short, witty Twitter replies."},
            {"role": "user", "content": f"Tweet: {tweet_text}\n\nWrite a reply:"}
        ],
        "max_tokens": 60,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    for _ in range(4):  # retries
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            time.sleep(1)
        except:
            time.sleep(1)

    return None

# ================================
# PROCESS BATCH OF 2 URLs
# ================================
def process_batch(batch):
    results = []
    failed = []

    for url in batch:
        cleaned = clean_url(url)

        tweet_text = fetch_tweet_text(cleaned)
        if not tweet_text:
            failed.append({"url": cleaned, "reason": "Failed to fetch tweet text"})
            continue

        comment = generate_comment(tweet_text)
        if not comment:
            failed.append({"url": cleaned, "reason": "Failed to generate comment"})
            continue

        results.append({
            "url": cleaned,
            "tweet": tweet_text,
            "comment": comment
        })

    return results, failed

# ================================
# KEEP ALIVE (EVERY 10 MINUTES)
# ================================
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/", timeout=10)
        except:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()

# ================================
# ROUTES
# ================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CrownTALK backend running"})

@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json()
        urls = data.get("urls", [])

        if not urls or not isinstance(urls, list):
            return jsonify({"error": "Invalid URL list"}), 400

        all_results = []
        all_failed = []

        # Batch size = 2
        for i in range(0, len(urls), 2):
            batch = urls[i:i+2]
            results, failed = process_batch(batch)
            all_results.extend(results)
            all_failed.extend(failed)

        return jsonify({
            "results": all_results,
            "failed": all_failed
        })

    except Exception as e:
        return jsonify({"error": "Internal error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
