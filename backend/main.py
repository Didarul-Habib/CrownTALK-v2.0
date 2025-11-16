from flask import Flask, request, jsonify
import threading
import requests
import time
import re
import random

app = Flask(__name__)

# ---------------------------------------------------------
# Manual CORS (safe & error-free)
# ---------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------
# URL Cleaner
# ---------------------------------------------------------
def clean_url(url):
    if not isinstance(url, str):
        return ""

    url = url.strip()

    # Remove "1. https…" style numbering
    url = re.sub(r"^\d+\.\s*", "", url)

    # Strip query params
    url = url.split("?")[0]

    return url


# ---------------------------------------------------------
# Offline Comment Generator
# ---------------------------------------------------------
banned_words = {
    "amazing","awesome","incredible","finally","excited",
    "love this","empowering","game changer","transformative",
    "as an ai","in this digital age","slay","yass","bestie",
    "queen","thoughts","agree","who’s with me","who's with me"
}

synonyms = [
    "wild", "real talk", "lowkey wild", "tbh kinda true",
    "not gonna lie", "fr tho", "kinda true", "makes sense fr",
    "whole situation wild", "not wrong", "fair point ngl",
    "seeing it clearly now", "pretty spot on"
]

phrase_bank = [
    "this got me thinking",
    "never saw it like this before",
    "honestly true sometimes",
    "hard to ignore fr",
    "been seeing this everywhere",
    "cant lie this tracks",
    "i get the point here",
    "lowkey makes sense",
    "not mad at this take"
]

def generate_comment(text):
    try:
        base = random.sample(synonyms, 1) + random.sample(phrase_bank, 1)
        combined = " ".join(base)

        # Remove banned/flagged patterns
        for bad in banned_words:
            combined = combined.replace(bad, "")

        words = combined.split()
        random.shuffle(words)

        # 5–12 words enforced
        words = words[:random.randint(5, 12)]

        return " ".join(words)

    except:
        return "kinda makes sense tbh seeing this now"


def generate_two_comments(text):
    c1 = generate_comment(text)
    c2 = generate_comment(text)

    # Ensure different
    attempts = 0
    while c2 == c1 and attempts < 5:
        c2 = generate_comment(text)
        attempts += 1

    return [c1, c2]


# ---------------------------------------------------------
# VXTwitter Fetcher (Fully Patched)
# ---------------------------------------------------------
def fetch_tweet_text(url):
    try:
        match = re.search(r"https?://([^/]+)(/.*)", url)
        if not match:
            return None, "Invalid URL"

        host, path = match.groups()
        api_url = f"https://api.vxtwitter.com/{host}{path}"

        for _ in range(3):
            try:
                r = requests.get(api_url, timeout=10)
                if r.status_code != 200:
                    time.sleep(1)
                    continue

                data = r.json()

                # --- Format A: { "text": "" }
                if "text" in data and isinstance(data["text"], str):
                    return data["text"], None

                # --- Format B: { "full_text": "" }
                if "full_text" in data and isinstance(data["full_text"], str):
                    return data["full_text"], None

                # --- Format C: { "tweet": { "text": "" } }
                if "tweet" in data:
                    tweet_obj = data["tweet"]

                    if "text" in tweet_obj:
                        return tweet_obj["text"], None

                    if "full_text" in tweet_obj:
                        return tweet_obj["full_text"], None

                # --- Format D: Tweet unavailable
                if "error" in data:
                    return None, data["error"]

            except:
                pass

            time.sleep(1)

        return None, "Tweet text not found"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Keep Alive (Render)
# ---------------------------------------------------------
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except:
            pass
        time.sleep(600)


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/")
def home():
    return jsonify({"status": "ok"})


@app.post("/comment")
def comment():
    try:
        data = request.get_json(silent=True)

        if not data or "urls" not in data:
            return jsonify({"error": "Invalid request"}), 400

        urls = data["urls"]
        cleaned = [clean_url(u) for u in urls if isinstance(u, str) and u.strip()]

        # Batch of 2
        batches = [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]

        out = []

        for i, batch in enumerate(batches):
            batch_info = {
                "batch": i + 1,
                "results": [],
                "failed": []
            }

            for url in batch:
                text, err = fetch_tweet_text(url)

                if err:
                    batch_info["failed"].append({
                        "url": url,
                        "reason": err
                    })
                    continue

                comments = generate_two_comments(text)

                batch_info["results"].append({
                    "url": url,
                    "comments": comments
                })

            out.append(batch_info)

        return jsonify({"batches": out})

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
