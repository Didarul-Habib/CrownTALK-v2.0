from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import requests
import time
import re
import random

app = Flask(__name__)
CORS(app)   # <-- FIXED CORS (critical for frontend)

# ---------------------------------------------------------
# Clean URL function
# ---------------------------------------------------------
def clean_url(url):
    if not isinstance(url, str):
        return ""

    url = url.strip()

    # Remove numbering like "12. https://..."
    url = re.sub(r"^\d+\.\s*", "", url)

    # Remove query params
    url = url.split("?")[0]

    return url


# ---------------------------------------------------------
# Offline Comment Generator (Improved stability)
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
        # Build a random structure
        pick = random.sample(synonyms, k=1) + random.sample(phrase_bank, k=1)
        combined = " ".join(pick)

        # Remove banned words
        for bad in banned_words:
            combined = combined.replace(bad, "")

        # Shuffle words
        parts = combined.split()
        random.shuffle(parts)

        # Keep 5–12 words
        parts = parts[:random.randint(5, 12)]

        return " ".join(parts)

    except Exception:
        # Emergency fallback (never fail comment generation)
        return "kinda makes sense tbh seeing this now"


def generate_two_comments(text):
    c1 = generate_comment(text)
    c2 = generate_comment(text)

    tries = 0
    while c2 == c1 and tries < 5:
        c2 = generate_comment(text)
        tries += 1

    return [c1, c2]


# ---------------------------------------------------------
# Fetch tweet text from VXTwitter
# ---------------------------------------------------------
def fetch_tweet_text(url):
    try:
        match = re.search(r"https?://([^/]+)(/.*)", url)
        if not match:
            return None, "Invalid URL"

        host, path = match.groups()
        api_url = f"https://api.vxtwitter.com/{host}{path}"

        for _ in range(3):  # retry
            try:
                r = requests.get(api_url, timeout=10)
                if r.status_code == 200:
                    data = r.json()

                    # New VXTwitter format safety check
                    if "tweet" in data and "text" in data["tweet"]:
                        return data["tweet"]["text"], None
                    return None, "Tweet text missing"
            except:
                pass

            time.sleep(1)

        return None, "VX API unreachable"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Keep-alive ping (Render)
# ---------------------------------------------------------
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except:
            pass
        time.sleep(600)  # every 10 mins


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def home():
    return jsonify({"status": "ok"})


@app.post("/comment")
def comment():
    """
    Main route — handles batching + comment generation
    """

    # Ensure backend never crashes on malformed input
    try:
        data = request.get_json(silent=True)
        if not data or "urls" not in data:
            return jsonify({"error": "Invalid request"}), 400

        urls = data.get("urls", [])
        if not isinstance(urls, list):
            return jsonify({"error": "URLs must be an array"}), 400

        cleaned = [clean_url(u) for u in urls if u.strip()]

        batches = [cleaned[i:i+2] for i in range(0, len(cleaned), 2)]

        batch_outputs = []

        for batch_index, batch in enumerate(batches):
            batch_result = {
                "results": [],
                "failed": [],
                "batch": batch_index + 1
            }

            for url in batch:
                text, err = fetch_tweet_text(url)

                if err:
                    batch_result["failed"].append({
                        "url": url,
                        "reason": err
                    })
                    continue

                comments = generate_two_comments(text)
                batch_result["results"].append({
                    "url": url,
                    "comments": comments
                })

            batch_outputs.append(batch_result)

        return jsonify({"batches": batch_outputs})

    except Exception as e:
        # Backend will NEVER hang — always responds
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Start keep-alive thread only when running directly
    threading.Thread(target=keep_alive, daemon=True).start()

    app.run(host="0.0.0.0", port=10000)
