from flask import Flask, request, jsonify
import threading
import requests
import time
import re
import random

app = Flask(__name__)

# ---------------------------------------------------------
# Manual CORS (no external dependency needed)
# ---------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"]
    return response


# ---------------------------------------------------------
# URL Cleaner
# ---------------------------------------------------------
def clean_url(url):
    if not isinstance(url, str):
        return ""

    url = url.strip()

    # Remove numbering like "11. https://..."
    url = re.sub(r"^\d+\.\s*", "", url)

    # Remove ?query parameters
    url = url.split("?")[0]

    return url


# ---------------------------------------------------------
# Offline Comment Generator (Stable)
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
        parts = random.sample(synonyms, k=1) + random.sample(phrase_bank, k=1)
        combined = " ".join(parts)

        # Remove banned words
        for bad in banned_words:
            combined = combined.replace(bad, "")

        # Shuffle for randomness
        words = combined.split()
        random.shuffle(words)

        # Enforce 5–12 words
        words = words[:random.randint(5, 12)]

        return " ".join(words)

    except:
        # Emergency fallback (never fail)
        return "kinda makes sense tbh seeing this now"


def generate_two_comments(text):
    c1 = generate_comment(text)
    c2 = generate_comment(text)

    # Ensure difference
    tries = 0
    while c2 == c1 and tries < 5:
        c2 = generate_comment(text)
        tries += 1

    return [c1, c2]


# ---------------------------------------------------------
# VXTwitter Fetcher (Fully Patched)
# ---------------------------------------------------------
def fetch_tweet_text(url):
    try:
        # Parse host/path from URL
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

                # --- NEW VXTwitter formats ---
                if "text" in data and isinstance(data["text"], str):
                    return data["text"], None

                if "full_text" in data and isinstance(data["full_text"], str):
                    return data["full_text"], None

                # --- Old format ---
                if "tweet" in data:
                    tweet_obj = data["tweet"]

                    if "text" in tweet_obj:
                        return tweet_obj["text"], None

                    if "full_text" in tweet_obj:
                        return tweet_obj["full_text"], None

                if "error" in data:
                    return None, data["error"]

            except:
                pass

            time.sleep(1)

        return None, "Tweet text not found"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Render Keep-Alive Thread
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
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ---------------------------------------------------------
# MAIN ENTRY (runs only on Render)
# ---------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
