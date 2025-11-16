from flask import Flask, request, jsonify
import threading
import requests
import time
import re
import random

app = Flask(__name__)

# ---------------------------------------------------------
# Clean URL function
# ---------------------------------------------------------
def clean_url(url):
    url = url.strip()
    url = re.sub(r"^\d+\.\s*", "", url)
    url = url.split("?")[0]
    return url

# ---------------------------------------------------------
# Offline "Humanlike" Comment Generator
# ---------------------------------------------------------
banned_words = {
    "amazing","awesome","incredible","finally","excited",
    "love this","empowering","game changer","transformative",
    "as an ai","in this digital age","slay","yass","bestie",
    "queen","thoughts","agree","who’s with me","who's with me"
}

synonyms = [
    "wild", "honest take", "real talk", "lowkey wild",
    "not gonna lie", "tbh this hits", "fr tho", "kinda true",
    "whole situation crazy", "makes sense tbh", "not wrong",
    "fair point ngl", "seeing it clearly now", "pretty spot on",
]

phrase_bank = [
    "this got me thinking",
    "never saw it that way before",
    "hard to ignore fr",
    "real ones know this",
    "been seeing this everywhere",
    "honestly kinda true",
    "this take goes hard",
    "wild timing ngl",
    "cant lie this tracks",
    "i get what you're saying",
]

def generate_comment(text):
    words = []

    # Mix random synonyms + phrase parts
    source = random.sample(synonyms, k=2) + random.sample(phrase_bank, k=1)

    combined = " ".join(source)

    # Filter banned words
    for bad in banned_words:
        combined = combined.replace(bad, "")

    # Shuffle
    parts = combined.split()
    random.shuffle(parts)

    # Length constraint
    parts = parts[:random.randint(5, 12)]

    return " ".join(parts)

def generate_two_comments(tweet_text):
    c1 = generate_comment(tweet_text)
    c2 = generate_comment(tweet_text)

    # Ensure distinct comments
    while c2 == c1:
        c2 = generate_comment(tweet_text)

    return [c1, c2]

# ---------------------------------------------------------
# Fetch tweet text from VXTwitter
# ---------------------------------------------------------
def fetch_tweet_text(url):
    try:
        match = re.search(r"https?://([^/]+)(/.*)", url)
        if not match:
            return None, "Invalid URL format"

        host, path = match.groups()
        api_url = f"https://api.vxtwitter.com/{host}{path}"

        for _ in range(3):  # retry
            r = requests.get(api_url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "tweet" in data and "text" in data["tweet"]:
                    return data["tweet"]["text"], None
            time.sleep(1)

        return None, "VX API failed"

    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# Keep-alive pinger (Render)
# ---------------------------------------------------------
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except:
            pass
        time.sleep(600)  # every 10 min

threading.Thread(target=keep_alive, daemon=True).start()

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def home():
    return jsonify({"status": "ok"})

@app.post("/comment")
def comment():
    data = request.get_json()
    urls = data.get("urls", [])

    cleaned = [clean_url(u) for u in urls]
    batches = [cleaned[i:i+2] for i in range(0, len(cleaned), 2)]

    all_results = []
    all_failed = []

    # Streaming per batch
    batch_outputs = []

    for batch_index, batch in enumerate(batches):
        batch_result = {"results": [], "failed": [], "batch": batch_index + 1}
        
        for url in batch:
            text, err = fetch_tweet_text(url)
            if err:
                batch_result["failed"].append({"url": url, "reason": err})
                continue

            comments = generate_two_comments(text)
            batch_result["results"].append({
                "url": url,
                "comments": comments
            })

        batch_outputs.append(batch_result)

    return jsonify({"batches": batch_outputs})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
