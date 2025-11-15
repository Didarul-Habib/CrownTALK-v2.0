import os
import json
import time
import threading
import requests
import re
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# -------------------------------------
# REMOVE RENDER PROXY VARIABLES
# -------------------------------------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


# -------------------------------------
# CLEAN URL
# -------------------------------------
def clean_url(url):
    if not url:
        return None
    url = url.strip()

    # remove numbering (1., 2., etc)
    url = re.sub(r"^\d+\.\s*", "", url)

    # remove parameters
    url = url.split("?")[0]

    # convert x.com → twitter.com for vxtwitter
    url = url.replace("x.com/", "twitter.com/")

    # ensure correct form
    if "twitter.com" not in url:
        return None
    return url


# -------------------------------------
# FETCH TWEET TEXT FROM VXTWITTER
# -------------------------------------
def fetch_tweet_text(url):
    try:
        parts = url.split("twitter.com/")[1]
        api_url = f"https://api.vxtwitter.com/{parts}"

        r = requests.get(api_url, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        if "text" in data:
            return data["text"]

        if "tweet" in data and "text" in data["tweet"]:
            return data["tweet"]["text"]

        return None
    except:
        return None


# -------------------------------------
# GENERATE COMMENTS (STRICT MODE)
# -------------------------------------
def generate_comments(tweet_text):
    banned_words = [
        "game changer", "finally", "love to see", "amazing", "incredible", "awesome",
        "visionary", "groundbreaking", "transformative", "slay", "ate", "queen",
        "omg", "yass", "bestie", "fascinating perspective",
        "in today’s world", "as an ai"
    ]

    prompt = f"""
Generate two short humanlike comments. STRICT RULES:
- 5–12 words each
- 2 lines, one comment per line
- No emojis
- No punctuation at end
- No exclamation marks
- No hashtags
- Must NOT contain ANY banned words: {", ".join(banned_words)}
- Natural slang allowed (tbh, fr, ngl, lowkey)
- Make them DIFFERENT from each other
- Must be based on this tweet

Tweet:
{tweet_text}
"""

    for _ in range(4):
        try:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.7
            }

            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json=payload,
                timeout=20
            )

            if r.status_code != 200:
                continue

            content = r.json()["choices"][0]["message"]["content"]
            lines = content.strip().split("\n")

            results = []
            for line in lines:
                line = line.strip()
                line = re.sub(r"[.!?]+$", "", line)

                if len(line.split()) < 5 or len(line.split()) > 12:
                    continue

                skip = False
                for bad in banned_words:
                    if bad in line.lower():
                        skip = True
                        break
                if not skip:
                    results.append(line)

                if len(results) == 2:
                    return results

        except:
            continue

    return ["generation failed", "please retry"]


# -------------------------------------
# STREAM ENDPOINT — REAL-TIME BATCHING
# -------------------------------------
@app.route("/stream")
def stream():
    try:
        urls_json = request.args.get("urls", "[]")
        urls = json.loads(urls_json)
    except:
        return "invalid", 400

    cleaned = []
    for u in urls:
        cu = clean_url(u)
        if cu:
            cleaned.append(cu)

    cleaned = list(dict.fromkeys(cleaned))
    total_batches = (len(cleaned) + 1) // 2

    def event_stream():
        for i in range(0, len(cleaned), 2):
            batch = cleaned[i:i+2]
            results = []

            for url in batch:
                text = fetch_tweet_text(url)
                if not text:
                    results.append({"url": url, "comments": ["failed", "no tweet text"]})
                    continue

                comments = generate_comments(text)
                results.append({"url": url, "comments": comments})

            payload = {
                "batch": (i // 2) + 1,
                "total": total_batches,
                "results": results
            }

            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(2)

        yield "data: {\"done\": true}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/")
def home():
    return {"status": "CrownTALK backend running"}


# -------------------------------------
# KEEP ALIVE THREAD
# -------------------------------------
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/")
        except:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()


# -------------------------------------
# RUN
# -------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
