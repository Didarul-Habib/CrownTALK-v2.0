import os
import time
import re
import threading
import requests
from flask import Flask, request, jsonify

# ----------------------------------------------------
# REMOVE PROXY VARIABLES (Render injects them)
# ----------------------------------------------------
for p in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    if p in os.environ:
        del os.environ[p]

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


# ----------------------------------------------------
# KEEP RENDER AWAKE (EVERY 10 MIN)
# ----------------------------------------------------
def keep_awake():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/")
        except:
            pass
        time.sleep(600)  # 10 minutes


threading.Thread(target=keep_awake, daemon=True).start()


# ----------------------------------------------------
# FETCH TWEET TEXT USING VX API
# ----------------------------------------------------
def get_tweet_text(url):
    try:
        clean = url.replace("https://", "").replace("http://", "")
        api = f"https://api.vxtwitter.com/{clean}"

        r = requests.get(api, timeout=10)
        data = r.json()

        # structure: {"tweet": {"text": "..."}}
        if "tweet" in data and "text" in data["tweet"]:
            return data["tweet"]["text"]

        return None
    except Exception as e:
        print("VX error:", e)
        return None


# ----------------------------------------------------
# GENERATE COMMENTS (OpenAI REST)
# ----------------------------------------------------
def generate_comments(tweet_text):
    prompt = f"""
Generate two humanlike comments.
Rules:
- 5–12 words each
- no punctuation at end
- no emojis or hashtags
- natural slang allowed (tbh, fr, ngl, lowkey)
- comments must be different
- based on the tweet
- exactly 2 lines

Tweet:
{tweet_text}
"""

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.65,
        "max_tokens": 60,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(4):
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=20
            )

            data = r.json()

            if "choices" not in data:
                print("AI retry:", data)
                time.sleep(2)
                continue

            output = data["choices"][0]["message"]["content"]
            lines = [l.strip() for l in output.split("\n") if l.strip()]

            clean_out = []
            for c in lines:
                c = re.sub(r"[.,!?;:]+$", "", c)
                if 5 <= len(c.split()) <= 12:
                    clean_out.append(c)

            if len(clean_out) >= 2:
                return clean_out[:2]

        except Exception as e:
            print("AI error:", e)
            time.sleep(2)

    return ["generation failed", "please retry"]


# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.route("/")
def home():
    return jsonify({"status": "CrownTALK backend running"})


@app.route("/comment", methods=["POST"])
def comment_api():
    try:
        body = request.json
        urls = body.get("urls", [])

        if not urls:
            return jsonify({"error": "No URLs provided"}), 400

        # clean + dedupe
        clean_urls = []
        for u in urls:
            u = re.sub(r"\?.*$", "", u.strip())
            if u not in clean_urls:
                clean_urls.append(u)

        results = []
        failed = []

        # batch of 2
        for i in range(0, len(clean_urls), 2):
            batch = clean_urls[i:i + 2]
            print("Processing batch:", batch)

            for url in batch:
                text = get_tweet_text(url)

                if not text:
                    failed.append(url)
                    continue

                comments = generate_comments(text)

                results.append({
                    "url": url,
                    "comments": comments
                })

            time.sleep(2)

        return jsonify({
            "results": results,
            "failed": failed
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": "Server internal error"}), 500


# ----------------------------------------------------
# RUN
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
