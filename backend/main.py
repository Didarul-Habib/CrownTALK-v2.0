import os
import time
import threading
import requests
from flask import Flask, request, jsonify
from urllib.parse import urlparse

app = Flask(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# =====================================================
# Clean URL → remove tracking params, dupes, numbering
# =====================================================
def clean_url(u):
    u = u.strip()

    # Remove numbering: "1. http..." → "http..."
    if u[0].isdigit() and "." in u[:4]:
        u = u.split(".", 1)[1].strip()

    # Strip trailing params
    if "?" in u:
        u = u.split("?")[0]

    return u


# =====================================================
# Fetch tweet text using VXTwitter API
# =====================================================
def fetch_tweet_text(url):
    try:
        parsed = urlparse(url)
        path = parsed.path  # /user/status/123
        clean = f"https://api.vxtwitter.com{path}"

        r = requests.get(clean, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        return data.get("text") or data.get("tweet", {}).get("text")

    except:
        return None


# =====================================================
# Generate comments using OpenAI (Fast + Strict mode)
# =====================================================
def generate_comments(tweet_text):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Generate two short humanlike comments.
Rules:
- 5–12 words each
- No emojis
- No hashtags
- No punctuation at the end
- Natural slang allowed
- Comments must be different
- Exactly 2 lines
- Avoid hype/buzzwords (amazing, awesome, incredible, game changer, empowering, etc.)

Tweet:
{tweet_text}
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 60
    }

    max_retries = 2  # Option A
    for attempt in range(max_retries):

        r = requests.post(url, headers=headers, json=payload)

        # Success
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            lines = [l.strip() for l in content.split("\n") if l.strip()]

            cleaned = []
            for line in lines:
                # Remove punctuation at end
                while line.endswith((".", "!", "?", ",")):
                    line = line[:-1]

                words = line.split()
                if 5 <= len(words) <= 12:
                    cleaned.append(line)

                if len(cleaned) == 2:
                    return cleaned

            # If AI didn't produce 2 valid lines → fallback
            break

        # Rate limit → wait + retry
        if r.status_code == 429:
            time.sleep(2)
            continue

        # Any other error → skip fast
        break

    # Fallback (strict mode)
    return ["generation failed", "try again later"]


# =====================================================
# /comment endpoint — processes in batches of 2
# =====================================================
@app.route("/comment", methods=["POST"])
def comment():
    data = request.get_json()
    urls = data.get("urls", [])

    # Clean + unique
    cleaned = []
    for u in urls:
        cu = clean_url(u)
        if cu and cu not in cleaned:
            cleaned.append(cu)

    results = []
    failed = []

    # Process in batches of 2
    for i in range(0, len(cleaned), 2):
        batch = cleaned[i:i+2]

        for url in batch:
            text = fetch_tweet_text(url)
            if not text:
                failed.append({"url": url, "reason": "Tweet text fetch failed"})
                continue

            comments = generate_comments(text)
            if comments[0] == "generation failed":
                failed.append({"url": url, "reason": "OpenAI generation failed"})

            results.append({
                "url": url,
                "comments": comments
            })

        time.sleep(1)  # Prevent Render overload

    return jsonify({"results": results, "failed": failed})


# =====================================================
# Keep-alive ping (prevents Render sleep)
# =====================================================
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com", timeout=5)
        except:
            pass
        time.sleep(600)  # every 10 min


threading.Thread(target=keep_alive, daemon=True).start()

@app.route("/")
def home():
    return "OK"
