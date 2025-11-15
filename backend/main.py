from flask import Flask, request, jsonify
from flask_cors import CORS
import os, time, threading, requests
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")


# ===========================================
# CLEAN URL
# ===========================================
def clean_url(u):
    if not u:
        return None
    u = u.strip()

    # Remove numbering like "1. https://..."
    if u[0].isdigit() and "." in u[:4]:
        u = u.split(".", 1)[1].strip()

    # Remove URL params
    if "?" in u:
        u = u.split("?")[0]

    return u


# ===========================================
# FETCH TWEET TEXT SAFELY
# ===========================================
def fetch_tweet_text(url):
    try:
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path

        api_url = f"https://api.vxtwitter.com/{host}{path}"

        r = requests.get(api_url, timeout=10)
        if r.status_code != 200:
            return None

        ct = r.headers.get("content-type", "")
        if "application/json" not in ct:
            return None

        try:
            data = r.json()
        except:
            return None

        if isinstance(data, dict):
            if "text" in data:
                return data["text"]
            if "tweet" in data and "text" in data["tweet"]:
                return data["tweet"]["text"]

        return None

    except:
        return None


# ===========================================
# GENERATE COMMENTS (with DEBUG PRINT)
# ===========================================
def generate_comments(tweet_text):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer " + OPENAI_KEY,
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
- Avoid hype/buzzwords

Tweet:
{tweet_text}
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 50
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=8)

        if r.status_code != 200:
            return ["generation failed", "try again later"]

        data = r.json()
        content = data["choices"][0]["message"]["content"]

        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) < 2:
            return ["generation failed", "try again later"]

        cleaned = []
        for line in lines:
            while line.endswith((".", "!", "?", ",")):
                line = line[:-1]
            words = line.split()
            if 5 <= len(words) <= 12:
                cleaned.append(line)
            if len(cleaned) == 2:
                return cleaned

        return ["generation failed", "try again later"]

    except:
        return ["generation failed", "try again later"]

# ===========================================
# COMMENT ENDPOINT
# ===========================================
@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json(force=True)
    except:
        return jsonify({"results": [], "failed": []})

    urls = data.get("urls", [])
    cleaned = []

    for u in urls:
        cu = clean_url(u)
        if cu and cu not in cleaned:
            cleaned.append(cu)

    results = []
    failed = []

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

            results.append({"url": url, "comments": comments})

        time.sleep(0.1)

    return jsonify({"results": results, "failed": failed})


# ===========================================
# KEEP ALIVE
# ===========================================
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except:
            pass
        time.sleep(600)


threading.Thread(target=keep_alive, daemon=True).start()


@app.route("/")
def home():
    return "OK"
