import os
import re
import time
import threading
import requests
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==============================================
# REMOVE Render's injected proxy env vars
# ==============================================
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy"
]:
    os.environ.pop(k, None)

session = requests.Session()
session.trust_env = False  # ensures proxy vars are ignored

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================================
# URL CLEANING + PARSING
# ==============================================
def clean_url(url: str) -> str:
    """Normalize tweet URL, strip params, convert x.com -> twitter.com."""
    if not isinstance(url, str):
        return ""
    url = url.strip().split("?", 1)[0]

    # Normalize host
    url = url.replace("://x.com", "://twitter.com")
    url = url.replace("://www.x.com", "://twitter.com")

    # Ensure https://
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url

def extract_username_and_id(url: str):
    """
    Expect: https://twitter.com/<user>/status/<id>
    Returns (username, tweetID) or (None, None)
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/").split("/")
        if len(path) < 3:
            return None, None
        if path[1] != "status":
            return None, None
        username = path[0]
        tweet_id = path[2]
        return username, tweet_id
    except:
        return None, None

# ==============================================
# FETCH TWEET TEXT FROM VXTwitter
# ==============================================
def fetch_tweet_text(tweet_url: str):
    """Call VXTwitter API and extract `.text` field."""
    norm = clean_url(tweet_url)
    user, tid = extract_username_and_id(norm)

    if not user or not tid:
        return None, f"Invalid tweet URL structure: {tweet_url}"

    vx_api = f"https://api.vxtwitter.com/{user}/status/{tid}"

    headers = {
        "User-Agent": "CrownTALK/1.0 (https://crowntalk.netlify.app)",
        "Accept": "application/json",
    }

    last_error = None
    for attempt in range(3):
        try:
            r = session.get(vx_api, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                text = data.get("text") or data.get("tweet", {}).get("text")
                if text and isinstance(text, str):
                    return text.strip(), None
                last_error = "VX returned JSON but no text field"
            else:
                last_error = f"VX HTTP {r.status_code}: {r.text[:200]}"
        except Exception as ex:
            last_error = f"VX exception: {ex}"

        time.sleep(min(2 ** attempt, 4))  # backoff

    return None, last_error or "Unknown VX error"

# ==============================================
# OPENAI RAW REST CALL
# ==============================================
def generate_comment(tweet_text: str):
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY not configured on server"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You generate short, witty Twitter/X replies."},
            {"role": "user", "content": f"Tweet: {tweet_text}\n\nWrite a concise reply (max 25 words):"}
        ],
        "max_tokens": 80,
        "temperature": 0.7
    }

    last_error = None
    for attempt in range(4):
        try:
            r = session.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                j = r.json()
                msg = j["choices"][0]["message"]["content"].strip()
                return msg, None

            # Retry OpenAI soft errors
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"OpenAI {r.status_code}: {r.text[:200]}"
                time.sleep(min(2 ** attempt, 4))
                continue

            # Hard error
            try:
                err_msg = r.json().get("error", {}).get("message")
            except:
                err_msg = r.text[:200]
            return None, f"OpenAI {r.status_code}: {err_msg}"

        except Exception as ex:
            last_error = f"OpenAI exception: {ex}"
            time.sleep(min(2 ** attempt, 4))

    return None, last_error or "OpenAI request failed after retries"

# ==============================================
# PROCESS BATCH OF 2 URLS
# ==============================================
def process_batch(urls):
    results = []
    failed = []

    for url in urls:
        cleaned = clean_url(url)

        tweet_text, err = fetch_tweet_text(cleaned)
        if not tweet_text:
            failed.append({"url": cleaned, "reason": err})
            continue

        comment, err2 = generate_comment(tweet_text)
        if not comment:
            failed.append({"url": cleaned, "reason": err2})
            continue

        results.append({
            "url": cleaned,
            "tweet": tweet_text,
            "comment": comment
        })

    return results, failed

# ==============================================
# KEEP-ALIVE (PREVENT RENDER SLEEP)
# ==============================================
def keep_alive():
    target = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SELF_URL")
    if not target:
        return
    time.sleep(8)
    while True:
        try:
            session.get(target, timeout=10)
        except:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()

# ==============================================
# ROUTES
# ==============================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CrownTALK backend running"})

@app.route("/comment", methods=["POST"])
def comment():
    try:
        body = request.get_json(silent=True) or {}
        urls = body.get("urls")

        if not isinstance(urls, list) or not urls:
            return jsonify({"error": "Send JSON: {\"urls\": [\"url1\", ...]}"}), 400

        # Deduplicate while preserving order
        seen = set()
        cleaned = []
        for u in urls:
            cu = clean_url(u)
            if cu and cu not in seen:
                cleaned.append(cu)
                seen.add(cu)

        all_results = []
        all_failed = []

        # Process in batches of 2
        for i in range(0, len(cleaned), 2):
            batch = cleaned[i:i+2]
            r, f = process_batch(batch)
            all_results.extend(r)
            all_failed.extend(f)

        return jsonify({
            "results": all_results,
            "failed": all_failed
        })

    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# ==============================================
# RUN (development)
# ==============================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
