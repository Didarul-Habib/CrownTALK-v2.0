import os
import re
import time
import threading
import requests
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from flask_cors import CORS

# =========================================================
# DISABLE PROXY ENV VARS (Render injects these)
# =========================================================
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy"
]:
    os.environ.pop(k, None)

session = requests.Session()
session.trust_env = False  # ignore proxy environment


# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================================================
# URL CLEANING / PARSING
# =========================================================
def clean_url(url: str) -> str:
    """Normalize tweet URL and remove parameters."""
    if not isinstance(url, str):
        return ""
    url = url.strip().split("?", 1)[0]
    url = url.replace("://x.com", "://twitter.com")
    url = url.replace("://www.x.com", "://twitter.com")
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


def extract_username_and_id(url: str):
    """
    Expect: https://twitter.com/<user>/status/<id>
    Returns (username, tweet_id)
    """
    try:
        p = urlparse(url)
        parts = p.path.strip("/").split("/")
        if len(parts) < 3:
            return None, None
        if parts[1] != "status":
            return None, None
        return parts[0], parts[2]
    except:
        return None, None


# =========================================================
# VXTwitter API CALL
# =========================================================
def fetch_tweet_text(tweet_url: str):
    """
    Calls: https://api.vxtwitter.com/<username>/status/<tweetID>
    Extracts `.text`
    """
    url = clean_url(tweet_url)
    username, tid = extract_username_and_id(url)

    if not username or not tid:
        return None, f"Invalid tweet URL structure: {url}"

    vx_api = f"https://api.vxtwitter.com/{username}/status/{tid}"

    headers = {
        "User-Agent": "CrownTALK/1.0",
        "Accept": "application/json",
    }

    last_error = None

    for attempt in range(3):
        try:
            r = session.get(vx_api, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                text = data.get("text") or data.get("tweet", {}).get("text")
                if text:
                    return text.strip(), None
                last_error = "VX returned JSON but no text field"
            else:
                last_error = f"VX {r.status_code}: {r.text[:200]}"
        except Exception as ex:
            last_error = f"VX exception: {ex}"

        time.sleep(min(2 ** attempt, 4))

    return None, last_error


# =========================================================
# OPENAI — RAW REST CALL WITH FULL RATE-LIMIT HANDLING
# =========================================================
def generate_comment(tweet_text: str):
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY not set on server"

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You generate short, witty, human Twitter/X replies."},
            {"role": "user", "content": f"Tweet: {tweet_text}\n\nWrite a concise reply (max 25 words):"}
        ],
        "max_tokens": 80,
        "temperature": 0.7
    }

    for attempt in range(6):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=30)

            # SUCCESS
            if r.status_code == 200:
                j = r.json()
                reply = j["choices"][0]["message"]["content"].strip()
                return reply, None

            # ==================================================
            # RATE LIMITING — 429
            # ==================================================
            if r.status_code == 429:
                try:
                    detail = r.json()
                    retry_after = detail.get("error", {}).get("retry_after", 20)
                except:
                    retry_after = 20

                time.sleep(retry_after + 2)  # wait and retry
                continue

            # ==================================================
            # SERVER ERRORS — RETRY
            # ==================================================
            if r.status_code in (500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 4))
                continue

            # ==================================================
            # HARD FAILURE
            # ==================================================
            try:
                error_message = r.json().get("error", {}).get("message")
            except:
                error_message = r.text[:200]

            return None, f"OpenAI {r.status_code}: {error_message}"

        except Exception as ex:
            last_err = f"OpenAI exception: {ex}"
            time.sleep(min(2 ** attempt, 4))

    return None, last_err


# =========================================================
# BATCH PROCESSOR (batch size = 2)
# =========================================================
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

        # avoid slamming OpenAI (3 RPM limit on your org)
        time.sleep(1.2)

        results.append({
            "url": cleaned,
            "tweet": tweet_text,
            "comment": comment
        })

    return results, failed


# =========================================================
# KEEP ALIVE (EVERY 10 MINUTES)
# =========================================================
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


# =========================================================
# ROUTES
# =========================================================
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

        seen = set()
        cleaned = []
        for u in urls:
            cu = clean_url(u)
            if cu and cu not in seen:
                cleaned.append(cu)
                seen.add(cu)

        all_results = []
        all_failed = []

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


# =========================================================
# RUN LOCAL
# =========================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
