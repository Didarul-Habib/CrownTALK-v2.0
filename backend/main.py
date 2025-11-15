import os
import json
import time
import threading
from typing import List, Tuple
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================
# Block proxy envs (Render injects these)
# ============================================
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if k in os.environ:
        del os.environ[k]

# Use a single session and ensure it ignores env proxies even if any remain
session = requests.Session()
session.trust_env = False  # ignore *_PROXY env vars

app = Flask(__name__)
CORS(app)  # allow all origins so Netlify can POST

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================
# Helpers
# ============================================
def clean_url(url: str) -> str:
    """Remove query params and trim whitespace."""
    return url.split("?")[0].strip()

def chunked(items: List[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i+size]

def _sleep_backoff(attempt: int):
    # 0 -> 1s, 1 -> 2s, 2 -> 4s, 3 -> 4s
    delay = min(2 ** attempt, 4)
    time.sleep(delay)

# ============================================
# VX Twitter fetch
# ============================================
def fetch_tweet_text_once(tweet_url: str, timeout: int = 15) -> str | None:
    """
    Calls https://api.vxtwitter.com/{tweet_url} and extracts the tweet text.
    Handles multiple possible shapes of VX responses.
    """
    try:
        # VX expects the full tweet URL appended to the path
        resp = session.get(f"https://api.vxtwitter.com/{tweet_url}", timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()

        # Common fields seen in VX responses
        tweet = data.get("tweet", {}) if isinstance(data, dict) else {}
        candidates = [
            data.get("text") if isinstance(data, dict) else None,
            tweet.get("text"),
            tweet.get("full_text"),
            tweet.get("content"),
        ]
        for c in candidates:
            if c and isinstance(c, str) and c.strip():
                return c.strip()
        return None
    except Exception:
        return None

def fetch_tweet_text(tweet_url: str) -> str | None:
    """Retry VX up to 3x with backoff."""
    for attempt in range(3):
        text = fetch_tweet_text_once(tweet_url)
        if text:
            return text
        _sleep_backoff(attempt)
    return None

# ============================================
# OpenAI raw REST (chat.completions)
# ============================================
def generate_comment(tweet_text: str) -> str | None:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY or ''}",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You generate short, friendly, on-topic Twitter/X replies."},
            {"role": "user", "content": f"Tweet: {tweet_text}\n\nWrite a concise, witty reply (max ~25 words):"}
        ],
        "max_tokens": 80,
        "temperature": 0.7,
    }

    for attempt in range(4):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=30)
            # 200 → parse; 429/5xx → retry; 4xx others → don't retry
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            elif r.status_code in (429, 500, 502, 503, 504):
                _sleep_backoff(attempt)
                continue
            else:
                # likely 400/401/403 — bad payload or missing API key
                return None
        except Exception:
            _sleep_backoff(attempt)

    return None

# ============================================
# Batch processing
# ============================================
def process_batch(batch: List[str]) -> Tuple[list, list]:
    results = []
    failed = []

    for url in batch:
        cleaned = clean_url(url)

        tweet_text = fetch_tweet_text(cleaned)
        if not tweet_text:
            failed.append({"url": cleaned, "reason": "Failed to fetch tweet text from VX API"})
            continue

        if not OPENAI_API_KEY:
            failed.append({"url": cleaned, "reason": "OPENAI_API_KEY is not configured on the server"})
            continue

        comment = generate_comment(tweet_text)
        if not comment:
            failed.append({"url": cleaned, "reason": "Failed to generate comment"})
            continue

        results.append({
            "url": cleaned,
            "tweet": tweet_text,
            "comment": comment
        })

    return results, failed

# ============================================
# Keep-alive thread (every 10 minutes)
# Uses RENDER_EXTERNAL_URL if available, otherwise SELF_URL if you set it.
# ============================================
def keep_alive():
    target = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SELF_URL")
    if not target:
        # Nothing to ping; skip silently
        return
    # Give the service a few seconds to fully boot before pinging
    time.sleep(10)
    while True:
        try:
            session.get(target, timeout=10)
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()

# ============================================
# Routes
# ============================================
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "CrownTALK backend running"})

@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json(silent=True) or {}
        urls = data.get("urls")

        if not isinstance(urls, list) or len(urls) == 0:
            return jsonify({"error": "Invalid body. Expect {\"urls\": [\"https://x.com/...\", ...]}"}), 400

        # normalize, dedupe while preserving order
        seen = set()
        cleaned_urls = []
        for u in urls:
            if not isinstance(u, str): 
                continue
            cu = clean_url(u)
            if cu and cu not in seen:
                cleaned_urls.append(cu)
                seen.add(cu)

        all_results, all_failed = [], []
        for batch in chunked(cleaned_urls, 2):  # batch size = 2
            r, f = process_batch(batch)
            all_results.extend(r)
            all_failed.extend(f)

        return jsonify({"results": all_results, "failed": all_failed})

    except Exception as e:
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

if __name__ == "__main__":
    # Local dev convenience. In Render, gunicorn will run this app and bind to $PORT.
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
