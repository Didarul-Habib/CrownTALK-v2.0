import os
import re
import time
import threading
from typing import List, Tuple, Optional
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
# Disable proxies (upper/lower)
# -----------------------------
for k in [
    "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","NO_PROXY",
    "http_proxy","https_proxy","all_proxy","no_proxy"
]:
    os.environ.pop(k, None)

session = requests.Session()
session.trust_env = False  # ignore proxy envs
UA_HEADERS = {
    "User-Agent": "CrownTALK/1.0 (+https://crowntalk.netlify.app)",
    "Accept": "application/json",
}

app = Flask(__name__)
CORS(app)  # allow Netlify → Render

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Helpers
# -----------------------------
def clean_url(url: str) -> str:
    """
    1) Trim & strip query params; 2) Normalize x.com -> twitter.com
    """
    s = (url or "").strip()
    s = s.split("?", 1)[0]
    # normalize host
    s = s.replace("://x.com/", "://twitter.com/")
    return s

def chunk2(xs: List[str]):
    for i in range(0, len(xs), 2):
        yield xs[i:i+2]

def backoff(attempt: int):
    time.sleep(min(2 ** attempt, 4))

def extract_tweet_id(u: str) -> Optional[str]:
    m = re.search(r"/status/(\d+)", u)
    return m.group(1) if m else None

# -----------------------------
# VX Twitter fetch (primary)
# -----------------------------
def vx_fetch(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Call https://api.vxtwitter.com/{tweet_url} and extract text.
    Returns (text, error_reason)
    """
    try:
        r = session.get(f"https://api.vxtwitter.com/{url}", headers=UA_HEADERS, timeout=15)
        if r.status_code != 200:
            snippet = (r.text or "")[:120]
            return None, f"VX {r.status_code}: {snippet}"
        j = r.json()
        # common fields observed
        for path in [
            ("text",),
            ("tweet", "text"),
            ("tweet", "full_text"),
            ("tweet", "content"),
        ]:
            d = j
            for p in path:
                d = d.get(p) if isinstance(d, dict) else None
                if d is None:
                    break
            if isinstance(d, str) and d.strip():
                return d.strip(), None
        return None, "VX 200 but no text field"
    except Exception as ex:
        return None, f"VX exception: {ex}"

# -----------------------------
# Twitter CDN embed fetch (fallback)
# -----------------------------
def cdn_fetch(tweet_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    GET https://cdn.syndication.twimg.com/widgets/tweet?id=<id>
    Returns (text, error_reason)
    """
    try:
        url = f"https://cdn.syndication.twimg.com/widgets/tweet?id={tweet_id}"
        r = session.get(url, headers=UA_HEADERS, timeout=15)
        if r.status_code != 200:
            snippet = (r.text or "")[:120]
            return None, f"CDN {r.status_code}: {snippet}"
        j = r.json()
        text = j.get("text") or j.get("full_text") or j.get("renderedText")
        if isinstance(text, str) and text.strip():
            return text.strip(), None
        return None, "CDN 200 but no text field"
    except Exception as ex:
        return None, f"CDN exception: {ex}"

# -----------------------------
# Unified tweet text fetch with fallback
# -----------------------------
def fetch_tweet_text(url: str) -> Tuple[Optional[str], Optional[str]]:
    # normalize URL
    norm = clean_url(url)

    # 1) Try VX
    text, err_vx = vx_fetch(norm)
    if text:
        return text, None

    # 2) Fallback: try Twitter CDN embed
    tid = extract_tweet_id(norm)
    if not tid:
        return None, f"{err_vx or 'VX failed'}; no tweet id in URL"
    text2, err_cdn = cdn_fetch(tid)
    if text2:
        return text2, None

    return None, f"{err_vx or 'VX failed'}; {err_cdn or 'CDN failed'}"

# -----------------------------
# OpenAI (raw REST)
# -----------------------------
def openai_comment(tweet_text: str) -> Tuple[Optional[str], Optional[str]]:
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY is not configured on the server"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You generate short, on-topic Twitter/X replies."},
            {"role": "user", "content": f"Tweet: {tweet_text}\n\nWrite a concise, witty reply (≤25 words):"}
        ],
        "max_tokens": 80,
        "temperature": 0.7,
    }

    for attempt in range(4):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                j = r.json()
                return j["choices"][0]["message"]["content"].strip(), None
            if r.status_code in (429, 500, 502, 503, 504):
                backoff(attempt)
                continue
            try:
                err_msg = r.json().get("error", {}).get("message")
            except Exception:
                err_msg = r.text[:200]
            return None, f"OpenAI {r.status_code}: {err_msg or 'Request failed'}"
        except Exception as ex:
            last = str(ex)
            backoff(attempt)

    return None, "OpenAI request failed after retries"

# -----------------------------
# Batch processing (size=2)
# -----------------------------
def process_batch(urls: List[str]) -> Tuple[list, list]:
    results, failed = [], []
    for raw in urls:
        cleaned = clean_url(raw)

        text, text_err = fetch_tweet_text(cleaned)
        if not text:
            failed.append({"url": cleaned, "reason": text_err or "Failed to fetch tweet text"})
            continue

        comment, err = openai_comment(text)
        if not comment:
            failed.append({"url": cleaned, "reason": err or "Failed to generate comment"})
            continue

        results.append({"url": cleaned, "tweet": text, "comment": comment})
    return results, failed

# -----------------------------
# Keep-alive every 10 minutes
# -----------------------------
def keep_alive():
    base = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SELF_URL")
    if not base:
        return
    time.sleep(8)
    while True:
        try:
            session.get(base, timeout=10)
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CrownTALK backend running"})

@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json(silent=True) or {}
        urls = data.get("urls")
        if not isinstance(urls, list) or not urls:
            return jsonify({"error": "Invalid body. Expect {\"urls\": [\"https://x.com/...\", ...]}"}), 400

        # dedupe while preserving order
        seen, cleaned = set(), []
        for u in urls:
            if isinstance(u, str):
                cu = clean_url(u)
                if cu and cu not in seen:
                    cleaned.append(cu); seen.add(cu)

        all_results, all_failed = [], []
        for batch in chunk2(cleaned):  # size = 2
            r, f = process_batch(batch)
            all_results.extend(r)
            all_failed.extend(f)

        return jsonify({"results": all_results, "failed": all_failed})
    except Exception as e:
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

if __name__ == "__main__":
    # In Render, gunicorn runs this app and binds to $PORT
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
