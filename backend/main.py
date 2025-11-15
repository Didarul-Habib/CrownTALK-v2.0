import os
import time
import json
import math
import threading
from datetime import datetime
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ──────────────────────────────────────────────────────────────────────────────
# Hard safety: nuke proxies Render sometimes injects
for k in list(os.environ.keys()):
    if k.lower().endswith("_proxy") or k in ("HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")  # must be set in Render
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Tunables
BATCH_SIZE = 2                 # <= keep as requested
RETRY_OPENAI = 2               # gentle retries on 429/network
SLEEP_AFTER_BATCH = 1.0        # light breath between batches
REQ_TIMEOUT = 12               # seconds for web calls
MAX_TWEET_CHARS = 750          # avoid huge prompts

HEADERS_OA = {
    "Authorization": f"Bearer {OPENAI_KEY}" if OPENAI_KEY else "",
    "Content-Type": "application/json",
}

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def clean_url(u: str):
    """Remove numbering, whitespace, & URL params. Return None if empty."""
    if not u:
        return None
    u = u.strip()
    # Handle "1. https://..." style prefixes
    if u and u[0].isdigit() and "." in u[:4]:
        u = u.split(".", 1)[1].strip()
    if "?" in u:
        u = u.split("?", 1)[0]
    return u or None


def vx_url_for(any_twitter_url: str) -> str:
    """Map an X/Twitter URL to the VXTwitter API endpoint."""
    p = urlparse(any_twitter_url)
    host, path = p.netloc, p.path  # e.g., x.com, /user/status/123
    return f"https://api.vxtwitter.com/{host}{path}"


def fetch_tweet_text(url: str):
    """
    Return (text, diag) where text is None if we failed.
    diag is a small dict for optional debugging.
    """
    api_url = vx_url_for(url)
    diag = {"api_url": api_url, "status": None, "ct": None}
    try:
        r = requests.get(api_url, timeout=REQ_TIMEOUT)
        diag["status"] = r.status_code
        diag["ct"] = r.headers.get("content-type", "")
        if r.status_code != 200:
            return None, diag
        if "application/json" not in diag["ct"]:
            return None, diag

        data = r.json()
        if isinstance(data, dict):
            if "text" in data:
                return data["text"], diag
            if "tweet" in data and isinstance(data["tweet"], dict) and "text" in data["tweet"]:
                return data["tweet"]["text"], diag
        return None, diag
    except Exception as e:
        diag["error"] = str(e)
        return None, diag


BLOCKLIST = set([
    # hype / corporate
    "amazing", "awesome", "incredible", "great", "epic", "so good",
    "game changer", "empowering", "transformative", "visionary", "groundbreaking",
    "love to see", "love that", "can’t wait", "cant wait", "excited",
    # cringe
    "literally shaking", "so true bestie", "slay", "ate", "yass", "yasss", "queen",
    # ai giveaway
    "as an ai", "as a language model", "in today’s world", "in this digital age",
    "fascinating perspective",
    # engagement bait
    "thoughts?", "agree?", "anyone else?", "who’s with me?", "who's with me?"
])

def scrub_line(line: str) -> str:
    line = line.strip()
    # strip terminal punctuation
    while line.endswith((".", ",", "!", "?")):
        line = line[:-1]
    return line.strip()

def acceptable_line(line: str) -> bool:
    w = line.split()
    if not (5 <= len(w) <= 12):
        return False
    lw = line.lower()
    for banned in BLOCKLIST:
        if banned in lw:
            return False
    return True


def offline_comments(tweet_text: str):
    """Deterministic local fallback (no network)."""
    txt = (tweet_text or "").strip().replace("\n", " ")
    if len(txt) > 160:
        txt = txt[:160]
    seeds = [
        "lowkey this makes sense ngl",
        "fr this hits different not gonna lie",
        "kinda clean execution can’t even pretend",
        "honestly this goes hard i respect it",
        "no cap this tracks i vibe with it",
        "wild take but i’m here for it",
        "ngl the details carry this a lot",
        "smart move overall i can see it",
    ]
    a = hash(txt) % len(seeds)
    b = (hash(txt[::-1]) + 3) % len(seeds)
    c1 = scrub_line(seeds[a])
    c2 = scrub_line(seeds[b] if b != a else seeds[(b+1) % len(seeds)])
    return [c1, c2]


def openai_comments(tweet_text: str):
    """
    Try OpenAI (raw REST) a few times.
    Returns (comments:list[str], source:str) — source in {"openai","offline"}.
    """
    if not OPENAI_KEY:
        return offline_comments(tweet_text), "offline"

    prompt = f"""
Generate two short humanlike comments.
Rules:
- 5–12 words each
- No emojis
- No hashtags
- No punctuation at the end
- Natural slang allowed (tbh, fr, ngl, lowkey)
- Comments must be different
- Exactly 2 lines
- Avoid hype/buzzwords (amazing, awesome, incredible, game changer, empowering, etc.)

Tweet:
{tweet_text}
""".strip()

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 60,
    }

    for attempt in range(RETRY_OPENAI):
        try:
            r = requests.post(OPENAI_URL, headers=HEADERS_OA, json=payload, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                lines = [scrub_line(x) for x in content.split("\n") if x.strip()]
                valid = []
                for ln in lines:
                    if acceptable_line(ln):
                        valid.append(ln)
                    if len(valid) == 2:
                        return valid, "openai"
                # If format off, just bail to offline
                return offline_comments(tweet_text), "offline"

            if r.status_code == 429:
                time.sleep(2)  # gentle backoff
                continue
            # other errors → fallback
            break
        except Exception:
            time.sleep(1)

    return offline_comments(tweet_text), "offline"


# ──────────────────────────────────────────────────────────────────────────────
# Classic JSON endpoint (kept)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/comment", methods=["POST"])
def comment():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"results": [], "failed": []})

    raw_urls = payload.get("urls", []) or []
    cleaned = []
    for u in raw_urls:
        cu = clean_url(u)
        if cu and cu not in cleaned:
            cleaned.append(cu)

    results = []
    failed = []

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i:i + BATCH_SIZE]
        for url in batch:
            text, vx = fetch_tweet_text(url)
            if not text:
                failed.append({"url": url, "reason": "vx_text_not_found"})
                continue
            text = text.strip()
            if len(text) > MAX_TWEET_CHARS:
                text = text[:MAX_TWEET_CHARS]
            comments, source = openai_comments(text)
            results.append({"url": url, "comments": comments, "source": source})

        time.sleep(SLEEP_AFTER_BATCH)

    return jsonify({"results": results, "failed": failed})


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Streaming endpoint — NDJSON per batch
# ──────────────────────────────────────────────────────────────────────────────
def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"

@app.route("/comment_stream", methods=["POST"])
def comment_stream():
    """Streams newline-delimited JSON (NDJSON) as batches finish."""
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        payload = {}

    raw_urls = payload.get("urls", []) or []
    cleaned = []
    for u in raw_urls:
        cu = clean_url(u)
        if cu and cu not in cleaned:
            cleaned.append(cu)

    total = len(cleaned)
    total_batches = math.ceil(total / BATCH_SIZE) if total else 0

    def generator():
        yield _ndjson({"type": "start", "total": total, "batches": total_batches, "ts": datetime.utcnow().isoformat() + "Z"})
        if total == 0:
            yield _ndjson({"type": "done"})
            return

        batch_index = 0
        for i in range(0, total, BATCH_SIZE):
            batch_index += 1
            batch = cleaned[i:i + BATCH_SIZE]
            # Announce we’re starting a batch
            yield _ndjson({"type": "progress", "stage": "processing", "batch": batch_index, "batches": total_batches})

            batch_results = []
            batch_failed = []
            for url in batch:
                text, vx = fetch_tweet_text(url)
                if not text:
                    batch_failed.append({"url": url, "reason": "vx_text_not_found"})
                    continue
                text = text.strip()
                if len(text) > MAX_TWEET_CHARS:
                    text = text[:MAX_TWEET_CHARS]
                comments, source = openai_comments(text)
                batch_results.append({"url": url, "comments": comments, "source": source})

            yield _ndjson({
                "type": "batch",
                "batch": batch_index,
                "batches": total_batches,
                "results": batch_results,
                "failed": batch_failed
            })

            time.sleep(SLEEP_AFTER_BATCH)

        yield _ndjson({"type": "done"})

    return Response(stream_with_context(generator()),
                    mimetype="application/x-ndjson")


# ──────────────────────────────────────────────────────────────────────────────
# Keep-alive + health
# ──────────────────────────────────────────────────────────────────────────────
def keep_alive():
    url = "https://crowntalk-v2-0.onrender.com/"
    while True:
        try:
            requests.get(url, timeout=5)
        except Exception:
            pass
        time.sleep(600)  # every 10 minutes

threading.Thread(target=keep_alive, daemon=True).start()

@app.route("/")
def home():
    return "OK"

@app.route("/keytest")
def keytest():
    return f"KEY LOADED: {bool(OPENAI_KEY)}"

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat() + "Z"})
