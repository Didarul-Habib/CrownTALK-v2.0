import os
import time
from flask import Flask, request, jsonify
from utils import (
    BATCH_SIZE,
    PER_URL_SLEEP_SECONDS,
    generate_two_comments,
)

app = Flask(__name__)

# Provider order env (comma-separated)
DEFAULT_ORDER = "groq,openai,gemini,offline"
ORDER = [p.strip() for p in os.getenv("CROWNTALK_LLM_ORDER", DEFAULT_ORDER).split(",") if p.strip()]

@app.get("/")
def root():
    # Minimal body so client warmup pings are cheap and cache-busting isn’t needed.
    return "ok", 200

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, providers=ORDER), 200

def _process_urls(urls):
    results = []
    failed = []
    for url in urls:
        comments, errors = generate_two_comments(url, ORDER)
        if comments:
            results.append({"url": url, "comments": [{"lang": "en", "text": c} for c in comments]})
        else:
            failed.append({"url": url, "reason": "; ".join(errors) or "Unknown error"})
        time.sleep(PER_URL_SLEEP_SECONDS)
    return results, failed

@app.post("/comment")
def comment():
    data = (request.get_json(silent=True) or {})
    urls = data.get("urls") or []
    if not isinstance(urls, list) or not urls:
        return jsonify(error="Provide a non-empty 'urls' array."), 400

    # clamp batch size a bit for free tiers
    batch = max(1, min(BATCH_SIZE, len(urls)))
    urls = urls[:batch]

    results, failed = _process_urls(urls)
    return jsonify(results=results, failed=failed), 200

@app.post("/reroll")
def reroll():
    data = (request.get_json(silent=True) or {})
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="Provide 'url'."), 400

    comments, errors = generate_two_comments(url, ORDER)
    if not comments:
        return jsonify(error="All upstreams failed.", details=errors), 502

    return jsonify(url=url, comments=[{"lang": "en", "text": c} for c in comments]), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
