import os
import time
import re
from functools import wraps
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------
# Rate Limiting (Lightweight for free tier)
# -------------------------------------------------
RATE_LIMIT_WINDOW = 10   # seconds
MAX_REQUESTS = 5
recent_requests = {}

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr
        now = time.time()

        recent_requests.setdefault(ip, [])
        recent_requests[ip] = [
            t for t in recent_requests[ip]
            if now - t < RATE_LIMIT_WINDOW
        ]

        if len(recent_requests[ip]) >= MAX_REQUESTS:
            return jsonify({"error": "Slow down — rate limit reached"}), 429

        recent_requests[ip].append(now)
        return func(*args, **kwargs)
    return wrapper

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def clean_link(url):
    return url.split("?")[0].strip()

def chunk_list(lst, size=1):  # batch size = 1 for Koyeb stability
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def load_style_guide():
    with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
        return f.read()

def generate_comments(tweet_url, style_guide):
    prompt = f"""
Follow these rules exactly:

{style_guide}

Generate exactly 2 short human-like replies (7–10 words).
No punctuation, no emojis, no hashtags.
Do not repeat structure between replies.
Replies must sound natural and different.

Tweet: {tweet_url}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.65,
    )

    # Clean output
    text = response.choices[0].message.content.strip()
    lines = [l.strip("-• ").strip() for l in text.split("\n") if l.strip()]
    lines = [l for l in lines if len(l.split()) >= 4]

    return lines[:2]

# -------------------------------------------------
# Routes
# -------------------------------------------------

# HEALTH CHECK route — passes instantly on deploy
@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
@rate_limit
def process():
    raw_links = request.form.get("links", "").strip()

    if not raw_links:
        return jsonify({"error": "No links provided"}), 400

    links = list({clean_link(x) for x in raw_links.splitlines() if x.strip()})
    style_guide = load_style_guide()
    batches = list(chunk_list(links, size=1))  # small batches = safe

    results = {}
    failed = []

    for idx, batch in enumerate(batches, start=1):
        batch_key = f"batch_{idx}"
        batch_results = {}

        for url in batch:
            try:
                comments = generate_comments(url, style_guide)
                batch_results[url] = comments
            except Exception as e:
                failed.append(url)
                batch_results[url] = ["Generation failed", str(e)]

        results[batch_key] = {
            "status": "success" if len(batch_results) else "failed",
            "links": batch_results
        }

        # tiny sleep to avoid overloading Koyeb
        time.sleep(0.1)

    return jsonify({
        "results": results,
        "failed": failed,
        "total_batches": len(batches),
    })


# -------------------------------------------------
# Koyeb start
# -------------------------------------------------
if __name__ == "__main__":
    # LOCAL run
    app.run(host="0.0.0.0", port=8080, debug=False)
