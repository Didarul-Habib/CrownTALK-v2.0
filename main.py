import os
import time
import re
from functools import wraps
from flask import Flask, request, render_template, jsonify, abort
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------- Rate Limiting (per-IP, simple) ------------- #
RATE_LIMIT_WINDOW = 10      # seconds
MAX_REQUESTS = 3
recent_requests = {}

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr
        now = time.time()

        if ip not in recent_requests:
            recent_requests[ip] = []

        # purge old entries
        recent_requests[ip] = [
            t for t in recent_requests[ip] if now - t < RATE_LIMIT_WINDOW
        ]

        if len(recent_requests[ip]) >= MAX_REQUESTS:
            return jsonify({
                "error": "Rate limit exceeded. Wait a few seconds and try again."
            }), 429

        recent_requests[ip].append(now)
        return func(*args, **kwargs)
    return wrapper

# ------------ Helpers ------------ #

def clean_link(url):
    return url.split("?")[0].strip()

def chunk_list(lst, size=2):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def load_style_guide():
    with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
        return f.read()

def generate_comments(tweet_url, style_guide):
    prompt = f"""
Follow these rules exactly:

{style_guide}

Generate exactly 2 replies.
Each reply:
• 7–10 words
• no punctuation
• no emojis or hashtags
• no hype language
• must sound natural and not similar to each other

Tweet: {tweet_url}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.65,
    )

    text = response.choices[0].message.content.strip()
    all_lines = [l.strip("-• ").strip() for l in text.split("\n") if l.strip()]
    all_lines = [l for l in all_lines if len(l.split()) >= 4]

    return all_lines[:2]

# ------------ Routes ------------ #

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
    batches = list(chunk_list(links, size=2))

    results = {}
    failed = []

    for i, batch in enumerate(batches, start=1):
        attempts = 0
        batch_key = f"batch_{i}"

        while attempts < 3:
            try:
                batch_results = {}

                for url in batch:
                    comments = generate_comments(url, style_guide)
                    batch_results[url] = comments

                results[batch_key] = {
                    "status": "success",
                    "links": batch_results
                }
                break

            except Exception:
                attempts += 1
                time.sleep(2)

        else:
            failed.extend(batch)
            results[batch_key] = {"status": "failed", "links": batch}

        time.sleep(1)

    return jsonify({
        "results": results,
        "failed": failed,
        "total_batches": len(batches),
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
