import re
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_url_path="", static_folder="static")


# -------------------------------------------------
# CLEAN & NORMALIZE TWITTER URLs
# -------------------------------------------------
def clean_url(url):
    if not url:
        return None

    # Remove spaces
    url = url.strip()

    # Remove index numbers like "1. https://"
    url = re.sub(r"^\d+\.\s*", "", url)

    # Remove tracking junk
    url = re.sub(r"\?.*$", "", url)

    # Force standard x.com format
    replacements = [
        ("mobile.twitter.com", "x.com"),
        ("twitter.com", "x.com"),
        ("m.twitter.com", "x.com"),
        ("www.twitter.com", "x.com")
    ]

    for old, new in replacements:
        url = url.replace(old, new)

    return url


# -------------------------------------------------
# FETCH SYSTEM: TRY VX → THEN FX → THEN FAIL
# -------------------------------------------------
def fetch_from_vx(tweet_url):
    try:
        api_url = tweet_url.replace("x.com", "api.vxtwitter.com")
        res = requests.get(api_url, timeout=7)

        if res.status_code != 200:
            return None

        data = res.json()

        # VX format changed many times — we normalize it always
        tweet_text = (
            data.get("text") or
            data.get("tweet", {}).get("text") or
            None
        )

        if not tweet_text:
            return None

        return {
            "text": tweet_text,
            "author": data.get("user_name") or data.get("user", {}).get("screen_name"),
            "id": data.get("id") or data.get("tweet", {}).get("id")
        }

    except:
        return None


def fetch_from_fx(tweet_url):
    try:
        api_url = tweet_url.replace("x.com", "api.fxtwitter.com")
        res = requests.get(api_url, timeout=7)

        if res.status_code != 200:
            return None

        j = res.json()
        tw = j.get("tweet") or {}

        tweet_text = tw.get("text")
        if not tweet_text:
            return None

        return {
            "text": tweet_text,
            "author": tw.get("author", {}).get("screen_name"),
            "id": tw.get("id")
        }

    except:
        return None


# -------------------------------------------------
# MAIN FETCH CONTROLLER
# -------------------------------------------------
def fetch_tweet(tweet_url):
    tweet_url = clean_url(tweet_url)

    if not tweet_url:
        return {"error": "Invalid URL"}

    # 1️⃣ Try VX first
    data = fetch_from_vx(tweet_url)
    if data:
        return {"ok": True, "data": data}

    # 2️⃣ Try FX fallback
    data = fetch_from_fx(tweet_url)
    if data:
        return {"ok": True, "data": data}

    # 3️⃣ Both failed
    return {"error": "Could not fetch tweet (private / deleted / API blocked)"}


# -------------------------------------------------
# API ROUTE
# -------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def generate():
    content = request.json
    links = content.get("links", [])

    results = []

    for link in links:
        cleaned = clean_url(link)
        fetched = fetch_tweet(cleaned)
        results.append({
            "url": cleaned,
            "result": fetched
        })

    return jsonify({"success": True, "results": results})


# -------------------------------------------------
# SERVE FRONTEND (static/index.html)
# -------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
