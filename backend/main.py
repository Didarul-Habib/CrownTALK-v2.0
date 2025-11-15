import os
import re
import time
import threading
import requests
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS

# =========================================================
# DISABLE PROXIES (REQUIRED FOR RENDER)
# =========================================================
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy"
]:
    os.environ.pop(k, None)

session = requests.Session()
session.trust_env = False  # ignore proxy env


# =========================================================
# FLASK SETUP
# =========================================================
app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================================================
# STRICT BLOCKLIST (NO HYPE / NO CRINGE / NO CORPORATE)
# =========================================================
STRICT_BLOCKLIST = {
    # generic hype / overused
    "amazing", "awesome", "incredible", "great", "so good",
    "epic", "fire", "cool", "nice", "wow", "excited",
    "cant wait", "can’t wait", "love to see", "love that",
    "finally", "game changer", "insane", "crazy",

    # corporate cringe
    "innovation", "innovative", "transformative", "visionary",
    "empowering", "groundbreaking",

    # tiktok slang cringe
    "slay", "yass", "yasss", "queen", "sis", "ate",
    "literally shaking", "bestie",

    # ai giveaways
    "as an ai", "as a language model",
    "in today’s world", "in this digital age",

    # engagement bait
    "thoughts?", "agree?", "anyone else?", "who’s with me"
}


# =========================================================
# URL CLEANING & PARSING (OFFICIAL VX API MODE)
# =========================================================
def clean_url(url: str) -> str:
    """Normalize tweet URL, strip parameters, fix x.com."""
    if not isinstance(url, str):
        return ""
    url = url.strip().split("?", 1)[0]
    url = url.replace("://x.com", "://twitter.com")
    url = url.replace("://www.x.com", "://twitter.com")

    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


def extract_user_id(url: str):
    """
    Expect: https://twitter.com/<user>/status/<id>
    Return: (user, id)
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
# FETCH TWEET TEXT FROM VX TWITTER (OFFICIAL API)
# =========================================================
def fetch_tweet_text(url: str):
    norm = clean_url(url)
    user, tid = extract_user_id(norm)

    if not user or not tid:
        return None, f"Invalid tweet URL: {url}"

    api_url = f"https://api.vxtwitter.com/{user}/status/{tid}"

    headers = {
        "User-Agent": "CrownTALK/1.0",
        "Accept": "application/json"
    }

    last_error = None

    for attempt in range(3):
        try:
            r = session.get(api_url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                text = data.get("text") or data.get("tweet", {}).get("text")
                if text:
                    return text.strip(), None
                last_error = "VX JSON missing .text"
            else:
                last_error = f"VX {r.status_code}: {r.text[:200]}"
        except Exception as ex:
            last_error = f"VX error: {ex}"

        time.sleep(min(2 ** attempt, 4))

    return None, last_error


# =========================================================
# OPENAI RAW REST — WITH 429 HANDLING + RETRIES
# =========================================================
def call_openai(prompt):
    """Low-level OpenAI REST call with retries + 429 backoff."""
    if not OPENAI_API_KEY:
        return None, "Missing OPENAI_API_KEY"

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You generate humanlike Twitter/X comments."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 120,
        "temperature": 0.7
    }

    last_error = None

    for attempt in range(4):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=30)

            # SUCCESS
            if r.status_code == 200:
                out = r.json()["choices"][0]["message"]["content"]
                return out, None

            # RATE LIMIT
            if r.status_code == 429:
                try:
                    retry_after = r.json()["error"].get("retry_after", 20)
                except:
                    retry_after = 20
                time.sleep(retry_after + 2)
                continue

            # SERVER RETRY
            if r.status_code in (500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 4))
                continue

            # HARD FAIL
            try:
                msg = r.json().get("error", {}).get("message")
            except:
                msg = r.text
            return None, f"OpenAI {r.status_code}: {msg}"

        except Exception as ex:
            last_error = f"OpenAI exception: {ex}"
            time.sleep(min(2 ** attempt, 4))

    return None, last_error


# =========================================================
# COMMENT VALIDATION / CLEANING
# =========================================================
def strip_punctuation(s):
    return re.sub(r"[!?,.;:]+$", "", s).strip()


def remove_emojis(text):
    return re.sub(r"[\U00010000-\U0010ffff]", "", text)


def is_banned(line):
    low = line.lower()
    for w in STRICT_BLOCKLIST:
        if w in low:
            return True
    return False


def valid_comment(line):
    line = line.strip().lower()

    # no hashtags
    if "#" in line:
        return False

    # no emojis
    if re.search(r"[\U00010000-\U0010ffff]", line):
        return False

    # strict blocklist
    if is_banned(line):
        return False

    # word count
    wc = len(line.split())
    if wc < 5 or wc > 12:
        return False

    return True


def clean_and_validate(raw):
    """Take AI output, produce EXACTLY 2 valid lines."""
    lines = raw.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # remove emojis
        line = remove_emojis(line)

        # strip punctuation
        line = strip_punctuation(line)

        # remove extra spaces
        line = re.sub(r"\s+", " ", line)

        if valid_comment(line):
            cleaned.append(line)

        if len(cleaned) == 2:
            break

    return cleaned


# =========================================================
# GENERATE 2 STRICT COMMENTS FOR EACH TWEET
# =========================================================
def generate_comments(tweet_text):
    prompt = f"""
Generate two humanlike comments based strictly on this tweet.

Rules:
- 2 lines ONLY
- each line 5–12 words
- no emojis
- no hashtags
- no punctuation at end
- natural slang allowed (tbh, fr, ngl, bro, lowkey)
- avoid hype/corporate/cringe words
- lines must be different

Tweet:
{tweet_text}
"""

    for attempt in range(4):
        raw, err = call_openai(prompt)
        if not raw:
            continue

        cleaned = clean_and_validate(raw)
        if len(cleaned) == 2:
            return cleaned, None

    # fallback
    return ["generation failed", "please retry"], None


# =========================================================
# BATCH PROCESSING (2 per batch + cooldown)
# =========================================================
def process_batch(urls):
    results = []
    failed = []

    for url in urls:
        text, err = fetch_tweet_text(url)
        if not text:
            failed.append(url)
            continue

        comments, _ = generate_comments(text)

        # spacing to avoid rate-limit
        time.sleep(1.2)

        results.append({
            "url": url,
            "comments": comments
        })

    return results, failed


# =========================================================
# KEEP ALIVE FOR RENDER
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
        data = request.get_json(silent=True) or {}
        urls = data.get("urls")

        if not isinstance(urls, list) or not urls:
            return jsonify({"error": "Send JSON: {\"urls\": [...]}"})

        # Clean + dedupe
        seen = set()
        clean_list = []

        for u in urls:
            cu = clean_url(u)
            if cu and cu not in seen:
                clean_list.append(cu)
                seen.add(cu)

        all_results = []
        all_failed = []

        # BATCH of 2
        for i in range(0, len(clean_list), 2):
            batch = clean_list[i:i+2]

            r, f = process_batch(batch)
            all_results.extend(r)
            all_failed.extend(f)

            # cooldown between batches
            time.sleep(2)

        return jsonify({
            "results": all_results,
            "failed": all_failed
        })

    except Exception as e:
        return jsonify({"error": "Internal error", "detail": str(e)}), 500


# =========================================================
# LOCAL RUN
# =========================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
