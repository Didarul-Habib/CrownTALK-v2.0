import os
import re
import time
import random
import requests
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__, template_folder="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# 🔥 BLACKLIST (your rules + AI pattern blockers)
# -------------------------------------------------------
BLACKLIST = {
    "finally", "curious", "love", "loving", "love to",
    "feels like", "excited", "is wild", "hit different",
    "game changer", "insane", "iconic", "masterpiece",
    "huge w", "massive w", "phenomenal", "incredible",
    "beautiful", "well said", "impressive", "stunning",
    "amazing work", "this aged well", "this goes crazy",
    "this slaps", "no way", "this blew my mind",
    "interesting perspective", "well articulated",
    "insightful", "truly", "definitely", "pretty",
    "overall", "amazing", "incredible", "great"
}

# -------------------------------------------------------
# 🔥 SLANG POOL (medium → high)
# -------------------------------------------------------
SLANG = [
    "fr", "ngl", "tbh", "lowkey", "kinda", "btw",
    "no lie", "rn", "for real", "bro", "man",
    "real talk", "not gonna lie", "honestly",
    "deadass", "lowkey feels", "idk tho"
]

# -------------------------------------------------------
# 🔥 Extract tweet ID from URL
# -------------------------------------------------------
def extract_tweet_id(url):
    clean = url.split("?")[0]
    match = re.search(r"/status/(\d+)", clean)
    return match.group(1) if match else None

# -------------------------------------------------------
# 🔥 Fetch tweet text using vx → fx → fallback
# -------------------------------------------------------
def fetch_tweet_content(tweet_id):
    endpoints = [
        f"https://api.vxtwitter.com/{tweet_id}",
        f"https://api.fxtwitter.com/{tweet_id}"
    ]

    for endpoint in endpoints:
        try:
            r = requests.get(endpoint, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if "text" in data:
                    return data["text"]
        except:
            continue

    # Fallback scraper (super lightweight)
    try:
        r = requests.get(f"https://cdn.fxtwitter.com/{tweet_id}", timeout=8)
        if r.status_code == 200:
            text = r.text
            cleaned = re.sub("<.*?>", "", text)
            return cleaned[:280]  # limit to tweet length
    except:
        pass

    return None

# -------------------------------------------------------
# 🔥 Generate 2 human-like comments
# -------------------------------------------------------
def generate_comments(tweet_text):
    # Remove blacklisted words from context to avoid influence
    safe_text = tweet_text
    for bad in BLACKLIST:
        safe_text = safe_text.replace(bad, "")

    slang_sample = random.sample(SLANG, k=3)

    prompt = f"""
You are CrownTALK 👑 — a medium-edgy, humorous, human-like commenter.
Generate **two** short comments for this tweet:

Tweet: "{safe_text}"

RULES:
- 5 to 12 words each
- No punctuation (no . , ! ?)
- No emojis
- No hashtags
- No blacklisted phrases
- Avoid AI patterns or generic hype
- Vary tone naturally
- Use slang naturally (e.g. {', '.join(slang_sample)})
- Make both comments DIFFERENT styles
- Must be based on the tweet context
- NO repeated structure
- NO overhype words
- Humor allowed

OUTPUT:
Write the two comments on separate lines. No labels.
"""

    for _ in range(3):  # retry Safe
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.78,
            )

            raw = response.choices[0].message["content"].strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip()]

            # Enforce EXACT 2 comments
            if len(lines) >= 2:
                c1, c2 = lines[0], lines[1]

                # Remove punctuation again (double safety)
                c1 = re.sub(r"[^\w\s]", "", c1)
                c2 = re.sub(r"[^\w\s]", "", c2)

                # Word count enforcement
                if 5 <= len(c1.split()) <= 12 and 5 <= len(c2.split()) <=12:
                    return c1, c2

        except Exception as e:
            time.sleep(1)

    return "couldnt craft comment rn", "try again in a sec"

# -------------------------------------------------------
# 🔥 ROUTE: Home Page
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------------------------------------
# 🔥 ROUTE: Process tweet links
# -------------------------------------------------------
@app.route("/process", methods=["POST"])
def process():
    data = request.json
    urls = data.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    results = []
    batch_size = 2
    total_batches = (len(urls) + batch_size - 1) // batch_size

    for batch_index in range(total_batches):
        batch = urls[batch_index * batch_size : (batch_index + 1) * batch_size]

        for url in batch:
            tweet_id = extract_tweet_id(url)
            if not tweet_id:
                results.append({
                    "url": url,
                    "error": "Invalid tweet link"
                })
                continue

            tweet_text = fetch_tweet_content(tweet_id)
            if not tweet_text:
                results.append({
                    "url": url,
                    "error": "Could not fetch tweet (private/deleted)"
                })
                continue

            c1, c2 = generate_comments(tweet_text)
            results.append({
                "url": url,
                "comment1": c1,
                "comment2": c2
            })

            # Cooldown between tweets
            time.sleep(random.uniform(2, 3))

        # Delay between batches
        if batch_index < total_batches - 1:
            time.sleep(random.uniform(10, 12))

    return jsonify({"results": results})

# -------------------------------------------------------
# 🔥 GUNICORN ENTRY
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
