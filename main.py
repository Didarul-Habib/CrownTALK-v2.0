from flask import Flask, request, jsonify
from openai import OpenAI
import requests, threading, time, re

app = Flask(__name__)
client = OpenAI()

# -------- KEEP SERVER AWAKE --------
def keep_alive():
    while True:
        try:
            requests.get("https://YOUR-RENDER-APP.onrender.com/")
            print("Ping sent.")
        except Exception as e:
            print("Ping failed:", e)
        time.sleep(600)  # ping every 10 mins

threading.Thread(target=keep_alive, daemon=True).start()

# -------- ROOT --------
@app.route("/")
def home():
    return "CrownTALK server running"

# -----------------------------------------------------------
# COMMENT GENERATION WITH RETRIES (VERY STABLE)
# -----------------------------------------------------------
def generate_comments(prompt, retries=3, delay=6):
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are CrownTALK. Write humanlike 5–10 word comments. "
                            "No punctuation, no emojis, no repeated phrases, no hype words. "
                            "Two comments only. Different tone each time."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=80,
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Retry {attempt+1}] Error:", e)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None

# -----------------------------------------------------------
# COMMENT ENDPOINT
# -----------------------------------------------------------
@app.route("/comment", methods=["POST"])
def comment_api():
    try:
        urls = request.json.get("urls", [])
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400

        # Clean URLs
        cleaned = []
        for u in urls:
            u = re.sub(r"\?.*$", "", u).strip()
            if u not in cleaned:
                cleaned.append(u)

        results = []
        failed = []

        batch_size = 2
        total_batches = (len(cleaned) + batch_size - 1) // batch_size

        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            print(f"Batch {i//batch_size+1}/{total_batches}: {batch}")

            for url in batch:
                try:
                    # Fetch tweet using VX API
                    api_url = f"https://api.vxtwitter.com/{url.replace('https://', '')}"
                    r = requests.get(api_url, timeout=12)
                    data = r.json()

                    if "text" not in data:
                        failed.append(url)
                        continue

                    tweet_text = data["text"]

                    prompt = (
                        f"Tweet: {tweet_text}\n"
                        f"Write exactly two natural comments based on this tweet."
                    )

                    comments = generate_comments(prompt)

                    if not comments:
                        failed.append(url)
                        continue

                    results.append({
                        "url": url,
                        "comments": comments
                    })

                except:
                    failed.append(url)

            time.sleep(8)  # small delay between batches

        return jsonify({
            "processed": len(results),
            "failed": failed,
            "results": results
        })

    except Exception as e:
        print("Critical error:", e)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
