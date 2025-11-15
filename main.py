from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    links = data.get("links", [])
    api_key = data.get("api_key")

    results = []

    for link in links:
        try:
            clean = link.split("?")[0].strip()
            tweet_id = clean.split("/")[-1]

            prompt = f"Write 2 short, casual humanlike Twitter replies for this tweet: {clean}"

            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 80
                }
            )

            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                replies = [x.strip("- ").strip() for x in text.split("\n") if x.strip()]
                results.append({"tweet": clean, "replies": replies})
            else:
                results.append({"tweet": clean, "replies": ["Could not fetch this tweet (private/deleted)"]})

        except:
            results.append({"tweet": link, "replies": ["Error processing link"]})

    return jsonify({"results": results})


@app.errorhandler(404)
def not_found(e):
    return send_from_directory('static', 'index.html')
