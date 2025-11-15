from flask import Flask, request, jsonify
from flask_cors import CORS
import os, time, threading, requests
from urllib.parse import urlparse

# =============================
# Strictly disable proxies (Render "proxies" bug)
# =============================
for _v in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if os.environ.get(_v):
        os.environ.pop(_v, None)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

# =============================
# Helpers
# =============================
def clean_url(u: str):
    if not u:
        return None
    u = u.strip()
    # remove numbering like "1. https://..."
    if u and u[0].isdigit() and "." in u[:4]:
        u = u.split(".", 1)[1].strip()
    # strip params
    if "?" in u:
        u = u.split("?", 1)[0]
    return u

def summarize_openai_error(resp: requests.Response) -> str:
    try:
        j = resp.json()
        err = j.get("error") or {}
        if isinstance(err, dict):
            # common fields: message, type, code
            msg = err.get("message") or str(err)
        else:
            msg = str(j)
        return f"{resp.status_code} {msg}"
    except Exception:
        # not JSON (e.g., HTML)
        text = (resp.text or "").strip()
        if len(text) > 300:
            text = text[:300] + " …"
        return f"{resp.status_code} {text or 'non-JSON error'}"

def fetch_tweet_text(tweet_url: str):
    """Return tweet text or None. Never raises."""
    try:
        parsed = urlparse(tweet_url)
        host = parsed.netloc
        path = parsed.path  # /user/status/123...
        api_url = f"https://api.vxtwitter.com/{host}{path}"
        r = requests.get(api_url, timeout=10)
        ct = r.headers.get("content-type", "")
        print(f"[VX] GET {api_url} -> {r.status_code} ct={ct}")
        if r.status_code != 200:
            return None
        if "application/json" not in ct:
            return None
        try:
            data = r.json()
        except Exception:
            return None
        if isinstance(data, dict):
            if "text" in data:
                return data["text"]
            if "tweet" in data and isinstance(data["tweet"], dict) and "text" in data["tweet"]:
                return data["tweet"]["text"]
        return None
    except Exception as e:
        print("[VX] ERROR:", repr(e))
        return None

def generate_comments(tweet_text: str):
    """
    Returns (comments_list, error_reason).
    On success: (['line1','line2'], None)
    On failure: (['generation failed','try again later'], 'reason string')
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
Generate two short humanlike comments.
Rules:
- 5–12 words each
- No emojis
- No hashtags
- No punctuation at the end
- Natural slang allowed (ngl, fr, lowkey, tbh)
- Comments must be different
- Exactly 2 lines
- Avoid hype/buzzwords (amazing, awesome, incredible, game changer, empowering, etc.)

Tweet:
{tweet_text}
"""

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 50,
    }

    # Option A strict: 2 quick tries, no long sleeps
    for attempt in range(2):
        try:
            r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=8)
        except Exception as e:
            print(f"[OA] REQUEST ERROR attempt {attempt+1}:", repr(e))
            # try once more
            continue

        if r.status_code == 200:
            try:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
            except Exception as e:
                print("[OA] PARSE ERROR:", repr(e))
                return (["generation failed", "try again later"], "openai_parse_error")
            lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            cleaned = []
            for line in lines:
                while line.endswith((".", "!", "?", ",")):
                    line = line[:-1]
                words = line.split()
                if 5 <= len(words) <= 12:
                    cleaned.append(line)
                if len(cleaned) == 2:
                    return (cleaned, None)
            return (["generation failed", "try again later"], "openai_invalid_output")

        # Log exact OpenAI response
        detail = summarize_openai_error(r)
        print(f"[OA] ERROR attempt {attempt+1}: {detail}")
        # quick backoff only for 429
        if r.status_code == 429:
            time.sleep(2)

    return (["generation failed", "try again later"], "openai_error")

# =============================
# API Endpoints
# =============================
@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"results": [], "failed": []})

    urls = data.get("urls", [])
    cleaned = []
    for u in urls:
        cu = clean_url(u)
        if cu and cu not in cleaned:
            cleaned.append(cu)

    results = []
    failed = []

    # batches of 2
    for i in range(0, len(cleaned), 2):
        batch = cleaned[i:i+2]
        for url in batch:
            text = fetch_tweet_text(url)
            if not text:
                failed.append({"url": url, "reason": "vx_text_not_found"})
                # still add placeholder in results so UI shows something
                results.append({"url": url, "comments": ["generation failed", "try again later"]})
                continue

            comments, err = generate_comments(text)
            if err:
                failed.append({"url": url, "reason": err})
            results.append({"url": url, "comments": comments})

        # tiny delay to avoid hammering
        time.sleep(0.2)

    return jsonify({"results": results, "failed": failed})

@app.route("/keytest")
def keytest():
    return f"KEY LOADED: {bool(OPENAI_KEY)}"

@app.route("/diag")
def diag():
    """
    Quick end-to-end diagnostics.
    Optional query param: ?tweet=https://x.com/user/status/123
    """
    diag = {
        "env": {
            "openai_key_present": bool(OPENAI_KEY),
            "proxies_unset": not any(os.environ.get(v) for v in
                                     ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]),
        },
        "openai": {},
        "vxtwitter": {},
        "app": {"model": OPENAI_MODEL, "version": "diag-1.0"},
    }

    # --- OpenAI check ---
    if OPENAI_KEY:
        test_payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
        }
        try:
            rr = requests.post(OPENAI_URL,
                               headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                               json=test_payload, timeout=6)
            ok = rr.status_code == 200
            err = None if ok else summarize_openai_error(rr)
            diag["openai"] = {"ok": ok, "status": rr.status_code, "detail": err}
        except Exception as e:
            diag["openai"] = {"ok": False, "error": repr(e)}
    else:
        diag["openai"] = {"ok": False, "error": "OPENAI_API_KEY missing"}

    # --- VXTwitter check ---
    test_tweet = request.args.get("tweet") or "https://x.com/Twitter/status/1577730467436138524"
    try:
        parsed = urlparse(test_tweet)
        api_url = f"https://api.vxtwitter.com/{parsed.netloc}{parsed.path}"
        vr = requests.get(api_url, timeout=8)
        ct = vr.headers.get("content-type", "")
        v_ok = (vr.status_code == 200 and "application/json" in ct)
        keys = []
        if v_ok:
            try:
                jj = vr.json()
                if isinstance(jj, dict):
                    keys = list(jj.keys())[:6]
            except Exception:
                v_ok = False
        diag["vxtwitter"] = {"ok": v_ok, "status": vr.status_code, "content_type": ct, "keys": keys, "api_url": api_url}
    except Exception as e:
        diag["vxtwitter"] = {"ok": False, "error": repr(e)}

    return jsonify(diag)

@app.route("/")
def home():
    return "OK"

# =============================
# Keep-alive (Render)
# =============================
def keep_alive():
    while True:
        try:
            requests.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()
