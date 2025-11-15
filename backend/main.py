from flask import Flask, request, jsonify
from flask_cors import CORS
import os, time, threading, requests, datetime, json, re
from urllib.parse import urlparse

# =============================
# Hard-disable proxies (Render)
# =============================
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"]:
    os.environ.pop(k, None)

http = requests.Session()
http.trust_env = False

# =============================
# App + Config
# =============================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

OPENAI_KEY   = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

OPENAI_URL   = "https://api.openai.com/v1/chat/completions"
OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "8"))
VX_TIMEOUT     = float(os.environ.get("VX_TIMEOUT", "8"))

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def safe_trunc(s, n=900):
    if s is None: return None
    s = str(s)
    return s if len(s) <= n else s[:n] + "...(truncated)"

# =============================
# URL clean
# =============================
def clean_url(u: str):
    if not u: return None
    u = u.strip()
    if u and u[0].isdigit() and "." in u[:4]:
        u = u.split(".", 1)[1].strip()
    if "?" in u:
        u = u.split("?", 1)[0]
    return u

# =============================
# VXTwitter fetch with fallback
# =============================
_TWEET_ID_RE = re.compile(r"/status/(\d+)")

def fetch_tweet_text(url):
    """
    Returns (text, info) — never raises.
    Strategy:
      1) try api.vxtwitter.com/{host}{path}
      2) if not JSON, extract tweet id and try api.vxtwitter.com/Twitter/status/{id}
    """
    try:
        parsed = urlparse(url)
        host, path = parsed.netloc, parsed.path

        # First attempt: host+path
        api1 = f"https://api.vxtwitter.com/{host}{path}"
        r1 = http.get(api1, timeout=VX_TIMEOUT)
        ct1 = r1.headers.get("content-type", "")
        print(f"[{now_iso()}] VX #1 {api1} -> {r1.status_code} ct={ct1}")

        if r1.status_code == 200 and "application/json" in ct1:
            try:
                d = r1.json()
                if isinstance(d, dict):
                    if "text" in d: return d["text"], {"path":"host_path", "status":200}
                    if "tweet" in d and isinstance(d["tweet"], dict) and "text" in d["tweet"]:
                        return d["tweet"]["text"], {"path":"host_path", "status":200}
            except Exception as e:
                print(f"[{now_iso()}] VX #1 JSON decode error: {e}")

        # Fallback: /Twitter/status/{id}
        m = _TWEET_ID_RE.search(path or "")
        if m:
            tid = m.group(1)
            api2 = f"https://api.vxtwitter.com/Twitter/status/{tid}"
            r2 = http.get(api2, timeout=VX_TIMEOUT)
            ct2 = r2.headers.get("content-type", "")
            print(f"[{now_iso()}] VX #2 {api2} -> {r2.status_code} ct={ct2}")
            if r2.status_code == 200 and "application/json" in ct2:
                try:
                    d2 = r2.json()
                    if isinstance(d2, dict):
                        if "text" in d2: return d2["text"], {"path":"twitter_status", "status":200}
                        if "tweet" in d2 and isinstance(d2["tweet"], dict) and "text" in d2["tweet"]:
                            return d2["tweet"]["text"], {"path":"twitter_status", "status":200}
                except Exception as e:
                    print(f"[{now_iso()}] VX #2 JSON decode error: {e}")

        return None, {"status": r1.status_code, "ct1": ct1}
    except Exception as e:
        print(f"[{now_iso()}] VX exception: {e}")
        return None, {"error": str(e)}

# =============================
# OpenAI — single tweet (fast fail)
# =============================
def generate_comments_single(text):
    headers = {"Authorization": f"Bearer {OPENAI_KEY or ''}", "Content-Type": "application/json"}
    prompt = f"""
Generate two short humanlike comments.
Rules:
- 5–12 words each
- no emojis
- no hashtags
- no punctuation at the end
- natural slang allowed (ngl, fr, lowkey, tbh)
- comments must be different
- exactly 2 lines
- avoid hype/buzzwords

Tweet:
{text}
""".strip()

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role":"user","content":prompt}],
        "temperature": 0.6,
        "max_tokens": 50
    }
    try:
        r = http.post(OPENAI_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
    except Exception as e:
        print(f"[{now_iso()}] OpenAI request exception: {e}")
        return ["generation failed","try again later"], {"exception": str(e)}

    if r.status_code != 200:
        print(f"[{now_iso()}] OpenAI SINGLE ERR {r.status_code}: {safe_trunc(r.text)}")
        return ["generation failed","try again later"], {"status": r.status_code, "body": safe_trunc(r.text)}

    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        return ["generation failed","try again later"], {"status": r.status_code, "note": f"parse error: {e}", "body": safe_trunc(r.text)}

    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    cleaned = []
    for line in lines:
        while line.endswith((".",",","!","?")):
            line = line[:-1]
        w = line.split()
        if 5 <= len(w) <= 12:
            cleaned.append(line)
        if len(cleaned)==2:
            return cleaned, {"status": r.status_code}

    return ["generation failed","try again later"], {"status": r.status_code, "note": "invalid output"}

# =============================
# OpenAI — batch (one call for all tweets)
# =============================
def generate_batch_comments(texts):
    """
    texts: list[str]
    Returns (dict: idx-> [c1,c2], err_info)
    """
    headers = {"Authorization": f"Bearer {OPENAI_KEY or ''}", "Content-Type": "application/json"}

    # Ask for strict JSON we can parse.
    system = "You only output strict JSON. No prose."
    user_obj = {
        "instruction": "For each tweet, write two comments obeying the rules and return JSON.",
        "rules": [
            "5-12 words each",
            "no emojis",
            "no hashtags",
            "no punctuation at the end",
            "natural slang allowed (ngl, fr, lowkey, tbh)",
            "comments must be different",
            "exactly 2 lines per tweet",
            "avoid hype/buzzwords"
        ],
        "tweets": [{"i": i, "text": t} for i, t in enumerate(texts)]
    }
    user = (
        "Return JSON object with key 'results' only, like:\n"
        '{"results":[{"i":0,"c":["first comment","second comment"]}, ...]}.\n'
        "No markdown fences, no extra fields.\n\n" +
        json.dumps(user_obj, ensure_ascii=False)
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role":"system","content": system},
            {"role":"user","content": user}
        ],
        "temperature": 0.6,
        "max_tokens": 120 + 50*len(texts)  # allow enough for N pairs
    }

    try:
        r = http.post(OPENAI_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
    except Exception as e:
        print(f"[{now_iso()}] OpenAI BATCH exception: {e}")
        return {}, {"exception": str(e)}

    if r.status_code != 200:
        print(f"[{now_iso()}] OpenAI BATCH ERR {r.status_code}: {safe_trunc(r.text)}")
        return {}, {"status": r.status_code, "body": safe_trunc(r.text)}

    # Parse JSON from model (strip any possible junk, though we asked for strict JSON)
    try:
        content = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {}, {"status": r.status_code, "note": f"parse error: {e}", "body": safe_trunc(r.text)}

    # Try direct JSON; if fails, try to extract first {...} block.
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return {}, {"status": r.status_code, "note": "no json object", "content": safe_trunc(content)}
        try:
            obj = json.loads(m.group(0))
        except Exception as e:
            return {}, {"status": r.status_code, "note": f"json decode fail: {e}", "content": safe_trunc(content)}

    res_map = {}
    try:
        for item in obj.get("results", []):
            i = item.get("i")
            c = item.get("c", [])
            if isinstance(i, int) and isinstance(c, list) and len(c) >= 2:
                # enforce end-punct removal + length rule
                out = []
                for line in c[:2]:
                    s = str(line).strip()
                    while s.endswith((".",",","!","?")):
                        s = s[:-1]
                    if 5 <= len(s.split()) <= 12:
                        out.append(s)
                if len(out) == 2:
                    res_map[i] = out
    except Exception as e:
        return {}, {"status":"ok", "note": f"postprocess error: {e}", "content": safe_trunc(obj)}

    return res_map, {"status":"ok"}

# =============================
# POST /comment — batch-first
# =============================
@app.route("/comment", methods=["POST"])
def comment():
    try:
        data = request.get_json(force=True) or {}
    except:
        return jsonify({"results": [], "failed": []})

    raw_urls = data.get("urls", [])
    cleaned = []
    for u in raw_urls:
        cu = clean_url(u)
        if cu and cu not in cleaned:
            cleaned.append(cu)

    # Fetch all texts first
    texts, idx_map, failed = [], [], []
    for idx, url in enumerate(cleaned):
        text, vx_info = fetch_tweet_text(url)
        if not text:
            failed.append({"url": url, "reason": "vx_text_not_found", "vx": vx_info})
        else:
            idx_map.append((idx, url))
            texts.append(text)

    results = []

    # If we have no texts, return early
    if not texts:
        return jsonify({"results": results, "failed": failed})

    # ---- Batch call first (ONE OpenAI request) ----
    batch_map, batch_info = generate_batch_comments(texts)

    if batch_map:  # success for some/all indices
        for local_i, (global_idx, url) in enumerate(idx_map):
            if local_i in batch_map:
                results.append({"url": url, "comments": batch_map[local_i]})
            else:
                # fallback per-tweet for the ones missing
                comments, oa = generate_comments_single(texts[local_i])
                if comments[0] == "generation failed":
                    failed.append({"url": url, "reason": "openai_single_failed", "openai": oa})
                results.append({"url": url, "comments": comments})
    else:
        # batch failed (likely rate limit). record info and fall back per-tweet
        failed.append({"url": "ALL", "reason": "openai_batch_failed", "openai": batch_info})
        for local_i, (global_idx, url) in enumerate(idx_map):
            comments, oa = generate_comments_single(texts[local_i])
            if comments[0] == "generation failed":
                failed.append({"url": url, "reason": "openai_single_failed", "openai": oa})
            results.append({"url": url, "comments": comments})

    return jsonify({"results": results, "failed": failed})

# =============================
# Diagnostics
# =============================
@app.route("/diag/env")
def diag_env():
    return jsonify({
        "time": now_iso(),
        "openai_key_loaded": bool(OPENAI_KEY),
        "openai_key_prefix": (OPENAI_KEY[:7] + "…") if OPENAI_KEY else None,
        "model": OPENAI_MODEL
    })

@app.route("/diag/openai")
def diag_openai():
    headers = {"Authorization": f"Bearer {OPENAI_KEY or ''}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":"ping"}], "max_tokens":1, "temperature":0}
    try:
        r = http.post(OPENAI_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
        body = None
        try:
            body = r.json()
        except Exception:
            body = {"raw": safe_trunc(r.text)}
        return jsonify({"status": r.status_code, "body": body})
    except Exception as e:
        return jsonify({"status": "exception", "error": str(e)})

@app.route("/diag/vx")
def diag_vx():
    url = clean_url(request.args.get("url",""))
    text, info = fetch_tweet_text(url) if url else (None, {"error":"missing url"})
    return jsonify({"url": url, "vx_info": info, "text_preview": safe_trunc(text, 200)})

@app.route("/diag/comment")
def diag_comment():
    url = clean_url(request.args.get("url",""))
    if not url:
        return jsonify({"error":"missing url"}), 400
    text, vx_info = fetch_tweet_text(url)
    if not text:
        return jsonify({"ok": False, "reason":"vx_text_not_found", "vx": vx_info})
    comments, oa = generate_comments_single(text)
    ok = comments[0] != "generation failed"
    return jsonify({"ok": ok, "url": url, "comments": comments, "openai": oa})

# =============================
# Keep-alive
# =============================
def keep_alive():
    while True:
        try:
            http.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()

@app.route("/")
def home():
    return "OK"
