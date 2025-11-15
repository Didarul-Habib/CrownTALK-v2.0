from flask import Flask, request, jsonify
from flask_cors import CORS
import os, time, threading, requests, datetime, json, re, random
from urllib.parse import urlparse

# ──────────────────────────────────────────────────────────────────────────────
# Kill any proxy env (Render sometimes injects) to avoid "proxies" bugs
# ──────────────────────────────────────────────────────────────────────────────
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY",
          "ALL_PROXY","all_proxy","NO_PROXY","no_proxy"]:
    os.environ.pop(k, None)

http = requests.Session()
http.trust_env = False

# ──────────────────────────────────────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

OPENAI_KEY        = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL        = "https://api.openai.com/v1/chat/completions"
OPENAI_TIMEOUT_S  = float(os.environ.get("OPENAI_TIMEOUT", "12"))

# Micro‑batching for OpenAI (cut RPM/TPM)
MICRO_BATCH_SIZE  = int(os.environ.get("MICRO_BATCH_SIZE", "12"))
MAX_TWEET_CHARS   = int(os.environ.get("MAX_TWEET_CHARS", "220"))
RETRIES_429       = int(os.environ.get("OPENAI_RETRIES_429", "1"))
BACKOFF_CAP_S     = float(os.environ.get("OPENAI_BACKOFF_MAX", "18"))
SLEEP_BETWEEN_MB  = float(os.environ.get("SLEEP_BETWEEN_MICROBATCH", "0.4"))

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def trunc(s, n=300):
    if s is None: return None
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"

# ──────────────────────────────────────────────────────────────────────────────
# URL clean
# ──────────────────────────────────────────────────────────────────────────────
def clean_url(u: str):
    if not u: return None
    u = u.strip()
    if u and u[0].isdigit() and "." in u[:4]:
        u = u.split(".", 1)[1].strip()
    if "?" in u:
        u = u.split("?", 1)[0]
    return u

# ──────────────────────────────────────────────────────────────────────────────
# VXTwitter fetch with fallback
# ──────────────────────────────────────────────────────────────────────────────
_TWEET_ID_RE = re.compile(r"/status/(\d+)")

def fetch_tweet_text(url):
    """Return (text, info) — never raises."""
    try:
        p = urlparse(url)
        host, path = p.netloc, p.path

        # Attempt 1: host+path
        api1 = f"https://api.vxtwitter.com/{host}{path}"
        r1 = http.get(api1, timeout=8)
        ct1 = r1.headers.get("content-type", "")
        print(f"[{now_iso()}] VX #1 {api1} -> {r1.status_code} ct={ct1}")
        if r1.status_code == 200 and "application/json" in ct1:
            try:
                j = r1.json()
                if isinstance(j, dict):
                    if "text" in j: return j["text"], {"path":"host_path", "status":200}
                    if "tweet" in j and isinstance(j["tweet"], dict) and "text" in j["tweet"]:
                        return j["tweet"]["text"], {"path":"host_path", "status":200}
            except Exception as e:
                print(f"[VX] JSON decode error #1: {e}")

        # Attempt 2: /Twitter/status/{id}
        m = _TWEET_ID_RE.search(path or "")
        if m:
            tid = m.group(1)
            api2 = f"https://api.vxtwitter.com/Twitter/status/{tid}"
            r2 = http.get(api2, timeout=8)
            ct2 = r2.headers.get("content-type", "")
            print(f"[{now_iso()}] VX #2 {api2} -> {r2.status_code} ct={ct2}")
            if r2.status_code == 200 and "application/json" in ct2:
                try:
                    j2 = r2.json()
                    if isinstance(j2, dict):
                        if "text" in j2: return j2["text"], {"path":"twitter_status", "status":200}
                        if "tweet" in j2 and isinstance(j2["tweet"], dict) and "text" in j2["tweet"]:
                            return j2["tweet"]["text"], {"path":"twitter_status", "status":200}
                except Exception as e:
                    print(f"[VX] JSON decode error #2: {e}")

        return None, {"status": r1.status_code, "ct": ct1}
    except Exception as e:
        print(f"[VX] exception: {e}")
        return None, {"error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI JSON batch with short prompt (min tokens)
# ──────────────────────────────────────────────────────────────────────────────
def parse_retry_after_seconds(msg: str) -> float:
    if not msg: return 5.0
    m = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
    if m:
        try: return max(1.0, float(m.group(1)))
        except: pass
    m2 = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", msg)
    if m2:
        try: return max(1.0, 60*float(m2.group(1)) + float(m2.group(2)))
        except: pass
    return 5.0

def generate_batch_comments(texts):
    """
    texts: list[str] (already truncated)
    Returns (map: local_index -> [c1,c2], err or None)
    """
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}

    # Ultra‑short prompt to cut TPM
    system = "Only output strict JSON."
    schema = {
        "rules":[
            "5-12 words each",
            "no emojis",
            "no hashtags",
            "no punctuation at the end",
            "natural slang allowed (ngl, fr, lowkey, tbh)",
            "comments must be different"
        ],
        "tweets":[{"i":i,"t":t} for i,t in enumerate(texts)]
    }
    user = (
        "Return strictly: {\"results\":[{\"i\":0,\"c\":[\"line1\",\"line2\"]},...]}\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    payload = {
        "model": OPENAI_MODEL,
        "messages":[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        "response_format":{"type":"json_object"},
        "temperature":0.5,
        "max_tokens": 80 + 30*len(texts)  # very small
    }

    for attempt in range(RETRIES_429 + 1):
        try:
            r = http.post(OPENAI_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_S)
        except Exception as e:
            print(f"[OA BATCH] request exception: {e}")
            if attempt < RETRIES_429:
                time.sleep(3)
                continue
            return {}, {"exception": str(e)}

        if r.status_code == 200:
            try:
                content = r.json()["choices"][0]["message"]["content"]
                obj = json.loads(content)
            except Exception as e:
                print(f"[OA BATCH] parse error: {e}; raw={trunc(r.text,900)}")
                return {}, {"status": r.status_code, "parse_error": str(e), "raw": trunc(r.text, 900)}

            res_map = {}
            for item in obj.get("results", []):
                i = item.get("i")
                c = item.get("c", [])
                if isinstance(i,int) and isinstance(c,list) and len(c)>=2:
                    cleaned = []
                    for line in c[:2]:
                        s = str(line).strip()
                        while s.endswith((".",",","!","?")):
                            s = s[:-1]
                        if 5 <= len(s.split()) <= 12:
                            cleaned.append(s)
                    if len(cleaned)==2:
                        res_map[i]=cleaned
            return res_map, None

        # non‑200
        try:
            err = r.json().get("error", {})
            msg = err.get("message", "")
        except Exception:
            err, msg = {}, r.text

        print(f"[OA BATCH] ERR {r.status_code}: {trunc(msg,900)}")
        if r.status_code == 429 and attempt < RETRIES_429:
            time.sleep(min(BACKOFF_CAP_S, parse_retry_after_seconds(msg)))
            continue
        return {}, {"status": r.status_code, "body": trunc(r.text, 900)}

# ──────────────────────────────────────────────────────────────────────────────
# OFFLINE FALLBACK generator (no OpenAI)
# ──────────────────────────────────────────────────────────────────────────────
STOPWORDS = set("""
a an and are as at be but by for from has have i if in is it its of on or our so that the this to was were will with you your
""".split())

BANNED = [
    "game changer","finally","love to see","love that","can’t wait","cant wait","excited",
    "amazing","great","wow","so good","awesome","epic","incredible","innovation",
    "disruptive","empowering","transformative","visionary","groundbreaking","literally shaking",
    "omg","so true bestie","slay","ate","yass","yasss","sis","queen","as an ai",
    "as a language model","in today’s world","in this digital age","fascinating perspective",
    "thoughts?","agree?","anyone else?","who’s with me?"
]

OPENERS = ["lowkey","ngl","tbh","fr","no cap","kinda","highkey","honestly","deadass"]
VERBS   = ["hits","slaps","goes hard","goes crazy","makes sense","is clean","is wild","is tough","is valid","sounds right"]
TAILS   = ["for real","not mad at it","i vibe with this","i’m here for it","i respect it","no lie","can’t even pretend"]

def _keywords(text):
    toks = re.findall(r"[A-Za-z0-9']+", text.lower())
    kws = [t for t in toks if t not in STOPWORDS and len(t) > 2][:6]
    return kws or ["this"]

def _bad(line):
    l = line.lower()
    for b in BANNED:
        if b in l: return True
    return False

def fallback_comments(text):
    kws = _keywords(text)
    def build(prev_first=None):
        for _ in range(10):  # try a few times to meet rules + blocklist
            opener = random.choice(OPENERS)
            verb   = random.choice(VERBS)
            k1 = random.choice(kws)
            k2 = random.choice(kws) if len(kws)>1 else ""
            parts = [opener, k1, verb]
            if k2 and k2 != k1 and random.random() < 0.5:
                parts.insert(2, k2)
            if random.random() < 0.6:
                parts += [random.choice(TAILS)]
            line = " ".join([p for p in parts if p]).strip()
            while line.endswith((".",",","!","?")):
                line = line[:-1]
            words = line.split()
            if prev_first and len(words)>1 and " ".join(words[:2]) == prev_first:
                continue
            if 5 <= len(words) <= 12 and not _bad(line):
                return line
        # hard fallback
        line = f"{random.choice(OPENERS)} {random.choice(kws)} {random.choice(VERBS)}"
        return line

    l1 = build()
    l2 = build(prev_first=" ".join(l1.split()[:2]))
    return [l1, l2]

# ──────────────────────────────────────────────────────────────────────────────
# POST /comment — batch with offline fallback
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/comment", methods=["POST"])
def comment():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"results": [], "failed": []})

    raw_urls = payload.get("urls", []) or []
    cleaned = []
    for u in raw_urls:
        cu = clean_url(u)
        if cu and cu not in cleaned:
            cleaned.append(cu)

    # 1) fetch texts first (pair-wise rhythm to stay friendly with VX)
    texts, idx_map, failed = [], [], []
    for idx, url in enumerate(cleaned):
        txt, vx = fetch_tweet_text(url)
        if not txt:
            failed.append({"url": url, "reason": "vx_text_not_found", "vx": vx})
            continue
        txt = txt.strip()
        if len(txt) > MAX_TWEET_CHARS:
            txt = txt[:MAX_TWEET_CHARS]
        idx_map.append((idx, url))
        texts.append(txt)
        time.sleep(0.2)  # light pacing

    results = []

    if not texts:
        return jsonify({"results": results, "failed": failed})

    # 2) micro-batch into one or few OpenAI calls
    for start in range(0, len(texts), MICRO_BATCH_SIZE):
        sub_texts = texts[start:start+MICRO_BATCH_SIZE]
        sub_pairs = idx_map[start:start+MICRO_BATCH_SIZE]

        res_map, err = ({}, {"status": 0}) if not OPENAI_KEY else generate_batch_comments(sub_texts)

        if res_map:
            for local_i, (gidx, url) in enumerate(sub_pairs):
                if local_i in res_map:
                    results.append({"url": url, "comments": res_map[local_i]})
                else:
                    # Offline fallback for any missing rows in the batch result
                    comments = fallback_comments(sub_texts[local_i])
                    results.append({"url": url, "comments": comments})
        else:
            # Whole batch failed (e.g., 429 / no key / model issue) → offline for all in this slice
            if err:
                # keep one batch-level record for transparency (UI can show it)
                failed.append({"url": "BATCH", "reason": "openai_batch_failed", "openai": err})
            for local_i, (gidx, url) in enumerate(sub_pairs):
                comments = fallback_comments(sub_texts[local_i])
                results.append({"url": url, "comments": comments})

        time.sleep(SLEEP_BETWEEN_MB)

    return jsonify({"results": results, "failed": failed})

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/keytest")
def keytest():
    return f"KEY LOADED: {bool(OPENAI_KEY)}"

@app.route("/diag")
def diag():
    return jsonify({
        "time": now_iso(),
        "env": {"openai_key_present": bool(OPENAI_KEY), "model": OPENAI_MODEL}
    })

@app.route("/diag/openai")
def diag_openai():
    headers = {"Authorization": f"Bearer {OPENAI_KEY or ''}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":"ping"}], "max_tokens": 1}
    try:
        r = http.post(OPENAI_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_S)
        try:
            body = r.json()
        except Exception:
            body = {"raw": trunc(r.text, 900)}
        return jsonify({"status": r.status_code, "body": body})
    except Exception as e:
        return jsonify({"status": "exception", "error": str(e)})

@app.route("/diag/vx")
def diag_vx():
    url = request.args.get("url")
    t, info = fetch_tweet_text(url) if url else (None, {"error": "missing url"})
    return jsonify({"ok": bool(t), "vx": info, "text_preview": trunc(t, 200)})

@app.route("/")
def home():
    return "OK"

# ──────────────────────────────────────────────────────────────────────────────
# Keep-alive (Render)
# ──────────────────────────────────────────────────────────────────────────────
def keep_alive():
    while True:
        try:
            http.get("https://crowntalk-v2-0.onrender.com/", timeout=5)
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=keep_alive, daemon=True).start()
