import re
import os
import time
import json
import threading
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, make_response

# -------- Optional provider SDKs (import only if keys are present) --------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Import lazily so missing packages don't crash boot if a provider isn't used.
_groq = _openai = _genai = None

if GROQ_API_KEY:
    try:
        from groq import Groq  # groq>=0.11.0
        _groq = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"[boot] Groq SDK unavailable: {e}")

if OPENAI_API_KEY:
    try:
        from openai import OpenAI  # openai>=1.x
        _openai = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"[boot] OpenAI SDK unavailable: {e}")

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai  # google-generativeai
        genai.configure(api_key=GEMINI_API_KEY)
        _genai = genai
    except Exception as e:
        print(f"[boot] Gemini SDK unavailable: {e}")

# -------- Config --------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
PER_URL_SLEEP_SECONDS = float(os.getenv("PER_URL_SLEEP_SECONDS", "0.1"))
UPSTREAM_MIN_GAP_SECONDS = float(os.getenv("UPSTREAM_MIN_GAP_SECONDS", "0.5"))

# model ids (you already set these in Render env)
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# provider order, e.g. "groq,openai,gemini,offline"
LLM_ORDER = [p.strip() for p in os.getenv(
    "CROWNTALK_LLM_ORDER", "groq,openai,gemini,offline"
).split(",") if p.strip()]

# network-ish timeouts
PROVIDER_TIMEOUT = float(os.getenv("PROVIDER_TIMEOUT", "45"))

# simple per-provider cool-down to avoid hammering rate limits
_last_call_at: Dict[str, float] = {"groq": 0.0, "openai": 0.0, "gemini": 0.0}

# app
app = Flask(__name__)
_app_started_at = time.time()


# -------- Helpers --------
def _allow_cors(resp):
    # Very permissive CORS for your Netlify → Render front-end call
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.after_request
def _after(resp):
    return _allow_cors(resp)


@app.route("/", methods=["GET"])
def root():
    return _allow_cors(make_response(jsonify({
        "ok": True,
        "name": "CrownTALK backend",
        "uptime_sec": round(time.time() - _app_started_at, 2),
        "providers": {
            "groq": bool(_groq),
            "openai": bool(_openai),
            "gemini": bool(_genai),
        },
        "order": LLM_ORDER,
    }), 200))


@app.route("/healthz", methods=["GET"])
def healthz():
    return _allow_cors(make_response(jsonify({
        "ok": True,
        "uptime_sec": round(time.time() - _app_started_at, 2)
    }), 200))


@app.route("/ping", methods=["GET"])
def ping():
    # Small endpoint your frontend or a pinger can hit to keep the dyno warm
    return _allow_cors(make_response(jsonify({"pong": True, "ts": time.time()}), 200))


def _normalize(urls: List[str]) -> List[str]:
    out = []
    for u in urls or []:
        s = (u or "").strip()
        if s:
            out.append(s)
    return out[: max(1, BATCH_SIZE) * 3]  # sanity cap


# ---------- Prompts ----------
SYSTEM_PROMPT = (
    "You are CrownTALK, a friendly, concise social reply writer. "
    "Write short, humanlike, positive comments for the given tweet URL. "
    "Return exactly two lines, each line is the final comment only. "
    "No headings, no labels, no numbering, no quotes, no emojis, no hashtags, "
    "and no punctuation at all. each comment must be different tone and style like a human based on the post context."
)


USER_TEMPLATE = "Tweet URL: {url}\nTask: Write 2 short humanlike comments in English."


# ---------- Providers ----------
def _respect_gap(provider: str):
    now = time.time()
    last = _last_call_at.get(provider, 0.0)
    gap = max(UPSTREAM_MIN_GAP_SECONDS, 0.1)
    delta = now - last
    if delta < gap:
        time.sleep(gap - delta)
    _last_call_at[provider] = time.time()


def _split_comments(text: str) -> List[str]:
    """
    Extract exactly two plain comment lines from model output.
    - Removes preambles like 'Here are two comments:'
    - Strips list labels like 'Comment 1:', '1)', '-', etc.
    - Removes ALL punctuation characters from the final lines.
    """

    if not text:
        return []

    def _clean_labels(line: str) -> str:
        s = line.strip()
        # Remove obvious bullets/dashes first
        s = s.strip("•-–—").strip()

        # Drop common preambles if they appear at line start
        s = re.sub(r"^\s*(here (are|is).{0,60}comment[s]?\s*:)\s*", "", s, flags=re.I)
        s = re.sub(r"^\s*(sample|possible|two|the two|some)\s+comment[s]?\s*:\s*", "", s, flags=re.I)

        # Remove explicit labels / numbering
        s = re.sub(r"^\s*(comment|reply|option|idea|line)\s*\d+\s*[:\-\.)]\s*", "", s, flags=re.I)
        s = re.sub(r"^\s*\(?\s*\d+\s*[:\-\.)]\s*", "", s)     # 1) 1. 1-
        s = re.sub(r"^\s*[A-Z]\s*[:\-\.)]\s*", "", s)        # A) A. A-
        return s.strip()

    def _to_plain(line: str) -> str:
        # Remove ALL punctuation (keep letters, numbers, spaces)
        line = re.sub(r"[^\w\s]", "", line, flags=re.UNICODE)
        # Collapse multiple spaces
        line = re.sub(r"\s+", " ", line).strip()
        return line

    # 1) Line-oriented parse
    lines: List[str] = []
    for row in text.splitlines():
        row = _clean_labels(row)
        if row:
            lines.append(row)

    # Remove any leftover preamble-y lines
    lines = [ln for ln in lines if not re.match(r"^(here (are|is)|two comments|possible comments)", ln, re.I)]

    if len(lines) < 2:
        # 2) Try splitting inside one paragraph on obvious separators
        chunks = re.split(r"(?:\s(?:(?:comment|reply)\s*\d+|(?:\d+|[A-Z])[:\-\.)]|-)\s*)", text, flags=re.I)
        chunks = [_clean_labels(c) for c in chunks if _clean_labels(c)]
        lines.extend(chunks)

    if len(lines) < 2:
        # 3) Sentence-ish fallback
        parts = [p.strip() for p in re.split(r"[.?!]\s+", text) if p.strip()]
        parts = [_clean_labels(p) for p in parts if _clean_labels(p)]
        lines.extend(parts)

    # Dedup while preserving order
    seen = set()
    uniq = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)

    # Convert to plain (no punctuation) and keep non-empty
    plain = [p for p in (_to_plain(ln) for ln in uniq) if p]

    # Guarantee at least one line if everything was stripped
    if not plain and text.strip():
        plain = [_to_plain(text)]

    return plain[:2]


def call_groq(url: str) -> Optional[List[str]]:
    if not _groq:
        return None
    _respect_gap("groq")
    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(url=url)},
            ],
            temperature=0.7,
            max_tokens=180,
            # groq python often ignores per-call timeout; safe to leave, or remove if it errors
            timeout=PROVIDER_TIMEOUT,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _split_comments(text)
    except Exception as e:
        print(f"[groq] error for {url}: {e}")
        return None

def call_openai(url: str) -> Optional[List[str]]:
    if not _openai:
        return None
    _respect_gap("openai")
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(url=url)},
            ],
            temperature=0.7,
            max_tokens=180,
            timeout=PROVIDER_TIMEOUT,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _split_comments(text)
    except Exception as e:
        print(f"[openai] error for {url}: {e}")
        return None

def call_gemini(url: str) -> Optional[List[str]]:
    if not _genai:
        return None
    _respect_gap("gemini")
    try:
        model = _genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(url=url)}"
        resp = model.generate_content(prompt, request_options={"timeout": PROVIDER_TIMEOUT})
        text = (resp.text or "").strip()
        return _split_comments(text)
    except Exception as e:
        print(f"[gemini] error for {url}: {e}")
        return None

def call_offline(url: str) -> List[str]:
    base = [
        "Sounds like a solid direction—keen to see where this goes.",
        "Interesting take—curious how this unfolds over time.",
        "Love the clarity here. Appreciate you sharing this.",
        "Great points. What are the first steps to make it happen?",
        "This is promising—how can people get involved?",
    ]
    h = abs(hash(url))
    return [base[h % len(base)], base[(h // 7) % len(base)]]

PROVIDER_FUNCS = {
    "groq":   call_groq,
    "openai": call_openai,
    "gemini": call_gemini,
    "offline": lambda url: call_offline(url),
}

def generate_for_url(url: str) -> Dict[str, Any]:
    for name in LLM_ORDER:
        fn = PROVIDER_FUNCS.get(name)
        if not fn:
            continue
        out = fn(url)
        if out and len(out) >= 1:
            comments = [{"lang": "en", "text": c.strip()} for c in out[:2] if c and c.strip()]
            if comments:
                return {"url": url, "comments": comments, "provider": name}
    return {"url": url, "comments": [], "provider": None}

# -------- API: /comment --------
@app.route("/comment", methods=["POST", "OPTIONS"])
def comment():
    if request.method == "OPTIONS":
        return _allow_cors(make_response("", 204))
    try:
        payload = request.get_json(silent=True) or {}
        urls = _normalize(payload.get("urls", []))
        if not urls:
            return _allow_cors(make_response(jsonify({
                "results": [], "failed": [{"url": "(none)", "reason": "No URLs provided"}]
            }), 400))
        results: List[Dict[str, Any]] = []
        failed: List[Dict[str, str]] = []
        for i, url in enumerate(urls):
            try:
                item = generate_for_url(url)
                if item["comments"]:
                    results.append(item)
                else:
                    failed.append({"url": url, "reason": "No comments from providers"})
            except Exception as e:
                print(f"[comment] error on {url}: {e}")
                failed.append({"url": url, "reason": "Internal error"})
            time.sleep(max(PER_URL_SLEEP_SECONDS, 0.0))
        return _allow_cors(make_response(jsonify({
            "results": results,
            "failed": failed
        }), 200))
    except Exception as e:
        print(f"[comment] fatal: {e}")
        return _allow_cors(make_response(jsonify({
            "results": [], "failed": [{"url": "(all)", "reason": "Server error"}]
        }), 500))

# -------- API: /reroll --------
@app.route("/reroll", methods=["POST", "OPTIONS"])
def reroll():
    if request.method == "OPTIONS":
        return _allow_cors(make_response("", 204))
    try:
        payload = request.get_json(silent=True) or {}
        url = (payload.get("url") or "").strip()
        if not url:
            return _allow_cors(make_response(jsonify({"error": "Missing url"}), 400))
        item = generate_for_url(url)
        return _allow_cors(make_response(jsonify({
            "url": url,
            "comments": item.get("comments", []),
            "provider": item.get("provider")
        }), 200))
    except Exception as e:
        print(f"[reroll] fatal: {e}")
        return _allow_cors(make_response(jsonify({
            "url": url if "url" in locals() else None,
            "comments": [], "error": "Server error"
        }), 500))

# -------- Warm-up on boot (non-blocking) --------
def _warm_once():
    try:
        # A tiny self-ping after boot so first user isn't cold
        time.sleep(1.5)
        with app.test_client() as c:
            c.get("/healthz")   # removed invalid timeout kwarg
    except Exception:
        pass

if __name__ == "__main__":
    # local dev: python main.py
    threading.Thread(target=_warm_once, daemon=True).start()
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
