from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, request, jsonify

from utils import (
    CrownTALKError,
    Tweet,
    fetch_tweet_data,
    clean_and_normalize_urls,
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowntalk")

PORT = int(os.environ.get("PORT", "10000"))
DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
PER_URL_SLEEP = float(os.environ.get("PER_URL_SLEEP_SECONDS", "0.08"))
KEEP_ALIVE_INTERVAL = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))

# Provider config (Groq → OpenAI → Gemini → Offline)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

USE_GROQ = bool(GROQ_API_KEY)
USE_OPENAI = bool(OPENAI_API_KEY)
USE_GEMINI = bool(GEMINI_API_KEY)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

groq_client = None
openai_client = None
gemini_model = None

if USE_GROQ:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.warning("Groq init failed: %s", e)
        USE_GROQ = False

if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning("OpenAI init failed: %s", e)
        USE_OPENAI = False

if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logger.warning("Gemini init failed: %s", e)
        USE_GEMINI = False

# DB init + locking (portable)
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False


def _acquire_lock(f):
    def wrapper(*args, **kwargs):
        if not _HAS_FCNTL:
            return f(*args, **kwargs)
        with open(DB_PATH + ".lock", "a+") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            try:
                return f(*args, **kwargs)
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)
    return wrapper


@_acquire_lock
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS comments_seen(
                hash TEXT PRIMARY KEY,
                created_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS style_signatures_seen(
                sig TEXT PRIMARY KEY,
                created_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS comments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                handle TEXT,
                lang TEXT,
                comment TEXT,
                style_sig TEXT,
                created_at INTEGER
            );
            """
        )
        conn.commit()

# ===== Language detection (CJK-aware) =====
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_JA_RE  = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uFF66-\uFF9D]")
_KO_RE  = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]")

def detect_lang(tweet_lang_hint: str, text: str) -> str:
    t = (text or "")
    if _KO_RE.search(t): return "ko"
    if _JA_RE.search(t): return "ja"
    if _CJK_RE.search(t): return "zh"
    hint = (tweet_lang_hint or "").lower()
    if hint in {"zh", "zh-cn", "zh-tw", "ja", "ko"}:
        return {"zh-cn": "zh", "zh-tw": "zh"}.get(hint, hint)
    return "en"

# ===== Human-only filters =====
# Emoji ranges (broad)
EMOJI_RE = re.compile(
    "["                       # start class
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002700-\U000027BF"   # dingbats
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]",
    re.UNICODE,
)

# Punctuation removal per language
def strip_nonhuman(text: str, lang: str) -> str:
    s = (text or "")
    s = re.sub(r"https?://\S+", "", s)              # no links
    s = EMOJI_RE.sub("", s)                         # no emoji
    if lang in {"zh", "ja", "ko"}:
        s = re.sub(r"[^\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\s]", "", s)
        s = re.sub(r"\s+", "", s)                   # typical CJK: no spaces
    else:
        s = re.sub(r"[^A-Za-z0-9\s]", "", s)        # EN: only words and spaces
        s = re.sub(r"\s+", " ", s).strip()
    return s.strip()

# Blacklist of common AI-isms (lowercased)
BLACKLIST_PHRASES = {
    "as an ai",
    "as a language model",
    "as an ai language model",
    "i cannot",
    "i can not",
    "i cant",
    "i'm unable",
    "i am unable",
    "unable to",
    "i apologize",
    "sorry but",
    "i'm sorry",
    "assistant",
    "chatgpt",
    "openai",
    "gpt",
    "language model",
    "knowledge cutoff",
    "my training",
    "training data",
    "hallucinate",
    "according to my training",
    "i do not have access",
    "i dont have access",
    "ethical guidelines",
    "cannot assist",
    "cannot provide",
    "not able to browse",
    "as a bot",
    "model response",
    # stylistic AI-ish words (overformal)
    "moreover", "furthermore", "thus", "therefore", "hence",
    "in conclusion", "in summary", "delve", "utilize", "leverage",
    "paradigm", "robust", "holistic", "synergy", "pipeline",
}

# Add extra blacklist via env (comma-separated)
_extra = os.getenv("BLACKLIST_EXTRA", "")
if _extra.strip():
    for tok in _extra.split(","):
        tok = tok.strip().lower()
        if tok:
            BLACKLIST_PHRASES.add(tok)

def contains_blacklisted(text: str) -> bool:
    s = (text or "").lower()
    if not s: return False
    # simple substring check for phrases
    for p in BLACKLIST_PHRASES:
        if p in s:
            return True
    return False

# ===== OTP rules & uniqueness =====
OTP_RULES = {
    "length": {"words_min": 6, "words_max": 13, "cjk_chars_min": 12, "cjk_chars_max": 28},
    "style":  {"one_thought": True, "no_hashtags": True, "no_urls": True, "no_emojis_spam": True, "speak_human": True},
    "diversity": {
        "ban_reused_openers": True,
        "ban_reused_last_bigram": True,
        "ban_reused_punct_pattern": True,
        "ban_same_after_normalize": True,
    },
}

BUCKETS = {
    "en": {
        "agree": ["Fair point not gonna lie", "Yeah this tracks from my side", "I can get behind this take", "Honestly this lands pretty well", "Hard to argue with that framing"],
        "pushback": ["I get it but missing key context", "Respectfully numbers suggest otherwise", "Not sure the premise fully holds", "I think this overstates the downside", "Counterpoint incentives drive the result"],
        "curious": ["Curious how this scales in practice", "What happens under edge cases here", "Would love more detail on assumptions", "How does this compare historically", "What signal are we actually seeing"],
        "personal": ["This lines up with my experience", "Seen similar outcomes on small teams", "Ive shipped with the same constraint", "We tried this results were mixed", "This improved things for us a ton"],
        "praise": ["Clear framing seriously helpful thread", "Love the nuance you brought here", "Elegant summary thanks for sharing", "Solid synthesis saved me a read", "Concise and useful well done"],
        "question": ["What would change your mind here", "Where would this approach break", "How do you measure success here", "What tradeoff are you accepting", "What risks are you discounting"],
    },
    "zh": {
        "agree": ["确实有道理", "这个说法挺贴切", "思路很清晰", "很认同这个观点", "讲得挺到位"],
        "pushback": ["但有点忽略背景", "数据可能不完全支持", "前提有待验证", "结论有点早", "风险被低估了"],
        "curious": ["想看看实际落地", "细节还有哪些", "边界条件怎么处理", "历史上怎么做的", "数据口径是什么"],
        "personal": ["和我实践类似", "团队也遇到过", "我们试过效果一般", "我们这么做提升明显", "跟我的经验相符"],
        "praise": ["总结很到位", "表达很清楚", "信息量挺大", "很有启发", "思考很细致"],
        "question": ["你会如何评估", "如果失败怎么办", "替代方案有吗", "成本如何量化", "关键假设是什么"],
    },
    "ja": {
        "agree": ["たしかに筋が通ってる", "この視点はしっくりくる", "納得感がある話", "言い回しが絶妙", "整理がうまい"],
        "pushback": ["前提がやや弱いかも", "背景の説明が足りない", "数字が追いついていない", "結論が早い気がする", "リスクが軽め"],
        "curious": ["実運用でどう回るのか", "前提条件を知りたい", "比較の軸はどこか", "どこで崩れるのか", "成功指標は何か"],
        "personal": ["自分の経験とも近い", "チームでも似た話があった", "試したが結果は微妙だった", "導入して改善した", "体感と合っている"],
        "praise": ["要点がわかりやすい", "視野が広い整理", "示唆が多い", "学びが大きい", "良いまとめ"],
        "question": ["何が転換点になる", "どの条件で破綻する", "成功の定義は何", "検証方法は何", "代替案はある"],
    },
    "ko": {
        "agree": ["말이 꽤 설득력 있다", "이 관점은 꽤 맞는 듯", "맥락이 잘 잡혀 있다", "정리 깔끔해서 좋다", "이건 인정하게 된다"],
        "pushback": ["배경 설명이 부족해 보임", "데이터가 아직 약한 듯", "결론이 조금 빠른 편", "리스크를 가볍게 봄", "전제가 애매한 느낌"],
        "curious": ["실무에서 어떻게 굴릴지", "경계 조건은 어디인지", "비교 기준이 뭔지", "어디서부터 흔들리는지", "성공 지표는 뭔지"],
        "personal": ["내 경험과도 비슷함", "팀에서도 비슷한 일 있었음", "해봤는데 결과는 애매했음", "도입하고 꽤 개선됨", "체감과 잘 맞음"],
        "praise": ["핵심 정리가 좋다", "표현이 명료해서 좋다", "배울 점이 많다", "통찰이 인상적이다", "요약이 훌륭하다"],
        "question": ["여기서 기준은 뭐지", "어떤 상황에선 깨질까", "검증은 어떻게 하지", "대안은 뭘로 보나", "가정은 무엇인가"],
    },
}

FILLERS_EN = ["tbh","right now","in practice","for real","in context","from experience","in the wild","no fluff","genuinely"]

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
def _words_en(s:str): return WORD_RE.findall(s or "")

def _punct_pattern(s:str) -> str:
    return re.sub(r"[A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\s]", "", s or "")

def _last_bigram(s:str, lang:str) -> str:
    if lang in {"zh","ja","ko"}:
        t = re.sub(r"\s+","", s or "")
        return t[-2:] if len(t)>=2 else t
    ws = _words_en(s); return " ".join(ws[-2:]) if len(ws)>=2 else " ".join(ws)

def _normalize_cmp(s:str)->str:
    s = (s or "").strip().lower()
    s = re.sub(r"[#@]\w+","", s); s = re.sub(r"https?://\S+","", s)
    s = re.sub(r"[^\w\s\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]","", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def _style_signature(text:str, lang:str)->str:
    opener = (text.strip().split(" ",1)[0] if " " in text else text[:2]).lower()
    pat = _punct_pattern(text); lb = _last_bigram(text, lang)
    return hashlib.sha1(f"{lang}|{opener}|{pat}|{lb}".encode()).hexdigest()[:24]

def _sha(text:str)->str: return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
def _now()->int: return int(time.time())
def _is_cjk(lang:str)->bool: return lang in {"zh","ja","ko"}

def enforce_length(text:str, lang:str)->Optional[str]:
    t = (text or "").strip()
    if not t: return None
    if _is_cjk(lang):
        t2 = re.sub(r"\s+","", t); n = len(t2)
        mn, mx = OTP_RULES["length"]["cjk_chars_min"], OTP_RULES["length"]["cjk_chars_max"]
        if n < mn:
            add = ""  # no punctuation padding for CJK per rules
            while n < mn and len(t2) < mx:
                # pad with last char if absolutely needed (rare)
                t2 += t2[-1] if t2 else "好"
                n = len(t2)
            t = t2
        elif n > mx:
            t = t2[:mx]
        return t
    ws = _words_en(t); mn, mx = OTP_RULES["length"]["words_min"], OTP_RULES["length"]["words_max"]
    if len(ws) < mn:
        while len(ws) < mn and len(ws)+1 <= mx:
            ws.append(random.choice(FILLERS_EN))
        t = " ".join(ws)
    elif len(ws) > mx:
        ws = ws[:mx]; t = " ".join(ws)
    return t

def sanitize_and_validate(text: str, lang: str) -> Optional[str]:
    if not text: return None
    s = strip_nonhuman(text, lang)
    if not s: return None
    if contains_blacklisted(s): return None
    # hashtags or @mentions not possible now, but double-check
    if "#" in s or "@" in s: return None
    # enforce strict lengths after cleanup
    s2 = enforce_length(s, lang) or ""
    if not s2: return None
    # final guard: ensure no punctuation or emoji slipped through
    if EMOJI_RE.search(s2): return None
    if lang in {"zh","ja","ko"}:
        if re.search(r"[^\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\s]", s2): return None
        if len(re.sub(r"\s+","", s2)) < OTP_RULES["length"]["cjk_chars_min"]: return None
    else:
        if re.search(r"[^A-Za-z0-9\s]", s2): return None
        wc = len(_words_en(s2))
        if wc < OTP_RULES["length"]["words_min"] or wc > OTP_RULES["length"]["words_max"]: return None
    return s2.strip()

class OfflineGenerator:
    def __init__(self, db_path:str): self.db_path = db_path
    def _seen_text(self, text:str)->bool:
        h = _sha(_normalize_cmp(text))
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute("SELECT 1 FROM comments_seen WHERE hash=?", (h,)).fetchone()
            if c: return True
            conn.execute("INSERT OR REPLACE INTO comments_seen(hash, created_at) VALUES(?,?)",(h,_now(),)); conn.commit()
        return False
    def _seen_sig(self, sig:str)->bool:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute("SELECT 1 FROM style_signatures_seen WHERE sig=?", (sig,)).fetchone()
            if c: return True
            conn.execute("INSERT OR REPLACE INTO style_signatures_seen(sig, created_at) VALUES(?,?)",(sig,_now(),)); conn.commit()
        return False
    def _candidate(self, lang:str)->str:
        table = BUCKETS.get(lang) or BUCKETS["en"]
        bucket = random.choice(list(table.values())); base = random.choice(bucket)
        return base.strip()
    def _generate_unique(self, lang:str)->Optional[str]:
        for _ in range(32):
            c = self._candidate(lang)
            c = sanitize_and_validate(c, lang) or ""
            if not c: continue
            sig = _style_signature(c, lang)
            if not self._seen_sig(sig) and not self._seen_text(c): return c
        return None
    def two_comments(self, lang:str)->List[str]:
        out: List[str] = []
        # try to get two unique, clean comments
        for _ in range(20):
            c = self._generate_unique(lang)
            if c and c not in out: out.append(c)
            if len(out) == 2: break
        # fallback to English if needed
        while len(out) < 2:
            c = self._generate_unique("en") or "Genuinely curious how this plays out"
            c = sanitize_and_validate(c, "en") or c
            out.append(c)
        return out[:2]

offline_gen = OfflineGenerator(DB_PATH)

SYSTEM_PROMPT = (
    "You write very short, natural human comments.\n"
    "Rules:\n"
    "1) Output exactly TWO comments as a JSON array of strings.\n"
    "2) No hashtags, no links, no emoji, no punctuation at all.\n"
    "3) One thought only, human tone, no boilerplate.\n"
    "4) Keep them distinct from each other.\n"
)

def build_user_prompt(post_text: str, lang: str) -> str:
    base = f"Post:\n{post_text.strip()}\n\n"
    if lang in {"zh","ja","ko"}:
        base += (
            "Language: write in the post's native language (ZH/JA/KO accordingly).\n"
            "Length: each comment 12–28 native characters.\n"
            "Do not use punctuation or emoji, only characters and optional spaces.\n"
        )
    else:
        base += (
            "Language: English.\n"
            "Length: each comment 6–13 words.\n"
            "Do not use punctuation or emoji, only words and spaces.\n"
        )
    base += 'Return ONLY the JSON array, e.g., ["first comment","second comment"].'
    return base

def parse_json_list(text: str) -> Optional[List[str]]:
    try:
        m = re.search(r"\[[\s\S]*\]", text)
        if not m: return None
        data = json.loads(m.group(0))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        return None
    return None

def _clean_two(arr: List[str], lang: str) -> List[str]:
    out: List[str] = []
    for c in arr:
        s = sanitize_and_validate(c, lang)
        if s and s not in out:
            out.append(s)
        if len(out) == 2:
            break
    return out[:2]

def provider_groq_two(text: str, lang: str) -> Optional[List[str]]:
    if not USE_GROQ or not groq_client: return None
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": build_user_prompt(text, lang)}],
            temperature=0.8, max_tokens=128,
        )
        content = (resp.choices[0].message.content or "").strip()
        arr = parse_json_list(content) or []
        two = _clean_two(arr, lang)
        return two if len(two) == 2 else None
    except Exception as e:
        logger.warning("Groq failed: %s", e); return None

def provider_openai_two(text: str, lang: str) -> Optional[List[str]]:
    if not USE_OPENAI or not openai_client: return None
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": build_user_prompt(text, lang)}],
            temperature=0.8, max_tokens=128,
        )
        content = (resp.choices[0].message.content or "").strip()
        arr = parse_json_list(content) or []
        two = _clean_two(arr, lang)
        return two if len(two) == 2 else None
    except Exception as e:
        logger.warning("OpenAI failed: %s", e); return None

def provider_gemini_two(text: str, lang: str) -> Optional[List[str]]:
    if not USE_GEMINI or not gemini_model: return None
    try:
        resp = gemini_model.generate_content([{"text": SYSTEM_PROMPT},{"text": build_user_prompt(text, lang)}])
        content = (resp.text or "").strip()
        arr = parse_json_list(content) or []
        two = _clean_two(arr, lang)
        return two if len(two) == 2 else None
    except Exception as e:
        logger.warning("Gemini failed: %s", e); return None

def gen_two_with_providers(text: str, lang: str) -> List[str]:
    for fn in (provider_groq_two, provider_openai_two, provider_gemini_two):
        arr = fn(text, lang)
        if arr and len(arr) == 2:
            return arr
    return offline_gen.two_comments(lang)

@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status":"ok","groq":bool(USE_GROQ)}), 200

@app.route("/ping", methods=["GET"])
def ping(): return ("",204)

def parse_json_request() -> Dict[str, Any]:
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        raise CrownTALKError("invalid_json", code="invalid_json")
    if not isinstance(data, dict):
        raise CrownTALKError("invalid_body", code="invalid_body")
    return data

def make_comment_record(url: str, handle: str, lang: str, text: str) -> Dict[str, Any]:
    sig = _style_signature(text, lang)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO comments(url,handle,lang,comment,style_sig,created_at) VALUES(?,?,?,?,?,?)",
                     (url, handle, lang, text, sig, _now()))
        conn.commit()
    return {"lang": lang, "text": text}

@app.route("/comment", methods=["POST", "OPTIONS"])
def comment():
    if request.method == "OPTIONS": return ("",204)
    try:
        data = parse_json_request()
        urls_raw = data.get("urls") or ""
        handle = (data.get("handle") or "").strip().lstrip("@")[:32]
        urls = clean_and_normalize_urls(urls_raw)
        if not urls: raise CrownTALKError("no_urls", code="no_urls")

        results, failed = [], []
        for i in range(0, len(urls), BATCH_SIZE):
            batch = urls[i:i+BATCH_SIZE]
            for url in batch:
                try:
                    t = fetch_tweet_data(url)
                except CrownTALKError as e:
                    failed.append({"url": url, "error": e.code}); continue
                except Exception:
                    logger.exception("fetch_tweet_data failed for %s", url)
                    failed.append({"url": url, "error": "fetch_failed"}); continue

                time.sleep(PER_URL_SLEEP)
                tgt_lang = detect_lang(t.lang, t.text)

                try:
                    two = gen_two_with_providers(t.text, tgt_lang)
                    # extra guard: sanitize again here
                    clean_two = []
                    for c in two:
                        sc = sanitize_and_validate(c, tgt_lang)
                        if sc: clean_two.append(sc)
                    if len(clean_two) < 2:
                        clean_two = offline_gen.two_comments(tgt_lang)
                except Exception:
                    logger.exception("generation failed for %s", url)
                    clean_two = offline_gen.two_comments(tgt_lang)

                comments_payload = [make_comment_record(url, t.handle or "", tgt_lang, c) for c in clean_two[:2]]

                results.append({
                    "url": url,
                    "handle": t.handle or "",
                    "lang": tgt_lang,
                    "author_name": t.author_name or "",
                    "comments": comments_payload,
                })

        return jsonify({"results": results, "failed": failed}), 200

    except CrownTALKError as e:
        return jsonify({"error": str(e), "code": e.code}), 400
    except Exception:
        logger.exception("Unhandled error during /comment")
        return jsonify({"error": "internal_error", "code": "internal_error"}), 500

@app.route("/reroll", methods=["POST"])
def reroll():
    try:
        data = parse_json_request()
        url = (data.get("url") or "").strip()
        handle = (data.get("handle") or "").strip().lstrip("@")[:32]
        if not url: raise CrownTALKError("no_url", code="no_url")

        try:
            t = fetch_tweet_data(url)
        except CrownTALKError as e:
            return jsonify({"url": url, "error": e.code, "comments": []}), 502
        except Exception:
            logger.exception("fetch_tweet_data failed for %s", url)
            return jsonify({"url": url, "error": "fetch_failed", "comments": []}), 502

        tgt_lang = detect_lang(t.lang, t.text)
        try:
            arr = gen_two_with_providers(t.text, tgt_lang)
            clean_two = []
            for c in arr:
                sc = sanitize_and_validate(c, tgt_lang)
                if sc: clean_two.append(sc)
            if len(clean_two) < 2:
                clean_two = offline_gen.two_comments(tgt_lang)
        except Exception:
            logger.exception("reroll generation failed for %s", url)
            clean_two = offline_gen.two_comments(tgt_lang)

        comments = [make_comment_record(url, t.handle or "", tgt_lang, c) for c in clean_two[:2]]
        return jsonify({"url": url, "comments": comments}), 200

    except CrownTALKError as e:
        return jsonify({"url": url, "error": str(e), "comments": [], "code": e.code}), 502
    except Exception:
        logger.exception("Unhandled error during reroll for %s", url)
        return jsonify({"url": url, "error": "internal_error", "comments": [], "code": "internal_error"}), 500

def main() -> None:
    init_db()
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
