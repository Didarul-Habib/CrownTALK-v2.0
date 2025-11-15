import re
import time
import requests
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langdetect import detect
from openai import OpenAI

app = FastAPI()

client = OpenAI()

# ----------------------------
# Mount static + templates
# ----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----------------------------
# Load Style Guide
# ----------------------------
with open("comment_style_guide.txt", "r", encoding="utf-8") as f:
    STYLE_GUIDE = f.read()


# ----------------------------
# Clean URLs
# ----------------------------
def clean_url(url: str):
    url = url.strip()
    url = re.sub(r"\?.*$", "", url)
    url = url.replace("mobile.", "")
    return url


# ----------------------------
# Fetch tweet text (VxTwitter)
# ----------------------------
def fetch_tweet_text(url):
    url = clean_url(url)
    api_url = "https://api.vxtwitter.com/" + url.split("twitter.com/")[-1].split("x.com/")[-1]

    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        if "tweet" in data and "text" in data["tweet"]:
            return data["tweet"]["text"]

        return None
    except:
        return None


# ----------------------------
# Translate to English
# ----------------------------
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate this to English only."},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    except:
        return text


# ----------------------------
# Generate AI comments
# ----------------------------
def generate_comments(tweet_text):
    prompt = f"""
You are CrownTALK 👑 — a humanlike comment generator.

Rules:
- Two comments
- Each 5–12 words
- No punctuation at end
- No emojis
- No hype words
- No repeated patterns
- Natural human slang allowed
- Must follow the style guide exactly

STYLE GUIDE:
{STYLE_GUIDE}

Tweet:
"{tweet_text}"

Return exactly two lines.
"""

    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.65,
            )
            out = r.choices[0].message.content.strip().split("\n")
            out = [re.sub(r"[.,!?;:]+$", "", x).strip() for x in out]
            out = [x for x in out if 5 <= len(x.split()) <= 12]

            if len(out) >= 2:
                return out[:2]
        except:
            time.sleep(1)

    return ["could not generate reply", "generator failed"]


# ----------------------------
# PAGE ROUTE
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ----------------------------
# COMMENT API
# ----------------------------
@app.post("/comment")
async def comment_api(request: Request):
    data = await request.json()

    if "tweets" not in data:
        return JSONResponse({"error": "Invalid request"}, status_code=400)

    raw_links = data["tweets"]
    cleaned = [clean_url(x) for x in raw_links]

    results = []

    for url in cleaned:
        # Retry tweet fetching 3 times
        txt = None
        for _ in range(3):
            txt = fetch_tweet_text(url)
            if txt:
                break
            time.sleep(1)

        if not txt:
            results.append({"error": "Could not fetch this tweet (private/deleted)", "url": url})
            continue

        en = translate_to_english(txt)
        comments = generate_comments(en)

        results.append({
            "url": url,
            "comments": comments
        })

    return JSONResponse({"results": results, "ok": True})
