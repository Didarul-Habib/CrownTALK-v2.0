import os
import re
import time
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------
#  CLEAN TWEET URL
# ---------------------------------------
def clean_url(url: str) -> str:
    return url.split("?")[0].strip()

# ---------------------------------------
#  FETCH METHOD A — TWXT (Primary)
# ---------------------------------------
async def fetch_twxt(url: str):
    api = f"https://api.twxt.live/v1/tweet?url={url}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(api)
            if r.status_code != 200:
                return None
            data = r.json()
            if "text" in data and data["text"]:
                return data["text"]
    except:
        return None
    return None

# ---------------------------------------
#  FETCH METHOD B — VXTwitter
# ---------------------------------------
async def fetch_vx(url: str):
    try:
        api = url.replace("twitter.com", "vx-twitter.com").replace("x.com", "vx-twitter.com")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(api)
            if r.status_code != 200:
                return None
            html = r.text
            match = re.search(r'<meta property="og:description" content="(.*?)"', html)
            if match:
                return match.group(1)
    except:
        return None
    return None

# ---------------------------------------
#  FETCH METHOD C — CATX API (Very reliable)
# ---------------------------------------
async def fetch_catx(url: str):
    try:
        api = f"https://api.catx.one/tweet?url={url}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(api)
            if r.status_code != 200:
                return None
            j = r.json()
            if "text" in j and j["text"]:
                return j["text"]
    except:
        return None
    return None

# ---------------------------------------
#  FETCH METHOD D — Raw HTML Scrape
# ---------------------------------------
async def fetch_html(url: str):
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            html = r.text
            # OG tag (works surprisingly often)
            match = re.search(r'<meta property="og:description" content="(.*?)"', html)
            if match:
                return match.group(1)
    except:
        return None
    return None

# ---------------------------------------
#  MASTER FETCHER (Rotates sources)
# ---------------------------------------
async def get_tweet_text(url: str):
    url = clean_url(url)
    sources = [fetch_twxt, fetch_vx, fetch_catx, fetch_html]

    for fetcher in sources:
        text = await fetcher(url)
        if text:
            return text

        # micro delay (protect from rate-limit)
        await asyncio.sleep(0.04)

    return None

# ---------------------------------------
#  OPENAI COMMENT GENERATOR
# ---------------------------------------
async def generate_comment(text: str):
    prompt = (
        "Write a short, casual, humanlike reply to this tweet:\n\n"
        f"Tweet: {text}\n\n"
        "Reply:"
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 60,
                    "temperature": 0.8,
                },
            )

            if r.status_code != 200:
                return None

            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    except:
        return None

# ---------------------------------------
#  REQUEST MODEL
# ---------------------------------------
class TweetBatch(BaseModel):
    urls: list

# ---------------------------------------
#  MAIN PROCESSOR
# ---------------------------------------
@app.post("/process")
async def process(data: TweetBatch):
    results = []

    for url in data.urls:
        cleaned = clean_url(url)

        tweet_text = await get_tweet_text(cleaned)

        if tweet_text:
            comment = await generate_comment(tweet_text)
        else:
            comment = None

        results.append({
            "url": cleaned,
            "tweet": tweet_text,
            "comment": comment,
            "status": "success" if comment else "failed"
        })

        # small pause protects from bans
        await asyncio.sleep(0.05)

    return {"results": results}
