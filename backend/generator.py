import random
import re
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from utils import CrownTALKError, TweetData, fetch_tweet_data, naive_lang_detect, safe_excerpt


# ===== Theme Engine =====


@dataclass
class ThemeConfig:
    id: str
    label: str
    tone: str
    emoji: str
    intensity: str  # "low", "medium", "high"


THEMES: Dict[str, ThemeConfig] = {
    "default": ThemeConfig(
        id="default",
        label="Default Crown",
        tone="supportive and lightly hyped",
        emoji="👑",
        intensity="medium",
    ),
    "hype": ThemeConfig(
        id="hype",
        label="Hyper Hype",
        tone="ultra-energetic, big hype, friendly",
        emoji="🚀",
        intensity="high",
    ),
    "savage": ThemeConfig(
        id="savage",
        label="Savage Roast",
        tone="playfully savage but not genuinely cruel",
        emoji="🔥",
        intensity="high",
    ),
    "wholesome": ThemeConfig(
        id="wholesome",
        label="Wholesome",
        tone="soft, kind, and encouraging",
        emoji="🌸",
        intensity="low",
    ),
    "sarcastic": ThemeConfig(
        id="sarcastic",
        label="Dry Sarcasm",
        tone="dry, witty, but not hateful",
        emoji="🙃",
        intensity="medium",
    ),
    "poetic": ThemeConfig(
        id="poetic",
        label="Poetic",
        tone="lyrical, reflective, slightly dramatic",
        emoji="🎭",
        intensity="medium",
    ),
    "minimal": ThemeConfig(
        id="minimal",
        label="Minimal",
        tone="short, punchy, minimal words",
        emoji="✨",
        intensity="low",
    ),
}


def get_theme_ids() -> List[str]:
    return list(THEMES.keys())


class LanguageMode(str, Enum):
    EN = "en"
    BN = "bn"
    DUAL = "dual"


def normalize_language_mode(value: Any) -> str:
    """
    Normalize user-provided language mode into one of: "en", "bn", "dual".
    """
    if not isinstance(value, str):
        return LanguageMode.DUAL.value
    v = value.lower().strip()
    if v in {"en", "english"}:
        return LanguageMode.EN.value
    if v in {"bn", "bangla", "bengali"}:
        return LanguageMode.BN.value
    if v in {"dual", "both"}:
        return LanguageMode.DUAL.value
    return LanguageMode.DUAL.value


# ===== Anti-Repetition Cache =====


class AntiRepetitionCache:
    """
    Simple in-memory cache to avoid repeating the exact same comment text.
    """

    def __init__(self, max_size: int = 200) -> None:
        self.max_size = max_size
        self._queue: Deque[str] = deque()
        self._set = set()

    def contains(self, text: str) -> bool:
        key = self._normalize(text)
        return key in self._set

    def add(self, text: str) -> None:
        key = self._normalize(text)
        if key in self._set:
            return
        self._queue.append(key)
        self._set.add(key)
        if len(self._queue) > self.max_size:
            old = self._queue.popleft()
            self._set.discard(old)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()


ANTI_REP_CACHE = AntiRepetitionCache(max_size=300)


# ===== Context Engine =====


@dataclass
class ContextFeatures:
    mood: str
    is_question: bool
    has_link: bool
    keywords: List[str]
    base_lang: str


def extract_keywords(text: str, limit: int = 4) -> List[str]:
    """
    Tiny keyword extractor: pick notable words, not stopwords, not emojis.
    """
    text = re.sub(r"http\S+", "", text)
    tokens = re.findall(r"[A-Za-z\u0980-\u09FF]{3,}", text)
    lower_tokens = [t.lower() for t in tokens]

    stopwords = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "have",
        "from",
        "just",
        "been",
        "about",
        "your",
        "their",
        "will",
        "what",
        "when",
        "where",
        "how",
        "এবং",
        "করে",
        "কেন",
        "কোথায়",
    }

    keywords: List[str] = []
    for tok in lower_tokens:
        if tok in stopwords:
            continue
        if tok not in keywords:
            keywords.append(tok)
        if len(keywords) >= limit:
            break
    return keywords


def detect_mood(text: str) -> str:
    text_l = text.lower()
    if any(x in text_l for x in ["?", "what do you think", "any thoughts"]):
        return "curious"
    if any(x in text_l for x in ["sad", "tired", "lonely", "broken"]):
        return "sad"
    if any(x in text_l for x in ["win", "proud", "promotion", "success", "goal"]):
        return "celebratory"
    if any(x in text_l for x in ["angry", "mad", "furious", "annoyed"]):
        return "angry"
    if any(x in text_l for x in ["love", "grateful", "thankful"]):
        return "grateful"
    return "neutral"


def build_context_features(tweet: TweetData) -> ContextFeatures:
    is_question = "?" in tweet.text
    has_link = bool(re.search(r"http\S+", tweet.text))
    keywords = extract_keywords(tweet.text)
    base_lang = naive_lang_detect(tweet.text)

    mood = detect_mood(tweet.text)

    return ContextFeatures(
        mood=mood,
        is_question=is_question,
        has_link=has_link,
        keywords=keywords,
        base_lang=base_lang,
    )


# ===== Tiny Bangla “translation” helpers (offline, approximate) =====

BN_TEMPLATES = {
    "soft_support": [
        "ভালো বলেছো, একদম মনের কথা বলেছো।",
        "তুমি যা বলছো, অনেকের ইচ্ছে কিন্তু ঠিক এমনটাই।",
        "ভালো লাগলো তোমার এই ভাবনা, অনেক রিলেটেবল।",
    ],
    "hype": [
        "ওহ, একদম 🔥 লেভেলের কথা বলেছো!",
        "এইটা তো পুরো পাওয়ার প্যাকড স্টেটমেন্ট!",
        "এই এনার্জি আর হার্ডওয়ার্ক থাকলে অনেক দূর যাবে।",
    ],
    "savage": [
        "এভাবে বলে সরাসরি হার্টে গিয়ে লাগলো।",
        "ভাই, আজকে তো কারো ঘুম উড়ে যাবে এই টুইট দেখে।",
        "এইটা দেখে কেউ কেউ গোপনে কাঁদবে, নিশ্চিত।",
    ],
}


def rough_bn_comment(mood: str, theme: ThemeConfig) -> str:
    """
    Generate a short Bangla side-comment, mood-aware and theme-aware.
    """
    if theme.id == "wholesome":
        bucket = BN_TEMPLATES["soft_support"]
    elif theme.id in {"hype", "poetic"}:
        bucket = BN_TEMPLATES["hype"]
    elif theme.id == "savage":
        bucket = BN_TEMPLATES["savage"]
    else:
        bucket = BN_TEMPLATES["soft_support"] + BN_TEMPLATES["hype"]

    base = random.choice(bucket)
    if mood == "sad" and theme.id != "savage":
        base += " হাল ছেড়ো না, ধীরে ধীরে সব ঠিক হয়ে যাবে।"
    elif mood == "celebratory":
        base += " এইভাবে চলতে থাকো, সামনে আরও ভালো কিছু আসবে।"
    return base


# ===== Core Generator =====


def build_english_comment(
    tweet: TweetData,
    ctx: ContextFeatures,
    theme: ThemeConfig,
) -> str:
    """
    Build an English comment line based on context + theme.
    """
    # Build a soft subject line from keywords
    if ctx.keywords:
        key_phrase = ", ".join(ctx.keywords[:2])
    else:
        key_phrase = "this"

    author = tweet.author_name or "you"
    mood = ctx.mood
    t_emoji = theme.emoji or "👑"

    templates: List[str] = []

    if theme.id == "hype":
        templates = [
            f"{t_emoji} {author}, this take on {key_phrase} is crazy on point. Keep that energy.",
            f"{t_emoji} Lowkey obsessed with how you framed {key_phrase}. This is big.",
            f"{t_emoji} This isn’t just a tweet, it’s a whole mood. {key_phrase.title()} on max volume.",
        ]
    elif theme.id == "savage":
        templates = [
            f"{t_emoji} Somewhere out there, someone read this and took it *very* personally.",
            f"{t_emoji} You really woke up and chose {key_phrase} today huh.",
            f"{t_emoji} This tweet is a free therapy session and a jump scare at the same time.",
        ]
    elif theme.id == "wholesome":
        templates = [
            f"{t_emoji} Love how gently you put this, {author}. Internet needs more of this energy.",
            f"{t_emoji} This feels like a warm message to anyone scrolling through in silence.",
            f"{t_emoji} Quiet tweet, loud comfort. Thanks for sharing this.",
        ]
    elif theme.id == "sarcastic":
        templates = [
            f"{t_emoji} Ah yes, just another totally normal day on the timeline.",
            f"{t_emoji} This tweet is doing the emotional labor my brain refused to do.",
            f"{t_emoji} Reading this like ‘wow, so we all agreed not to be normal anymore?’",
        ]
    elif theme.id == "poetic":
        templates = [
            f"{t_emoji} Reads like a tiny chapter from a bigger story nobody sees but you.",
            f"{t_emoji} There’s a quiet kind of cinema in the way you talk about {key_phrase}.",
            f"{t_emoji} This tweet feels like a late-night city street: a little loud, but very real.",
        ]
    elif theme.id == "minimal":
        templates = [
            f"{t_emoji} This hits.",
            f"{t_emoji} Understood.",
            f"{t_emoji} Loud and clear.",
        ]
    else:
        templates = [
            f"{t_emoji} Can’t lie, this take on {key_phrase} actually lands.",
            f"{t_emoji} Solid tweet. Simple, honest, and very scroll-stopping.",
            f"{t_emoji} Timeline needed this one more than it realized.",
        ]

    # Mood tweaks
    if mood == "sad" and theme.id not in {"savage", "sarcastic"}:
        templates.append(
            f"{t_emoji} This feels heavy, but also very human. Hope you’re taking care of yourself."
        )
    elif mood == "celebratory":
        templates.append(
            f"{t_emoji} Big congrats energy here. Proud of you even from the timeline."
        )

    return random.choice(templates)


def ensure_not_repeated(text: str, theme: ThemeConfig) -> str:
    """
    Slightly mutate the comment if we've already used it recently.
    """
    if not ANTI_REP_CACHE.contains(text):
        ANTI_REP_CACHE.add(text)
        return text

    # Add a soft variant tag to differentiate
    suffixes = [
        " (still true tbh)",
        " (had to say it again)",
        " (no notes)",
        " (this stays in my head)",
    ]
    mutated = text + random.choice(suffixes)
    ANTI_REP_CACHE.add(mutated)
    return mutated


def build_dual_language_comment(
    en_comment: str,
    ctx: ContextFeatures,
    theme: ThemeConfig,
) -> Dict[str, str]:
    """
    Combine English + Bangla in a structured way.
    """
    bn = rough_bn_comment(ctx.mood, theme)
    return {
        "en": en_comment,
        "bn": bn,
    }


def build_single_output(
    tweet: TweetData,
    theme: ThemeConfig,
    language_mode: str,
) -> Dict[str, Any]:
    """
    Produce the final structured payload for one tweet.
    """
    ctx = build_context_features(tweet)
    en_comment = build_english_comment(tweet, ctx, theme)
    en_comment = ensure_not_repeated(en_comment, theme)

    lang_mode_enum = LanguageMode(language_mode)

    if lang_mode_enum == LanguageMode.EN:
        body: Dict[str, Any] = {"en": en_comment}
    elif lang_mode_enum == LanguageMode.BN:
        # Bangla-only: we still generate EN internally and then only ship BN
        dual = build_dual_language_comment(en_comment, ctx, theme)
        body = {"bn": dual["bn"]}
    else:
        dual = build_dual_language_comment(en_comment, ctx, theme)
        body = dual

    meta = {
        "keywords": ctx.keywords,
        "mood": ctx.mood,
        "is_question": ctx.is_question,
        "base_lang": ctx.base_lang,
        "excerpt": safe_excerpt(tweet.text),
        "author_name": tweet.author_name,
        "theme_id": theme.id,
        "theme_label": theme.label,
        "theme_tone": theme.tone,
    }

    return {
        "url": tweet.url,
        "comment": body,
        "meta": meta,
    }


def generate_comments_for_urls(
    urls: List[str],
    theme_id: str,
    language_mode: str,
) -> List[Dict[str, Any]]:
    """
    Top-level API used by main.py.

    For each URL:
    - Fetch tweet data via VXTwitter
    - Run context engine
    - Generate multilingual, theme-aware comments
    """
    theme = THEMES.get(theme_id) or THEMES["default"]
    lang_mode = normalize_language_mode(language_mode)

    results: List[Dict[str, Any]] = []

    for url in urls:
        try:
            tweet = fetch_tweet_data(url)
        except CrownTALKError as e:
            results.append(
                {
                    "url": url,
                    "error": e.code,
                    "message": str(e),
                }
            )
            continue
        except Exception:
            # Fully guard against unexpected parse/HTTP errors for any single URL
            results.append(
                {
                    "url": url,
                    "error": "unexpected_fetch_error",
                    "message": "Unexpected error while fetching tweet.",
                }
            )
            continue

        item = build_single_output(tweet, theme, lang_mode)
        results.append(item)

        # Slight jitter so patterns in logs are less uniform
        time.sleep(0.05 + random.random() * 0.05)

    return results
