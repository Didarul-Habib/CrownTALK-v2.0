import random
import re
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List

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


class LanguageMode(str, Enum):
    EN = "en"
    BN = "bn"
    DUAL = "dual"

    # NOTE: language_mode from the client is effectively ignored now;
    # we always do "auto + english". Kept only for backwards-compat.


def normalize_language_mode(value: Any) -> str:
    """
    Kept for backwards compatibility with older frontends.

    We do not rely on this anymore; generator is "auto + english" regardless.
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

    def __init__(self, max_size: int = 300) -> None:
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


ANTI_REP_CACHE = AntiRepetitionCache(max_size=400)


# ===== Tiny NLP helpers =====

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "have",
    "from",
    "just",
    "about",
    "your",
    "their",
    "will",
    "what",
    "when",
    "where",
    "how",
    "youre",
    "theyre",
    "cant",
    "dont",
    "doesnt",
    "im",
    "its",
    "are",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "bullish",
    "win",
    "winning",
    "pump",
    "clean",
    "progress",
    "love",
    "excited",
    "proud",
    "amazing",
    "fire",
    "smooth",
}

NEGATIVE_WORDS = {
    "bad",
    "bearish",
    "risk",
    "dump",
    "rug",
    "scam",
    "annoying",
    "angry",
    "tired",
    "broken",
    "fear",
    "red",
    "down",
    "pain",
}

# No explicit hate slurs here; keep it generic.
BANNED_PHRASES = {
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "bastard",
}

FILLER_TOKENS = [
    "lowkey",
    "ngl",
    "fr",
    "tbh",
    "no cap",
    "not gonna lie",
    "for real",
]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Very small keyword extractor — English-ish side only.
    """
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s#]", " ", text)
    words = [w for w in text.split() if w and w not in STOPWORDS]
    if not words:
        return []

    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    # take top by frequency then by order
    sorted_words = sorted(counts.items(), key=lambda x: (-x[1], words.index(x[0])))
    keywords: List[str] = []
    for w, _ in sorted_words:
        if w.startswith("#"):
            continue
        if w.isdigit():
            continue
        keywords.append(w)
        if len(keywords) >= max_keywords:
            break
    return keywords


def guess_topic(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("giveaway", "airdrop", "retweet to win", "tag 3 friends")):
        return "giveaway"
    if any(k in t for k in ("chart", "support", "resistance", "ath", "price target", "%", "market cap")):
        return "chart"
    if "🧵" in text or len(text) > 220:
        return "thread"
    if len(text) < 80:
        return "one_liner"
    return "generic"


def is_crypto_tweet(text: str) -> bool:
    t = text.lower()
    crypto_tokens = [
        "btc",
        "eth",
        "sol",
        "bnb",
        "dex",
        "cex",
        "defi",
        "token",
        "airdop",
        "airdrop",
        "meme",
        "altcoin",
        "onchain",
        "pump",
        "dump",
    ]
    if any(tok in t for tok in crypto_tokens):
        return True
    # simple ticker guess: ALL CAPS 3–5 chars with digit or not
    return bool(re.search(r"\b[A-Z0-9]{3,5}\b", text))


def detect_mood(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["sad", "tired", "lonely", "broken", "exhausted"]):
        return "sad"
    if any(x in t for x in ["angry", "mad", "furious", "annoyed", "pissed"]):
        return "angry"
    if any(x in t for x in ["win", "promotion", "proud", "achieved", "milestone"]):
        return "celebratory"
    if any(x in t for x in ["love", "grateful", "thankful"]):
        return "grateful"
    if "?" in t:
        return "curious"
    return "neutral"


def analyze_sentiment(text: str) -> str:
    t = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    if pos > neg + 1:
        return "positive"
    if neg > pos + 1:
        return "negative"
    return "neutral"


@dataclass
class ContextFeatures:
    mood: str
    is_question: bool
    has_link: bool
    keywords: List[str]
    base_lang: str
    topic: str
    is_crypto: bool
    sentiment: str


def build_context_features(tweet: TweetData) -> ContextFeatures:
    text = tweet.text
    base_lang = naive_lang_detect(text)
    return ContextFeatures(
        mood=detect_mood(text),
        is_question=("?" in text),
        has_link=bool(re.search(r"http\S+", text)),
        keywords=extract_keywords(text),
        base_lang=base_lang,
        topic=guess_topic(text),
        is_crypto=is_crypto_tweet(text),
        sentiment=analyze_sentiment(text),
    )


# ===== Comment post-processing =====


def clean_and_constrain_comment(
    comment: str,
    target_min: int = 8,
    target_max: int = 22,
) -> str:
    """
    Remove banned phrases, normalize spacing, enforce length window.
    """
    if not comment:
        comment = "lowkey trying to process all this"

    # remove banned phrases (case-insensitive)
    lowered = comment.lower()
    for bad in BANNED_PHRASES:
        if bad in lowered:
            pattern = re.compile(re.escape(bad), flags=re.IGNORECASE)
            lowered = pattern.sub("", lowered)

    comment = re.sub(r"\s+", " ", lowered).strip()

    words = comment.split()
    if len(words) < target_min:
        while len(words) < target_min:
            words.append(random.choice(FILLER_TOKENS))
    elif len(words) > target_max:
        words = words[:target_max]

    comment = " ".join(words).strip()

    # basic punctuation ending
    if comment and comment[-1] not in ".!?…":
        comment += "."

    return comment


def ensure_not_repeated(text: str) -> str:
    """
    Slightly mutate the comment if we've already used it recently.
    """
    if not ANTI_REP_CACHE.contains(text):
        ANTI_REP_CACHE.add(text)
        return text

    suffixes = [
        " still hits tbh.",
        " kind of living rent free.",
        " cant lie that stuck with me.",
        " been thinking about this all day.",
    ]
    mutated = text + random.choice(suffixes)
    ANTI_REP_CACHE.add(mutated)
    return mutated


# ===== Core English generator =====


def _pick_key_phrase(ctx: ContextFeatures) -> str:
    if ctx.keywords:
        if len(ctx.keywords) >= 2:
            return ", ".join(ctx.keywords[:2])
        return ctx.keywords[0]
    return "this"


def build_english_comment(
    tweet: TweetData,
    ctx: ContextFeatures,
    theme: ThemeConfig,
) -> str:
    author = tweet.author_name or "you"
    key_phrase = _pick_key_phrase(ctx)
    t_emoji = theme.emoji or "👑"

    base_templates: List[str] = []

    if theme.id == "hype":
        if ctx.is_crypto or ctx.topic == "chart":
            base_templates = [
                f"{t_emoji} {author}, this {key_phrase} setup looks way too clean to ignore.",
                f"{t_emoji} Timeline is not ready for how {key_phrase} plays out ngl.",
                f"{t_emoji} Lowkey bullish on the way you framed {key_phrase} here.",
            ]
        else:
            base_templates = [
                f"{t_emoji} {author}, this take on {key_phrase} is doing laps in my head.",
                f"{t_emoji} The energy around {key_phrase} in this tweet is actually wild.",
                f"{t_emoji} This isn’t just a post, it’s a whole mood around {key_phrase}.",
            ]
    elif theme.id == "savage":
        base_templates = [
            f"{t_emoji} Somewhere, someone read this {key_phrase} line and felt very attacked.",
            f"{t_emoji} You really dropped this {key_phrase} bomb and walked away huh.",
            f"{t_emoji} This tweet is free therapy and a tiny jump scare for anyone into {key_phrase}.",
        ]
    elif theme.id == "wholesome":
        base_templates = [
            f"{t_emoji} The way you talk about {key_phrase} here is strangely comforting.",
            f"{t_emoji} This feels like a soft little reminder about {key_phrase} for anyone scrolling in silence.",
            f"{t_emoji} Quiet tweet, loud comfort around {key_phrase}.",
        ]
    elif theme.id == "sarcastic":
        base_templates = [
            f"{t_emoji} Love how casually you dropped a whole thesis on {key_phrase} like it’s nothing.",
            f"{t_emoji} This {key_phrase} take has strong ‘I’m tired of everyone but still online’ energy.",
            f"{t_emoji} Reading this like ‘oh cool, so we gave up on being normal about {key_phrase}’.",
        ]
    elif theme.id == "poetic":
        base_templates = [
            f"{t_emoji} Feels like a late-night monologue about {key_phrase} that only the timeline gets to hear.",
            f"{t_emoji} There’s a small cinema playing in my head after reading this on {key_phrase}.",
            f"{t_emoji} This tweet turns {key_phrase} into a little story more than a take.",
        ]
    elif theme.id == "minimal":
        base_templates = [
            f"{t_emoji} This on {key_phrase} just hits.",
            f"{t_emoji} Understood the {key_phrase} vibe instantly.",
            f"{t_emoji} Short but heavy on {key_phrase}.",
        ]
    else:
        base_templates = [
            f"{t_emoji} Solid angle on {key_phrase}, feels honest and very scroll-stopping.",
            f"{t_emoji} Can’t lie, this way of framing {key_phrase} actually lands.",
            f"{t_emoji} This tweet puts {key_phrase} into words a lot of people couldn’t.",
        ]

    # Mood / sentiment nudges
    if ctx.mood == "sad":
        base_templates.append(
            f"{t_emoji} This feels heavy around {key_phrase}, hope you’re giving yourself some room to breathe."
        )
    elif ctx.mood == "celebratory":
        base_templates.append(
            f"{t_emoji} Big congrats vibes here, you’ve earned this moment around {key_phrase}."
        )
    elif ctx.sentiment == "positive" and ctx.is_crypto:
        base_templates.append(
            f"{t_emoji} For a {key_phrase} setup this actually looks surprisingly clean ngl."
        )
    elif ctx.sentiment == "negative":
        base_templates.append(
            f"{t_emoji} You can hear the frustration around {key_phrase} between every line of this tweet."
        )

    return random.choice(base_templates)


# ===== Native-language helper =====


def build_native_comment(
    tweet: TweetData,
    ctx: ContextFeatures,
    theme: ThemeConfig,
    lang_code: str,
) -> str | None:
    """
    Very small offline native snippets, keyed off first keyword + mood.
    """
    kw = ctx.keywords[0] if ctx.keywords else ""

    if lang_code == "bn":
        kw_local = kw or "এটা"
        bn_templates = [
            "এই {kw_local} নিয়ে যা বলেছো, অনেকের মনের কথা।",
            "{kw_local} নিয়ে তোমার এই ভিউটা একদম honest লাগলো।",
            "টাইমলাইনে {kw_local} নিয়ে এমন টুইট খুব কম দেখি।",
            "{kw_local} বিষয়টা তুমি যেভাবে ধরেছো, সেটা বেশ real লেগেছে।",
            "শান্ত voice-এ {kw_local} নিয়ে অনেক strong point তুলে এনেছো।",
        ]
        tmpl = random.choice(bn_templates)
        return tmpl.format(kw_local=kw_local)

    if lang_code == "hi":
        kw_local = kw or "ये चीज"
        hi_templates = [
            "{kw_local} पर जो बात लिखी है, वो काफी real लग रही।",
            "{kw_local} वाला point तुमने काफी साफ तरीके से रख दिया।",
            "आजकल {kw_local} पर ऐसे honest take कम ही दिखते हैं।",
            "{kw_local} को जिस tone में लिखा है, वो काफी relatable है।",
            "सीधे तरीके से {kw_local} की जो बात उठाई है, वो सोचने लायक है।",
        ]
        tmpl = random.choice(hi_templates)
        return tmpl.format(kw_local=kw_local)

    if lang_code == "zh":
        kw_local = kw or "这个"
        zh_templates = [
            "{kw_local} 这段话 看着挺真 实的。",
            "你这样写 {kw_local} 感觉挺到位 的。",
            "最近关于 {kw_local} 的声音很多，这条很有感觉。",
            "你说的 {kw_local} 这点，挺值得慢慢想的。",
            "这样聊 {kw_local} ，气氛刚刚好，不吵不闹。",
        ]
        tmpl = random.choice(zh_templates)
        return tmpl.format(kw_local=kw_local)

    # unsupported native language
    return None


# ===== Final assembly =====


def build_single_output(
    tweet: TweetData,
    theme: ThemeConfig,
    language_mode: str,  # kept for compatibility, effectively ignored
) -> Dict[str, Any]:
    ctx = build_context_features(tweet)

    # 1) English comment (always present)
    en_comment_raw = build_english_comment(tweet, ctx, theme)
    en_comment = clean_and_constrain_comment(en_comment_raw)
    en_comment = ensure_not_repeated(en_comment)

    # 2) Native comment if tweet is not in English and we support that script
    native_lang = ctx.base_lang if ctx.base_lang in {"bn", "hi", "zh"} else None
    native_comment = None
    if native_lang:
        nc_raw = build_native_comment(tweet, ctx, theme, native_lang)
        if nc_raw:
            native_comment = clean_and_constrain_comment(nc_raw, target_min=6, target_max=20)
            native_comment = ensure_not_repeated(native_comment)

    comment_payload: Dict[str, str] = {"en": en_comment}
    if native_lang and native_comment:
        comment_payload[native_lang] = native_comment

    meta = {
        "keywords": ctx.keywords,
        "mood": ctx.mood,
        "is_question": ctx.is_question,
        "base_lang": ctx.base_lang,
        "topic": ctx.topic,
        "is_crypto": ctx.is_crypto,
        "sentiment": ctx.sentiment,
        "excerpt": safe_excerpt(tweet.text),
        "author_name": tweet.author_name,
        "theme_id": theme.id,
        "theme_label": theme.label,
        "theme_tone": theme.tone,
    }

    return {
        "url": tweet.url,
        "comment": comment_payload,
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

    language_mode is kept for backwards-compat but we always behave as:
    "original language + english".
    """
    theme = THEMES.get(theme_id) or THEMES["default"]
    _ = language_mode  # intentionally unused

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
            results.append(
                {
                    "url": url,
                    "error": "unexpected_fetch_error",
                    "message": "Unexpected error while fetching tweet.",
                }
            )
            continue

        item = build_single_output(tweet, theme, language_mode)
        results.append(item)

        time.sleep(0.05 + random.random() * 0.05)

    return results
