from __future__ import annotations

import os

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

MAX_URLS_DEFAULT = 50
MAX_TWEET_TEXT_LEN = 4000
MAX_PRESET_LEN = 64
MAX_LANG_LEN = 12

QualityMode = Literal["fast", "balanced", "pro"]

class CommentRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, description="List of X/Twitter status URLs")
    preset: Optional[str] = Field(default=None, max_length=MAX_PRESET_LEN)
    # style controls (accepted for forward-compat; backend may apply selectively)
    tone: Optional[str] = Field(default=None, max_length=32)
    intent: Optional[str] = Field(default=None, max_length=32)
    voice: Optional[int] = Field(default=None, ge=0, le=4)
    tone_match: Optional[bool] = Field(default=False)
    thread_ready: Optional[bool] = Field(default=False)
    anti_cringe: Optional[bool] = Field(default=False)
    # generation preferences
    output_language: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN, description="Preferred output language (e.g., en, bn, hi)")
    quality_mode: Optional[QualityMode] = Field(default=None, description="Quality preset: fast/balanced/pro")
    fast: Optional[bool] = Field(default=False, description="Fast mode: fewer tokens/variants when possible")
    include_alternates: Optional[bool] = Field(default=False)
    # dual-language output (tweet mode already supports these in payload)
    lang_en: Optional[bool] = Field(default=True)
    lang_native: Optional[bool] = Field(default=False)
    native_lang: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN)

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, v: List[str]) -> List[str]:
        if len(v) > MAX_URLS_DEFAULT:
            raise ValueError(f"Too many URLs; max {MAX_URLS_DEFAULT}")
        # light sanitization
        cleaned=[]
        for u in v:
            if not isinstance(u, str):
                continue
            u=u.strip()
            if not u:
                continue
            if len(u) > 2048:
                continue
            cleaned.append(u)
        if not cleaned:
            raise ValueError("No valid URLs provided")
        return cleaned

class StreamCommentRequest(BaseModel):
    url: Optional[str] = None
    urls: Optional[List[str]] = None
    preset: Optional[str] = Field(default=None, max_length=MAX_PRESET_LEN)
    tone: Optional[str] = Field(default=None, max_length=32)
    intent: Optional[str] = Field(default=None, max_length=32)
    voice: Optional[int] = Field(default=None, ge=0, le=4)
    tone_match: Optional[bool] = Field(default=False)
    thread_ready: Optional[bool] = Field(default=False)
    anti_cringe: Optional[bool] = Field(default=False)
    output_language: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN)
    quality_mode: Optional[QualityMode] = Field(default=None, description="Quality preset: fast/balanced/pro")
    fast: Optional[bool] = False
    include_alternates: Optional[bool] = Field(default=False)
    lang_en: Optional[bool] = Field(default=True)
    lang_native: Optional[bool] = Field(default=False)
    native_lang: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN)

    @model_validator(mode="after")
    def _normalize_and_validate_urls(self) -> "StreamCommentRequest":
        """Ensure that either `url` or `urls` is provided and normalize both.

        If `urls` is provided, we clean/strip entries and set `url=None` to make
        it explicit that this is a batch request.
        """
        # Normalize single url field.
        if self.url:
            u = (self.url or "").strip()
            self.url = u or None

        # Normalize urls list (if present).
        if self.urls:
            cleaned: list[str] = []
            for item in self.urls:
                if not isinstance(item, str):
                    continue
                u = item.strip()
                if not u or len(u) > 2048:
                    continue
                cleaned.append(u)
            self.urls = cleaned or None

        # If urls is set, treat this as the source of truth and clear url.
        if self.urls:
            self.url = None

        if not self.url and not self.urls:
            raise ValueError("Either 'url' or 'urls' must be provided")

        # Enforce max URLs when batching.
        if self.urls and len(self.urls) > MAX_URLS_DEFAULT:
            raise ValueError(f"Too many URLs; max {MAX_URLS_DEFAULT}")

        return self

class UrlCommentRequest(BaseModel):
    """Generate comments from a single URL (X thread, article, blog, etc.)."""

    source_url: str = Field(..., min_length=4, max_length=2048)
    preset: Optional[str] = Field(default=None, max_length=MAX_PRESET_LEN)
    tone: Optional[str] = Field(default=None, max_length=32)
    intent: Optional[str] = Field(default=None, max_length=32)
    voice: Optional[int] = Field(default=None, ge=0, le=4)
    tone_match: Optional[bool] = Field(default=False)
    thread_ready: Optional[bool] = Field(default=False)
    anti_cringe: Optional[bool] = Field(default=False)
    output_language: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN)
    fast: Optional[bool] = Field(default=False)
    quote_mode: Optional[bool] = Field(default=False, description="If true, prefer quoting/citing claims")
    include_alternates: Optional[bool] = Field(default=False)
    lang_en: Optional[bool] = Field(default=True)
    lang_native: Optional[bool] = Field(default=False)
    native_lang: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN)

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("source_url is required")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("source_url must start with http:// or https://")
        return v

class VerifyAccessRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=128)

class SignupRequest(BaseModel):
    # Back-compat: some clients used email; current UI uses x_link.
    x_link: Optional[str] = Field(default=None, min_length=1, max_length=320)
    email: Optional[str] = Field(default=None, min_length=3, max_length=320)
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    password: str = Field(..., min_length=8, max_length=256)

class LoginRequest(BaseModel):
    x_link: Optional[str] = Field(default=None, min_length=1, max_length=320)
    email: Optional[str] = Field(default=None, min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=256)


class CancelRunRequest(BaseModel):
    """Request body schema for /run/cancel endpoint."""

    run_id: str = Field(..., min_length=8, max_length=128)


class ApiError(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ApiEnvelope(BaseModel):
    success: bool
    requestId: str
    apiVersion: str = Field(default_factory=lambda: os.getenv("CROWNTALK_API_VERSION", "2026-02-22.1"))
    data: Optional[Any] = None
    error: Optional[ApiError] = None
