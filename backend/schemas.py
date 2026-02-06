from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator

MAX_URLS_DEFAULT = 50
MAX_TWEET_TEXT_LEN = 4000
MAX_PRESET_LEN = 64
MAX_LANG_LEN = 12

class CommentRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, description="List of X/Twitter status URLs")
    preset: Optional[str] = Field(default=None, max_length=MAX_PRESET_LEN)
    # generation preferences
    output_language: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN, description="Preferred output language (e.g., en, bn, hi)")
    fast: Optional[bool] = Field(default=False, description="Fast mode: fewer tokens/variants when possible")

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
    url: str
    preset: Optional[str] = Field(default=None, max_length=MAX_PRESET_LEN)
    output_language: Optional[str] = Field(default=None, max_length=MAX_LANG_LEN)
    fast: Optional[bool] = False

class VerifyAccessRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=128)

class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=256)

class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=256)

class ApiError(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ApiEnvelope(BaseModel):
    success: bool
    requestId: str
    data: Optional[Any] = None
    error: Optional[ApiError] = None
