from __future__ import annotations

import json

import os

_MAX_OUTPUT_CHARS = int(os.getenv("CROWNTALK_MAX_OUTPUT_CHARS", "6000"))

def _truncate_str(s: str) -> str:
    s = s or ""
    if _MAX_OUTPUT_CHARS and isinstance(s, str) and len(s) > _MAX_OUTPUT_CHARS:
        return s[:_MAX_OUTPUT_CHARS].rstrip() + "\n\n[truncated]"
    return s

def _truncate_any(x: Any):
    # Best-effort recursive truncation for very large outputs
    try:
        if isinstance(x, str):
            return _truncate_str(x)
        if isinstance(x, list):
            return [_truncate_any(i) for i in x]
        if isinstance(x, dict):
            return {k: _truncate_any(v) for k, v in x.items()}
    except Exception:
        pass
    return x

import time
from typing import Any, Dict, Optional, Tuple

from flask import Response, jsonify, g

from schemas import ApiEnvelope, ApiError

def api_success(data: Any = None, status: int = 200, headers: Optional[Dict[str, str]] = None):
    env = ApiEnvelope(success=True, requestId=getattr(g, "request_id", ""), data=_truncate_any(data), error=None)
    resp = jsonify(env.model_dump())
    resp.status_code = status
    if headers:
        for k, v in headers.items():
            resp.headers[k] = v
    return resp

def api_error(code: str, message: str, status: int = 400, details: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
    env = ApiEnvelope(success=False, requestId=getattr(g, "request_id", ""), data=None, error=ApiError(code=code, message=message, details=details))
    resp = jsonify(env.model_dump())
    resp.status_code = status
    if headers:
        for k, v in headers.items():
            resp.headers[k] = v
    return resp

def sse_event(event: str, data: Any):
    payload = data
    if not isinstance(data, str):
        payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
