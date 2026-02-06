from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

from flask import Response, jsonify, g

from schemas import ApiEnvelope, ApiError

def api_success(data: Any = None, status: int = 200, headers: Optional[Dict[str, str]] = None):
    env = ApiEnvelope(success=True, requestId=getattr(g, "request_id", ""), data=data, error=None)
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
