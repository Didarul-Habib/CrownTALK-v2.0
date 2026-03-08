import os
import sys
from types import SimpleNamespace

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import main  # noqa: E402


class FakeResp:
    def __init__(self, text: str, content_type: str = "text/html; charset=utf-8"):
        self._text = text
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        yield self._text.encode("utf-8")


def test_fetch_source_preview_uses_local_values(monkeypatch):
    html = """
    <html>
      <head>
        <title>Preview Title</title>
        <meta name="description" content="Useful excerpt" />
      </head>
      <body>
        <article><p>Hello from article body.</p></article>
      </body>
    </html>
    """

    monkeypatch.setattr(main, "validate_public_http_url", lambda u: u)
    monkeypatch.setattr(main, "_ttl_get", lambda cache, key: None)
    monkeypatch.setattr(main, "_ttl_set", lambda cache, key, value, ttl: None)
    monkeypatch.setattr(main.requests, "get", lambda *args, **kwargs: FakeResp(html))
    monkeypatch.setattr(main, "_detect_lang_light", lambda text: "en")
    monkeypatch.setattr(main, "_citation_snippets", lambda text, url, title: [{"url": url, "title": title}])

    with main.app.test_request_context("/source_preview", method="POST"):
        out = main.fetch_source_preview("https://example.com/article")

    assert out["url"] == "https://example.com/article"
    assert out["input_url"] == "https://example.com/article"
    assert out["title"] == "Preview Title"
    assert out["excerpt"] == "Useful excerpt"
    assert out["language"] == "en"
    assert out["citations"][0]["url"] == "https://example.com/article"


def test_hmac_verifier_allows_unsigned_browser_requests(monkeypatch):
    monkeypatch.setattr(main, "CROWNTALK_HMAC_ENFORCE", True)
    monkeypatch.setattr(main, "CROWNTALK_HMAC_SECRET", b"secret")
    with main.app.test_request_context("/comment", method="POST"):
        assert main._verify_hmac_signature(b"{}") is True


def test_signup_invalid_name_uses_api_error_envelope(monkeypatch):
    monkeypatch.setattr(main, "_require_access_or_forbidden", lambda: None)
    client = main.app.test_client()
    res = client.post("/signup", json={"name": "A", "x_link": "https://x.com/test", "password": "strong-pass-123"})
    body = res.get_json()
    assert res.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "invalid_name"


def test_login_missing_fields_uses_api_error_envelope(monkeypatch):
    monkeypatch.setattr(main, "_require_access_or_forbidden", lambda: None)
    client = main.app.test_client()
    res = client.post("/login", json={"x_link": "", "password": ""})
    body = res.get_json()
    assert res.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "missing_fields"


def test_comment_stream_supports_multi_url_batches(monkeypatch):
    monkeypatch.setattr(main, "_require_access_or_forbidden", lambda: None)
    monkeypatch.setattr(main, "_require_user_or_unauthorized", lambda: None)
    monkeypatch.setattr(main, "_register_run_or_conflict", lambda user_key, run_id: True)
    monkeypatch.setattr(main, "_mark_run_done", lambda run_id: None)
    monkeypatch.setattr(main, "_is_run_cancelled", lambda run_id: False)
    monkeypatch.setattr(main, "fetch_tweet_data_retry", lambda url: SimpleNamespace(
        text=f"tweet for {url}", author_name="Author", handle="tester", tweet_id=url.rsplit("/", 1)[-1], lang="en", canonical_url=url
    ))
    monkeypatch.setattr(main, "fallback_tweet_data", lambda url: None)
    monkeypatch.setattr(main, "fetch_thread_context", lambda url, t: None)
    monkeypatch.setattr(main, "build_research_context_for_tweet", lambda *args, **kwargs: {"status": "ok", "projects": []})
    monkeypatch.setattr(main, "set_request_style_hints", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "set_request_voice", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "set_request_nonce", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "generate_two_comments_with_providers", lambda **kwargs: [{"text": f"reply for {kwargs['url']}", "lang": kwargs['target_lang']}])
    monkeypatch.setattr(main, "extract_entities", lambda text: {"cashtags": [], "handles": [], "numbers": []})
    monkeypatch.setattr(main, "_save_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_persist_generation", lambda *args, **kwargs: None)

    client = main.app.test_client()
    res = client.post(
        "/comment/stream",
        json={
            "urls": [
                "https://x.com/a/status/1",
                "https://twitter.com/a/status/1",
                "https://x.com/b/status/2",
            ],
            "lang_en": True,
            "lang_native": False,
        },
    )

    payload = res.get_data(as_text=True)
    assert res.status_code == 200
    assert res.headers["Content-Type"].startswith("text/event-stream")
    assert '"skipped_duplicates": 1' in payload
    assert payload.count('"type": "result"') == 2
    assert '"total": 2' in payload
    assert '"ok_count": 2' in payload
