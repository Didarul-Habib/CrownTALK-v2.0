import logging
import os
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

from generator import generate_comments_for_urls, get_theme_ids, normalize_language_mode
from utils import (
    clean_and_normalize_urls,
    CrownTALKError,
    chunk_list,
)

BACKEND_URL = os.getenv("CROWNTALK_BACKEND_URL", "https://crowntalk-v2-0.onrender.com")
BATCH_SIZE = 2  # EXTREME spec: 2 per batch


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    )
    app.logger.setLevel(log_level)

    @app.route("/", methods=["GET"])
    def index() -> Any:
        """
        Simple index for sanity check / debugging.
        """
        return jsonify(
            {
                "service": "CrownTALK Backend EXTREME",
                "version": "v3",
                "backend_url": BACKEND_URL,
                "endpoints": {
                    "health": "/health",
                    "generate": "/api/generate",
                },
            }
        )

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        """
        Healthcheck endpoint for Render.
        """
        return jsonify({"ok": True, "status": "healthy", "service": "CrownTALK"}), 200

    @app.route("/api/generate", methods=["POST"])
    def api_generate() -> Any:
        """
        Main generation endpoint.

        Expected JSON body:
        {
            "urls": ["https://x.com/...","..."],
            "theme": "default",          # optional
            "language_mode": "dual"      # "en" | "bn" | "dual"
        }
        """
        try:
            payload: Dict[str, Any] = request.get_json(force=True, silent=False)  # type: ignore
        except Exception:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "invalid_json",
                        "message": "Request body must be valid JSON.",
                    }
                ),
                400,
            )

        if not isinstance(payload, dict):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "invalid_body",
                        "message": "JSON body must be an object.",
                    }
                ),
                400,
            )

        urls_raw = payload.get("urls")
        if not isinstance(urls_raw, list) or not urls_raw:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "missing_urls",
                        "message": "Field 'urls' must be a non-empty list of URLs.",
                    }
                ),
                400,
            )

        # Clean URLs
        try:
            cleaned_urls: List[str] = clean_and_normalize_urls(urls_raw)
        except CrownTALKError as e:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": e.code,
                        "message": str(e),
                    }
                ),
                400,
            )

        if len(cleaned_urls) > 20:
            # Safety limit
            cleaned_urls = cleaned_urls[:20]

        requested_theme = payload.get("theme") or "default"
        language_mode = normalize_language_mode(payload.get("language_mode"))

        if requested_theme not in get_theme_ids():
            # Fall back to default if unknown
            requested_theme = "default"

        batches_payload: List[Dict[str, Any]] = []
        batch_index = 0

        for url_batch in chunk_list(cleaned_urls, BATCH_SIZE):
            try:
                items = generate_comments_for_urls(
                    urls=url_batch,
                    theme_id=requested_theme,
                    language_mode=language_mode,
                )
            except CrownTALKError as e:
                # One bad URL shouldn't kill the whole request; we add an error item for that URL
                items = [
                    {
                        "url": bad_url,
                        "error": e.code,
                        "message": str(e),
                    }
                    for bad_url in url_batch
                ]
            except Exception as e:
                # Generic failure protection per batch
                app.logger.exception("Unexpected error during batch generation")
                items = [
                    {
                        "url": u,
                        "error": "batch_failure",
                        "message": "Unexpected error during generation.",
                    }
                    for u in url_batch
                ]

            batches_payload.append(
                {
                    "batch_index": batch_index,
                    "items": items,
                }
            )
            batch_index += 1

        response_body: Dict[str, Any] = {
            "ok": True,
            "batches": batches_payload,
            "meta": {
                "total_urls": len(cleaned_urls),
                "batch_size": BATCH_SIZE,
                "theme": requested_theme,
                "language_mode": language_mode,
            },
        }
        return jsonify(response_body), 200

    return app


app = create_app()

if __name__ == "__main__":
    # Local dev only; Render will use gunicorn / similar
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
