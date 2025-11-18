# backend/gunicorn_config.py
import multiprocessing
import os

# Render injects PORT; default for local runs
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Sensible defaults for IO-bound Flask APIs
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
threads = int(os.environ.get("WEB_THREADS", "2"))
worker_class = "gthread"

# Timeouts: VXTwitter calls + cold starts can be slow on free tier
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

# Limit request size if needed (in bytes); None = unlimited
limit_request_line = 0
limit_request_fields = 32768
limit_request_field_size = 0

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Reduce noisy worker restart churn
max_requests = 0  # disable by default
max_requests_jitter = 0

# Ensure threads don’t fight over DNS (safe default)
preload_app = False
