# gunicorn_config.py
import os

# Workers & threading (safe for Render free tier)
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
worker_class = "gthread"
threads = int(os.getenv("WEB_THREADS", "2"))

# Networking: bind to $PORT (Render provides), default 8000 for local dev
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Timeouts
timeout = int(os.getenv("WEB_TIMEOUT", "120"))
keepalive = 30

# Process naming
proc_name = "crowntalk"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Security-ish defaults
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190
