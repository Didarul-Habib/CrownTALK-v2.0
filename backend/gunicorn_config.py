import os

#
# Workers & threading (safe defaults for small Render instances)
#
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
worker_class = "gthread"
threads = int(os.getenv("WEB_THREADS", "2"))

# Networking: bind to $PORT (Render provides), default 8000 for local dev
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

#
# Timeouts
#
# Hard timeout: if a worker takes longer than this, kill it.
timeout = int(os.getenv("WEB_TIMEOUT", "120"))

# Graceful timeout: how long to wait for workers to finish requests
# after receiving a reload/quit signal before force-killing.
graceful_timeout = int(os.getenv("WEB_GRACEFUL_TIMEOUT", str(timeout + 30)))

# Keep-alive for HTTP connections
keepalive = int(os.getenv("WEB_KEEPALIVE", "30"))

#
# Process naming
#
proc_name = os.getenv("WEB_PROC_NAME", "crowntalk")

#
# Logging
#
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("WEB_LOG_LEVEL", "info")

#
# Security-ish defaults
#
limit_request_line = int(os.getenv("WEB_LIMIT_REQUEST_LINE", "8190"))
limit_request_fields = int(os.getenv("WEB_LIMIT_REQUEST_FIELDS", "100"))
limit_request_field_size = int(os.getenv("WEB_LIMIT_REQUEST_FIELD_SIZE", "8190"))

#
# Worker recycling (helps avoid memory leaks on long-running dynos)
#
max_requests = int(os.getenv("WEB_MAX_REQUESTS", "1000"))          # restart worker after N requests
max_requests_jitter = int(os.getenv("WEB_MAX_REQUESTS_JITTER", "50"))

#
# Optional tuning
#
# Preload the app code before forking workers. This can reduce memory a bit
# via copy-on-write and make worker startup faster, but if your init is heavy
# you can disable via env.
preload_app = os.getenv("WEB_PRELOAD_APP", "true").lower() in {"1", "true", "yes"}

# Where to put temporary worker files (use /tmp by default; /dev/shm on some hosts is faster)
worker_tmp_dir = os.getenv("WEB_WORKER_TMP_DIR", "/tmp")
