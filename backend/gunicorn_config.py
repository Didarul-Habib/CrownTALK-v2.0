# gunicorn_config.py

# Workers & threading (safe for Render free tier)
workers = 2
worker_class = "gthread"
threads = 2

# Networking
bind = "0.0.0.0:10000"  # ignored on Render (Procfile uses $PORT), useful for local dev
timeout = 120
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
