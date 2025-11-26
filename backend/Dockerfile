# ===== Base =====
FROM python:3.11-slim

# Faster, cleaner Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

...
RUN useradd -m appuser
USER appuser

# Render sets $PORT; default to 10000 for local runs
ENV PORT=10000
EXPOSE 10000

# Start the server (same as Procfile)
CMD gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:$PORT main:app
