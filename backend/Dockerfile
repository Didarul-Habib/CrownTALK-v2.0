# ===== Base =====
FROM python:3.11-slim

# Faster, cleaner Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (minimal but with curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== Install deps first (better caching) =====
# If your file is named differently on Render, copy/rename it to requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== App code =====
# Copy only what we need (adjust if your layout differs)
COPY . /app

# Optional: non-root user
RUN useradd -m appuser
USER appuser

# Render sets $PORT; default to 10000 for local runs
ENV PORT=10000
EXPOSE 10000

# Optional: basic container healthcheck (hits the Flask/Gunicorn "/" route)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/" || exit 1

# Start the server
# Use gunicorn_config.py so worker/thread/timeouts are controlled from code
CMD gunicorn -c gunicorn_config.py main:app
