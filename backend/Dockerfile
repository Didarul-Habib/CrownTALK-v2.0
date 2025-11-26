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

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY main.py utils.py gunicorn_config.py /app/
COPY static /app/static
COPY index.html /app/index.html

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Render sets $PORT; default to 10000 for local runs
ENV PORT=10000
EXPOSE 10000

# Start the server (matches Procfile)
CMD gunicorn -c gunicorn_config.py main:app
