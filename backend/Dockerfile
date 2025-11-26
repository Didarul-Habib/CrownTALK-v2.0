# ===== Base =====
FROM python:3.11-slim

# Fast, clean Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project (frontend + backend)
COPY . /app

# Create non-root user and fix ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Port + runtime settings
ENV PORT=10000
EXPOSE 10000

# Start the server (matches Procfile)
CMD gunicorn -c gunicorn_config.py main:app
