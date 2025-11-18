# backend/Dockerfile
FROM python:3.11-slim

# System basics (and a non-root user)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080
WORKDIR /app

# Install runtime deps (gcc not needed for this stack; keep image slim)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install gunicorn

# Copy backend code
COPY . /app

# (Optional) show useful defaults in logs
ENV GUNICORN_CMD_ARGS="--config /app/gunicorn_config.py"

# Render will inject $PORT; expose is informational
EXPOSE 8080

# Run the Flask app via gunicorn (module: main.py -> app)
CMD ["gunicorn", "main:app"]
