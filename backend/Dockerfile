FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# layer-cached deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# app code
COPY . /app

# Render expects 0.0.0.0:8080
EXPOSE 8080

# Flask app object is "app" inside main.py
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "8", "-b", "0.0.0.0:8080", "main:app"]
