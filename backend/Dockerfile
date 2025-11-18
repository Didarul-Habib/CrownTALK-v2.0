FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Keep this separate so pip layer caches when only code changes
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Now copy the app
COPY . /app

EXPOSE 8080

# Free tier: 1 worker, a few threads -> faster boot & less CPU thrash
CMD ["gunicorn", "--workers", "1", "--threads", "4", "-k", "gthread", "-b", "0.0.0.0:8080", "main:app"]
