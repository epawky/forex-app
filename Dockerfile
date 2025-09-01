FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# tzdata for timezones; libgomp1 for scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + static UI
COPY fx_pipeline.py ./fx_pipeline.py
COPY app.py ./app.py
COPY static ./static

# Create volumes
RUN mkdir -p /app/data /app/models

# Runtime env (change schedule via env at run-time)
ENV DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    TICKS_GLOB=/app/data/ticks/*.csv \
    ENABLE_SCHEDULER=1 \
    SCHEDULE_UTC_HOUR=06 \
    SCHEDULE_UTC_MINUTE=10

EXPOSE 8000

# Start API (APScheduler runs inside this process)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:application"]
