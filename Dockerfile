# Pin to the same Python as your stable venv
FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# tzdata for time zones; libgomp1 for scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
      tzdata libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --prefer-binary -r requirements.txt

# Copy app code
COPY fx_pipeline.py ./fx_pipeline.py
COPY app.py ./app.py

# If you build the UI, copy the built assets (adjust if you actually use ./static)
# Comment out whichever you don't use.
# COPY ui/dist ./ui/dist
COPY static ./static

# Data/model dirs (bind-mounted in compose, but OK to create)
RUN mkdir -p /app/data /app/models

# Runtime environment
ENV DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    TICKS_GLOB=/app/data/ticks/*.csv \
    SERVE_UI=1

EXPOSE 5001

# Gunicorn (single worker is fine; training happens in subprocess)
CMD ["gunicorn", "app:app", "-w", "1", "-b", "0.0.0.0:5001", "--timeout", "600"]

