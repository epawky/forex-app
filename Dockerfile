FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# tzdata for timezones; libgomp1 for scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your pipeline
COPY fx_pipeline.py .

# Create run-time dirs (you will mount host dirs here)
RUN mkdir -p /app/data /app/models

ENTRYPOINT ["python", "fx_pipeline.py"]
CMD ["--help"]
