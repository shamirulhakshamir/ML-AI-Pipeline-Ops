# ---------- build stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY . .

# Run tests at build time to guarantee the image ships a passing codebase
RUN PYTHONPATH=/app pip install --no-cache-dir pytest \
    && PYTHONPATH=/app pytest -q tests/

# ---------- runtime stage ----------
FROM python:3.11-slim

LABEL maintainer="Shamirul Hak"
LABEL description="Picnic ML Pipeline POC — fraud detection with A/B testing and monitoring"

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Train the model on startup and print evaluation metrics
CMD ["python", "-m", "src.pipeline"]
