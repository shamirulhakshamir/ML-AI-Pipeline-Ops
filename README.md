# ML AI Pipeline Ops

A production-grade ML pipeline framework implementing automated model training, deployment workflows, A/B testing, and monitoring frameworks — built to demonstrate end-to-end ML platform engineering.

## What This Demonstrates

| Capability | Module | Description |
|---|---|---|
| **ML Pipeline** | `src/pipeline.py` | Trains a fraud detection classifier on synthetic transaction data, serialises the model, and exposes a prediction interface |
| **A/B Testing** | `src/ab_testing.py` | Compares two model versions (champion vs. challenger) using configurable traffic splits and statistical significance checks |
| **Monitoring** | `src/monitoring.py` | Tracks prediction drift (PSI), latency percentiles, and sliding-window accuracy — logs alerts when thresholds are breached |
| **Configuration** | `src/config.py` | Central configuration for all modules |
| **Tests** | `tests/` | Pytest suite covering every module |
| **Containerisation** | `Dockerfile` | Multi-stage Docker build ready for Kubernetes deployment |

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python -m src.pipeline

# 4. Run A/B test simulation
python -m src.ab_testing

# 5. Run monitoring simulation
python -m src.monitoring

# 6. Run all tests
pytest -q
```

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data Layer  │───>│   Pipeline   │───>│  Model Store │
│  (synthetic) │    │  (train/eval)│    │  (.joblib)   │
└─────────────┘    └──────┬───────┘    └──────┬───────┘
                          │                    │
                   ┌──────▼───────┐     ┌──────▼───────┐
                   │  A/B Testing │     │  Monitoring   │
                   │  (traffic    │     │  (drift, lat- │
                   │   splitting) │     │   ency, acc.) │
                   └──────────────┘     └──────────────┘
```

## Docker / Kubernetes

Build and run locally:

```bash
docker build -t ml-ai-pipeline-ops:latest .
docker run --rm ml-ai-pipeline-ops:latest
```

For Kubernetes deployment, the container is designed to work behind a service mesh. Key considerations implemented in this POC:

- **Health probes**: The monitoring module exposes health-check compatible metrics (drift score, latency p99) that a liveness/readiness probe can consume.
- **Config injection**: All tunables live in `src/config.py` and can be overridden via environment variables, making them compatible with ConfigMaps.
- **Stateless inference**: The trained model is serialised to a file, which in production would be stored in an object store (S3/GCS) and mounted as a volume or fetched at startup.
- **A/B via traffic splitting**: The A/B testing module implements weight-based routing, which maps directly to Istio VirtualService traffic policies.

## Project Structure

```
ml-ai-pipeline-ops/
├── src/
│   ├── __init__.py
│   ├── config.py          # Central configuration
│   ├── pipeline.py        # Model training and inference
│   ├── ab_testing.py      # A/B testing framework
│   └── monitoring.py      # Production monitoring
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_ab_testing.py
│   └── test_monitoring.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — GradientBoosting classifier for fraud detection
- **pandas / numpy** — Data manipulation and synthetic data generation
- **joblib** — Model serialisation
- **pytest** — Testing
- **Docker** — Containerisation
