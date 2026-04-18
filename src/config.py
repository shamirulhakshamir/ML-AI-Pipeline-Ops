"""
Central configuration for the Picnic ML Pipeline POC.

All tunables are defined here so they can be overridden via environment
variables when deployed inside Kubernetes (e.g. via ConfigMaps).
"""

import os

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
N_SAMPLES: int = int(os.getenv("N_SAMPLES", "5000"))
FRAUD_RATIO: float = float(os.getenv("FRAUD_RATIO", "0.08"))
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
N_ESTIMATORS: int = int(os.getenv("N_ESTIMATORS", "120"))
MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "6"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.1"))
MODEL_PATH: str = os.getenv("MODEL_PATH", "model.joblib")

# ---------------------------------------------------------------------------
# A/B testing
# ---------------------------------------------------------------------------
AB_CHAMPION_WEIGHT: float = float(os.getenv("AB_CHAMPION_WEIGHT", "0.8"))
AB_SIGNIFICANCE_LEVEL: float = float(os.getenv("AB_SIGNIFICANCE_LEVEL", "0.05"))
AB_MIN_SAMPLES: int = int(os.getenv("AB_MIN_SAMPLES", "100"))

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
LATENCY_P99_THRESHOLD_MS: float = float(os.getenv("LATENCY_P99_THRESHOLD_MS", "50.0"))
ACCURACY_WINDOW: int = int(os.getenv("ACCURACY_WINDOW", "200"))
ACCURACY_MIN_THRESHOLD: float = float(os.getenv("ACCURACY_MIN_THRESHOLD", "0.80"))
PSI_BUCKETS: int = int(os.getenv("PSI_BUCKETS", "10"))
