"""
Model Monitoring Module.

Tracks three production-critical signals:
  1. **Prediction drift** — Population Stability Index (PSI) between a
     reference distribution and the live prediction distribution.
  2. **Latency** — Per-prediction wall-clock time, reported as p50/p95/p99.
  3. **Accuracy** — Sliding-window accuracy over the most recent N
     labelled predictions.

Alerts are logged when any metric exceeds its configured threshold.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional

import numpy as np

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# PSI (Population Stability Index)
# ------------------------------------------------------------------


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    buckets: int = config.PSI_BUCKETS,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI quantifies how much the prediction distribution has shifted
    relative to the training-time baseline.  Values > 0.2 typically
    indicate significant drift.
    """
    eps = 1e-6
    breakpoints = np.linspace(0, 1, buckets + 1)

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


# ------------------------------------------------------------------
# Monitor
# ------------------------------------------------------------------


@dataclass
class HealthStatus:
    """Snapshot of model health at a point in time."""

    drift_psi: float
    drift_alert: bool
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_alert: bool
    accuracy: float
    accuracy_alert: bool

    @property
    def healthy(self) -> bool:
        return not (self.drift_alert or self.latency_alert or self.accuracy_alert)


class ModelMonitor:
    """Lightweight production monitor for a deployed model.

    Parameters
    ----------
    reference_predictions : np.ndarray
        Baseline prediction distribution captured at training time.
    predict_fn : callable
        Function that takes a feature vector and returns a probability.
    drift_threshold : float
        PSI value above which a drift alert fires.
    latency_p99_threshold_ms : float
        p99 latency (ms) above which a latency alert fires.
    accuracy_window : int
        Number of recent labelled predictions used for the accuracy check.
    accuracy_min : float
        Minimum sliding-window accuracy before an alert fires.
    """

    def __init__(
        self,
        reference_predictions: np.ndarray,
        predict_fn: Callable,
        drift_threshold: float = config.DRIFT_THRESHOLD,
        latency_p99_threshold_ms: float = config.LATENCY_P99_THRESHOLD_MS,
        accuracy_window: int = config.ACCURACY_WINDOW,
        accuracy_min: float = config.ACCURACY_MIN_THRESHOLD,
    ):
        self.reference_predictions = reference_predictions
        self.predict_fn = predict_fn
        self.drift_threshold = drift_threshold
        self.latency_p99_threshold_ms = latency_p99_threshold_ms
        self.accuracy_window = accuracy_window
        self.accuracy_min = accuracy_min

        # Rolling buffers
        self.live_predictions: List[float] = []
        self.latencies_ms: List[float] = []
        self._accuracy_buffer: Deque[bool] = deque(maxlen=accuracy_window)

    # ----- recording -----

    def record_prediction(
        self,
        features: np.ndarray,
        actual_label: Optional[int] = None,
    ) -> float:
        """Run the model, measure latency, and optionally record accuracy.

        Returns the predicted probability.
        """
        start = time.perf_counter()
        prob = float(self.predict_fn(features))
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.live_predictions.append(prob)
        self.latencies_ms.append(elapsed_ms)

        if actual_label is not None:
            predicted_label = int(prob >= 0.5)
            self._accuracy_buffer.append(predicted_label == actual_label)

        return prob

    # ----- health check -----

    def health_check(self) -> HealthStatus:
        """Evaluate all three monitoring dimensions and return a status."""
        # Drift
        if len(self.live_predictions) >= 30:
            psi = compute_psi(
                self.reference_predictions,
                np.array(self.live_predictions),
            )
        else:
            psi = 0.0
        drift_alert = psi > self.drift_threshold

        # Latency
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        p50 = float(np.percentile(lat, 50))
        p95 = float(np.percentile(lat, 95))
        p99 = float(np.percentile(lat, 99))
        latency_alert = p99 > self.latency_p99_threshold_ms

        # Accuracy
        if self._accuracy_buffer:
            acc = sum(self._accuracy_buffer) / len(self._accuracy_buffer)
        else:
            acc = 1.0  # no data yet — assume healthy
        accuracy_alert = acc < self.accuracy_min

        status = HealthStatus(
            drift_psi=psi,
            drift_alert=drift_alert,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            latency_alert=latency_alert,
            accuracy=acc,
            accuracy_alert=accuracy_alert,
        )

        if drift_alert:
            logger.warning("DRIFT ALERT — PSI=%.4f (threshold=%.2f)", psi, self.drift_threshold)
        if latency_alert:
            logger.warning("LATENCY ALERT — p99=%.2fms (threshold=%.1fms)", p99, self.latency_p99_threshold_ms)
        if accuracy_alert:
            logger.warning("ACCURACY ALERT — %.4f (threshold=%.2f)", acc, self.accuracy_min)

        return status


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    from src.pipeline import generate_synthetic_data, train_model, FEATURE_COLS

    # Train
    df = generate_synthetic_data()
    result = train_model(df, model_path="model_monitor_demo.joblib")
    model = result.model

    # Build reference distribution from training predictions
    ref_probs = model.predict_proba(df[FEATURE_COLS])[:, 1]

    monitor = ModelMonitor(
        reference_predictions=ref_probs,
        predict_fn=lambda x: model.predict_proba(x.reshape(1, -1))[0, 1],
    )

    # Simulate live traffic
    live_df = generate_synthetic_data(n_samples=500, seed=99)
    for _, row in live_df.iterrows():
        features = row[FEATURE_COLS].values.astype(float)
        monitor.record_prediction(features, actual_label=int(row["is_fraud"]))

    status = monitor.health_check()
    print(f"\nHealthy: {status.healthy}")
    print(f"Drift PSI: {status.drift_psi:.4f}  (alert={status.drift_alert})")
    print(f"Latency p50={status.latency_p50_ms:.2f}ms  p95={status.latency_p95_ms:.2f}ms  p99={status.latency_p99_ms:.2f}ms  (alert={status.latency_alert})")
    print(f"Accuracy: {status.accuracy:.4f}  (alert={status.accuracy_alert})")
