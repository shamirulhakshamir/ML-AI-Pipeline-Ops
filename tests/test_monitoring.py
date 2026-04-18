"""Tests for src.monitoring."""

import numpy as np
import pytest

from src.monitoring import ModelMonitor, compute_psi


class TestComputePSI:
    def test_identical_distributions(self):
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, size=1000)
        psi = compute_psi(data, data)
        assert psi < 0.01  # near-zero drift

    def test_shifted_distribution_high_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.uniform(0.0, 0.5, size=1000)
        cur = rng.uniform(0.5, 1.0, size=1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.5  # very different distributions

    def test_slightly_shifted(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0.4, 0.1, size=2000).clip(0, 1)
        cur = rng.normal(0.45, 0.1, size=2000).clip(0, 1)
        psi = compute_psi(ref, cur)
        assert 0.0 < psi < 0.5  # moderate drift


class TestModelMonitor:
    @staticmethod
    def _build_monitor(accuracy=0.95, seed=42):
        """Create a monitor with a mock model that is correct `accuracy`
        fraction of the time when the true label matches a coin flip."""
        rng = np.random.RandomState(seed)
        ref = rng.uniform(0, 1, size=500)

        def mock_predict(x):
            return float(rng.uniform(0, 1))

        return ModelMonitor(
            reference_predictions=ref,
            predict_fn=mock_predict,
            drift_threshold=0.2,
            latency_p99_threshold_ms=500.0,  # generous for tests
            accuracy_window=100,
            accuracy_min=0.5,
        )

    def test_record_prediction_stores_values(self):
        monitor = self._build_monitor()
        features = np.array([1.0, 2.0])

        monitor.record_prediction(features, actual_label=1)
        assert len(monitor.live_predictions) == 1
        assert len(monitor.latencies_ms) == 1

    def test_health_check_no_alerts_initially(self):
        monitor = self._build_monitor()
        status = monitor.health_check()
        # No predictions yet — everything should be healthy
        assert status.healthy

    def test_health_check_after_predictions(self):
        monitor = self._build_monitor()
        features = np.array([1.0, 2.0])

        for _ in range(100):
            monitor.record_prediction(features, actual_label=1)

        status = monitor.health_check()
        assert status.latency_p50_ms >= 0
        assert status.latency_p99_ms >= 0

    def test_drift_alert_fires_on_shifted_data(self):
        rng = np.random.RandomState(42)
        ref = rng.uniform(0.0, 0.3, size=500)  # low predictions

        # Model that always predicts high
        monitor = ModelMonitor(
            reference_predictions=ref,
            predict_fn=lambda x: 0.9,
            drift_threshold=0.2,
            latency_p99_threshold_ms=500.0,
            accuracy_window=100,
            accuracy_min=0.0,
        )

        features = np.array([1.0])
        for _ in range(100):
            monitor.record_prediction(features)

        status = monitor.health_check()
        assert status.drift_alert is True
        assert status.drift_psi > 0.2

    def test_accuracy_alert_fires_on_bad_model(self):
        rng = np.random.RandomState(42)
        ref = rng.uniform(0, 1, size=500)

        # Model that always predicts 0.9 (fraud) — bad when actual is 0
        monitor = ModelMonitor(
            reference_predictions=ref,
            predict_fn=lambda x: 0.9,
            drift_threshold=10.0,  # disable drift alert
            latency_p99_threshold_ms=500.0,
            accuracy_window=50,
            accuracy_min=0.80,
        )

        features = np.array([1.0])
        # Feed it all-negative labels so the model is always wrong
        for _ in range(60):
            monitor.record_prediction(features, actual_label=0)

        status = monitor.health_check()
        assert status.accuracy_alert is True
        assert status.accuracy < 0.80

    def test_healthy_property_false_on_any_alert(self):
        rng = np.random.RandomState(42)
        ref = rng.uniform(0, 1, size=500)

        monitor = ModelMonitor(
            reference_predictions=ref,
            predict_fn=lambda x: 0.9,
            drift_threshold=10.0,
            latency_p99_threshold_ms=500.0,
            accuracy_window=50,
            accuracy_min=0.99,  # impossible to meet
        )

        features = np.array([1.0])
        for _ in range(60):
            monitor.record_prediction(features, actual_label=0)

        status = monitor.health_check()
        assert status.healthy is False
