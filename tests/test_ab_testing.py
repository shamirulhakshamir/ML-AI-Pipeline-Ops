"""Tests for src.ab_testing."""

import numpy as np
import pytest

from src.ab_testing import ABTestRunner, VariantMetrics


class TestVariantMetrics:
    def test_accuracy_no_data(self):
        vm = VariantMetrics(name="test")
        assert vm.accuracy == 0.0

    def test_accuracy_correct(self):
        vm = VariantMetrics(name="test", correct=8, total=10)
        assert vm.accuracy == pytest.approx(0.8)

    def test_f1_computation(self):
        vm = VariantMetrics(
            name="test",
            predictions=[0.9, 0.8, 0.1, 0.2, 0.95, 0.05],
            actuals=[1, 1, 0, 0, 1, 0],
        )
        assert vm.f1 > 0.9

    def test_roc_auc_needs_both_classes(self):
        vm = VariantMetrics(
            name="test",
            predictions=[0.5, 0.6],
            actuals=[1, 1],
        )
        # Only one class present — should return 0.0 gracefully
        assert vm.roc_auc == 0.0


class TestABTestRunner:
    @staticmethod
    def _make_runner(champion_acc=0.90, challenger_acc=0.85, seed=42):
        """Helper: build a runner with deterministic mock models."""
        rng_c = np.random.RandomState(seed)
        rng_ch = np.random.RandomState(seed + 1)

        def champion_model(x):
            return 0.9 if rng_c.random() < champion_acc else 0.1

        def challenger_model(x):
            return 0.9 if rng_ch.random() < challenger_acc else 0.1

        return ABTestRunner(
            champion_model=champion_model,
            challenger_model=challenger_model,
            champion_weight=0.5,
            min_samples=30,
            seed=seed,
        )

    def test_routing_respects_weight(self):
        runner = ABTestRunner(
            champion_model=lambda x: 0.5,
            challenger_model=lambda x: 0.5,
            champion_weight=1.0,  # all traffic to champion
            seed=42,
        )
        routes = [runner.route() for _ in range(100)]
        assert all(r == "champion" for r in routes)

    def test_record_updates_metrics(self):
        runner = ABTestRunner(
            champion_model=lambda x: 0.9,
            challenger_model=lambda x: 0.1,
            champion_weight=0.5,
            seed=42,
        )
        features = np.array([1.0, 2.0, 3.0])
        for _ in range(50):
            runner.record(features, actual_label=1)

        total = runner.champion_metrics.total + runner.challenger_metrics.total
        assert total == 50

    def test_evaluate_returns_result(self):
        runner = self._make_runner()
        features = np.array([1.0])

        for _ in range(200):
            label = int(np.random.randint(0, 2))
            runner.record(features, actual_label=label)

        result = runner.evaluate()
        assert result.winner in ("champion", "challenger")
        assert 0.0 <= result.p_value <= 1.0

    def test_insufficient_samples(self):
        runner = self._make_runner()
        features = np.array([1.0])

        # Record fewer than min_samples
        for _ in range(10):
            runner.record(features, actual_label=1)

        result = runner.evaluate()
        assert result.p_value == 1.0
        assert result.is_significant is False
        assert result.winner == "champion"

    def test_champion_wins_when_equal(self):
        """When there is no significant difference, champion should be kept."""
        runner = self._make_runner(champion_acc=0.85, challenger_acc=0.85)
        features = np.array([1.0])

        for _ in range(200):
            label = int(np.random.randint(0, 2))
            runner.record(features, actual_label=label)

        result = runner.evaluate()
        # With equal accuracy the test should not be significant
        # so the winner defaults to champion
        assert result.winner == "champion"
