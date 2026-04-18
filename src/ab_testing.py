"""
A/B Testing Framework.

Supports traffic splitting between a champion and a challenger model,
collects per-variant metrics, and evaluates statistical significance
to decide whether the challenger should be promoted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VariantMetrics:
    """Accumulated metrics for a single model variant."""

    name: str
    predictions: List[float] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def f1(self) -> float:
        if not self.actuals:
            return 0.0
        binary_preds = [int(p >= 0.5) for p in self.predictions]
        return f1_score(self.actuals, binary_preds, zero_division=0)

    @property
    def roc_auc(self) -> float:
        if len(set(self.actuals)) < 2 or not self.actuals:
            return 0.0
        return roc_auc_score(self.actuals, self.predictions)


@dataclass
class ABTestResult:
    """Outcome of an A/B test comparison."""

    champion: VariantMetrics
    challenger: VariantMetrics
    p_value: float
    is_significant: bool
    winner: str


class ABTestRunner:
    """Run an A/B test between two models on a stream of labelled data.

    Parameters
    ----------
    champion_model : callable
        A function that takes feature rows and returns fraud probabilities.
    challenger_model : callable
        Same signature as *champion_model*.
    champion_weight : float
        Fraction of traffic routed to the champion (default from config).
    significance_level : float
        Alpha for the two-proportion z-test (default from config).
    min_samples : int
        Minimum observations per variant before significance is evaluated.
    """

    def __init__(
        self,
        champion_model,
        challenger_model,
        champion_weight: float = config.AB_CHAMPION_WEIGHT,
        significance_level: float = config.AB_SIGNIFICANCE_LEVEL,
        min_samples: int = config.AB_MIN_SAMPLES,
        seed: int = config.RANDOM_SEED,
    ):
        self.champion_model = champion_model
        self.challenger_model = challenger_model
        self.champion_weight = champion_weight
        self.significance_level = significance_level
        self.min_samples = min_samples
        self.rng = np.random.RandomState(seed)

        self.champion_metrics = VariantMetrics(name="champion")
        self.challenger_metrics = VariantMetrics(name="challenger")

    # ----- routing -----

    def route(self) -> str:
        """Decide which variant handles the next request."""
        return "champion" if self.rng.random() < self.champion_weight else "challenger"

    # ----- recording -----

    def record(
        self,
        features: np.ndarray,
        actual_label: int,
    ) -> Tuple[str, float]:
        """Route one observation, score it, and record the outcome.

        Returns the variant name and the predicted probability.
        """
        variant = self.route()

        if variant == "champion":
            prob = float(self.champion_model(features))
            metrics = self.champion_metrics
        else:
            prob = float(self.challenger_model(features))
            metrics = self.challenger_metrics

        predicted_label = int(prob >= 0.5)
        metrics.predictions.append(prob)
        metrics.actuals.append(actual_label)
        metrics.total += 1
        if predicted_label == actual_label:
            metrics.correct += 1

        return variant, prob

    # ----- evaluation -----

    def evaluate(self) -> ABTestResult:
        """Compare champion vs. challenger using a two-proportion z-test
        on accuracy, and report the winner.
        """
        c = self.champion_metrics
        ch = self.challenger_metrics

        # Two-proportion z-test on accuracy
        n1, n2 = c.total, ch.total
        p1, p2 = c.accuracy, ch.accuracy

        if n1 < self.min_samples or n2 < self.min_samples:
            logger.warning(
                "Not enough samples (champion=%d, challenger=%d, min=%d). "
                "Cannot determine significance.",
                n1, n2, self.min_samples,
            )
            return ABTestResult(
                champion=c,
                challenger=ch,
                p_value=1.0,
                is_significant=False,
                winner="champion",
            )

        # Pooled proportion
        p_pool = (c.correct + ch.correct) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 else 1.0
        z = (p2 - p1) / se if se > 0 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))  # two-tailed

        is_significant = p_value < self.significance_level
        winner = "challenger" if (is_significant and p2 > p1) else "champion"

        logger.info(
            "A/B result — champion_acc=%.4f  challenger_acc=%.4f  p=%.4f  significant=%s  winner=%s",
            p1, p2, p_value, is_significant, winner,
        )

        return ABTestResult(
            champion=c,
            challenger=ch,
            p_value=p_value,
            is_significant=is_significant,
            winner=winner,
        )


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    from src.pipeline import generate_synthetic_data, train_model, predict, FEATURE_COLS

    # Train two slightly different models (different hyperparams)
    df = generate_synthetic_data()

    original_depth = config.MAX_DEPTH
    result_a = train_model(df, model_path="model_champion.joblib")

    config.MAX_DEPTH = 4
    config.N_ESTIMATORS = 80
    result_b = train_model(df, model_path="model_challenger.joblib")
    config.MAX_DEPTH = original_depth

    model_a = result_a.model
    model_b = result_b.model

    runner = ABTestRunner(
        champion_model=lambda x: model_a.predict_proba(x.reshape(1, -1))[0, 1],
        challenger_model=lambda x: model_b.predict_proba(x.reshape(1, -1))[0, 1],
        champion_weight=0.5,  # 50/50 for demo
    )

    # Simulate traffic
    test_df = generate_synthetic_data(n_samples=1000, seed=99)
    for _, row in test_df.iterrows():
        features = row[FEATURE_COLS].values.astype(float)
        runner.record(features, int(row["is_fraud"]))

    outcome = runner.evaluate()
    print(f"\nWinner: {outcome.winner}  (p-value={outcome.p_value:.4f})")
    print(f"Champion accuracy: {outcome.champion.accuracy:.4f}  F1: {outcome.champion.f1:.4f}")
    print(f"Challenger accuracy: {outcome.challenger.accuracy:.4f}  F1: {outcome.challenger.f1:.4f}")
