"""
ML Pipeline — fraud detection classifier.

Generates synthetic transaction data, trains a GradientBoosting model,
evaluates it, serialises the artefact, and exposes a prediction interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# Feature names used throughout the pipeline
FEATURE_COLS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_risk_score",
    "transaction_count_24h",
    "avg_amount_30d",
    "distance_from_home",
    "is_international",
]


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------


def generate_synthetic_data(
    n_samples: int = config.N_SAMPLES,
    fraud_ratio: float = config.FRAUD_RATIO,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Create a synthetic fraud-detection dataset.

    The generator produces realistic-looking distributions: fraud
    transactions tend to have higher amounts, unusual hours, higher
    merchant risk, and longer distances from the cardholder's home.
    """
    rng = np.random.RandomState(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def _block(n: int, is_fraud: bool) -> pd.DataFrame:
        if is_fraud:
            amount = rng.exponential(scale=800, size=n) + 50
            hour = rng.choice([0, 1, 2, 3, 4, 22, 23], size=n)
            merchant_risk = rng.uniform(0.6, 1.0, size=n)
            distance = rng.exponential(scale=500, size=n) + 100
            tx_count = rng.poisson(lam=12, size=n)
        else:
            amount = rng.exponential(scale=80, size=n) + 5
            hour = rng.choice(range(7, 23), size=n)
            merchant_risk = rng.uniform(0.0, 0.5, size=n)
            distance = rng.exponential(scale=20, size=n)
            tx_count = rng.poisson(lam=3, size=n)

        return pd.DataFrame(
            {
                "amount": amount,
                "hour_of_day": hour,
                "day_of_week": rng.randint(0, 7, size=n),
                "merchant_risk_score": merchant_risk,
                "transaction_count_24h": tx_count,
                "avg_amount_30d": amount * rng.uniform(0.6, 1.4, size=n),
                "distance_from_home": distance,
                "is_international": rng.binomial(1, 0.3 if is_fraud else 0.05, size=n),
                "is_fraud": int(is_fraud),
            }
        )

    df = pd.concat(
        [_block(n_legit, False), _block(n_fraud, True)], ignore_index=True
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


@dataclass
class TrainResult:
    """Container for training artefacts and evaluation metrics."""

    model: GradientBoostingClassifier
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    report: str


def train_model(
    df: pd.DataFrame | None = None,
    model_path: str | None = None,
) -> TrainResult:
    """Train a GradientBoosting fraud classifier and persist it."""
    if df is None:
        df = generate_synthetic_data()

    X = df[FEATURE_COLS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        learning_rate=config.LEARNING_RATE,
        random_state=config.RANDOM_SEED,
    )

    logger.info("Training GradientBoosting (n_estimators=%d, max_depth=%d) ...",
                config.N_ESTIMATORS, config.MAX_DEPTH)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = TrainResult(
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        report=classification_report(y_test, y_pred),
    )

    save_path = model_path or config.MODEL_PATH
    joblib.dump(model, save_path)
    logger.info("Model saved to %s", save_path)
    logger.info(
        "Evaluation — accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  roc_auc=%.4f",
        result.accuracy, result.precision, result.recall, result.f1, result.roc_auc,
    )

    return result


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------


def load_model(model_path: str | None = None) -> GradientBoostingClassifier:
    """Load a persisted model from disk."""
    path = model_path or config.MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"No model found at {path}. Train one first.")
    return joblib.load(path)


def predict(
    model: GradientBoostingClassifier,
    transactions: pd.DataFrame,
) -> np.ndarray:
    """Return fraud probability scores for each transaction row."""
    return model.predict_proba(transactions[FEATURE_COLS])[:, 1]


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    result = train_model()
    print("\n" + result.report)
