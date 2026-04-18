"""Tests for src.pipeline."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    FEATURE_COLS,
    generate_synthetic_data,
    load_model,
    predict,
    train_model,
)


class TestSyntheticData:
    def test_shape(self):
        df = generate_synthetic_data(n_samples=500)
        assert len(df) == 500

    def test_columns_present(self):
        df = generate_synthetic_data(n_samples=100)
        for col in FEATURE_COLS + ["is_fraud"]:
            assert col in df.columns

    def test_fraud_ratio(self):
        df = generate_synthetic_data(n_samples=10000, fraud_ratio=0.10, seed=7)
        actual_ratio = df["is_fraud"].mean()
        assert 0.08 <= actual_ratio <= 0.12

    def test_deterministic_with_seed(self):
        df1 = generate_synthetic_data(n_samples=200, seed=123)
        df2 = generate_synthetic_data(n_samples=200, seed=123)
        pd.testing.assert_frame_equal(df1, df2)


class TestTrainModel:
    def test_trains_and_returns_metrics(self, tmp_path):
        model_file = str(tmp_path / "test_model.joblib")
        df = generate_synthetic_data(n_samples=600, seed=42)
        result = train_model(df, model_path=model_file)

        assert result.accuracy > 0.80
        assert result.roc_auc > 0.80
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert os.path.exists(model_file)

    def test_model_file_loadable(self, tmp_path):
        model_file = str(tmp_path / "test_model.joblib")
        df = generate_synthetic_data(n_samples=600, seed=42)
        train_model(df, model_path=model_file)

        model = load_model(model_file)
        assert hasattr(model, "predict_proba")


class TestPredict:
    def test_predict_returns_probabilities(self, tmp_path):
        model_file = str(tmp_path / "test_model.joblib")
        df = generate_synthetic_data(n_samples=600, seed=42)
        result = train_model(df, model_path=model_file)

        sample = df.head(10)
        probs = predict(result.model, sample)

        assert len(probs) == 10
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_load_model_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")
