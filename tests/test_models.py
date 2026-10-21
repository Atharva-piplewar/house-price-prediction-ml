"""
Integration and Sanity Tests for Trained Models and Evaluation Metrics
Academic Project: House Price Prediction Using Machine Learning
"""

import os
import json
import pytest
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def test_model_artifacts_exist():
    """Verify all expected joblib model files exist."""
    expected_files = [
        "preprocessing_pipeline.joblib",
        "linear_regression.joblib",
        "random_forest.joblib",
        "support_vector_regression.joblib",
        "gradient_boosting.joblib",
        "xgboost.joblib",
    ]
    for filename in expected_files:
        filepath = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(filepath), f"Missing model artifact: {filename}"
        assert os.path.getsize(filepath) > 0, f"Model artifact is empty: {filename}"


def test_model_metadata_validity():
    """Verify metadata JSON structure and best model selection."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["best_model"] == "XGBoost"
    assert len(meta["models_evaluated"]) == 5
    assert len(meta["test_metrics"]) == 5
    assert len(meta["cv_metrics"]) == 5


def test_model_performance_benchmarks():
    """Verify all models pass academic performance baseline standards."""
    metrics_csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    assert os.path.exists(metrics_csv)
    df = pd.read_csv(metrics_csv)
    
    # All models should have R² > 0.70
    for _, row in df.iterrows():
        assert row["R2"] >= 0.70, f"Model {row['Model']} has sub-par R² score: {row['R2']}"
        assert row["RMSE"] < 45000, f"Model {row['Model']} has excessive RMSE: {row['RMSE']}"
        
    # Best model should achieve R² >= 0.90
    xgb_row = df[df["Model"] == "XGBoost"].iloc[0]
    assert xgb_row["R2"] >= 0.90, f"XGBoost R² should be >= 0.90, got {xgb_row['R2']}"


def test_all_figures_generated():
    """Verify all 8 publication figures were produced."""
    expected_figures = [
        "model_mae_comparison.png",
        "model_rmse_comparison.png",
        "model_r2_comparison.png",
        "actual_vs_predicted.png",
        "residual_distribution.png",
        "residuals_vs_predicted.png",
        "feature_importance.png",
        "cv_comparison.png",
    ]
    for fig in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig)
        assert os.path.exists(fig_path), f"Missing evaluation figure: {fig}"
        assert os.path.getsize(fig_path) > 10000, f"Figure {fig} file seems too small"
