"""
Model Training Module for House Price Prediction
Academic Project: House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models

This module implements:
1. Training of five regression algorithms:
   - Linear Regression (Baseline Parametric Model)
   - Random Forest Regressor (Ensemble Bagging)
   - Support Vector Regression (Kernelized SVR)
   - Gradient Boosting Regressor (Sequential Boosting)
   - XGBoost Regressor (Extreme Gradient Boosting)
2. 5-Fold Cross-Validation on the training set.
3. Strict evaluation on held-out test data (20% split).
4. Calculation of MAE, MSE, RMSE, R², Training Time, and Prediction Time.
5. Automated selection of the best-performing model based on minimal RMSE.
6. Persistence of all trained models and evaluation summaries.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.preprocessing import prepare_train_test_data

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def get_model_definitions(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns initialized dictionary of the five regression models.
    Hyperparameters are carefully chosen to balance bias-variance tradeoff without test-set leakage.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Regression": SVR(
            kernel="rbf",
            C=15000.0,
            epsilon=0.05,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.07,
            max_depth=4,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.85,
            random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=160,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_and_evaluate_all_models() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Trains all five models, evaluates them on CV and test sets, and saves artifacts.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[STEP 1] Preparing train-test split and preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor, summary, feature_names = prepare_train_test_data(
        save_artifacts=True
    )

    models = get_model_definitions(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    results_list = []
    cv_records = []
    saved_model_paths = {}

    print("\n[STEP 2] Training and evaluating five regression models...")
    print("-" * 85)
    print(f"{'Model':<28} | {'MAE ($)':<10} | {'RMSE ($)':<10} | {'R²':<8} | {'Train(s)':<8}")
    print("-" * 85)

    for name, model in models.items():
        # 1. 5-Fold Cross Validation on Training Split
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1)
        cv_neg_mse = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1
        )
        cv_rmse_scores = np.sqrt(-cv_neg_mse)

        cv_records.append({
            "Model": name,
            "CV_R2_Mean": float(np.mean(cv_r2_scores)),
            "CV_R2_Std": float(np.std(cv_r2_scores)),
            "CV_RMSE_Mean": float(np.mean(cv_rmse_scores)),
            "CV_RMSE_Std": float(np.std(cv_rmse_scores)),
        })

        # 2. Train on full training split and record training time
        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t_start

        # 3. Test set inference and record prediction time
        t_pred_start = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = time.perf_counter() - t_pred_start

        # 4. Metric calculations
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        # 5. Save model artifact
        filename_slug = name.lower().replace(" ", "_")
        model_file = f"{filename_slug}.joblib"
        save_path = os.path.join(MODELS_DIR, model_file)
        joblib.dump(model, save_path)
        saved_model_paths[name] = save_path

        results_list.append({
            "Model": name,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "Training_Time_Sec": round(train_time, 4),
            "Prediction_Time_Sec": round(pred_time, 5),
            "Model_File": model_file,
        })

        print(f"{name:<28} | ${mae:<9,.0f} | ${rmse:<9,.0f} | {r2:<8.4f} | {train_time:<8.3f}")

    print("-" * 85)

    # Convert to DataFrames
    results_df = pd.DataFrame(results_list)
    cv_df = pd.DataFrame(cv_records)

    # Select Best Model using lowest held-out RMSE
    best_row = results_df.loc[results_df["RMSE"].idxmin()]
    best_model_name = best_row["Model"]

    print(f"\n[BEST MODEL SELECTED] {best_model_name} (Lowest Test RMSE: ${best_row['RMSE']:,.2f}, R²: {best_row['R2']:.4f})")

    # Save summary tables to CSV
    metrics_csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    cv_csv_path = os.path.join(RESULTS_DIR, "cv_scores.csv")
    results_df.to_csv(metrics_csv_path, index=False)
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"[SUCCESS] Metrics summary saved to: {metrics_csv_path}")
    print(f"[SUCCESS] Cross-validation scores saved to: {cv_csv_path}")

    # Save metadata JSON
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_model_name,
        "selection_criterion": "Lowest held-out Root Mean Squared Error (RMSE) on 20% test split",
        "models_evaluated": list(models.keys()),
        "test_metrics": results_df.to_dict(orient="records"),
        "cv_metrics": cv_df.to_dict(orient="records"),
        "transformed_feature_count": len(feature_names),
        "transformed_feature_names": feature_names,
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUCCESS] Model metadata saved to: {metadata_path}")

    return results_df, cv_df, best_model_name


if __name__ == "__main__":
    train_and_evaluate_all_models()
