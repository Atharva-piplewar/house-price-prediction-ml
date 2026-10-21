"""
Preprocessing Module for House Price Prediction
Academic Project: House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models

This module handles:
1. Feature extraction and domain-informed feature engineering.
2. Missing value imputation (median for numerical, constant 'None' for categorical).
3. One-hot encoding of categorical variables with unseen category safety.
4. Standardization of numerical variables.
5. Construction of a Scikit-Learn ColumnTransformer pipeline.
6. Clean 80:20 train-test splitting with zero data leakage.
7. Generation of detailed preprocessing summary statistics.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_ames_housing

# Selected Raw Feature Sets
NUMERICAL_RAW_FEATURES: List[str] = [
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "YrSold",
    "TotalBsmtSF",
    "BedroomAbvGr",
    "FullBath",
    "HalfBath",
    "GarageCars",
    "GarageArea",
    "Fireplaces",
]

CATEGORICAL_FEATURES: List[str] = [
    "HouseStyle",
    "KitchenQual",
    "ExterQual",
    "Neighborhood",
    "Street",
    "Alley",
    "Condition1",
    "LandContour",
    "LotConfig",
    "CentralAir",
]

# Features after domain feature engineering
NUMERICAL_FINAL_FEATURES: List[str] = [
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "TotalBsmtSF",
    "BedroomAbvGr",
    "FullBath",
    "HalfBath",
    "GarageCars",
    "GarageArea",
    "Fireplaces",
    "PropertyAge",
    "YearsSinceRemodel",
    "IsRemodeled",
    "TotalBath",
    "TotalSF",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain feature engineering to raw housing data.
    
    Derives:
    - PropertyAge: Calculated as YrSold - YearBuilt (or CurrentYear - YearBuilt for inference)
    - YearsSinceRemodel: Calculated as YrSold - YearRemodAdd
    - IsRemodeled: Binary flag (1 if YearRemodAdd > YearBuilt else 0)
    - TotalBath: FullBath + 0.5 * HalfBath
    - TotalSF: GrLivArea + TotalBsmtSF
    """
    data = df.copy()
    
    # Use YrSold if present; otherwise default to reference year (2026 or 2010)
    current_year = 2026
    sold_year = data["YrSold"] if "YrSold" in data.columns else current_year
    
    # Calculate property age (prevent negative values from data entry discrepancies)
    data["PropertyAge"] = np.maximum(0, sold_year - data["YearBuilt"])
    data["YearsSinceRemodel"] = np.maximum(0, sold_year - data["YearRemodAdd"])
    data["IsRemodeled"] = (data["YearRemodAdd"] > data["YearBuilt"]).astype(int)
    
    # Composite living metrics
    data["TotalBath"] = data["FullBath"] + 0.5 * data["HalfBath"]
    data["TotalSF"] = data["GrLivArea"] + data["TotalBsmtSF"].fillna(0)
    
    # Keep only the engineered numerical and categorical features
    all_features = NUMERICAL_FINAL_FEATURES + CATEGORICAL_FEATURES
    return data[all_features]


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a Scikit-Learn ColumnTransformer pipeline.
    
    Pipeline guarantees:
    - Numerical: Median imputation followed by standard scaling.
    - Categorical: Constant imputation ('None' representing absence) followed by
      One-Hot Encoding with handle_unknown='ignore' to safely handle previously unseen categories.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FINAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_train_test_data(
    data_path: str = None,
    test_size: float = 0.20,
    random_state: int = 42,
    save_artifacts: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, Dict[str, Any], List[str]]:
    """
    Loads raw data, executes feature engineering, splits into train/test sets,
    fits the preprocessor ONLY on the training split to eliminate data leakage,
    and returns transformed arrays along with a comprehensive summary.
    """
    raw_df = load_ames_housing(data_path)
    
    # Exclude ID column and separate target
    if "Id" in raw_df.columns:
        df_no_id = raw_df.drop(columns=["Id"])
    else:
        df_no_id = raw_df.copy()
        
    y = df_no_id["SalePrice"].copy()
    
    # Raw feature matrix
    required_cols = NUMERICAL_RAW_FEATURES + CATEGORICAL_FEATURES
    X_raw = df_no_id[required_cols].copy()
    
    # Inspect missing values in raw subset
    missing_before = X_raw.isnull().sum()
    missing_before_dict = {col: int(cnt) for col, cnt in missing_before.items() if cnt > 0}
    
    # Domain feature engineering
    X_engineered = engineer_features(X_raw)
    
    # Strict 80:20 Train-Test Split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=test_size, random_state=random_state
    )
    
    # Instantiate and fit preprocessor on TRAINING DATA ONLY (Zero Data Leakage)
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract feature names after one-hot encoding for interpretability
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_transformed_feature_names = NUMERICAL_FINAL_FEATURES + cat_feature_names
    
    # Preprocessing Summary
    summary = {
        "dataset_name": "Ames Housing Dataset (Kaggle Advanced Regression)",
        "total_raw_samples": int(len(raw_df)),
        "total_raw_columns": int(len(raw_df.columns)),
        "selected_numerical_features_raw": len(NUMERICAL_RAW_FEATURES),
        "selected_categorical_features": len(CATEGORICAL_FEATURES),
        "engineered_numerical_features": len(NUMERICAL_FINAL_FEATURES),
        "total_transformed_features": int(X_train_transformed.shape[1]),
        "train_samples": int(X_train_transformed.shape[0]),
        "test_samples": int(X_test_transformed.shape[0]),
        "train_split_ratio": round(1.0 - test_size, 2),
        "test_split_ratio": round(test_size, 2),
        "random_state": random_state,
        "missing_values_before_cleaning": missing_before_dict,
        "missing_values_after_cleaning": 0,
        "target_variable": "SalePrice (USD)",
        "target_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "median": float(y.median()),
            "min": float(y.min()),
            "max": float(y.max()),
            "q25": float(y.quantile(0.25)),
            "q75": float(y.quantile(0.75)),
        },
        "preprocessing_decisions": [
            "Numerical features imputed with training set median to resist skew/outliers.",
            "Numerical features standardized with StandardScaler (zero mean, unit variance).",
            "Categorical missing values imputed with constant 'None' to denote absence of feature (e.g. Alley).",
            "Categoricals encoded using OneHotEncoder with handle_unknown='ignore' to guarantee safe inference on unseen categories.",
            "Transformations fitted strictly on 80% training split to eliminate data leakage.",
            "Derived property age and remodel age from construction years to reflect physical depreciation.",
        ],
    }
    
    if save_artifacts:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save fitted preprocessor
        pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
        joblib.dump(preprocessor, pipeline_path)
        print(f"[SUCCESS] Saved preprocessing pipeline to: {pipeline_path}")
        
        # Save summary JSON
        summary_path = os.path.join(results_dir, "preprocessing_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[SUCCESS] Saved preprocessing summary to: {summary_path}")
        
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, summary, all_transformed_feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preproc, summary, feat_names = prepare_train_test_data()
    print("\n--- Preprocessing Execution Complete ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of encoded feature dimensions: {len(feat_names)}")
    print(f"Sample encoded features: {feat_names[:8]}")
