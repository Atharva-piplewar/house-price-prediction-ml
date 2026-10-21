"""
Unit Tests for Preprocessing and Feature Engineering Module
Academic Project: House Price Prediction Using Machine Learning
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.preprocessing import (
    engineer_features,
    build_preprocessor,
    prepare_train_test_data,
    NUMERICAL_FINAL_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def sample_raw_df():
    """Provides a synthetic DataFrame matching raw schema for quick tests."""
    return pd.DataFrame({
        "GrLivArea": [1500, 2200],
        "LotArea": [9000, 11000],
        "OverallQual": [7, 8],
        "OverallCond": [5, 6],
        "YearBuilt": [2000, 1990],
        "YearRemodAdd": [2005, 1990],
        "YrSold": [2020, 2010],
        "TotalBsmtSF": [800, 1200],
        "BedroomAbvGr": [3, 4],
        "FullBath": [2, 2],
        "HalfBath": [1, 1],
        "GarageCars": [2, 3],
        "GarageArea": [500, 700],
        "Fireplaces": [1, 2],
        "HouseStyle": ["2Story", "1Story"],
        "KitchenQual": ["Gd", "Ex"],
        "ExterQual": ["Gd", "Ex"],
        "Neighborhood": ["CollgCr", "NridgHt"],
        "Street": ["Pave", "Pave"],
        "Alley": ["None", "None"],
        "Condition1": ["Norm", "Norm"],
        "LandContour": ["Lvl", "Lvl"],
        "LotConfig": ["Inside", "CulDSac"],
        "CentralAir": ["Y", "Y"],
    })


def test_engineer_features(sample_raw_df):
    """Verify feature engineering calculations."""
    engineered = engineer_features(sample_raw_df)
    
    # Check derived columns
    assert "PropertyAge" in engineered.columns
    assert "YearsSinceRemodel" in engineered.columns
    assert "IsRemodeled" in engineered.columns
    assert "TotalBath" in engineered.columns
    assert "TotalSF" in engineered.columns
    
    # Check math: Row 0: 2020 - 2000 = 20
    assert engineered["PropertyAge"].iloc[0] == 20
    # Remodel: 2020 - 2005 = 15
    assert engineered["YearsSinceRemodel"].iloc[0] == 15
    # IsRemodeled: 2005 > 2000 -> 1
    assert engineered["IsRemodeled"].iloc[0] == 1
    # Row 1: 1990 built and 1990 remod -> 0
    assert engineered["IsRemodeled"].iloc[1] == 0
    # TotalBath: 2 + 0.5*1 = 2.5
    assert engineered["TotalBath"].iloc[0] == 2.5
    # TotalSF: 1500 + 800 = 2300
    assert engineered["TotalSF"].iloc[0] == 2300


def test_preprocessor_pipeline_transformation(sample_raw_df):
    """Verify preprocessor fits and transforms without NaNs."""
    preprocessor = build_preprocessor()
    engineered = engineer_features(sample_raw_df)
    
    X_trans = preprocessor.fit_transform(engineered)
    assert isinstance(X_trans, np.ndarray)
    assert not np.isnan(X_trans).any(), "Transformed output must not contain NaN values"
    assert X_trans.shape[0] == 2


def test_unseen_categories_safe_handling(sample_raw_df):
    """Verify handle_unknown='ignore' avoids runtime errors on novel category strings."""
    preprocessor = build_preprocessor()
    engineered = engineer_features(sample_raw_df)
    preprocessor.fit(engineered)
    
    novel_df = sample_raw_df.copy()
    novel_df["Neighborhood"] = ["UnknownColony1", "UnknownColony2"]
    novel_df["HouseStyle"] = ["FuturisticTower", "TreeHouse"]
    novel_engineered = engineer_features(novel_df)
    
    # Must not raise an error
    trans = preprocessor.transform(novel_engineered)
    assert trans.shape[0] == 2
    assert not np.isnan(trans).any()


def test_prepare_train_test_split():
    """Verify train-test split shapes and zero data leakage."""
    X_train, X_test, y_train, y_test, preproc, summary, feat_names = prepare_train_test_data(
        save_artifacts=False
    )
    assert X_train.shape[0] == 1168  # 80% of 1460
    assert X_test.shape[0] == 292    # 20% of 1460
    assert len(y_train) == 1168
    assert len(y_test) == 292
    assert summary["missing_values_after_cleaning"] == 0
    assert len(feat_names) == X_train.shape[1]
