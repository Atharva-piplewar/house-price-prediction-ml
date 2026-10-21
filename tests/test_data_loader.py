"""
Unit Tests for Data Loader Module
Academic Project: House Price Prediction Using Machine Learning
"""

import os
import pytest
import pandas as pd
from src.data_loader import load_ames_housing, load_california_fallback, LOCAL_DATA_FILE


def test_ames_housing_file_exists():
    """Verify train.csv is present in data directory."""
    assert os.path.exists(LOCAL_DATA_FILE), f"Dataset file missing at {LOCAL_DATA_FILE}"


def test_load_ames_housing_schema():
    """Verify Ames Housing dataset loads with valid shape and target column."""
    df = load_ames_housing()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1460, f"Expected 1460 records, got {len(df)}"
    assert "SalePrice" in df.columns, "Target column 'SalePrice' must be present"
    assert "GrLivArea" in df.columns
    assert "OverallQual" in df.columns
    assert "YearBuilt" in df.columns
    assert df["SalePrice"].min() > 0, "SalePrice must be strictly positive"


def test_california_fallback():
    """Verify fallback loader returns valid feature DataFrame and target Series."""
    X, y = load_california_fallback()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 1000
    assert "MedInc" in X.columns
