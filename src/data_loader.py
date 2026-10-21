"""
Data Loader Module for House Price Prediction Project
Dataset: Ames Housing Dataset (Kaggle Advanced Regression Techniques)

This module handles:
1. Downloading the Ames Housing dataset from verified remote sources if not already present.
2. Loading and validating the raw dataset schema.
3. Fallback support for California Housing dataset with explicit documentation of differences.
"""

import os
import urllib.request
import pandas as pd
from typing import Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOCAL_DATA_FILE = os.path.join(DATA_DIR, "train.csv")

# Reliable mirror of the authentic Kaggle Ames Housing train.csv (1460 rows, 81 columns)
PRIMARY_DATA_URL = "https://raw.githubusercontent.com/krishnaik06/Advanced-House-Price-Prediction-/master/train.csv"
KAGGLE_SOURCE_URL = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"


def download_dataset(target_path: str = LOCAL_DATA_FILE, source_url: str = PRIMARY_DATA_URL) -> str:
    """
    Downloads the Ames Housing dataset if it is not already available locally.
    
    Args:
        target_path: Local path where train.csv should be stored.
        source_url: URL to download the dataset from.
        
    Returns:
        Path to the saved CSV file.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        print(f"[INFO] Dataset already exists locally at: {target_path}")
        return target_path
    
    print(f"[INFO] Downloading Ames Housing dataset from: {source_url}")
    try:
        # User-agent header to prevent HTTP 403 blocks
        req = urllib.request.Request(
            source_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"[SUCCESS] Dataset successfully saved to: {target_path}")
        return target_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset from {source_url}. Error: {e}\n"
            f"Manual Setup: Please download 'train.csv' from Kaggle:\n"
            f"{KAGGLE_SOURCE_URL}\n"
            f"and place it inside the 'data/' folder as '{target_path}'."
        )


def load_ames_housing(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates the Ames Housing dataset.
    
    Args:
        data_path: Optional local path to train.csv. If None, uses default.
        
    Returns:
        DataFrame containing the raw Ames Housing records.
    """
    if data_path is None:
        data_path = LOCAL_DATA_FILE
    
    if not os.path.exists(data_path):
        download_dataset(data_path)
    
    df = pd.read_csv(data_path)
    
    # Schema validation
    if "SalePrice" not in df.columns:
        raise ValueError("Invalid dataset schema: Target column 'SalePrice' not found in dataset.")
    if len(df) == 0:
        raise ValueError("Dataset is empty.")
        
    print(f"[INFO] Loaded Ames Housing dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def load_california_fallback() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fallback loader for California Housing dataset via scikit-learn.
    
    IMPORTANT LIMITATION & DIFFERENCE:
    - California Housing represents census block groups (districts) from 1990, NOT individual homes.
    - Target 'MedHouseVal' is measured in $100,000 units (e.g., 2.0 = $200,000), not exact USD.
    - Features represent district-level aggregates (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude).
    - It lacks property-specific features (basement, garage, renovation year, neighborhood style).
    - Evaluation metrics MUST NOT be mixed with Ames Housing results.
    """
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing(as_frame=True)
    df = california.data.copy()
    y = california.target.copy()
    print("[WARNING] California Housing fallback invoked. Features and target scale differ from Ames Housing.")
    return df, y


if __name__ == "__main__":
    df = load_ames_housing()
    print("\n--- Dataset Summary ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Target 'SalePrice' Mean: ${df['SalePrice'].mean():,.2f}")
    print(f"Target 'SalePrice' Median: ${df['SalePrice'].median():,.2f}")
    print(f"Target 'SalePrice' Min: ${df['SalePrice'].min():,.2f}")
    print(f"Target 'SalePrice' Max: ${df['SalePrice'].max():,.2f}")
    print("\nFirst 5 Rows Preview:")
    print(df[["Id", "OverallQual", "GrLivArea", "GarageCars", "YearBuilt", "SalePrice"]].head())
