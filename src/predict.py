"""
Inference and Prediction Module for House Price Prediction
Academic Project: House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models

This module provides:
1. Input validation for property characteristics.
2. Feature transformation through the saved preprocessing pipeline.
3. Inference using any of the 5 saved models or the automatically selected best model.
4. Consistent prediction format denominated in USD ($).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import engineer_features

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def load_model_metadata() -> Dict[str, Any]:
    """Loads metadata describing available models and the selected best model."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Model metadata not found at {METADATA_PATH}. Train models first.")
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def load_model(model_name: Optional[str] = None):
    """
    Loads a specified model artifact by name, or defaults to the best performing model.
    """
    metadata = load_model_metadata()
    if model_name is None:
        model_name = metadata["best_model"]
        
    slug = model_name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Available models: {metadata['models_evaluated']}")
        
    model = joblib.load(model_path)
    return model, model_name


def load_preprocessor():
    """Loads the fitted ColumnTransformer pipeline."""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessing pipeline not found at {PREPROCESSOR_PATH}.")
    return joblib.load(PREPROCESSOR_PATH)


def validate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates property details dictionary against structural constraints.
    """
    errors = []
    
    # Area checks
    if data.get("GrLivArea", 0) <= 0:
        errors.append("Above-ground living area (GrLivArea) must be greater than 0.")
    if data.get("LotArea", 0) <= 0:
        errors.append("Lot area (LotArea) must be greater than 0.")
        
    # Quality and Condition rating checks (1 to 10 scale)
    if not (1 <= data.get("OverallQual", 0) <= 10):
        errors.append("Overall Quality (OverallQual) must be between 1 and 10.")
    if not (1 <= data.get("OverallCond", 0) <= 10):
        errors.append("Overall Condition (OverallCond) must be between 1 and 10.")
        
    # Temporal checks
    year_built = data.get("YearBuilt", 0)
    year_remod = data.get("YearRemodAdd", year_built)
    current_year = 2026
    
    if not (1850 <= year_built <= current_year):
        errors.append(f"Year Built must be between 1850 and {current_year}.")
    if year_remod < year_built:
        errors.append("Year of Remodeling cannot be earlier than Year Built.")
        
    # Count checks
    if data.get("BedroomAbvGr", 0) < 0:
        errors.append("Bedrooms cannot be negative.")
    if data.get("FullBath", 0) < 0:
        errors.append("Full Bathrooms cannot be negative.")
    if data.get("GarageCars", 0) < 0:
        errors.append("Garage car capacity cannot be negative.")

    if errors:
        raise ValueError("Input Validation Error(s):\n - " + "\n - ".join(errors))
        
    return data


def predict_house_price(
    property_features: Dict[str, Any],
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predicts house sale price for a single property dictionary.
    
    Args:
        property_features: Dictionary of raw property inputs.
        model_name: Optional specific model name; defaults to best model.
        
    Returns:
        Dictionary containing:
        - predicted_price_usd: Rounded prediction in USD.
        - model_used: Name of model algorithm.
        - property_age: Derived age of the house.
        - total_living_sf: Combined above-ground and basement square footage.
    """
    # 1. Validate inputs
    validated = validate_input(property_features.copy())
    
    # 2. Build single-row DataFrame
    input_df = pd.DataFrame([validated])
    
    # Ensure default reference year for sold year if missing
    if "YrSold" not in input_df.columns:
        input_df["YrSold"] = 2026
        
    # 3. Apply exact domain feature engineering used during training
    engineered_df = engineer_features(input_df)
    
    # 4. Transform through fitted preprocessing pipeline
    preprocessor = load_preprocessor()
    transformed_features = preprocessor.transform(engineered_df)
    
    # 5. Load model and predict
    model, selected_model_name = load_model(model_name)
    prediction = model.predict(transformed_features)[0]
    
    # Ensure non-negative valuation
    predicted_val = float(max(10000.0, prediction))
    
    return {
        "predicted_price_usd": round(predicted_val, 2),
        "model_used": selected_model_name,
        "property_age": int(engineered_df["PropertyAge"].iloc[0]),
        "total_living_sf": float(engineered_df["TotalSF"].iloc[0]),
        "total_bathrooms": float(engineered_df["TotalBath"].iloc[0]),
        "currency": "USD ($)",
    }


def get_default_property_input() -> Dict[str, Any]:
    """Returns a realistic median sample property for testing and default UI values."""
    return {
        "GrLivArea": 1500,
        "LotArea": 9500,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2005,
        "YrSold": 2026,
        "TotalBsmtSF": 900,
        "BedroomAbvGr": 3,
        "FullBath": 2,
        "HalfBath": 1,
        "GarageCars": 2,
        "GarageArea": 500,
        "Fireplaces": 1,
        "HouseStyle": "2Story",
        "KitchenQual": "Gd",
        "ExterQual": "Gd",
        "Neighborhood": "CollgCr",
        "Street": "Pave",
        "Alley": "None",
        "Condition1": "Norm",
        "LandContour": "Lvl",
        "LotConfig": "Inside",
        "CentralAir": "Y",
    }


if __name__ == "__main__":
    sample = get_default_property_input()
    res = predict_house_price(sample)
    print("\n--- Sample Prediction Result ---")
    print(f"Model Used: {res['model_used']}")
    print(f"Predicted Price: ${res['predicted_price_usd']:,.2f} USD")
    print(f"Property Age: {res['property_age']} years")
    print(f"Total Living Area: {res['total_living_sf']:,.0f} sq ft")
