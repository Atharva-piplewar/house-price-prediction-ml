"""
Unit Tests for Inference and Prediction Module
Academic Project: House Price Prediction Using Machine Learning
"""

import pytest
from src.predict import (
    validate_input,
    predict_house_price,
    get_default_property_input,
    load_model_metadata,
)


def test_validate_input_valid():
    """Verify standard property input passes validation."""
    sample = get_default_property_input()
    validated = validate_input(sample)
    assert validated["GrLivArea"] == 1500


def test_validate_input_negative_area():
    """Verify input validation fails on negative area."""
    bad_sample = get_default_property_input()
    bad_sample["GrLivArea"] = -100
    with pytest.raises(ValueError, match="Input Validation Error"):
        validate_input(bad_sample)


def test_validate_input_invalid_quality():
    """Verify input validation fails on quality out of 1-10 range."""
    bad_sample = get_default_property_input()
    bad_sample["OverallQual"] = 15
    with pytest.raises(ValueError, match="Overall Quality"):
        validate_input(bad_sample)


def test_validate_input_remodel_before_built():
    """Verify input validation fails if remodel year < built year."""
    bad_sample = get_default_property_input()
    bad_sample["YearBuilt"] = 2005
    bad_sample["YearRemodAdd"] = 1995
    with pytest.raises(ValueError, match="Year of Remodeling"):
        validate_input(bad_sample)


def test_predict_house_price_default():
    """Verify end-to-end prediction returns expected fields and reasonable price."""
    sample = get_default_property_input()
    res = predict_house_price(sample)
    
    assert "predicted_price_usd" in res
    assert "model_used" in res
    assert "property_age" in res
    assert "total_living_sf" in res
    assert res["predicted_price_usd"] > 30000, f"Unrealistically low price: {res['predicted_price_usd']}"
    assert res["predicted_price_usd"] < 1000000, f"Unrealistically high price: {res['predicted_price_usd']}"


@pytest.mark.parametrize("model_name", [
    "Linear Regression",
    "Random Forest",
    "Support Vector Regression",
    "Gradient Boosting",
    "XGBoost",
])
def test_predict_with_each_model(model_name):
    """Verify prediction works with each of the 5 saved models."""
    sample = get_default_property_input()
    res = predict_house_price(sample, model_name=model_name)
    assert res["model_used"] == model_name
    assert res["predicted_price_usd"] > 10000
