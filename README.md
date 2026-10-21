# House Price Prediction Using Machine Learning
### Comparative Analysis of Regression Models & Interactive Automated Valuation System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-brightgreen.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-21%20passed-success.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Project Overview
This project presents an academic, production-grade machine learning system for residential real estate valuation. Using the authentic **Ames Housing Dataset** (1,460 residential transactions across 81 attributes), we evaluate and compare five distinct regression paradigms to forecast property prices:
1. **Linear Regression** (Parametric Baseline)
2. **Random Forest Regressor** (Ensemble Bagging)
3. **Support Vector Regression (SVR)** (Kernelized Machine with RBF)
4. **Gradient Boosting Regressor (GBR)** (Sequential Boosting)
5. **XGBoost Regressor** (Extreme Gradient Boosting — **Top Performer**)

The project enforces **strict zero data leakage** by fitting all imputation and standard scaling transformers solely on the 80% training split, uses **5-Fold Cross-Validation** to assess parameter stability, produces **8 publication-quality diagnostic figures**, and features a rich, responsive **Streamlit Web Application** for real-time valuation and multi-model benchmarking.

---

## 🏆 Comparative Performance Benchmark

All models were evaluated on the identical held-out 20% test split ($N_{test} = 292$ records) and validated via 5-Fold Cross-Validation:

| Regression Algorithm | Paradigm | Test MAE ($) | Test RMSE ($) | Test R² Score | Train Time (s) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **Extreme Gradient Boosting** | **$16,394.81** | **$25,634.78** | **0.9143** | **0.267s** | **1.63 ms** |
| **Gradient Boosting** | Sequential Boosting | $16,566.74 | $26,119.64 | 0.9111 | 0.702s | 1.85 ms |
| **Random Forest** | Ensemble Bagging | $17,603.63 | $29,308.19 | 0.8880 | 0.317s | 43.54 ms |
| **Linear Regression** | Baseline Parametric | $20,152.59 | $33,169.42 | 0.8566 | 0.014s | 0.36 ms |
| **Support Vector Regressor**| Kernel Machine (RBF) | $18,350.07 | $40,270.69 | 0.7886 | 0.097s | 33.78 ms |

> 🥇 **Champion Algorithm**: **XGBoost** achieved the highest $R^2$ (**0.9143**) and lowest RMSE (**$25,635**), explaining **91.4% of the variance** in residential property prices while delivering real-time inference speeds of **1.63 ms per sample**.

---

## 🏗️ Project Architecture & Directory Layout

```
ml final project/
├── app.py                          # Full-featured interactive Streamlit Web Application
├── requirements.txt                # Production Python dependencies
├── pytest.ini                      # Pytest discovery configuration
├── README.md                       # Project master documentation
├── data/
│   ├── train.csv                   # Authentic Ames Housing dataset (1,460 records)
│   └── README.md                   # Dataset provenance, attributes & limitations
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Auto-download, schema validation & California fallback
│   ├── preprocessing.py            # Zero-leakage ColumnTransformer pipeline & feature engineering
│   ├── train_models.py             # 5-model training, 5-fold CV & artifact serialization
│   ├── evaluate_models.py          # 8 publication-grade visualization figures generator
│   └── predict.py                  # Input validation & single/batch inference engine
├── models/
│   ├── preprocessing_pipeline.joblib # Fitted ColumnTransformer pipeline
│   ├── linear_regression.joblib    # Serialized Linear Regression model
│   ├── random_forest.joblib        # Serialized Random Forest model
│   ├── support_vector_regression.joblib # Serialized SVR model
│   ├── gradient_boosting.joblib    # Serialized Gradient Boosting model
│   ├── xgboost.joblib              # Serialized XGBoost model
│   └── model_metadata.json         # Training timestamp, best model & metrics metadata
├── results/
│   ├── preprocessing_summary.json  # Imputation, scaling & split statistics
│   ├── metrics_summary.csv         # Comparative held-out test evaluation table
│   └── cv_scores.csv               # 5-Fold cross-validation scores (Mean ± Std Dev)
├── figures/                        # 8 High-resolution evaluation charts
│   ├── model_mae_comparison.png    # MAE bar chart
│   ├── model_rmse_comparison.png   # RMSE bar chart
│   ├── model_r2_comparison.png     # R² score bar chart
│   ├── actual_vs_predicted.png     # Scatterplot with ideal 45° reference line
│   ├── residual_distribution.png   # Residual distribution histogram & KDE
│   ├── residuals_vs_predicted.png  # Homoscedasticity check scatterplot
│   ├── feature_importance.png      # Top 15 predictor features ranking (XGBoost)
│   └── cv_comparison.png           # Cross-validation mean RMSE ± 1 std dev
├── tests/                          # Automated Pytest suite (21 unit & integration tests)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_loader.py         # Schema, existence & fallback tests
│   ├── test_preprocessing.py       # Imputation, encoding & feature engineering tests
│   ├── test_predict.py             # Input bounds validation & multi-model inference tests
│   └── test_models.py              # Artifact integrity & benchmark threshold tests
└── reports/
    └── final_report.md             # In-depth academic capstone final report
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
Clone the repository and install the dependencies:
```bash
# Optional: create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Full Model Training Pipeline
Train all five models, perform 5-fold cross-validation, and save all joblib artifacts:
```bash
python src/train_models.py
```

### 3. Generate Publication Figures
Generate all 8 high-resolution diagnostic charts in `figures/`:
```bash
python src/evaluate_models.py
```

### 4. Execute the Test Suite
Run the 21 automated unit and integration tests:
```bash
pytest -v tests/
```

### 5. Launch the Interactive Web Application
Launch the Streamlit dashboard in your default browser:
```bash
streamlit run app.py
```

---

## 💻 Python API Usage Example

You can use the inference engine programmatically in your own scripts:

```python
from src.predict import predict_house_price, get_default_property_input

# 1. Fetch realistic baseline inputs
property_features = get_default_property_input()

# 2. Customize property parameters
property_features["GrLivArea"] = 2200        # Living area (sq ft)
property_features["OverallQual"] = 8          # Quality rating (1 to 10)
property_features["YearBuilt"] = 2010         # Construction year
property_features["Neighborhood"] = "NridgHt" # Northridge Heights

# 3. Predict price using the champion model (Auto: XGBoost)
result = predict_house_price(property_features)

print(f"Algorithm Used: {result['model_used']}")
print(f"Valuation: ${result['predicted_price_usd']:,.2f} USD")
print(f"Property Age: {result['property_age']} years")
print(f"Total Living Area: {result['total_living_sf']:,.0f} sq ft")
```

---

## ⚠️ Academic Disclosures and Domain Boundaries

To uphold scientific integrity and academic transparency, please note the following constraints:
1. **Geographic and Currency Context**:
   - The model is calibrated on property transactions from **Ames, Iowa (USA)** denominated in **US Dollars ($ USD)**.
   - Predictions cannot be transferred directly to other international real estate markets (e.g., Indian Real Estate in INR), which depend on localized variables like RERA approvals, circle rates, floor rise index, and metro transit proximity.
2. **Absence of Coastline / Ocean Views**:
   - Ames is an inland Midwestern city. There are **no ocean or coastal attributes** in this dataset. We do not fabricate synthetic coastline parameters.
3. **Constructed Residential Units vs. Vacant Plots**:
   - All records represent single-family or attached constructed homes. Valuations for vacant agricultural or commercial land are out of scope.
4. **California Housing Fallback**:
   - The codebase includes a fallback loader via Scikit-Learn's `fetch_california_housing()` for restricted offline environments. However, California data reflects 1990 district census blocks measured in $100,000 units and its metrics are never mixed with Ames Housing results.

---

## 👥 Authors and Acknowledgments
- **Dataset**: Compiled by Prof. Dean De Cock (Truman State University) and hosted by Kaggle.
- **Reference**: Kaggle House Prices: Advanced Regression Techniques.
- **Project Structure**: Conforms to reproducible machine learning research standards.
